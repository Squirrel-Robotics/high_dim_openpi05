"""Offline 58D policy evaluation against stored absolute target chunks.

Use --checkpoint for JAX inference, or --predictions for an existing NPZ archive.
Both modes score physical (unnormalized) actions, never send robot commands, and
use only complete valid samples. See README.md for input/output conventions.
"""

import argparse
import dataclasses
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from openpi.shared import teleop_raw

if __package__:
    from . import metrics
    from . import validate_dataset
else:  # Support "python examples/umi/evaluate.py" as well as python -m.
    import metrics
    import validate_dataset


@dataclasses.dataclass(frozen=True)
class Samples:
    ids: np.ndarray  # [N,2]: episode_index, frame_index, not filtered row indices.
    states: np.ndarray
    targets: np.ndarray
    task_indices: np.ndarray


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Load data-only NPZ; require explicit action space and unique original row IDs."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"episode_index", "frame_index", "actions", "action_space"}
        if not required.issubset(archive.files):
            raise ValueError(f"Prediction archive needs keys {sorted(required)}.")
        episode, frame = archive["episode_index"], archive["frame_index"]
        actions, space = archive["actions"], archive["action_space"]
    if episode.ndim != 1 or frame.shape != episode.shape or len(episode) == 0:
        raise ValueError("Prediction episode_index/frame_index must be nonempty vectors of the same length.")
    if any(value.dtype.kind not in "iu" or np.any(value < 0) for value in (episode, frame)):
        raise ValueError("Prediction row IDs must be nonnegative integers.")
    ids = np.stack((episode, frame), axis=-1)
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError("Prediction archive contains duplicate episode/frame IDs.")
    if actions.shape != (len(ids), 50, 58) or actions.dtype.kind not in "fiu" or not np.isfinite(actions).all():
        raise ValueError("Prediction actions must be finite numeric [N,50,58] in physical units.")
    if space.shape != () or space.dtype.kind != "U" or str(space) not in ("absolute", "relative"):
        raise ValueError("action_space must be a scalar string 'absolute' or 'relative'.")
    return ids, actions, str(space)


def select_samples(
    root: Path,
    *,
    episodes: list[int] | None = None,
    stride: int = 1,
    max_samples: int = 100,
    prediction_ids: np.ndarray | None = None,
) -> Samples:
    """Choose deterministic rows, preserving original episode/frame identities.

    max_samples=0 means all; otherwise evenly subsample the eligible rows after
    applying the per-episode valid-row stride. Archive mode only selects IDs
    actually present in the archive, but rejects every unknown/invalid archive ID.
    """
    if stride < 1 or max_samples < 0:
        raise ValueError("stride must be positive and max_samples must be nonnegative.")
    info, episode_table, _ = validate_dataset.read_metadata(root)
    episode_ids = {row["episode_index"] for row in episode_table}
    requested = episode_ids if episodes is None else set(episodes)
    if not requested or not requested.issubset(episode_ids):
        raise ValueError(f"Requested episodes must be present in this dataset: {sorted(episode_ids)}.")
    archive_ids = None if prediction_ids is None else {tuple(map(int, row)) for row in prediction_ids}
    all_valid, candidates = set(), []
    for episode in episode_table:
        ep = episode["episode_index"]
        data = validate_dataset.load_episode(
            root, info, ep, columns=["frame_index", "sample_valid_h50", "action_is_pad"]
        )
        mask = (data["sample_valid_h50"].reshape(-1) == 1) & ~data["action_is_pad"].astype(bool).any(axis=1)
        valid_ids = [(ep, int(frame)) for frame in data["frame_index"].reshape(-1)[mask]]
        all_valid.update(valid_ids)
        if ep in requested:
            candidates.extend(key for key in valid_ids[::stride] if archive_ids is None or key in archive_ids)
    if archive_ids is not None and not archive_ids.issubset(all_valid):
        bad = sorted(archive_ids - all_valid)[:5]
        raise ValueError(f"Predictions reference unknown, padded, or invalid dataset rows: {bad}.")
    if not candidates:
        raise ValueError("No complete valid samples match the requested selection.")
    if max_samples and len(candidates) > max_samples:
        selected = np.linspace(0, len(candidates) - 1, max_samples, dtype=np.int64)
        candidates = [candidates[i] for i in selected]
    ids = np.asarray(candidates, dtype=np.int64)
    states, targets, task_indices = [], [], []
    for ep in np.unique(ids[:, 0]):
        data = validate_dataset.load_episode(root, info, int(ep))
        frames = ids[ids[:, 0] == ep, 1]
        # Validation established frame_index == arange(length), within each episode.
        states.extend(data["observation.state"][frames])
        targets.extend(data["action"][frames])
        task_indices.extend(data["task_index"].reshape(-1)[frames])
    return Samples(ids, np.asarray(states), np.asarray(targets), np.asarray(task_indices, dtype=np.int64))


def make_observation(record: dict, prompt: str) -> dict:
    """Build the existing Teleop adapter input contract, excluding all target data."""
    observation = {
        "observation/state": np.asarray(record["observation.state"]).copy(),
        "prompt": prompt,
    }
    destinations = ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image")
    for key, destination in zip(teleop_raw.CAMERA_KEYS, destinations, strict=True):
        image = np.asarray(record[key])
        # LeRobot video loader returns float CHW in [0,1]; clients use uint8 HWC.
        if image.ndim != 3:
            raise ValueError(f"{key}: expected a three-dimensional RGB image.")
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.shape[-1] != 3:
            raise ValueError(f"{key}: expected three RGB channels.")
        if image.dtype.kind == "f":
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f"{key}: floating images must be finite in [0,1].")
            image = np.rint(image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 or floating image.")
        observation[destination] = np.ascontiguousarray(image).copy()
    return observation


def infer_samples(
    policy, dataset, samples: Samples, tasks: dict[int, str], *, seed: int = 0
) -> tuple[np.ndarray, list]:
    """Only observations enter infer(); deterministic per-row Gaussian noise."""
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    # The raw LeRobot dataset is not filtered. Resolve IDs using its stored columns
    # instead of assuming a filtered sample position is a physical video frame.
    rows = dataset.hf_dataset.select_columns(["episode_index", "frame_index"]).with_format("numpy")[:]
    ids = zip(rows["episode_index"].reshape(-1), rows["frame_index"].reshape(-1), strict=True)
    lookup = {tuple(map(int, key)): i for i, key in enumerate(ids)}
    predictions, elapsed_ms = [], []
    previous_episode = None
    for index, ((ep, frame), task_index) in enumerate(zip(samples.ids, samples.task_indices, strict=True)):
        if previous_episode != int(ep) and hasattr(policy, "reset"):
            policy.reset()
        previous_episode = int(ep)
        record = dataset[lookup[(int(ep), int(frame))]]
        # Detect a reader/index mismatch before sending the wrong images/state to a model.
        if not np.allclose(np.asarray(record["observation.state"]), samples.states[index], atol=1e-6, rtol=0):
            raise ValueError(f"LeRobot state does not match selected Parquet row {(int(ep), int(frame))}.")
        observation = make_observation(record, tasks[int(task_index)])
        noise = np.random.default_rng(np.random.SeedSequence([seed, int(ep), int(frame)])).standard_normal(
            (50, 58), dtype=np.float32
        )
        start = time.perf_counter()
        result = policy.infer(observation, noise=noise)
        if not isinstance(result, dict) or "actions" not in result:
            raise ValueError(f"Policy output at {(int(ep), int(frame))} must contain an actions array.")
        actions = np.asarray(result["actions"])
        elapsed_ms.append((time.perf_counter() - start) * 1000)
        if actions.shape != (50, 58) or actions.dtype.kind not in "fiu" or not np.isfinite(actions).all():
            raise ValueError(f"Policy output at {(int(ep), int(frame))} must be finite [50,58].")
        predictions.append(actions.copy())
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(samples.ids):
            print(f"Inferred {index + 1}/{len(samples.ids)} samples.", flush=True)
    return np.asarray(predictions), elapsed_ms


def load_checkpoint_policy(checkpoint: Path, config_name: str, repo_id: str, num_steps: int):
    """Use the training policy pipeline and checkpoint-owned normalization stats."""
    # Lazy imports keep validation and prediction-file evaluation CPU-only.
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.shared import normalize  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    config = training_config.get_config(config_name)
    if config.model.action_dim != 58 or config.model.action_horizon != 50 or not getattr(config.model, "pi05", False):
        raise ValueError("Evaluation requires a pi0.5 config with action_dim=58 and action_horizon=50.")
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=repo_id))
    try:
        data_config = config.data.create(config.assets_dirs, config.model)
    except NotImplementedError as error:
        raise ValueError(
            "The Teleop policy adapter is not connected yet. Connect the same input/output transforms used in "
            "training before checkpoint evaluation; --predictions can already evaluate exported physical actions."
        ) from error
    if not (checkpoint / "params").is_dir() or (checkpoint / "model.safetensors").exists():
        raise ValueError("checkpoint must be a JAX checkpoint directory containing params/.")
    if not data_config.asset_id:
        raise ValueError("The training config must identify the checkpoint's normalization asset_id.")
    stats = normalize.load(checkpoint / "assets" / data_config.asset_id)
    fields = ("mean", "std", "q01", "q99") if data_config.use_quantile_norm else ("mean", "std")
    for name in ("state", "actions"):
        if name not in stats:
            raise ValueError(f"Checkpoint normalization stats are missing {name}.")
        for field in fields:
            value = np.asarray(getattr(stats[name], field))
            if value.shape != (58,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"Checkpoint {name}.{field} must be finite [58].")
        if np.any(stats[name].std < 0):
            raise ValueError(f"Checkpoint {name}.std cannot be negative.")
        if data_config.use_quantile_norm and np.any(stats[name].q01 > stats[name].q99):
            raise ValueError(f"Checkpoint {name} quantiles are reversed.")
    return policy_config.create_trained_policy(
        config, checkpoint, norm_stats=stats, sample_kwargs={"num_steps": num_steps}
    )


def _score(samples: Samples, actions: np.ndarray, prediction_space: str) -> dict:
    baseline = np.repeat(samples.states[:, None, :], 50, axis=1)
    result = {
        "model": metrics.compute_metrics(actions, samples.targets, samples.states, prediction_space=prediction_space),
        "hold_current_state_baseline": metrics.compute_metrics(baseline, samples.targets, samples.states),
        "per_episode": [],
    }
    for ep in np.unique(samples.ids[:, 0]):
        mask = samples.ids[:, 0] == ep
        result["per_episode"].append(
            {
                "episode_index": int(ep),
                "model": metrics.compute_metrics(
                    actions[mask], samples.targets[mask], samples.states[mask], prediction_space=prediction_space
                ),
                "hold_current_state_baseline": metrics.compute_metrics(
                    baseline[mask], samples.targets[mask], samples.states[mask]
                ),
            }
        )
    return result


def evaluate(
    dataset_dir: Path,
    output_dir: Path,
    *,
    predictions: Path | None = None,
    checkpoint: Path | None = None,
    config_name: str = "pi05_teleop_58d",
    repo_id: str = "local/teleop58",
    prediction_space: str | None = None,
    episodes: list[int] | None = None,
    stride: int = 1,
    max_samples: int = 100,
    seed: int = 0,
    num_steps: int = 10,
    skip_videos: bool = False,
) -> dict:
    if (predictions is None) == (checkpoint is None):
        raise ValueError("Specify exactly one of predictions or checkpoint.")
    if checkpoint is not None and prediction_space not in ("absolute", "relative"):
        raise ValueError("Checkpoint evaluation requires explicit --prediction-space absolute or relative.")
    if predictions is not None and prediction_space is not None:
        raise ValueError("Prediction archive already declares action_space; do not override --prediction-space.")
    if seed < 0 or num_steps < 1:
        raise ValueError("seed must be nonnegative and num_steps must be positive.")
    root, output = Path(dataset_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Evaluation output must be outside the source dataset directory tree.")
    validation = validate_dataset.validate_dataset(root, check_videos=not skip_videos)
    _, _, tasks = validate_dataset.read_metadata(root)
    archive_ids, archive_actions = None, None
    if predictions is not None:
        archive_ids, archive_actions, prediction_space = load_predictions(Path(predictions))
    samples = select_samples(
        root, episodes=episodes, stride=stride, max_samples=max_samples, prediction_ids=archive_ids
    )
    elapsed_ms = []
    if predictions is not None:
        lookup = {tuple(map(int, key)): i for i, key in enumerate(archive_ids)}
        actions = archive_actions[[lookup[tuple(map(int, key))] for key in samples.ids]]
    else:
        policy = load_checkpoint_policy(Path(checkpoint).expanduser().resolve(), config_name, repo_id, num_steps)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=None)
        actions, elapsed_ms = infer_samples(policy, dataset, samples, tasks, seed=seed)
    report = {
        "evaluation": "offline_chunk_prediction",
        "action_space": prediction_space,
        "dataset_dir": str(root),
        "source": str(Path(predictions if predictions is not None else checkpoint).expanduser().resolve()),
        "source_type": "prediction_archive" if predictions is not None else "jax_checkpoint",
        "config_name": config_name if checkpoint is not None else None,
        "selection": {
            "episodes": sorted(set(samples.ids[:, 0].tolist())),
            "stride": stride,
            "max_samples": max_samples,
            "num_samples": len(samples.ids),
            "archive_samples": None if archive_ids is None else len(archive_ids),
            "sample_ids": samples.ids.tolist(),
        },
        "seed": seed if checkpoint is not None else None,
        "denoising_steps": num_steps if checkpoint is not None else None,
        "latency_ms": None
        if not elapsed_ms
        else {
            "first_call_including_compilation": elapsed_ms[0],
            "subsequent_mean": float(np.mean(elapsed_ms[1:])) if len(elapsed_ms) > 1 else None,
        },
        "validation": validation,
        "limits": [
            "Offline errors are not robot task success rates.",
            "Overlapping action chunks are correlated; metrics weight each selected chunk equally.",
            "Train/validation separation must be established by episode before training.",
            "Hand errors retain native dataset units.",
        ],
        **_score(samples, actions, prediction_space),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(
            staging / "predictions.npz",
            episode_index=samples.ids[:, 0],
            frame_index=samples.ids[:, 1],
            actions=actions,
            action_space=np.asarray(prediction_space),
        )
        if output.exists():
            raise FileExistsError(f"Evaluation output appeared during scoring: {output}")
        staging.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Local JAX checkpoint with params/ and assets/.")
    source.add_argument("--predictions", type=Path, help="NPZ containing row IDs, actions, and action_space.")
    parser.add_argument("--config-name", default="pi05_teleop_58d")
    parser.add_argument(
        "--repo-id", default="local/teleop58", help="Identifier used with the explicit local dataset root."
    )
    parser.add_argument(
        "--prediction-space",
        choices=("absolute", "relative"),
        help="Required for checkpoint mode; physical policy output convention.",
    )
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=100, help="Evenly select at most N rows; 0 means all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10, help="Flow-matching sampling steps.")
    parser.add_argument(
        "--skip-videos", action="store_true", help="Skip video probing; inference still decodes images."
    )
    args = parser.parse_args()
    try:
        report = evaluate(**vars(args))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(f"Evaluated {report['selection']['num_samples']} valid samples. Results: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
