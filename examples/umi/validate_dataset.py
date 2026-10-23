"""Validate the local 58D absolute-EEF LeRobot v2.1 dataset contract.

This checks stored data and geometry before training/evaluation. It does not
verify robot calibration, the semantic meaning of recorded targets, or safety.
Rotation6D uses the first two columns of each rotation matrix, concatenated.
The three video streams must already be resampled onto the dataset frame grid.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import string
import subprocess
import sys

import numpy as np

from openpi.shared import teleop_raw
from openpi.shared import umi_pose

CAMERA_KEYS = teleop_raw.CAMERA_KEYS
HORIZON = 50
ACTION_DIM = 58
_SHAPES = {
    "observation.state": [58],
    "action": [50, 58],
    "action_is_pad": [50],
    "sample_valid_h50": [1],
    "timestamp": [1],
    "frame_index": [1],
    "episode_index": [1],
    "index": [1],
    "task_index": [1],
}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain nonempty JSON object rows.")
    return rows


def _integer(value, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}.")
    return value


def dataset_path(root: Path, info: dict, template_key: str, episode_index: int, video_key: str | None = None) -> Path:
    """Resolve a metadata path, rejecting absolute paths and directory escapes."""
    root = Path(root).expanduser().resolve()
    template = info.get(template_key)
    if not isinstance(template, str) or not template:
        raise ValueError(f"info.{template_key} must be a nonempty path template.")
    values = {
        "episode_index": _integer(episode_index, "episode_index"),
        "episode_chunk": episode_index // _integer(info.get("chunks_size"), "chunks_size", minimum=1),
        "video_key": video_key,
    }
    try:
        for _, field, _, _ in string.Formatter().parse(template):
            if field is not None and (field not in values or values[field] is None):
                raise ValueError(f"Unsupported or unavailable path field: {field!r}.")
        relative = Path(template.format(**values))
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid info.{template_key}: {error}") from error
    result = (root / relative).resolve()
    if relative.is_absolute() or not result.is_relative_to(root) or result == root:
        raise ValueError(f"info.{template_key} must resolve inside the dataset: {template!r}.")
    return result


def _video_shape(feature: dict, key: str) -> tuple[int, int]:
    shape, names = feature.get("shape"), feature.get("names")
    if (
        feature.get("dtype") != "video"
        or not isinstance(shape, list)
        or len(shape) != 3
        or not isinstance(names, list)
        or any(not isinstance(name, str) for name in names)
        or sorted(names) != ["channel", "height", "width"]
    ):
        raise ValueError(f"{key} must be a video feature with named height, width, channel axes.")
    axes = dict(zip(names, shape, strict=True))
    for name, size in axes.items():
        _integer(size, f"{key}.{name}", minimum=1)
    if axes["channel"] != 3:
        raise ValueError(f"{key} must have three RGB channels.")
    return axes["width"], axes["height"]


def read_metadata(dataset_dir: Path) -> tuple[dict, list[dict], dict[int, str]]:
    """Read and validate metadata without importing LeRobot, Torch, or JAX."""
    root = Path(dataset_dir).expanduser().resolve()
    info = _read_json(root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Expected LeRobot codebase_version='v2.1'.")
    for key in ("total_episodes", "total_frames", "total_tasks", "chunks_size", "total_chunks"):
        _integer(info.get(key), f"info.{key}", minimum=1)
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("info.fps must be finite and positive.")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.features must be an object.")
    for key, shape in _SHAPES.items():
        if not isinstance(features.get(key), dict) or features[key].get("shape") != shape:
            raise ValueError(f"{key} feature must have shape {shape}.")
        dtype = features[key].get("dtype", "")
        allowed = ("float",) if key in ("observation.state", "action", "timestamp") else ("int", "uint", "bool")
        if not isinstance(dtype, str) or not dtype.startswith(allowed):
            raise ValueError(f"{key} has incompatible dtype {dtype!r}.")
    for key in CAMERA_KEYS:
        if not isinstance(features.get(key), dict):
            raise ValueError(f"Missing video feature {key}.")
        _video_shape(features[key], key)
    video_keys = {key for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "video"}
    if video_keys != set(CAMERA_KEYS):
        raise ValueError(f"Expected exactly the three camera keys {CAMERA_KEYS}.")
    episodes = _read_jsonl(root / "meta/episodes.jsonl")
    for row in episodes:
        _integer(row.get("episode_index"), "episode_index")
        _integer(row.get("length"), "episode length", minimum=1)
    episodes.sort(key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in episodes] != list(range(info["total_episodes"])):
        raise ValueError("Episode indices must be unique and contiguous from zero, matching total_episodes.")
    if sum(row["length"] for row in episodes) != info["total_frames"]:
        raise ValueError("Episode lengths do not sum to total_frames.")
    if info.get("total_videos") != len(episodes) * len(CAMERA_KEYS):
        raise ValueError("total_videos must equal total_episodes times three cameras.")
    if info["total_chunks"] != (len(episodes) - 1) // info["chunks_size"] + 1:
        raise ValueError("total_chunks does not match episode count and chunks_size.")
    tasks = {}
    for row in _read_jsonl(root / "meta/tasks.jsonl"):
        index = _integer(row.get("task_index"), "task_index")
        task = row.get("task")
        if index in tasks or not isinstance(task, str) or not task.strip() or task in tasks.values():
            raise ValueError("Tasks must have unique indices and unique nonempty instructions.")
        tasks[index] = task
    if sorted(tasks) != list(range(info["total_tasks"])):
        raise ValueError("Task indices must be contiguous from zero, matching total_tasks.")
    for row in episodes:
        names = row.get("tasks")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or name not in tasks.values() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"Episode {row['episode_index']} must list unique known task instructions.")
    return info, episodes, tasks


def load_episode(
    root: Path, info: dict, episode_index: int, *, columns: list[str] | None = None
) -> dict[str, np.ndarray]:
    """Read stored rows as NumPy; action chunks are already [N, 50, 58]."""
    import datasets  # noqa: PLC0415

    path = dataset_path(root, info, "data_path", episode_index)
    if not path.is_file():
        raise ValueError(f"Missing episode Parquet: {path}")
    try:
        table = datasets.Dataset.from_parquet(str(path), columns=columns).with_format("numpy")[:]
    except Exception as error:
        raise ValueError(f"Cannot read episode {episode_index} from {path}: {error}") from error
    return {key: np.asarray(value) for key, value in table.items()}


def _scalar_column(data: dict, key: str, length: int) -> np.ndarray:
    value = np.asarray(data.get(key))
    if value.shape == (length, 1):
        value = value[:, 0]
    if value.shape != (length,) or value.dtype.kind not in "biuf" or not np.isfinite(value).all():
        raise ValueError(f"{key} must be a finite numeric scalar column with {length} rows.")
    return value


def _binary(values: np.ndarray, label: str) -> np.ndarray:
    if values.dtype.kind not in "biu" or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{label} must contain binary 0/1 or boolean values.")
    return values.astype(bool)


def _canonical_rotations(values: np.ndarray, label: str) -> None:
    for start in (3, 12):
        rotation = values[..., start : start + 6].astype(np.float64)
        first, second = rotation[..., :3], rotation[..., 3:]
        if (
            not np.allclose(np.linalg.norm(first, axis=-1), 1, atol=1e-3, rtol=0)
            or not np.allclose(np.linalg.norm(second, axis=-1), 1, atol=1e-3, rtol=0)
            or not np.allclose(np.sum(first * second, axis=-1), 0, atol=1e-3, rtol=0)
        ):
            raise ValueError(f"{label}: Rotation6D columns must be unit and orthogonal (tolerance 1e-3).")


def _validate_arrays(
    data: dict, episode: dict, tasks: dict, fps: float, start_index: int, action_offset: int | None
) -> dict:
    length, index = episode["length"], episode["episode_index"]
    for key, shape in (("observation.state", (length, 58)), ("action", (length, 50, 58))):
        value = np.asarray(data.get(key))
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"{key} must have finite floating-point values with shape {shape}.")
    state, action = data["observation.state"], data["action"]
    pad = np.asarray(data.get("action_is_pad"))
    if pad.shape != (length, HORIZON):
        raise ValueError(f"action_is_pad must have shape ({length}, {HORIZON}).")
    pad = _binary(pad, "action_is_pad")
    valid = _binary(_scalar_column(data, "sample_valid_h50", length), "sample_valid_h50")
    if np.any(valid & pad.any(axis=1)):
        raise ValueError("sample_valid_h50=1 includes padded actions.")
    if np.any(pad[:, :-1] & ~pad[:, 1:]):
        raise ValueError("action_is_pad must be a contiguous suffix within each chunk.")
    if action_offset is not None:
        expected_pad = np.arange(length)[:, None] + action_offset + np.arange(HORIZON)[None, :] >= length
        if not np.array_equal(pad, expected_pad):
            raise ValueError("action_is_pad does not match episode boundaries and declared action_offset_frames.")
    for key, expected in (
        ("frame_index", np.arange(length)),
        ("episode_index", np.full(length, index)),
        ("index", start_index + np.arange(length)),
    ):
        column = _scalar_column(data, key, length)
        if column.dtype.kind not in "iu" or not np.array_equal(column, expected):
            raise ValueError(f"{key} has inconsistent or non-contiguous values for episode {index}.")
    task_index = _scalar_column(data, "task_index", length)
    if task_index.dtype.kind not in "iu" or not np.isin(task_index, list(tasks)).all():
        raise ValueError("task_index contains an unknown or non-integer task.")
    if {tasks[int(value)] for value in task_index} != set(episode["tasks"]):
        raise ValueError("Parquet task_index values do not match episode task instructions.")
    timestamps = _scalar_column(data, "timestamp", length)
    # Account only for representational rounding of float32 timestamps, not dropped frames.
    tolerance = max(1e-6, 2 * float(np.spacing(np.float32((length - 1) / fps))))
    if np.any(np.diff(timestamps) <= 0) or not np.allclose(
        timestamps, np.arange(length, dtype=np.float64) / fps, atol=tolerance, rtol=0
    ):
        raise ValueError("timestamp must start at zero and follow frame_index / fps without gaps.")
    _canonical_rotations(state, "observation.state")
    _canonical_rotations(action[~pad], "non-padded action")
    maximum_error = 0.0
    valid_indices = np.flatnonzero(valid)
    for offset in range(0, len(valid_indices), 128):
        rows = valid_indices[offset : offset + 128]
        relative = umi_pose.absolute_actions_to_relative(action[rows], state[rows])
        absolute = umi_pose.relative_actions_to_absolute(relative, state[rows])
        # Compare physical transforms: legitimate rounding within the canonical
        # rotation tolerance is orthogonalized by the geometry helper.
        for pose in (slice(0, 9), slice(9, 18)):
            expected = umi_pose.pose9_to_matrix(action[rows, :, pose])
            actual = umi_pose.pose9_to_matrix(absolute[..., pose])
            error = float(np.max(np.abs(actual - expected)))
            maximum_error = max(maximum_error, error)
            if not np.allclose(actual, expected, atol=2e-5, rtol=2e-6):
                raise ValueError("Absolute/relative SE(3) roundtrip does not recover target transforms.")
        if not np.array_equal(relative[..., 18:], action[rows, :, 18:]) or not np.array_equal(
            absolute[..., 18:], action[rows, :, 18:]
        ):
            raise ValueError("SE(3) conversion modified hand channels.")
    return {
        "episode_index": index,
        "frames": length,
        "valid_samples": int(valid.sum()),
        "padded_rows": int(pad.any(axis=1).sum()),
        "roundtrip_max_transform_error": maximum_error,
    }


def _validate_stats(root: Path, info: dict, episodes: list[dict]) -> None:
    rows = _read_jsonl(root / "meta/episodes_stats.jsonl")
    by_episode = {}
    for row in rows:
        index = _integer(row.get("episode_index"), "statistics episode_index")
        if index in by_episode or not isinstance(row.get("stats"), dict):
            raise ValueError("Episode statistics must have unique indices and stats objects.")
        by_episode[index] = row["stats"]
    if set(by_episode) != {row["episode_index"] for row in episodes}:
        raise ValueError("Statistics episode indices must match episodes.jsonl.")
    for episode in episodes:
        for key, feature in info["features"].items():
            stats = by_episode[episode["episode_index"]].get(key)
            if not isinstance(stats, dict):
                raise ValueError(f"Missing statistics for episode {episode['episode_index']}: {key}.")
            shape = (3, 1, 1) if feature.get("dtype") == "video" else tuple(feature.get("shape", []))
            values = {}
            for name in ("min", "max", "mean", "std"):
                try:
                    value = np.asarray(stats.get(name), dtype=np.float64)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid {key}.{name} statistics.") from error
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"{key}.{name} statistics must be finite with shape {shape}.")
                values[name] = value
            if np.any(values["std"] < 0) or np.any(values["min"] > values["max"]):
                raise ValueError(f"Invalid min/max/std statistics for {key}.")
            count = stats.get("count")
            if feature.get("dtype") == "video":
                # Pinned LeRobot computes image statistics from a subset of frames.
                if not isinstance(count, list) or len(count) != 1:
                    raise ValueError(f"{key}.count statistics must contain one frame count.")
                if _integer(count[0], f"{key}.count", minimum=1) > episode["length"]:
                    raise ValueError(f"{key}.count statistics cannot exceed episode length.")
            elif count != [episode["length"]]:
                raise ValueError(f"{key}.count statistics must match episode length.")


def _validate_video(path: Path, feature: dict, length: int, fps: float) -> dict:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_streams",
                "-show_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,start_time,nb_read_frames:frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        probed = json.loads(result.stdout)
        stream = probed["streams"][0]
        frame_times = np.asarray([float(frame["best_effort_timestamp_time"]) for frame in probed["frames"]])
        count = int(stream["nb_read_frames"])
        actual_fps = float(Fraction(stream["avg_frame_rate"]))
        start = float(stream["start_time"])
        width, height = int(stream["width"]), int(stream["height"])
    except (
        OSError,
        subprocess.CalledProcessError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as error:
        raise ValueError(f"Cannot inspect video {path}: {error}") from error
    if count != length or not np.isclose(actual_fps, fps, atol=1e-6, rtol=0):
        raise ValueError(f"{path}: video frames/fps {count}/{actual_fps} do not match {length}/{fps}.")
    if not np.isfinite(start) or abs(start) > 1e-3:
        raise ValueError(f"{path}: video must start at timestamp zero, got {start}.")
    if (
        frame_times.shape != (length,)
        or not np.isfinite(frame_times).all()
        or not np.allclose(frame_times, np.arange(length, dtype=np.float64) / fps, atol=1e-3, rtol=0)
    ):
        raise ValueError(f"{path}: video frame timestamps must follow the dataset grid i / fps.")
    if (width, height) != _video_shape(feature, str(path)):
        raise ValueError(f"{path}: video dimensions do not match feature metadata.")
    details = feature.get("info", {})
    for key, expected in (("video.width", width), ("video.height", height), ("video.fps", fps)):
        if key in details and details[key] != expected:
            raise ValueError(f"{path}: {key} disagrees with the video.")
    return {"frames": count, "fps": actual_fps, "width": width, "height": height}


def _validate_manifest(root: Path, info: dict) -> tuple[list[str], int | None]:
    path = root / "meta/prepare_manifest.json"
    if not path.exists():
        return [
            "No prepare_manifest.json: absolute EEF semantics, units, Rotation6D column convention and action-offset boundary alignment remain unverified."
        ], None
    manifest = _read_json(path)
    expected = {
        "pose_representation": "absolute_eef_common_reference_frame",
        "rotation6d": "first_two_columns_concatenated",
        "position_units": "metres",
        "relative_pose_conversion_applied": False,
        "action_horizon": HORIZON,
        "fps": info["fps"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"prepare_manifest.{key} must be {value!r}; got {manifest.get(key)!r}.")
    action_offset = _integer(manifest.get("action_offset_frames"), "prepare_manifest.action_offset_frames")
    if action_offset not in (0, 1):
        raise ValueError("prepare_manifest.action_offset_frames must be zero or one.")
    return [], action_offset


def validate_dataset(dataset_dir: Path, *, check_videos: bool = True) -> dict:
    """Return a JSON-serializable validation report, or raise ValueError."""
    root = Path(dataset_dir).expanduser().resolve()
    info, episodes, tasks = read_metadata(root)
    warnings, action_offset = _validate_manifest(root, info)
    _validate_stats(root, info, episodes)
    reports = []
    start_index = 0
    seen_paths = set()
    for episode in episodes:
        index = episode["episode_index"]
        paths = [dataset_path(root, info, "data_path", index)]
        paths.extend(dataset_path(root, info, "video_path", index, key) for key in CAMERA_KEYS)
        if len(set(paths)) != len(paths) or seen_paths.intersection(paths):
            raise ValueError("Episode and video path templates must resolve to distinct files.")
        seen_paths.update(paths)
        for path in paths:
            if not path.is_file():
                raise ValueError(f"Missing dataset file: {path}")
        data = load_episode(root, info, index)
        try:
            report = _validate_arrays(data, episode, tasks, info["fps"], start_index, action_offset)
        except ValueError as error:
            raise ValueError(f"Episode {index}: {error}") from error
        report["videos"] = {}
        for key, path in zip(CAMERA_KEYS, paths[1:], strict=True):
            if check_videos:
                report["videos"][key] = _validate_video(path, info["features"][key], episode["length"], info["fps"])
        reports.append(report)
        start_index += episode["length"]
    valid_samples = sum(row["valid_samples"] for row in reports)
    if valid_samples == 0:
        raise ValueError("Dataset has no valid full-horizon samples.")
    if not check_videos:
        warnings.append("Video existence checked; decoding, frame counts, dimensions and fps were not checked.")
    return {
        "status": "passed",
        "dataset_dir": str(root),
        "total_frames": start_index,
        "valid_samples": valid_samples,
        "total_episodes": len(episodes),
        "fps": info["fps"],
        "action_shape": [HORIZON, ACTION_DIM],
        "action_offset_frames": action_offset,
        "check_videos": check_videos,
        "episodes": reports,
        "warnings": warnings,
        "scope": "Stored schema, indices, values, geometry, metadata and optional videos; no model or robot execution.",
        "limitations": [
            "Numerical checks cannot prove that rotations use the declared column convention or poses share a reference frame.",
            "Measured-state/commanded-target semantics, calibration, native hand units and source synchronization need collection-side verification.",
            "Passing does not verify policy adapter wiring, normalization assets, model quality or closed-loop robot behavior.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="New JSON report path outside the dataset; never overwrite.")
    parser.add_argument("--skip-videos", action="store_true", help="Skip ffprobe; still require all video files.")
    args = parser.parse_args()
    try:
        report_path = args.report.expanduser().resolve() if args.report is not None else None
        if report_path is not None:
            if report_path.is_relative_to(args.dataset_dir.expanduser().resolve()):
                raise ValueError("Report must be outside the source dataset directory.")
            if report_path.exists():
                raise ValueError(f"Refusing to overwrite existing report: {report_path}")
        report = validate_dataset(args.dataset_dir, check_videos=not args.skip_videos)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("x", encoding="utf-8") as output:
                output.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (ValueError, ImportError, OSError) as error:
        print(f"Dataset validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
