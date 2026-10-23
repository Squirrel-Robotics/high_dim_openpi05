"""Convert one absolute-EEF teleop episode into the project's LeRobot v2.1 format.

See examples/teleop_58d/README.md for the input contract and command line. This
writer stores absolute poses; observation-relative SE(3) belongs in the policy
data transform, before OpenPI normalization. It does not read raw Pico grip poses.

Run with the project's pinned datasets dependency and ffmpeg/ffprobe on PATH.
No JAX, Torch, GPU, or Hugging Face upload is needed for conversion.
"""

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from openpi.shared import teleop_raw


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _probe(path: Path, *, count_frames: bool = False) -> dict:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        command.append("-count_frames")
    command += ["-show_streams", "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    if not streams:
        raise ValueError(f"No video stream in {path}.")
    return streams[0]


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _resample_video(
    camera: teleop_raw.CameraStream,
    indices: np.ndarray,
    destination: Path,
    *,
    fps: int,
    image_width: int,
) -> tuple[dict, dict]:
    """Decode by frame index, repeat selected frames as needed, and encode at fps."""
    source = _probe(camera.video_path)
    width = image_width
    height = max(2, round(int(source["height"]) * width / int(source["width"]) / 2) * 2)
    if np.any(indices[1:] < indices[:-1]) or indices[0] < 0 or indices[-1] >= len(camera.times_ns):
        raise ValueError("Camera indices must be nondecreasing and within the recorded frame table.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    decode_command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-i",
        str(camera.video_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        f"scale={width}:{height}",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    encode_command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    total, squares = np.zeros(3), np.zeros(3)
    minimum, maximum = np.full(3, 255, dtype=np.uint8), np.zeros(3, dtype=np.uint8)
    frame_size = width * height * 3
    with tempfile.TemporaryFile() as decode_log, tempfile.TemporaryFile() as encode_log:
        decoder = subprocess.Popen(decode_command, stdout=subprocess.PIPE, stderr=decode_log)
        encoder = None
        try:
            encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE, stderr=encode_log)
            output_index, decoded_count = 0, 0
            while True:
                frame = decoder.stdout.read(frame_size)
                if not frame:
                    break
                if len(frame) != frame_size:
                    raise ValueError(f"Incomplete decoded frame {decoded_count} in {camera.video_path}.")
                if output_index < len(indices) and decoded_count == indices[output_index]:
                    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(-1, 3)
                    floating = pixels.astype(np.float64)
                    frame_sum = floating.sum(axis=0)
                    frame_squares = np.square(floating).sum(axis=0)
                    minimum = np.minimum(minimum, pixels.min(axis=0))
                    maximum = np.maximum(maximum, pixels.max(axis=0))
                    while output_index < len(indices) and decoded_count == indices[output_index]:
                        encoder.stdin.write(frame)
                        total += frame_sum
                        squares += frame_squares
                        output_index += 1
                decoded_count += 1
            if decoder.wait(timeout=30) != 0:
                raise ValueError(f"Could not decode {camera.video_path}.")
            if decoded_count != len(camera.times_ns) or output_index != len(indices):
                raise ValueError(
                    f"{camera.video_path}: decoded {decoded_count} frames but CSV declares {len(camera.times_ns)}; "
                    f"selected {output_index}/{len(indices)} output frames."
                )
            encoder.stdin.close()
            if encoder.wait(timeout=60) != 0:
                raise ValueError(f"Could not encode {destination}.")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            decode_log.seek(0)
            encode_log.seek(0)
            details = (decode_log.read() + encode_log.read()).decode(errors="replace")[-4000:]
            raise RuntimeError(f"Video conversion failed: {error}\n{details}") from error
        finally:
            _stop(decoder)
            decoder.stdout.close()
            if encoder is not None:
                _stop(encoder)
                with contextlib.suppress(BrokenPipeError):
                    encoder.stdin.close()
    output = _probe(destination, count_frames=True)
    if int(output["nb_read_frames"]) != len(indices):
        raise ValueError(f"Encoded frame count mismatch for {destination}.")
    pixel_count = len(indices) * width * height
    mean = total / pixel_count
    std = np.sqrt(np.maximum(0, squares / pixel_count - np.square(mean)))
    stats = {
        name: (value.astype(np.float64).reshape(3, 1, 1) / 255).tolist()
        for name, value in (("min", minimum), ("max", maximum), ("mean", mean), ("std", std))
    }
    stats["count"] = [len(indices)]
    feature = {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    return feature, stats


def _dimension_names(*, commands: bool) -> list[str]:
    pose = [
        f"{side}_eef_absolute_{suffix}"
        for side in ("left", "right")
        for suffix in ("x_m", "y_m", "z_m", *(f"rot6d_{i}" for i in range(6)))
    ]
    source = "command" if commands else "measured"
    hand = [f"{side}_hand_nid_{nid}_{source}_native" for side in ("left", "right") for nid in teleop_raw.HAND_NIDS]
    return pose + hand


def _numeric_data(arrays: dict, fps: int, action_offset: int) -> tuple[dict, dict]:
    count = len(arrays["times_ns"])
    data = {
        "observation.state": arrays["state"],
        "action": arrays["action"],
        "action_is_pad": arrays["action_is_pad"],
        "sample_valid_h50": arrays["sample_valid_h50"][:, 0],
        "source.host_monotonic_ns": arrays["times_ns"],
        "timestamp": np.arange(count, dtype=np.float32) / fps,
        "frame_index": np.arange(count, dtype=np.int64),
        "episode_index": np.zeros(count, dtype=np.int64),
        "index": np.arange(count, dtype=np.int64),
        "task_index": np.zeros(count, dtype=np.int64),
    }
    for key in teleop_raw.CAMERA_KEYS:
        source = f"source.{key.rsplit('.', 1)[-1]}"
        data[f"{source}.frame_index"] = arrays["camera_indices"][key]
        data[f"{source}.signed_delta_ms"] = arrays["camera_delta_ms"][key].astype(np.float32)
    features = {
        key: {"dtype": str(value.dtype), "shape": list(value.shape[1:]) or [1], "names": None}
        for key, value in data.items()
    }
    steps = [f"future_step_{i}" for i in range(action_offset, action_offset + teleop_raw.HORIZON)]
    features["observation.state"]["names"] = [_dimension_names(commands=False)]
    features["action"]["names"] = [steps, _dimension_names(commands=True)]
    features["action_is_pad"]["names"] = steps
    features["sample_valid_h50"]["names"] = ["complete_fresh_valid_h50"]
    return data, features


def _write_parquet(path: Path, data: dict, features: dict) -> None:
    import datasets  # noqa: PLC0415  --help and source validation do not need Arrow.

    hf_features = {}
    for key, feature in features.items():
        shape, dtype = feature["shape"], feature["dtype"]
        if dtype == "video":
            continue
        if shape == [1]:
            hf_features[key] = datasets.Value(dtype)
        elif len(shape) == 1:
            hf_features[key] = datasets.Sequence(datasets.Value(dtype), length=shape[0])
        else:
            hf_features[key] = datasets.Array2D(tuple(shape), dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_dict(data, features=datasets.Features(hf_features)).to_parquet(path)


def _numeric_stats(data: dict) -> dict:
    return {
        key: {
            "min": np.atleast_1d(value.min(axis=0)).tolist(),
            "max": np.atleast_1d(value.max(axis=0)).tolist(),
            "mean": np.atleast_1d(value.mean(axis=0, dtype=np.float64)).tolist(),
            "std": np.atleast_1d(value.std(axis=0, dtype=np.float64)).tolist(),
            "count": [len(value)],
        }
        for key, value in data.items()
    }


def _destination(repo_id: str, output_root: Path | None) -> Path:
    parts = repo_id.split("/")
    if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) for part in parts):
        raise ValueError("repo_id must have the form 'owner/dataset_name', without path traversal.")
    if output_root is None:
        hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
        output_root = Path(os.environ.get("HF_LEROBOT_HOME", str(hf_home / "lerobot")))
    return (output_root.expanduser() / repo_id).resolve()


def prepare_lerobot_dataset(
    raw_dir: Path,
    state_pose_csv: Path,
    target_pose_csv: Path,
    *,
    repo_id: str,
    task: str,
    output_root: Path | None = None,
    fps: int = 10,
    action_offset: int = 1,
    max_age_ms: float = 50,
    max_camera_delta_ms: float = 40,
    command_field: str = "position",
    image_width: int = 640,
) -> Path:
    """Write a new local dataset containing one episode; refuse existing output."""
    if not task.strip():
        raise ValueError("task must be a nonempty language instruction.")
    if isinstance(image_width, bool) or not isinstance(image_width, int) or image_width < 2 or image_width % 2:
        raise ValueError("image_width must be a positive even integer of at least 2.")
    output = _destination(repo_id, output_root)
    raw_dir, state_pose_csv, target_pose_csv = (
        Path(p).expanduser().resolve()
        for p in (
            raw_dir,
            state_pose_csv,
            target_pose_csv,
        )
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    if output.is_relative_to(raw_dir) or raw_dir.is_relative_to(output):
        raise ValueError("Output dataset and raw source must be separate directory trees.")
    if any(path.is_relative_to(output) for path in (state_pose_csv, target_pose_csv)):
        raise ValueError("EEF inputs must be outside the output dataset.")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is required on PATH.")
    arrays = teleop_raw.prepare_arrays(
        raw_dir,
        state_pose_csv,
        target_pose_csv,
        fps=fps,
        action_offset=action_offset,
        max_age_ms=max_age_ms,
        max_camera_delta_ms=max_camera_delta_ms,
        command_field=command_field,
    )
    data, features = _numeric_data(arrays, fps, action_offset)
    stats = _numeric_stats(data)
    count = len(arrays["times_ns"])
    valid_count = int(arrays["sample_valid_h50"].sum())
    print(f"Aligned {count} frames at {fps} Hz; {valid_count} valid 50-step training samples.", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-prepare-", dir=output.parent) as temporary:
        staging = Path(temporary)
        _write_parquet(staging / "data/chunk-000/episode_000000.parquet", data, features)
        for key in teleop_raw.CAMERA_KEYS:
            print(f"Resampling {key} ...", flush=True)
            features[key], stats[key] = _resample_video(
                arrays["cameras"][key],
                arrays["camera_indices"][key],
                staging / f"videos/chunk-000/{key}/episode_000000.mp4",
                fps=fps,
                image_width=image_width,
            )
        _json(
            staging / "meta/info.json",
            {
                "codebase_version": "v2.1",
                "robot_type": "dual_teleop_absolute_eef_rot6d_hand20_h50",
                "total_episodes": 1,
                "total_frames": count,
                "total_tasks": 1,
                "total_videos": 3,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": fps,
                "splits": {"train": "0:1"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": features,
            },
        )
        _jsonl(staging / "meta/tasks.jsonl", {"task_index": 0, "task": task.strip()})
        _jsonl(staging / "meta/episodes.jsonl", {"episode_index": 0, "tasks": [task.strip()], "length": count})
        _jsonl(staging / "meta/episodes_stats.jsonl", {"episode_index": 0, "stats": stats})
        _json(
            staging / "meta/prepare_manifest.json",
            {
                "repo_id": repo_id,
                "raw_dir": str(raw_dir),
                "state_pose_csv": str(state_pose_csv),
                "target_pose_csv": str(target_pose_csv),
                "pose_representation": "absolute_eef_common_reference_frame",
                "rotation6d": "first_two_columns_concatenated",
                "position_units": "metres",
                "hand_units": "native",
                "hand_nids": list(teleop_raw.HAND_NIDS),
                "hand_command_field": command_field,
                "clock": "host_CLOCK_MONOTONIC_ns",
                "non_image_sampling": "causal_zero_order_hold",
                "camera_sampling": "nearest_frame",
                "camera_signed_delta": "capture_time_minus_grid_time_ms",
                "start_ns": int(arrays["times_ns"][0]),
                "end_ns": int(arrays["times_ns"][-1]),
                "fps": fps,
                "action_offset_frames": action_offset,
                "action_horizon": teleop_raw.HORIZON,
                "max_age_ms": max_age_ms,
                "max_camera_delta_ms": max_camera_delta_ms,
                "total_frames": count,
                "valid_samples": valid_count,
                "image_width": image_width,
                "head_view": "full_frame_no_crop",
                "relative_pose_conversion_applied": False,
            },
        )
        if output.exists():
            raise FileExistsError(f"Output appeared during conversion; refusing overwrite: {output}")
        staging.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--state-pose-csv", type=Path, required=True, help="Measured absolute EEF poses; see README.")
    parser.add_argument(
        "--target-pose-csv", type=Path, required=True, help="Commanded absolute EEF targets; see README."
    )
    parser.add_argument("--repo-id", required=True, help="Local dataset ID, e.g. dzq/teleop_58d_example. No upload.")
    parser.add_argument("--task", required=True, help="Language instruction for this episode.")
    parser.add_argument("--output-root", type=Path, help="Defaults to HF_LEROBOT_HOME, then $HF_HOME/lerobot.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--action-offset",
        type=int,
        choices=(0, 1),
        default=1,
        help="First target offset in grid frames: 1 means t+dt; 0 means t.",
    )
    parser.add_argument("--max-age-ms", type=float, default=50)
    parser.add_argument("--max-camera-delta-ms", type=float, default=40)
    parser.add_argument("--command-field", choices=("position", "target_position"), default="position")
    parser.add_argument("--image-width", type=int, default=640, help="Even output width; preserve aspect ratio.")
    args = parser.parse_args()
    output = prepare_lerobot_dataset(**vars(args))
    print(f"Prepared local dataset: {output}")


if __name__ == "__main__":
    main()
