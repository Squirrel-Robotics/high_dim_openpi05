"""Read and align absolute-EEF teleoperation recordings without ML dependencies.

EEF measurements and commanded EEF targets must be supplied as separate CSVs in
the same reference frame and host CLOCK_MONOTONIC nanoseconds. The raw Pico grip
poses in a UMI recording are deliberately not substituted for these robot poses.
Rotation6D uses the first two columns concatenated, and positions are in metres.
Hand measurements/commands retain the native units recorded by the Wuji SDK.
"""

import csv
import dataclasses
import json
from pathlib import Path
import re

import numpy as np

from openpi.shared import umi_pose

HORIZON = 50
CAMERA_KEYS = (
    "observation.images.head_rgb",
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
)
HAND_NIDS = tuple(finger * 5 + joint for finger in range(5) for joint in range(1, 5))
_INT64_MAX = np.iinfo(np.int64).max


@dataclasses.dataclass(frozen=True)
class TimedValues:
    times_ns: np.ndarray
    values: np.ndarray
    valid: np.ndarray


@dataclasses.dataclass(frozen=True)
class CameraStream:
    times_ns: np.ndarray
    frame_indices: np.ndarray
    video_path: Path


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")
    result = int(str(value))
    if result > _INT64_MAX:
        raise ValueError(f"{name} exceeds int64; timestamps must be host monotonic nanoseconds.")
    return result


def _bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)) and str(value).lower() in ("true", "false", "1", "0"):
        return str(value).lower() in ("true", "1")
    raise ValueError(f"{name} must be a boolean (true/false or 1/0), got {value!r}.")


def _times(values: list[int], name: str) -> np.ndarray:
    if not values:
        raise ValueError(f"{name} contains no records.")
    result = np.asarray(values, dtype=np.int64)
    if np.any(result[1:] <= result[:-1]):
        raise ValueError(f"{name} timestamps must be strictly increasing without duplicates.")
    return result


def _float_values(values: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real numeric values.") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError(f"{name} must contain finite values representable as float32.")
    return array.astype(np.float32)


def _csv_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path}: missing or duplicate CSV column names.")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing CSV columns {sorted(missing)}.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows.")
    return rows


def read_pose_csv(path: Path) -> TimedValues:
    """Read measured or commanded absolute EEF poses, preserving their values.

    Columns: host_monotonic_ns, left_px_m/py_m/pz_m, left_rot6d_0..5,
    right_px_m/py_m/pz_m, right_rot6d_0..5, and optional valid (default true).
    """
    path = Path(path)
    columns = tuple(
        name
        for side in ("left", "right")
        for name in (
            *(f"{side}_p{axis}_m" for axis in "xyz"),
            *(f"{side}_rot6d_{index}" for index in range(6)),
        )
    )
    rows = _csv_rows(path, ("host_monotonic_ns", *columns))
    times = _times([_integer(row["host_monotonic_ns"], str(path)) for row in rows], str(path))
    values = _float_values([[row[name] for name in columns] for row in rows], (len(rows), 18), str(path))
    for side in (slice(0, 9), slice(9, 18)):
        try:
            umi_pose.pose9_to_matrix(values[:, side])
        except ValueError as error:
            raise ValueError(f"{path}: invalid absolute EEF pose: {error}") from error
    valid = np.asarray([_bool(row.get("valid", "true"), f"{path}: valid") for row in rows])
    return TimedValues(times, values, valid)


def _nid_order(nids: list[object], name: str) -> list[int]:
    parsed = [_integer(nid, f"{name}: nid") for nid in nids]
    if len(parsed) != 20 or set(parsed) != set(HAND_NIDS):
        raise ValueError(f"{name}: expected exactly the 20 unique Wuji joint NIDs {HAND_NIDS}.")
    return [parsed.index(nid) for nid in HAND_NIDS]


def read_hand_jsonl(path: Path, *, commands: bool, command_field: str = "position") -> TimedValues:
    """Read canonical Wuji joint order, using SDK callback/acceptance host time.

    Feedback is reordered by NID. Command arrays without explicit NIDs require
    adjacent metadata.json to declare thumb_index_middle_ring_pinky_S1_to_S4;
    this SDK ordering corresponds to HAND_NIDS. An explicit measurements.nids
    array is validated and reordered. Rejected commands remain invalid rows.
    """
    if command_field not in ("position", "target_position"):
        raise ValueError("command_field must be 'position' or 'target_position'.")
    path = Path(path)
    times, values, validity = [], [], []
    fixed_command_order_checked = False
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
                payload = row["payload"]
                measurements = payload["measurements"]
                valid = _bool(payload["valid"], f"{location}: valid")
                if commands:
                    if row["source_clock_domain"] != "host_monotonic":
                        raise ValueError(f"{location}: command timestamp must use host_monotonic.")
                    timestamp = _integer(row["source_timestamp"], location)
                    accepted = _bool(measurements["accepted_by_sdk"], f"{location}: accepted_by_sdk")
                    if accepted and _integer(measurements["accepted_mono_ns"], location) != timestamp:
                        raise ValueError(f"{location}: accepted_mono_ns does not match command source_timestamp.")
                    vector = _float_values(measurements[command_field], (20,), location)
                    if "nids" in measurements:
                        vector = vector[_nid_order(measurements["nids"], location)]
                    elif not fixed_command_order_checked:
                        metadata_path = path.with_name("metadata.json")
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if metadata.get("joint_order") != "thumb_index_middle_ring_pinky_S1_to_S4":
                            raise ValueError(f"{metadata_path}: command arrays require explicit SDK joint_order.")
                        fixed_command_order_checked = True
                    valid = valid and accepted
                else:
                    timestamp = _integer(row["host_receive_mono_ns"], location)
                    joints = measurements["joints"]
                    order = _nid_order([joint["nid"] for joint in joints], location)
                    vector = _float_values([joint["position"] for joint in joints], (20,), location)[order]
                times.append(timestamp)
                values.append(vector)
                validity.append(valid)
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{location}: malformed Wuji {'command' if commands else 'feedback'} record."
                ) from error
    return TimedValues(_times(times, str(path)), np.asarray(values, dtype=np.float32), np.asarray(validity))


def _frame_indices(rows: list[dict[str, str]], path: Path) -> np.ndarray:
    indices = np.asarray([_integer(row["frame_index"], str(path)) for row in rows], dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(rows))):
        raise ValueError(f"{path}: frame_index must be contiguous from zero in decoded video order.")
    return indices


def read_cameras(raw_dir: Path) -> dict[str, CameraStream]:
    """Read camera frame clocks; decoding is deferred to the LeRobot writer."""
    raw_dir = Path(raw_dir)
    result = {}
    head = raw_dir / "cameras/head_cam/rgb"
    model = json.loads((head / "clock_model.json").read_text(encoding="utf-8"))
    if model.get("status") != "ok" or model.get("target_clock_id") != "CLOCK_MONOTONIC":
        raise ValueError(f"{head}: head clock model must be ok and target CLOCK_MONOTONIC.")
    path = head / "aligned_frames.csv"
    rows = _csv_rows(path, ("frame_index", "part", "mapped_5b_monotonic_ns", "mapping_status"))
    if any(row["mapping_status"] != "ok" for row in rows):
        raise ValueError(f"{path}: every head frame needs an ok clock mapping.")
    if {_integer(row["part"], str(path)) for row in rows} != {1}:
        raise ValueError(f"{path}: only a single video part (part 1) is supported.")
    indices = _frame_indices(rows, path)
    raw_path = head / "frames.csv"
    raw_rows = _csv_rows(raw_path, ("frame_index", "part"))
    if {_integer(row["part"], str(raw_path)) for row in raw_rows} != {1}:
        raise ValueError(f"{raw_path}: only a single video part (part 1) is supported.")
    if not np.array_equal(_frame_indices(raw_rows, raw_path), indices):
        raise ValueError(f"{path}: aligned frames must match every raw frame_index.")
    times = _times([_integer(row["mapped_5b_monotonic_ns"], str(path)) for row in rows], str(path))
    video = head / "part-000001.mp4"
    if not video.is_file():
        video = head / "part-000001.h265"
    result[CAMERA_KEYS[0]] = CameraStream(times, indices, video)
    for side, key in zip(("left", "right"), CAMERA_KEYS[1:], strict=True):
        directory = raw_dir / "cameras" / side / "rgb"
        path = directory / "frames.csv"
        rows = _csv_rows(path, ("frame_index", "source_timestamp", "source_clock_domain"))
        if any(row["source_clock_domain"] != "v4l2_monotonic" for row in rows):
            raise ValueError(f"{path}: wrist source clock must be v4l2_monotonic.")
        times = _times([_integer(row["source_timestamp"], str(path)) for row in rows], str(path))
        result[key] = CameraStream(times, _frame_indices(rows, path), directory / f"{side}_cam.mp4")
    for camera in result.values():
        if not camera.video_path.is_file():
            raise FileNotFoundError(f"Missing camera video: {camera.video_path}")
    return result


def _hold(series: TimedValues, times: np.ndarray, max_age_ns: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(series.times_ns, times, side="right") - 1
    clipped = np.maximum(indices, 0)
    age = times - series.times_ns[clipped]
    valid = (indices >= 0) & (age >= 0) & (age <= max_age_ns) & series.valid[clipped]
    return series.values[clipped], valid


def prepare_arrays(
    raw_dir: Path,
    state_pose_csv: Path,
    target_pose_csv: Path,
    *,
    fps: int = 10,
    action_offset: int = 1,
    max_age_ms: float = 50,
    max_camera_delta_ms: float = 40,
    command_field: str = "position",
) -> dict:
    """Align one episode and build absolute [N,50,58] target chunks with masks.

    All non-image streams use zero-order hold, avoiding future feedback/command
    leakage into the current sample. Cameras use the nearest captured frame.
    Tail targets repeat the final target with action_is_pad=True; no rows are
    dropped here. sample_valid_h50 marks complete, fresh, valid training samples.
    """
    fps = _integer(fps, "fps")
    if fps == 0 or 1_000_000_000 % fps:
        raise ValueError("fps must be positive and divide 1,000,000,000 for an exact nanosecond grid.")
    action_offset = _integer(action_offset, "action_offset")
    for name, value in (("max_age_ms", max_age_ms), ("max_camera_delta_ms", max_camera_delta_ms)):
        if not np.isfinite(value) or value < 0 or value > _INT64_MAX / 1e6:
            raise ValueError(f"{name} must be finite, nonnegative, and fit int64 nanoseconds.")
    raw_dir = Path(raw_dir)
    state_pose, target_pose = read_pose_csv(state_pose_csv), read_pose_csv(target_pose_csv)
    feedback = [
        read_hand_jsonl(raw_dir / f"hands/wuji/{side}/joint_states.jsonl", commands=False) for side in ("left", "right")
    ]
    commands = [
        read_hand_jsonl(raw_dir / f"hands/wuji/{side}/commands.jsonl", commands=True, command_field=command_field)
        for side in ("left", "right")
    ]
    cameras = read_cameras(raw_dir)
    all_times = [item.times_ns for item in (state_pose, target_pose, *feedback, *commands, *cameras.values())]
    start, end = max(int(t[0]) for t in all_times), min(int(t[-1]) for t in all_times)
    if end < start:
        raise ValueError("The EEF, hand, and camera streams have no common host-monotonic time interval.")
    step = 1_000_000_000 // fps
    count = (end - start) // step + 1
    if count + action_offset + HORIZON > _INT64_MAX:
        raise ValueError("action_offset overflows int64 chunk indices.")
    times = start + np.arange(count, dtype=np.int64) * step
    state_samples = [_hold(series, times, max_age_ms * 1e6) for series in (state_pose, *feedback)]
    target_samples = [_hold(series, times, max_age_ms * 1e6) for series in (target_pose, *commands)]
    state = np.concatenate([sample[0] for sample in state_samples], axis=-1)
    target = np.concatenate([sample[0] for sample in target_samples], axis=-1)
    state_valid = np.logical_and.reduce([sample[1] for sample in state_samples])
    target_valid = np.logical_and.reduce([sample[1] for sample in target_samples])
    camera_indices, camera_delta_ms = {}, {}
    for key, camera in cameras.items():
        right = np.minimum(np.searchsorted(camera.times_ns, times), len(camera.times_ns) - 1)
        left = np.maximum(right - 1, 0)
        selected = np.where(
            np.abs(camera.times_ns[left] - times) <= np.abs(camera.times_ns[right] - times), left, right
        )
        delta_ms = (camera.times_ns[selected] - times) / 1e6
        camera_indices[key] = camera.frame_indices[selected]
        camera_delta_ms[key] = delta_ms
        state_valid &= np.abs(delta_ms) <= max_camera_delta_ms
    future = np.arange(count)[:, None] + action_offset + np.arange(HORIZON)[None, :]
    padded = future >= count
    clipped = np.minimum(future, count - 1)
    valid = state_valid & np.all(target_valid[clipped] & ~padded, axis=1)
    if not np.any(valid):
        raise ValueError("No valid 50-step training samples: check episode length, validity flags, and timestamp gaps.")
    return {
        "times_ns": times,
        "state": state,
        "action": target[clipped],
        "action_is_pad": padded,
        "sample_valid_h50": valid.astype(np.int64)[:, None],
        "camera_indices": camera_indices,
        "camera_delta_ms": camera_delta_ms,
        "cameras": cameras,
    }
