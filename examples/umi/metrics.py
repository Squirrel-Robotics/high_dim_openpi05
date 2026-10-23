"""Physical-unit offline metrics for the 58D dual-arm policy.

Targets and observation states are absolute, unnormalized values. Predictions
may be absolute or observation-relative; relative predictions are recovered
using the *same current observation* for every step of the 50-step chunk.
Rotation6D uses the first two matrix columns, as in ``openpi.shared.umi_pose``.

Position, orientation and hand errors are reported separately because their
units differ. Position RMSE is the root mean square Euclidean distance, not the
root mean square error over individual xyz coordinates. Hand units are retained
from the dataset; this module does not assume that they are radians.
"""

import numpy as np

from openpi.shared import umi_pose

HORIZON = 50


def _numeric_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values, got {array.dtype}.")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _rotations(actions: np.ndarray, start: int, name: str, arm: str) -> np.ndarray:
    try:
        return umi_pose.rot6d_to_matrix(actions[..., start + 3 : start + 9])
    except ValueError as exc:
        raise ValueError(f"Invalid {arm} Rotation6D in {name}: {exc}") from exc


def _scalar_error_summary(error: np.ndarray) -> dict:
    """Summarize one nonnegative physical error for every sample and horizon step."""
    return {
        "mean": float(np.mean(error)),
        "p95": float(np.percentile(error, 95)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "first_step_mean": float(np.mean(error[:, 0])),
        "last_step_mean": float(np.mean(error[:, -1])),
        "per_step_mean": np.mean(error, axis=0).tolist(),
    }


def _hand_error_summary(error: np.ndarray) -> dict:
    absolute_error = np.abs(error)
    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "first_step_mae": float(np.mean(absolute_error[:, 0])),
        "last_step_mae": float(np.mean(absolute_error[:, -1])),
        "per_step_mae": np.mean(absolute_error, axis=(0, 2)).tolist(),
        "per_joint_mae": np.mean(absolute_error, axis=(0, 1)).tolist(),
    }


def _orientation_error_degrees(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(prediction, -1, -2) @ target
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    skew = np.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        axis=-1,
    )
    sine = np.linalg.norm(skew, axis=-1) / 2
    return np.rad2deg(np.arctan2(sine, cosine))


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    states: np.ndarray,
    *,
    prediction_space: str = "absolute",
) -> dict:
    """Compare complete chunks using position metres, rotation degrees and hand units.

    Args:
        predictions: [N, 50, 58] physical model outputs, after unnormalization.
        targets: [N, 50, 58] absolute target poses and native hand commands.
        states: [N, 58] original absolute observations, before normalization.
        prediction_space: ``absolute`` or ``relative``. Only predictions are
            converted; targets must remain absolute in both modes. In relative
            mode, each arm's observation anchors all 50 predicted poses.

    The caller must exclude invalid samples and padded chunks before calling.
    This function does not infer masks, physical units, or normalization state.
    Invalid or undefined orientations raise ValueError instead of being dropped.
    Returned values are JSON-serializable Python numbers and lists.
    """
    if prediction_space not in {"absolute", "relative"}:
        raise ValueError(f"prediction_space must be 'absolute' or 'relative', got {prediction_space!r}.")
    predictions = _numeric_array(predictions, "predictions")
    targets = _numeric_array(targets, "targets")
    states = _numeric_array(states, "states")
    expected_chunk_shape = (HORIZON, umi_pose.ACTION_DIM)
    if predictions.ndim != 3 or predictions.shape[1:] != expected_chunk_shape:
        raise ValueError(f"predictions must have shape [N, 50, 58], got {predictions.shape}.")
    if predictions.shape[0] == 0:
        raise ValueError("At least one complete valid sample is required.")
    if targets.shape != predictions.shape:
        raise ValueError(f"targets must match predictions shape {predictions.shape}, got {targets.shape}.")
    if states.shape != (predictions.shape[0], umi_pose.ACTION_DIM):
        raise ValueError(f"states must have shape [{predictions.shape[0]}, 58], got {states.shape}.")

    # Validate anchors even in absolute mode: malformed observation geometry
    # usually indicates a dataset or Rotation6D convention problem.
    for arm, start in (("left", 0), ("right", 9)):
        _rotations(states, start, "states", arm)
        _rotations(predictions, start, "predictions", arm)
        _rotations(targets, start, "targets", arm)

    result = {
        "num_samples": int(predictions.shape[0]),
        "horizon": HORIZON,
        "action_dim": umi_pose.ACTION_DIM,
        "prediction_space": prediction_space,
    }
    try:
        with np.errstate(over="raise", invalid="raise"):
            if prediction_space == "relative":
                predictions = umi_pose.relative_actions_to_absolute(predictions, states)

            for arm, pose_start, hand_start in (("left", 0, 18), ("right", 9, 38)):
                position_error = np.linalg.norm(
                    predictions[..., pose_start : pose_start + 3] - targets[..., pose_start : pose_start + 3],
                    axis=-1,
                )
                rotation_error = _orientation_error_degrees(
                    _rotations(predictions, pose_start, "predictions", arm),
                    _rotations(targets, pose_start, "targets", arm),
                )
                hand_error = predictions[..., hand_start : hand_start + 20] - targets[..., hand_start : hand_start + 20]
                result[f"{arm}_position_m"] = _scalar_error_summary(position_error)
                result[f"{arm}_rotation_deg"] = _scalar_error_summary(rotation_error)
                result[f"{arm}_hand_native"] = _hand_error_summary(hand_error)
    except FloatingPointError as exc:
        raise ValueError("Physical inputs are too large to compute finite error metrics.") from exc
    return result
