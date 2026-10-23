"""NumPy pose conversions for 58D dual-arm teleoperation and UMI data.

Layout: left pose [0:9], right pose [9:18], left hand [18:38], right hand [38:58].
A pose is [x, y, z, Rotation6D]. Rotation6D concatenates the first two COLUMNS of
the rotation matrix: [R00, R10, R20, R01, R11, R21]. Confirm this convention at the
data-collection boundary; a feature name such as ``rot6d`` does not specify it.

Absolute poses map end-effector coordinates into the common reference frame.
Each arm's current observed pose anchors every target in the action chunk:
T_relative = inverse(T_observed) @ T_target. The inverse mapping recovers absolute
targets. These conversions operate on physical values before normalization (and
after unnormalization at inference). Hand commands retain their original units.

This module runs on the CPU in data transforms; it does not require JAX or Torch.
"""

import numpy as np

ACTION_DIM = 58
POSE_DIM = 9
LEFT_POSE_SLICE = slice(0, 9)
RIGHT_POSE_SLICE = slice(9, 18)
HAND_SLICE = slice(18, 58)

_COLLINEAR_TOL = 1e-6
_MATRIX_TOL = 1e-5


def _as_float_array(values: np.ndarray, name: str) -> np.ndarray:
    """Validate real, finite values and promote low-precision/integer inputs to floats."""
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values, got dtype {array.dtype}.")
    array = array.astype(np.result_type(array.dtype, np.float32), copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    # Scale before taking the norm to avoid overflow/underflow for finite inputs.
    scale = np.max(np.abs(vector), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Rotation6D contains a zero vector; its orientation is undefined.")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def rot6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert [..., 6] column-concatenated Rotation6D to proper [..., 3, 3] rotations.

    Gram-Schmidt orthogonalizes non-unit predictions. Zero and near-collinear
    vectors raise ValueError rather than silently producing an arbitrary rotation.
    Float32/float64 inputs retain their dtype; lower precision is promoted.
    """
    rotation_6d = _as_float_array(rotation_6d, "Rotation6D")
    if rotation_6d.ndim < 1 or rotation_6d.shape[-1] != 6:
        raise ValueError(f"Rotation6D must have shape [..., 6], got {rotation_6d.shape}.")

    # The vectors are small, so float64 intermediates cheaply reduce cancellation.
    work = rotation_6d.astype(np.float64)
    first = _unit_vector(work[..., :3])
    second = _unit_vector(work[..., 3:])
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm <= _COLLINEAR_TOL):
        raise ValueError("Rotation6D contains collinear or near-collinear vectors; its orientation is undefined.")
    second /= norm
    third = _unit_vector(np.cross(first, second))
    # Recompute the second column to remove residual non-orthogonality after subtraction.
    second = np.cross(third, first)
    return np.stack((first, second, third), axis=-1).astype(rotation_6d.dtype)


def matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Convert proper [..., 3, 3] rotation matrices to [..., 6] concatenated columns."""
    rotation = _as_float_array(rotation, "rotation matrix")
    if rotation.ndim < 2 or rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must have shape [..., 3, 3], got {rotation.shape}.")
    work = rotation.astype(np.float64)
    gram = np.swapaxes(work, -1, -2) @ work
    if not np.allclose(gram, np.eye(3), atol=_MATRIX_TOL, rtol=0) or not np.allclose(
        np.linalg.det(work), 1.0, atol=_MATRIX_TOL, rtol=0
    ):
        raise ValueError("rotation matrix must be orthonormal with determinant +1.")
    return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def pose9_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert [..., 9] position/Rotation6D poses to [..., 4, 4] rigid transforms."""
    pose = _as_float_array(pose, "pose")
    if pose.ndim < 1 or pose.shape[-1] != POSE_DIM:
        raise ValueError(f"pose must have shape [..., {POSE_DIM}], got {pose.shape}.")
    transform = np.zeros((*pose.shape[:-1], 4, 4), dtype=pose.dtype)
    transform[..., :3, :3] = rot6d_to_matrix(pose[..., 3:])
    transform[..., :3, 3] = pose[..., :3]
    transform[..., 3, 3] = 1
    return transform


def matrix_to_pose9(transform: np.ndarray) -> np.ndarray:
    """Convert [..., 4, 4] rigid transforms to [..., 9] position/Rotation6D poses."""
    transform = _as_float_array(transform, "pose matrix")
    if transform.ndim < 2 or transform.shape[-2:] != (4, 4):
        raise ValueError(f"pose matrix must have shape [..., 4, 4], got {transform.shape}.")
    if not np.allclose(transform[..., 3, :], [0, 0, 0, 1], atol=_MATRIX_TOL, rtol=0):
        raise ValueError("pose matrix must have homogeneous last row [0, 0, 0, 1].")
    rotation_6d = matrix_to_rot6d(transform[..., :3, :3])
    return np.concatenate((transform[..., :3, 3], rotation_6d), axis=-1)


def _convert_actions(actions: np.ndarray, state: np.ndarray, *, to_absolute: bool) -> np.ndarray:
    actions = _as_float_array(actions, "actions")
    state = _as_float_array(state, "state")
    if actions.ndim < 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [..., H, {ACTION_DIM}], got {actions.shape}.")
    if state.ndim < 1 or state.shape[-1] != ACTION_DIM:
        raise ValueError(f"state must have shape [..., {ACTION_DIM}], got {state.shape}.")
    if actions.shape[:-2] != state.shape[:-1]:
        raise ValueError(f"actions and state batch shapes must match: {actions.shape[:-2]} vs {state.shape[:-1]}.")
    if actions.shape[-2] == 0:
        raise ValueError("actions must have a positive horizon.")

    result = actions.copy()
    for pose_slice in (LEFT_POSE_SLICE, RIGHT_POSE_SLICE):
        anchor = pose9_to_matrix(state[..., pose_slice].astype(np.float64))
        targets = pose9_to_matrix(actions[..., pose_slice].astype(np.float64))
        if not to_absolute:
            # Analytic rigid inverse: R^-1 = R^T, p^-1 = -R^T p.
            inverse = np.zeros_like(anchor)
            inverse[..., :3, :3] = np.swapaxes(anchor[..., :3, :3], -1, -2)
            inverse[..., :3, 3] = -(inverse[..., :3, :3] @ anchor[..., :3, 3, None])[..., 0]
            inverse[..., 3, 3] = 1
            anchor = inverse
        # Only insert the horizon axis; batch dimensions were checked above.
        converted = anchor[..., None, :, :] @ targets
        converted_pose = matrix_to_pose9(converted)
        if np.any(np.abs(converted_pose) > np.finfo(result.dtype).max):
            raise ValueError(f"Converted actions exceed the finite range of {result.dtype}.")
        result[..., pose_slice] = converted_pose
    return result


def absolute_actions_to_relative(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Express each arm's absolute targets relative to its current observed pose.

    Args:
        actions: [..., H, 58] absolute target poses and absolute hand commands.
        state: [..., 58] current absolute observed state, in the same frame/units.
            Batch prefixes must match exactly; a single state [58] pairs with [H, 58].

    Returns:
        A new action array of the same shape, with T_relative = inv(T_state) @ T_target
        for each arm. All H targets share that arm's observation anchor. Hand channels
        [18:58] remain unchanged. Inputs are not modified.
    """
    return _convert_actions(actions, state, to_absolute=False)


def relative_actions_to_absolute(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Recover absolute target poses via T_target = T_state @ T_relative for each arm.

    Shapes, batch matching, dtype promotion and hand preservation match
    absolute_actions_to_relative. Pass unnormalized model actions and the original
    absolute observation state, not the previous predicted action.
    """
    return _convert_actions(actions, state, to_absolute=True)
