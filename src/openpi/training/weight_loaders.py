import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class ActionDimCheckpointWeightLoader(WeightLoader):
    """Loads pi05 weights while allowing the action dimension to change (e.g. 32 -> 58).

    Only the input projection kernel and output projection kernel/bias may change shape, and only
    along their action axes. These leaves retain the reference initialization; matching leaves are
    loaded normally. Missing checkpoint parameters are errors, except for newly added LoRA weights.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_action_dim_params(loaded_params, params)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


# Exact parameter paths mapped to (expected rank, action axis). In particular, action_in_proj/bias
# depends on the expert width, not the action dimension, and must always load from the checkpoint.
_ACTION_DIM_PARAMS = {
    "action_in_proj/kernel": (2, 0),
    "action_out_proj/kernel": (2, 1),
    "action_out_proj/bias": (1, 0),
}


def _merge_action_dim_params(loaded_params: at.Params, params: at.Params) -> at.Params:
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # All three projections must describe the same action dimension within each parameter tree.
    # This also ensures that a missing projection cannot silently fall back to initialization.
    for label, flat_params in (("checkpoint", flat_loaded), ("model", flat_ref)):
        action_dims = {}
        for key, (rank, axis) in _ACTION_DIM_PARAMS.items():
            if key not in flat_params:
                raise ValueError(f"Missing required parameter {key} in {label}")
            shape = flat_params[key].shape
            if len(shape) != rank:
                raise ValueError(f"Invalid rank for {key} in {label}: expected {rank}, got shape {shape}")
            action_dims[key] = shape[axis]
        if len(set(action_dims.values())) != 1:
            raise ValueError(f"Inconsistent action dimensions in {label}: {action_dims}")

    result = {}
    for key, ref in flat_ref.items():
        if key not in flat_loaded:
            if re.fullmatch(".*lora.*", key):
                result[key] = ref
                continue
            raise ValueError(f"Missing required parameter {key} in checkpoint")

        loaded = flat_loaded[key]
        if loaded.shape == ref.shape:
            result[key] = loaded.astype(ref.dtype) if loaded.dtype != ref.dtype else loaded
            continue

        spec = _ACTION_DIM_PARAMS.get(key)
        if spec is None:
            raise ValueError(f"Unexpected shape mismatch for {key}: checkpoint {loaded.shape}, model {ref.shape}")
        _, axis = spec
        if loaded.shape[:axis] + loaded.shape[axis + 1 :] != ref.shape[:axis] + ref.shape[axis + 1 :]:
            raise ValueError(f"Non-action dimensions differ for {key}: checkpoint {loaded.shape}, model {ref.shape}")

        # During JAX training ref is a ShapeDtypeStruct. scripts/train.py validates the complete tree,
        # then removes these placeholders so the model's fresh initialization is kept for these leaves.
        result[key] = ref
        logger.info("Re-initializing %s: checkpoint shape %s -> model shape %s", key, loaded.shape, ref.shape)

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
