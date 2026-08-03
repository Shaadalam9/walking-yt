"""Shared model loading helpers with safe optional 4 bit quantisation."""

from __future__ import annotations

import gc
from typing import Any, Callable, Dict, Tuple

import torch

from .shared import log


ModelLoader = Callable[..., Any]


def device_is_cuda(device: str) -> bool:
    """Return whether a configured device refers to CUDA."""
    return device.strip().lower().startswith("cuda")


def model_device_map(device: str) -> Any:
    """Build an explicit Hugging Face device map for one model worker."""
    normalised = device.strip().lower()
    if normalised == "auto":
        return "auto"
    return {"": device}


def _base_model_kwargs(device: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "device_map": model_device_map(device),
        "trust_remote_code": True,
    }
    if device_is_cuda(device) and torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32
    return kwargs


def _four_bit_model_kwargs(device: str) -> Dict[str, Any]:
    if not device_is_cuda(device) or not torch.cuda.is_available():
        raise RuntimeError("4 bit loading requires an available CUDA device")

    import bitsandbytes as bnb  # type: ignore
    from transformers import BitsAndBytesConfig  # type: ignore

    version = getattr(bnb, "__version__", "unknown")
    log(f"bitsandbytes version: {version}")
    return {
        "device_map": model_device_map(device),
        "trust_remote_code": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    }


def build_model_kwargs(
    device: str,
    load_in_4bit: bool,
) -> Tuple[Dict[str, Any], bool]:
    """Return loading arguments and whether 4 bit loading is active."""
    if not load_in_4bit:
        return _base_model_kwargs(device), False

    try:
        return _four_bit_model_kwargs(device), True
    except Exception as exc:
        log(
            "4 bit loading was requested but could not be configured: "
            f"{exc}. Falling back to BF16."
        )
        return _base_model_kwargs(device), False


def load_model_with_fallback(
    loader: ModelLoader,
    model_name: str,
    *,
    device: str,
    load_in_4bit: bool,
    model_label: str,
) -> Any:
    """Load a model and retry in BF16 when a 4 bit load fails."""
    kwargs, using_four_bit = build_model_kwargs(device, load_in_4bit)
    precision = "4 bit NF4" if using_four_bit else (
        "BF16" if device_is_cuda(device) and torch.cuda.is_available() else "FP32"
    )
    log(f"Loading {model_label} on {device} using {precision}")

    four_bit_error: str | None = None
    try:
        return loader(model_name, **kwargs)
    except Exception as exc:
        if not using_four_bit:
            raise
        four_bit_error = str(exc)

    log(
        f"The {model_label} 4 bit load failed: {four_bit_error}. "
        "Retrying once in BF16."
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    fallback_kwargs = _base_model_kwargs(device)
    return loader(model_name, **fallback_kwargs)
