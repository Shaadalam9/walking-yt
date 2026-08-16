"""Spike 1 and Run:ai runtime validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import settings
from .shared import log


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved execution mode for the GPUs visible inside the container."""

    mode: str
    gpu_count: int
    sequential_device: str
    text_device: str
    visual_device: str
    reason: str


def cuda_index(device: str) -> Optional[int]:
    """Return a CUDA index, or None when the value is not a CUDA device."""
    value = device.strip().lower()
    if value == "cuda":
        return 0
    if not value.startswith("cuda:"):
        return None
    try:
        index = int(value.split(":", 1)[1])
    except ValueError:
        return None
    return index if index >= 0 else None


def cuda_device_count() -> int:
    """Return the number of CUDA devices visible to this container."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _validate_device(device: str, gpu_count: int, variable_name: str) -> None:
    index = cuda_index(device)
    if index is not None:
        if index >= gpu_count:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
            raise RuntimeError(
                f"{variable_name}={device} is unavailable. Run:ai exposed "
                f"{gpu_count} CUDA device(s) to the container. "
                f"CUDA_VISIBLE_DEVICES={visible!r}. Use logical device "
                "indices such as cuda:0 and cuda:1 inside the pod."
            )
        return

    if device == "cpu":
        if settings.SPIKE1_REQUIRE_GPU:
            raise RuntimeError(
                f"{variable_name}=cpu is not allowed because "
                "SPIKE1_REQUIRE_GPU=1."
            )
        return

    raise RuntimeError(
        f"Unsupported {variable_name} value: {device!r}. "
        "Use cpu or cuda:N."
    )


def resolve_execution_plan() -> ExecutionPlan:
    """Resolve auto, sequential, or overlap against visible Run:ai GPUs."""
    gpu_count = cuda_device_count()
    if settings.SPIKE1_REQUIRE_GPU and gpu_count == 0:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        raise RuntimeError(
            "Spike 1 mode requires a visible CUDA GPU, but torch reports "
            f"zero devices. CUDA_VISIBLE_DEVICES={visible!r}. Confirm that "
            "the Run:ai workload requested a GPU and that the NVIDIA runtime "
            "is available in the container."
        )

    if settings.PIPELINE_MODE == "sequential":
        _validate_device(
            settings.SEQUENTIAL_DEVICE,
            gpu_count,
            "SEQUENTIAL_DEVICE",
        )
        return ExecutionPlan(
            mode="sequential",
            gpu_count=gpu_count,
            sequential_device=settings.SEQUENTIAL_DEVICE,
            text_device=settings.TEXT_DEVICE,
            visual_device=settings.VLM_DEVICE,
            reason="PIPELINE_MODE was explicitly set to sequential",
        )

    text_index = cuda_index(settings.TEXT_DEVICE)
    visual_index = cuda_index(settings.VLM_DEVICE)
    overlap_devices_are_distinct = (
        text_index is not None
        and visual_index is not None
        and text_index != visual_index
    )
    overlap_devices_are_visible = (
        overlap_devices_are_distinct
        and text_index < gpu_count  # type: ignore
        and visual_index < gpu_count  # type: ignore
    )

    if settings.PIPELINE_MODE == "overlap":
        if not overlap_devices_are_distinct:
            raise RuntimeError(
                "PIPELINE_MODE=overlap requires different CUDA devices for "
                "TEXT_DEVICE and VLM_DEVICE."
            )
        _validate_device(settings.TEXT_DEVICE, gpu_count, "TEXT_DEVICE")
        _validate_device(settings.VLM_DEVICE, gpu_count, "VLM_DEVICE")
        return ExecutionPlan(
            mode="overlap",
            gpu_count=gpu_count,
            sequential_device=settings.SEQUENTIAL_DEVICE,
            text_device=settings.TEXT_DEVICE,
            visual_device=settings.VLM_DEVICE,
            reason="PIPELINE_MODE was explicitly set to overlap",
        )

    # Automatic mode prefers application stage overlap only when both
    # configured devices are visible. Otherwise it safely uses one device.
    if overlap_devices_are_visible:
        return ExecutionPlan(
            mode="overlap",
            gpu_count=gpu_count,
            sequential_device=settings.SEQUENTIAL_DEVICE,
            text_device=settings.TEXT_DEVICE,
            visual_device=settings.VLM_DEVICE,
            reason="auto mode found two distinct visible CUDA devices",
        )

    _validate_device(
        settings.SEQUENTIAL_DEVICE,
        gpu_count,
        "SEQUENTIAL_DEVICE",
    )
    return ExecutionPlan(
        mode="sequential",
        gpu_count=gpu_count,
        sequential_device=settings.SEQUENTIAL_DEVICE,
        text_device=settings.TEXT_DEVICE,
        visual_device=settings.VLM_DEVICE,
        reason="auto mode did not find two configured visible CUDA devices",
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _check_writable_directory(path: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".walking_pipeline_write_test_{os.getpid()}"
    try:
        probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"{label} is not writable: {path}: {exc}") from exc
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass


def validate_storage() -> None:
    """Validate persistent data and model cache paths before processing."""
    if settings.SPIKE1_REQUIRE_PERSISTENT_STORAGE:
        if not settings.DATA_DIR.is_absolute():
            raise RuntimeError(
                "WALK_DATA_DIR in config must be an absolute mounted path "
                "in Spike 1 mode. "
                f"Received: {settings.DATA_DIR}"
            )

        persistent_outputs = {
            "WALK_STATE_JSON": settings.STATE_JSON,
            "WALK_OUTPUT_CSV": settings.OUTPUT_CSV,
            "WALK_VIDEO_DIR": settings.VIDEO_DIR,
            "WALK_GEOCODE_CACHE": settings.GEOCODE_CACHE_JSON,
        }
        for variable_name, path in persistent_outputs.items():
            if not _is_within(path, settings.DATA_DIR):
                raise RuntimeError(
                    f"{variable_name} resolves outside WALK_DATA_DIR: {path}. "
                    "Update the output paths in config so Spike 1 "
                    "outputs remain on persistent storage."
                )

    _check_writable_directory(settings.DATA_DIR, "WALK_DATA_DIR")
    _check_writable_directory(settings.VIDEO_DIR, "WALK_VIDEO_DIR")
    _check_writable_directory(settings.HF_HOME, "HF_HOME")
    _check_writable_directory(settings.HF_HUB_CACHE, "HF_HUB_CACHE")


def validate_runtime() -> ExecutionPlan:
    """Validate storage and compute, then report the resolved plan."""
    validate_storage()
    plan = resolve_execution_plan()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    log(
        "Runtime plan: "
        f"mode={plan.mode}, visible_gpus={plan.gpu_count}, "
        f"CUDA_VISIBLE_DEVICES={visible!r}, reason={plan.reason}"
    )
    if plan.mode == "sequential":
        log(f"Both models will run sequentially on {plan.sequential_device}")
    else:
        log(
            f"Text model will run on {plan.text_device}; visual model will "
            f"run on {plan.visual_device}"
        )
    log(f"Persistent data directory: {settings.DATA_DIR}")
    log(f"Hugging Face cache: {settings.HF_HUB_CACHE}")
    return plan
