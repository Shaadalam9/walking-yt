"""Tests for Spike 1 environment parsing and runtime selection."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from walking_pipeline import settings
from walking_pipeline.runtime import resolve_execution_plan, validate_storage


class EnvironmentParsingTests(unittest.TestCase):
    def test_boolean_values_are_strict(self) -> None:
        with patch.dict(os.environ, {"EXAMPLE_BOOL": "false"}):
            self.assertFalse(settings.env_bool("EXAMPLE_BOOL", True))
        with patch.dict(os.environ, {"EXAMPLE_BOOL": "yes"}):
            self.assertTrue(settings.env_bool("EXAMPLE_BOOL", False))
        with patch.dict(os.environ, {"EXAMPLE_BOOL": "maybe"}):
            with self.assertRaises(ValueError):
                settings.env_bool("EXAMPLE_BOOL", False)

    def test_optional_integer_accepts_unlimited(self) -> None:
        with patch.dict(os.environ, {"EXAMPLE_LIMIT": "unlimited"}):
            self.assertIsNone(
                settings.env_optional_int("EXAMPLE_LIMIT", 20)
            )


class RuntimeSelectionTests(unittest.TestCase):
    def _common_settings(self):
        return (
            patch.object(settings, "SPIKE1_REQUIRE_GPU", True),
            patch.object(settings, "SEQUENTIAL_DEVICE", "cuda:0"),
            patch.object(settings, "TEXT_DEVICE", "cuda:0"),
            patch.object(settings, "VLM_DEVICE", "cuda:1"),
        )

    def test_auto_uses_sequential_with_one_gpu(self) -> None:
        gpu, sequential, text, visual = self._common_settings()
        with gpu, sequential, text, visual, patch.object(
            settings, "PIPELINE_MODE", "auto"
        ), patch(
            "walking_pipeline.runtime.cuda_device_count", return_value=1
        ):
            plan = resolve_execution_plan()
        self.assertEqual(plan.mode, "sequential")
        self.assertEqual(plan.sequential_device, "cuda:0")

    def test_auto_uses_overlap_with_two_gpus(self) -> None:
        gpu, sequential, text, visual = self._common_settings()
        with gpu, sequential, text, visual, patch.object(
            settings, "PIPELINE_MODE", "auto"
        ), patch(
            "walking_pipeline.runtime.cuda_device_count", return_value=2
        ):
            plan = resolve_execution_plan()
        self.assertEqual(plan.mode, "overlap")
        self.assertEqual(plan.text_device, "cuda:0")
        self.assertEqual(plan.visual_device, "cuda:1")

    def test_explicit_overlap_fails_with_one_gpu(self) -> None:
        gpu, sequential, text, visual = self._common_settings()
        with gpu, sequential, text, visual, patch.object(
            settings, "PIPELINE_MODE", "overlap"
        ), patch(
            "walking_pipeline.runtime.cuda_device_count", return_value=1
        ):
            with self.assertRaises(RuntimeError):
                resolve_execution_plan()


class StorageValidationTests(unittest.TestCase):
    def test_spike_outputs_must_remain_in_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            data_dir = root / "data"
            outside_state = root / "outside.json"
            with patch.object(
                settings, "SPIKE1_REQUIRE_PERSISTENT_STORAGE", True
            ), patch.object(settings, "DATA_DIR", data_dir), patch.object(
                settings, "STATE_JSON", outside_state
            ), patch.object(
                settings, "OUTPUT_CSV", data_dir / "output.csv"
            ), patch.object(
                settings, "VIDEO_DIR", data_dir / "videos"
            ), patch.object(
                settings,
                "GEOCODE_CACHE_JSON",
                data_dir / "geocode.json",
            ), patch.object(
                settings, "HF_HOME", root / "huggingface"
            ), patch.object(
                settings,
                "HF_HUB_CACHE",
                root / "huggingface" / "hub",
            ), patch.dict(
                os.environ, {"WALK_DATA_DIR": str(data_dir)}
            ):
                with self.assertRaises(RuntimeError):
                    validate_storage()


if __name__ == "__main__":
    unittest.main()
