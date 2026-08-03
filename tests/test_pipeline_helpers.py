"""Unit tests for the Spike 1 execution helpers."""

from __future__ import annotations

import unittest
from collections import deque
from unittest.mock import patch

from walking_pipeline.model_loading import (
    build_model_kwargs,
    load_model_with_fallback,
)
from walking_pipeline.overlapped_pipeline import (
    _apply_text_result,
    _pending_visual_ids,
)


class ModelLoadingTests(unittest.TestCase):
    def test_non_quantised_loader_receives_dtype(self) -> None:
        calls = []

        def loader(model_name: str, **kwargs):
            calls.append((model_name, kwargs))
            return object()

        with patch("torch.cuda.is_available", return_value=False):
            result = load_model_with_fallback(
                loader,
                "example/model",
                device="cpu",
                load_in_4bit=False,
                model_label="test model",
            )

        self.assertIsNotNone(result)
        self.assertEqual(calls[0][0], "example/model")
        self.assertIn("torch_dtype", calls[0][1])

    def test_four_bit_request_falls_back_without_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            kwargs, using_four_bit = build_model_kwargs("cuda:0", True)

        self.assertFalse(using_four_bit)
        self.assertNotIn("quantization_config", kwargs)


class OverlapStateTests(unittest.TestCase):
    def test_pending_visual_records_include_retriable_statuses(self) -> None:
        state = {
            "videos": {
                "accepted": {
                    "text_decision": {"include": True},
                    "status": "text_accepted",
                },
                "failed": {
                    "text_decision": {"include": True},
                    "status": "download_failed",
                },
                "done": {
                    "text_decision": {"include": True},
                    "status": "complete",
                },
                "rejected": {
                    "text_decision": {"include": False},
                    "status": "text_rejected",
                },
            }
        }

        self.assertEqual(
            list(_pending_visual_ids(state)),
            ["accepted", "failed"],
        )

    def test_accepted_text_result_enters_visual_queue_once(self) -> None:
        state = {"videos": {"abc": {"metadata": {"title": "walk"}}}}
        ready = deque()
        known = set()
        message = {
            "video_id": "abc",
            "decision": {
                "include": True,
                "confidence": 0.9,
                "error": None,
            },
            "error": None,
        }

        _apply_text_result(state, message, ready, known)
        _apply_text_result(state, message, ready, known)

        self.assertEqual(list(ready), ["abc"])
        self.assertEqual(state["videos"]["abc"]["status"], "text_accepted")


if __name__ == "__main__":
    unittest.main()
