"""Regression tests for cut based visual review segmentation."""

from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

try:
    import torch  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")

    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    torch_stub.cuda = _CudaStub()  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_stub

from walking_pipeline import settings
from walking_pipeline.cut_detection import (
    build_segments,
    segment_duration_seconds,
    split_long_segments,
)
from walking_pipeline.video_filter import review_segments


class SegmentWindowTests(unittest.TestCase):
    def test_lisbon_long_segment_uses_fixed_review_windows(self) -> None:
        cut_times = [4.137, 7.107, 11.929, 16.066, 18.085, 24.074]
        cut_segments = build_segments(13986.541, cut_times)

        review_windows = split_long_segments(
            cut_segments,
            max_duration_seconds=300,
            min_duration_seconds=15,
        )

        long_windows = [
            segment
            for segment in review_windows
            if int(segment["start_time"]) >= 25
        ]
        self.assertEqual(len(long_windows), 47)
        self.assertEqual(long_windows[0]["start_time"], 25)
        self.assertEqual(long_windows[-1]["end_time"], 13986)
        self.assertTrue(
            all(
                15 <= segment_duration_seconds(segment) <= 300
                for segment in long_windows
            )
        )
        self.assertTrue(
            all(
                previous["end_time"] + 1 == following["start_time"]
                for previous, following in zip(
                    long_windows,
                    long_windows[1:],
                )
            )
        )

    def test_balancing_does_not_create_a_short_final_window(self) -> None:
        review_windows = split_long_segments(
            [{"start_time": 0, "end_time": 304}],
            max_duration_seconds=300,
            min_duration_seconds=15,
        )

        self.assertEqual(len(review_windows), 2)
        self.assertEqual(
            [segment_duration_seconds(segment) for segment in review_windows],
            [153, 152],
        )

    def test_short_segment_is_rejected_without_visual_model_call(self) -> None:
        class UnexpectedJudge:
            def judge_segment(self, *args: object, **kwargs: object) -> object:
                raise AssertionError("The visual model must not be called")

        with patch.object(settings, "MIN_SEGMENT_DURATION_SECONDS", 15):
            reviews = review_segments(
                "example",
                Path("unused.mp4"),
                100.0,
                {},
                [{"start_time": 0, "end_time": 13}],
                UnexpectedJudge(),  # type: ignore[arg-type]
            )

        self.assertEqual(len(reviews), 1)
        self.assertFalse(reviews[0].include)
        self.assertIsNone(reviews[0].error)
        self.assertEqual(reviews[0].quality_issues, ["segment_too_short"])

    def test_exactly_fifteen_seconds_reaches_clip_creation(self) -> None:
        with (
            patch.object(settings, "MIN_SEGMENT_DURATION_SECONDS", 15),
            patch(
                "walking_pipeline.video_filter.create_segment_review_clip",
                return_value=False,
            ) as create_clip,
        ):
            reviews = review_segments(
                "example",
                Path("unused.mp4"),
                100.0,
                {},
                [{"start_time": 0, "end_time": 14}],
                object(),  # type: ignore[arg-type]
            )

        create_clip.assert_called_once()
        self.assertEqual(reviews[0].error, "clip_creation_failed")


if __name__ == "__main__":
    unittest.main()