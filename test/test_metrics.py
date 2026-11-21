"""Quantitative metric tests — run with: .venv/bin/python -m unittest discover -s test -v"""

import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import beauty  # noqa: E402
import beauty_metrics  # noqa: E402

FACE = os.path.join(ROOT, "test-assets", "face1.jpg")
NO_FACE = os.path.join(tempfile.gettempdir(), "dsh-beauty-metrics-no-face.jpg")
cv2.imwrite(NO_FACE, np.full((360, 480, 3), 127, dtype=np.uint8))


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = beauty.load_image(FACE)
        cls.faces = beauty.FaceModel.get().detect(cls.image)

    def test_metrics_have_expected_shape_and_ranges(self):
        metrics = beauty_metrics.analyze_metrics(self.image, self.faces[0])
        for key in ("image", "pose", "geometry", "skin", "body", "lips", "quality"):
            self.assertIn(key, metrics)
        skin = metrics["skin"]
        self.assertTrue(skin["available"])
        self.assertGreater(skin["l_median"], 20)
        self.assertLess(skin["l_median"], 95)
        self.assertGreater(skin["l_std"], 0)
        self.assertTrue(0 < metrics["geometry"]["jaw_ratio"] < 1.5)
        self.assertIn("reliable", metrics["pose"])
        self.assertIn("geometry_confidence", metrics["pose"])
        self.assertIn("symmetry_reliable", metrics["pose"])
        confidence = metrics["pose"]["geometry_confidence"]
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
        self.assertGreaterEqual(metrics["skin"]["blemish_density"], 0)
        self.assertLessEqual(metrics["skin"]["blemish_density"], 1)

    def test_eye_aspect_is_the_mean_of_both_eyes(self):
        metrics = beauty_metrics.analyze_metrics(self.image, self.faces[0])
        geometry = metrics["geometry"]
        expected = (geometry["eye_aspect_l"] + geometry["eye_aspect_r"]) / 2
        self.assertAlmostEqual(geometry["eye_aspect"], expected, places=3)

    def test_body_delta_is_face_minus_body(self):
        metrics = beauty.analyze_report(FACE)["metrics"]
        body = metrics["body"]
        if not body["available"]:
            self.skipTest("test image has no body-skin region")
        expected = metrics["skin"]["l_median"] - body["l_median"]
        self.assertAlmostEqual(body["delta_l"], expected, places=2)

    def test_analyze_report_writes_three_vision_jpegs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = beauty.analyze_report(FACE, vision_out=tmp)
            self.assertTrue(report["ok"])
            self.assertEqual(report["faces"], 1)
            self.assertTrue(os.path.isfile(report["vision"]["full"]))
            self.assertTrue(os.path.isfile(report["vision"]["person"]))
            self.assertTrue(os.path.isfile(report["vision"]["face"]))
            full = cv2.imread(report["vision"]["full"])
            face = cv2.imread(report["vision"]["face"])
            self.assertLessEqual(max(full.shape[:2]), 1280)
            self.assertEqual(face.shape[:2], (768, 768))

    def test_no_face_report_is_clean(self):
        with mock.patch.object(beauty, "detect_faces", return_value=[]):
            report = beauty.analyze_report(NO_FACE)
        self.assertEqual(report["faces"], 0)
        self.assertFalse(report["metrics"]["skin"]["available"])
        self.assertTrue(any("no face" in note for note in report["notes"]))


if __name__ == "__main__":
    unittest.main()
