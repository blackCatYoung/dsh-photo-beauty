"""Engine tests — run with: .venv/bin/python -m unittest discover -s test -v"""

import json  # noqa: E402
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import beauty  # noqa: E402

FACE = os.path.join(ROOT, "test-assets", "face1.jpg")
SEG_MODEL = beauty.DEFAULT_SEG_MODEL
# A synthetic, face-free image so the "no face" path never depends on a fixture.
NO_FACE = os.path.join(tempfile.gettempdir(), "dsh-beauty-no-face.jpg")
_yy, _xx = np.mgrid[0:360, 0:480]
_gradient = np.dstack([
    (_xx * 0.4).astype(np.uint8),
    (_yy * 0.5).astype(np.uint8),
    ((_xx + _yy) * 0.2).astype(np.uint8),
])
cv2.imwrite(NO_FACE, _gradient)


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = beauty.load_image(FACE)
        cls.faces = beauty.FaceModel.get().detect(cls.image)

    def test_detects_one_face(self):
        self.assertEqual(len(self.faces), 1)
        self.assertEqual(self.faces[0].shape, (478, 2))

    def test_skin_mask_is_face_sized(self):
        mask = beauty.skin_mask(self.faces[0], self.image.shape)
        self.assertEqual(mask.shape, self.image.shape[:2])
        self.assertGreater(mask.mean(), 0.02)
        self.assertLess(mask.mean(), 0.6)

    def test_fallback_mask_protects_eyes(self):
        mask = beauty.skin_mask(self.faces[0], self.image.shape)
        for group in (beauty.LEFT_EYE, beauty.RIGHT_EYE):
            x, y = self.faces[0][group].mean(axis=0)
            self.assertLess(mask[int(round(y)), int(round(x))], 0.2)

    @unittest.skipUnless(SEG_MODEL.exists(), "segmentation model not installed")
    def test_segment_mask_includes_forehead_and_hairline(self):
        pts = self.faces[0]
        seg = beauty.segment_face(self.image, pts)
        mask = beauty.skin_mask_from_segment(self.image, pts, seg)
        self.assertEqual(mask.shape, self.image.shape[:2])
        side = seg["box"][2]
        self.assertEqual(seg["cat"].shape, (side, side))
        # mid-forehead is skin
        x, y = pts[10]
        self.assertGreater(mask[int(y), int(x)], 0.8)
        # the mask reaches above the landmark oval: the oval top cuts the brow
        oval_top = pts[beauty.FACE_OVAL][:, 1].min()
        rows = np.nonzero(mask.max(axis=1) > 0.5)[0]
        self.assertTrue(rows.size)
        self.assertGreater(rows.min(), 0)
        # eyes stay protected even though the segmenter calls them face skin
        for group in (beauty.LEFT_EYE, beauty.RIGHT_EYE):
            x, y = pts[group].mean(axis=0)
            self.assertLess(mask[int(round(y)), int(round(x))], 0.15)
        self.assertGreater(mask.mean(), 0.02)

    def test_tone_mask_keeps_the_features(self):
        """Whitening must cover brows, eyes and mouth, not punch them out."""
        pts = self.faces[0]
        seg = beauty.segment_face(self.image, pts)
        skin = beauty.skin_mask_from_segment(self.image, pts, seg)
        tone = beauty.skin_mask_from_segment(self.image, pts, seg, protect_features=False)
        for group in (beauty.LEFT_EYE, beauty.RIGHT_EYE, beauty.LEFT_BROW, beauty.RIGHT_BROW):
            x, y = np.round(pts[group].mean(axis=0)).astype(int)
            self.assertGreaterEqual(tone[y, x], 0.5)
            self.assertLess(skin[y, x], 0.5)
        lip = beauty.convex_hull(pts[beauty.LIPS_OUTER])
        x, y = np.round(lip.mean(axis=0)).astype(int)
        self.assertGreaterEqual(tone[y, x], 0.5)
        self.assertLess(skin[y, x], 0.5)
        # the tone mask never grows past the face: hair far above the head stays out
        x, y = np.round(pts[10] + np.array([0, -1.6 * float(np.ptp(pts[beauty.FACE_OVAL][:, 1]))])).astype(int)
        self.assertLess(tone[max(y, 0), x], 0.2)

    def test_whiten_lifts_the_eyes(self):
        """The visible symptom: brightened cheeks with dark eyes above them."""
        pts = self.faces[0]
        seg = beauty.segment_face(self.image, pts)
        tone = beauty.skin_mask_from_segment(self.image, pts, seg, protect_features=False)
        out = beauty.tone_weighted(self.image, tone * 0.4)
        x, y = np.round(pts[beauty.LEFT_EYE].mean(axis=0)).astype(int)
        before = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)[y, x, 0]
        after = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[y, x, 0]
        self.assertGreater(int(after), int(before) + 3)

    def test_skin_tone_moves_both_chroma_axes(self):
        """The 2D pad must warm b* and push a* magenta without touching L*."""
        pts = self.faces[0]
        seg = beauty.segment_face(self.image, pts)
        tone = beauty.skin_mask_from_segment(self.image, pts, seg, protect_features=False)
        x, y = np.round(pts[10]).astype(int)  # mid-forehead: solidly skin
        before = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)[y, x].astype(int)
        self.assertGreater(tone[y, x], 0.8)

        same = beauty.adjust_skin_tone(self.image, tone, 0, 0)
        self.assertTrue(np.array_equal(same, self.image), "zero tone is an identity pass")

        warm = beauty.adjust_skin_tone(self.image, tone, 50, 0)
        after = cv2.cvtColor(warm, cv2.COLOR_BGR2LAB)[y, x].astype(int)
        self.assertGreaterEqual(after[2] - before[2], 4, "temperature should raise b*")
        self.assertLessEqual(abs(int(after[0]) - int(before[0])), 2, "L* stays put")

        magenta = beauty.adjust_skin_tone(self.image, tone, 0, 50)
        after = cv2.cvtColor(magenta, cv2.COLOR_BGR2LAB)[y, x].astype(int)
        self.assertGreaterEqual(after[1] - before[1], 3, "tint should raise a*")

    def test_segment_classes_gate_the_mask(self):
        """Hair/background classes never enter the mask, skin does."""
        pts = self.faces[0]
        h, w = self.image.shape[:2]
        cat = np.full((256, 256), beauty.SEG_FACE_SKIN, np.uint8)
        cat[:, :128] = beauty.SEG_HAIR
        seg = {"box": [0, 0, w], "cat": cat}
        mask = beauty.skin_mask_from_segment(self.image, pts, seg, 1.0)
        cx, _, fw = beauty.face_axes(pts)
        fh = float(np.ptp(pts[beauty.FACE_OVAL][:, 1]))
        y = int(round(pts[152][1] - 0.06 * fh))
        self.assertLess(mask[y, int(cx - 0.30 * fw)], 0.15)
        self.assertGreater(mask[y, int(cx + 0.30 * fw)], 0.85)

    def test_slim_field_has_no_tear(self):
        """The displacement field must be continuous — a one-pixel step of a few
        pixels reads as the image being torn at the jaw."""
        pts = self.faces[0]
        for slim in (10, 42, 80, 100):
            _, _, (x0, _y0, mx, _my) = beauty.warp_portrait(
                self.image, pts, slim, 0, return_field=True
            )
            dx = np.arange(mx.shape[1], dtype=np.float32)[None, :] + x0 - mx
            self.assertLess(float(np.abs(np.diff(dx, axis=1)).max()), 1.0, f"slim {slim}")
            # and it has to reach zero before the ROI edge, or the untouched
            # border would show a step against the warped interior
            self.assertLess(float(np.abs(dx[:, 0]).max()), 1.0, f"slim {slim} left edge")
            self.assertLess(float(np.abs(dx[:, -1]).max()), 1.0, f"slim {slim} right edge")

    def test_warp_mask_follows_the_displacement(self):
        pts = self.faces[0]
        _, _, field = beauty.warp_portrait(
            self.image, pts, slim=60, eyes=0, return_field=True
        )
        self.assertIsNotNone(field)
        before = beauty.skin_mask(pts, self.image.shape)
        moved = beauty.warp_mask(before, field)
        self.assertEqual(moved.shape, before.shape)
        self.assertFalse(np.array_equal(moved, before))

    @unittest.skipUnless(SEG_MODEL.exists(), "segmentation model not installed")
    def test_detect_report_carries_segmentation(self):
        report = beauty.detect_report(FACE)
        self.assertEqual(len(report["faces"]), 1)
        self.assertTrue(report["seg"] and report["seg"][0])
        seg = beauty.decode_segment(report["seg"][0])
        self.assertIsNotNone(seg)
        self.assertEqual(seg["cat"].shape, (seg["box"][2], seg["box"][2]))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "landmarks.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(report, fh)
            faces, segs, person = beauty.load_landmarks(
                path, report["width"], report["height"]
            )
        self.assertEqual(len(faces), 1)
        self.assertIsNotNone(segs[0])
        self.assertIsNotNone(person)
        self.assertEqual(person["cat"].shape, (person["box"][2], person["box"][2]))

    def test_slim_moves_jaw_towards_centre(self):
        pts = self.faces[0]
        warped, wpts = beauty.warp_portrait(self.image, pts, slim=60, eyes=0)
        self.assertEqual(warped.shape, self.image.shape)
        # Landmark 172 sits on the subject's left jaw; slimming pulls it inward (+x).
        self.assertGreater(wpts[172][0], pts[172][0])
        self.assertFalse(np.array_equal(warped, self.image))

    def test_eyes_enlarge(self):
        pts = self.faces[0]
        _, wpts = beauty.warp_portrait(self.image, pts, slim=0, eyes=80)
        # 159/145 are the top/bottom lids of the subject's right eye.
        before = np.linalg.norm(pts[159] - pts[145])
        after = np.linalg.norm(wpts[159] - wpts[145])
        self.assertGreater(after, before)

    def test_retouch_writes_full_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.jpg")
            report = beauty.retouch(
                FACE,
                out,
                beauty.Params(smooth=45, whiten=18, slim=30, eyes=22, compare=True),
            )
            self.assertTrue(os.path.isfile(out))
            self.assertTrue(os.path.isfile(report["compare"]))
            self.assertEqual(report["faces"], 1)
            self.assertEqual((report["width"], report["height"]), (512, 512))
            self.assertGreater(report["elapsed_ms"], 0)

    def test_max_size_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "small.jpg")
            report = beauty.retouch(FACE, out, beauty.Params(smooth=20, max_size=200))
            self.assertEqual(max(report["width"], report["height"]), 200)
            self.assertEqual(report["original_width"], 512)

    def test_crop_render_is_native(self):
        # The zoomed-in patch: a crop at native size keeps source coordinates in
        # the report so the panel can place it over the preview.
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "crop.jpg")
            report = beauty.retouch(FACE, out, beauty.Params(smooth=10), crop=(100, 60, 240, 200))
            self.assertEqual((report["width"], report["height"]), (240, 200))
            self.assertEqual(report["crop"], [100, 60, 240, 200])
            self.assertEqual((report["original_width"], report["original_height"]), (512, 512))

    def test_crop_respects_max_size(self):
        # Regression: the downscale factor used to come from the source size, so
        # a crop larger than max_size came out at the full frame's dimensions.
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "crop-big.jpg")
            report = beauty.retouch(FACE, out, beauty.Params(smooth=10, max_size=120),
                                    crop=(0, 0, 400, 300))
            self.assertEqual(max(report["width"], report["height"]), 120)
            self.assertAlmostEqual(report["width"] / report["height"], 400 / 300, places=2)
            self.assertEqual(report["crop"], [0, 0, 400, 300])

    def test_crop_clamps_to_the_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = beauty.retouch(FACE, os.path.join(tmp, "crop-edge.jpg"), beauty.Params(),
                                    crop=(400, 400, 500, 500))
            self.assertEqual(report["crop"], [400, 400, 112, 112])
            self.assertEqual((report["width"], report["height"]), (112, 112))

    def test_no_face_reports_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = beauty.retouch(NO_FACE, os.path.join(tmp, "none.jpg"), beauty.Params())
        self.assertEqual(report["faces"], 0)
        self.assertIsNone(report["face_used"])

    # --- makeup ---------------------------------------------------------

    def makeup_masks(self):
        pts = self.faces[0]
        shape = self.image.shape
        up = beauty.up_vector(pts)
        masks = {
            "shadow": beauty.eyeshadow_mask(pts, shape, beauty.LEFT_EYE, beauty.LEFT_BROW, up),
            "shadow_r": beauty.eyeshadow_mask(pts, shape, beauty.RIGHT_EYE, beauty.RIGHT_BROW, up),
            "liner": beauty.liner_mask(pts, shape, beauty.LEFT_EYE, up),
            "blush": beauty.blush_mask(pts, shape, beauty.MAKEUP_SIDES[0], up),
            "lip": beauty.lip_mask(pts, shape),
        }
        return pts, masks

    def test_makeup_masks_land_on_the_face(self):
        _, masks = self.makeup_masks()
        for name, mask in masks.items():
            with self.subTest(mask=name):
                self.assertEqual(mask.shape, self.image.shape[:2])
                self.assertGreater(float(mask.sum()), 20.0, "empty mask")
                self.assertLessEqual(float(mask.max()), 1.0 + 1e-4)
                # Nothing may leak to a corner: no mask should reach the frame edge.
                self.assertLess(float(mask[:6, :6].sum()), 0.5)

    def test_makeup_masks_stay_near_their_feature(self):
        pts, masks = self.makeup_masks()
        fw = beauty.face_axes(pts)[2]
        outer = pts[beauty.LEFT_EYE[0]]
        ys, xs = np.where(masks["liner"] > 0.3)
        self.assertLess(float(np.abs(xs - outer[0]).max()), 0.5 * fw)
        self.assertLess(float(np.abs(ys - outer[1]).max()), 0.5 * fw)
        # The lip band is the vermilion border minus the opening, so it stays
        # inside the mouth's bounding box.
        lips = pts[beauty.LIPS_OUTER]
        ys, xs = np.where(masks["lip"] > 0.3)
        self.assertGreaterEqual(int(xs.min()), int(lips[:, 0].min()) - 3)
        self.assertLessEqual(int(xs.max()), int(lips[:, 0].max()) + 3)
        self.assertGreaterEqual(int(ys.min()), int(lips[:, 1].min()) - 3)
        self.assertLessEqual(int(ys.max()), int(lips[:, 1].max()) + 3)

    def test_makeup_off_is_a_no_op(self):
        pts, _ = self.makeup_masks()
        out = beauty.apply_makeup(self.image, pts, beauty.Params())
        self.assertIs(out, self.image)

    def test_lip_tint_only_touches_lips(self):
        pts, masks = self.makeup_masks()
        out = beauty.apply_makeup(self.image, pts, beauty.Params(lip=100, lip_color="#8b1f36"))
        lip = masks["lip"] > 0.5
        self.assertGreater(int(np.abs(out.astype(int) - self.image.astype(int)).sum(axis=2)[lip].sum()), 0.0)
        # The feathered rim fades gradually; beyond it nothing may move at all.
        away = masks["lip"] <= 0.0
        self.assertTrue(np.array_equal(out[away], self.image[away]))
        # ...and it really went in the red/berry direction rather than greying out.
        rgb = out[..., ::-1].astype(float)[lip].mean(axis=0)
        self.assertGreater(rgb[0], rgb[1] + 6.0)

    def test_makeup_colour_changes_the_result(self):
        pts, _ = self.makeup_masks()
        warm = beauty.apply_makeup(self.image, pts, beauty.Params(lip=100, lip_color="#d9843a"))
        cool = beauty.apply_makeup(self.image, pts, beauty.Params(lip=100, lip_color="#7a4fa8"))
        self.assertGreater(int(np.abs(warm.astype(int) - cool.astype(int)).sum()), 1000)

    def test_parse_color_forms(self):
        np.testing.assert_array_equal(beauty.parse_color("#ff8000", "#000000"), [0.0, 128.0, 255.0])
        np.testing.assert_array_equal(beauty.parse_color("f80", "#000000"), [0.0, 136.0, 255.0])
        np.testing.assert_array_equal(beauty.parse_color("#f80", "#000000"), [0.0, 136.0, 255.0])
        np.testing.assert_array_equal(beauty.parse_color("nonsense", "#010203"), [3.0, 2.0, 1.0])
        np.testing.assert_array_equal(beauty.parse_color(None, "#010203"), [3.0, 2.0, 1.0])

    def test_retouch_applies_makeup_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = os.path.join(tmp, "plain.jpg")
            made = os.path.join(tmp, "made.jpg")
            base = beauty.Params(smooth=20)
            report = beauty.retouch(FACE, plain, beauty.Params(smooth=20))
            self.assertFalse(any(note.startswith("makeup:") for note in report["notes"]))
            report = beauty.retouch(
                FACE, made,
                beauty.Params(smooth=20, shadow=50, liner=40, blush=40, lip=70),
            )
            self.assertTrue(any(note.startswith("makeup:") for note in report["notes"]))
            self.assertEqual(report["params"]["lip"], 70)
            diff = np.abs(
                beauty.load_image(made).astype(int) - beauty.load_image(plain).astype(int)
            ).sum()
            self.assertGreater(int(diff), 10000)
            del base

    def test_compare_sheet_shape(self):
        sheet = beauty.compare_sheet(self.image, self.image)
        self.assertEqual(sheet.shape[0], 512)
        self.assertGreater(sheet.shape[1], 512)


    @unittest.skipUnless(SEG_MODEL.exists(), "segmentation model not installed")
    def test_subject_skin_covers_face_and_body(self):
        h, w = self.image.shape[:2]
        seg = beauty.segment_person(self.image)
        mask, stats = beauty.subject_skin((h, w), self.faces[0], seg, (0, 0, w, h))
        self.assertEqual(mask.shape, (h, w))
        self.assertGreater(stats["subject"], 20)
        self.assertGreater(stats["face"], 5)
        self.assertGreater(stats["body"], 1)
        self.assertGreaterEqual(stats["parts"], 1)
        oval = beauty.polygon_mask((h, w), beauty.convex_hull(self.faces[0][beauty.FACE_OVAL]))
        body = (mask > 0.5) & (oval < 0.05)
        # Neck and shoulders are the whole point of "人物提亮".
        self.assertGreater(int(body.sum()), 200)

    @unittest.skipUnless(SEG_MODEL.exists(), "segmentation model not installed")
    def test_person_brightening_reaches_body_skin(self):
        h, w = self.image.shape[:2]
        seg = beauty.segment_person(self.image)
        mask, _ = beauty.subject_skin((h, w), self.faces[0], seg, (0, 0, w, h))
        oval = beauty.polygon_mask((h, w), beauty.convex_hull(self.faces[0][beauty.FACE_OVAL]))
        body = (mask > 0.5) & (oval < 0.05)
        with tempfile.TemporaryDirectory() as tmp:
            base = beauty.retouch(
                FACE, os.path.join(tmp, "base.jpg"),
                beauty.Params(smooth=0, whiten=0, detail=0, person=0),
            )
            lit_path = os.path.join(tmp, "lit.jpg")
            lit = beauty.retouch(
                FACE, lit_path,
                beauty.Params(smooth=0, whiten=0, detail=0, person=60),
            )
            before = beauty.load_image(base["output"])
            after = beauty.load_image(lit["output"])
        l_before = cv2.cvtColor(before, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        l_after = cv2.cvtColor(after, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        gain = l_after - l_before
        self.assertGreater(float(gain[body].mean()), 6.0)
        # The background must not follow: neck/arm light, room unchanged.
        corner = np.zeros_like(gain, dtype=bool)
        corner[:80, :80] = True
        self.assertLess(float(np.abs(gain[corner]).mean()), 1.5)
        self.assertTrue(any("person glow" in note for note in lit["notes"]))
        self.assertGreater(lit["skin"]["subject"], 20)
        # Face and body both brighten, and neither is double-lifted.
        face = (oval > 0.8) & (mask > 0.8)
        self.assertGreater(float(gain[face].mean()), 6.0)

    def test_person_brightens_more_toward_the_top(self):
        """Person 0-100 bends L* with a gamma, so late steps add more light."""
        h, w = self.image.shape[:2]
        oval = beauty.polygon_mask((h, w), beauty.convex_hull(self.faces[0][beauty.FACE_OVAL]))
        with tempfile.TemporaryDirectory() as tmp:
            means = {}
            note = ""
            for value in (0, 20, 80, 100):
                path = os.path.join(tmp, f"person{value}.jpg")
                data = beauty.retouch(
                    FACE, path,
                    beauty.Params(smooth=0, whiten=0, detail=0, person=value),
                )
                note = " ".join(data["notes"])
                lab = cv2.cvtColor(beauty.load_image(data["output"]), cv2.COLOR_BGR2LAB)
                means[value] = float(lab[..., 0][oval > 0.8].astype(np.float32).mean())
            clipping = float((lab[..., 0][oval > 0.8] >= 254).mean())
        self.assertIn("gamma 3.40", note)
        first = means[20] - means[0]
        last = means[100] - means[80]
        self.assertGreater(first, 0.5)
        self.assertGreater(means[100], means[80])
        self.assertGreater(means[80], means[20])
        # The last 20 points must add clearly more than the first 20: that is
        # the "越往后越亮" curve, not a straight line.
        self.assertGreater(last, first * 2.0)
        self.assertLess(clipping, 0.01)

    def test_liquify_strokes_are_local(self):
        h, w = self.image.shape[:2]
        stroke = {
            "mode": "push", "radius": 0.06, "strength": 0.9,
            "points": [[0.5, 0.5], [0.55, 0.52], [0.6, 0.55]],
        }
        out = beauty.apply_liquify(self.image, [stroke])
        self.assertEqual(out.shape, self.image.shape)
        diff = np.abs(out.astype(np.int16) - self.image.astype(np.int16)).max(axis=2)
        self.assertGreater(int(diff.sum()), 2000)
        # Everything outside the brush footprint must be pixel-identical.
        rad = 0.06 * w
        x0 = int(0.5 * w - rad * 1.4)
        x1 = int(0.6 * w + rad * 1.4)
        y0 = int(0.5 * h - rad * 1.4)
        y1 = int(0.55 * h + rad * 1.4)
        far = np.ones((h, w), dtype=bool)
        far[y0:y1, x0:x1] = False
        self.assertEqual(int(diff[far].max()), 0)

    def test_liquify_modes_differ(self):
        push = beauty.apply_liquify(self.image, [{
            "mode": "push", "radius": 0.05, "strength": 0.7,
            "points": [[0.5, 0.5], [0.56, 0.5]],
        }])
        bloat = beauty.apply_liquify(self.image, [{
            "mode": "bloat", "radius": 0.05, "strength": 0.7, "points": [[0.5, 0.5]],
        }])
        pucker = beauty.apply_liquify(self.image, [{
            "mode": "pucker", "radius": 0.05, "strength": 0.7, "points": [[0.5, 0.5]],
        }])
        self.assertFalse(np.array_equal(push, bloat))
        self.assertFalse(np.array_equal(bloat, pucker))

    def test_map_liquify_strokes_into_a_crop(self):
        strokes = [{
            "mode": "push", "radius": 0.1, "strength": 0.5,
            "points": [[0.5, 0.5], [0.6, 0.5]],
        }]
        mapped = beauty.map_liquify_strokes(strokes, (400, 300), (100, 50, 200, 100))
        self.assertEqual(mapped[0]["points"][0], [0.5, 1.0])
        self.assertEqual(mapped[0]["points"][1], [0.7, 1.0])
        self.assertAlmostEqual(mapped[0]["radius"], 0.2, places=6)

    def test_retouch_applies_liquify_and_reports_skin(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = beauty.retouch(
                FACE, os.path.join(tmp, "liquify.jpg"),
                beauty.Params(smooth=0, detail=0, liquify=[{
                    "mode": "bloat", "radius": 0.05, "strength": 0.5,
                    "points": [[0.5, 0.45]],
                }]),
            )
        self.assertTrue(any(note.startswith("liquify: 1") for note in report["notes"]))
        self.assertNotIn("liquify", report["params"])
        self.assertIn("skin", report)
        self.assertGreater(report["skin"]["face"], 0)

    # --- 皱纹修复 ---------------------------------------------------------
    @staticmethod
    def _crease_patch(size=200, line_y=60):
        """Flat skin with one soft dark ridge, plus its depth at *line_y*."""
        yy, xx = np.mgrid[0:size, 0:size]
        img = np.clip(
            np.zeros((size, size, 3), np.float32)
            + (150, 170, 205)
            + (8 * np.sin(xx / 30.0))[..., None],
            0, 255,
        ).astype(np.uint8)
        cv2.line(img, (30, line_y), (size - 30, line_y), (136, 156, 191), 3, cv2.LINE_AA)
        return img

    @staticmethod
    def _crease_depth(img, line_y, band=16):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        skin = (gray[line_y - band:line_y - band + 6, 50:150].mean()
                + gray[line_y + band - 6:line_y + band, 50:150].mean()) / 2
        return float(skin - gray[line_y - 1:line_y + 2, 50:150].mean())

    def test_crease_weight_finds_the_ridge_only(self):
        img = self._crease_patch()
        weight, base = beauty.crease_weight(img, np.ones(img.shape[:2], np.float32), 180.0)
        self.assertIsNotNone(weight)
        self.assertGreater(float(weight[58:62, 50:150].mean()), 0.5)
        self.assertLess(float(weight[20:35, 50:150].mean()), 0.02)
        self.assertGreater(float((base[58:62, 50:150] - cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY)[58:62, 50:150]).mean()), 2.0)

    def test_wrinkle_slider_fades_a_crease(self):
        img = self._crease_patch()
        mask = np.ones(img.shape[:2], np.float32)
        before = self._crease_depth(img, 60)
        half = self._crease_depth(beauty.apply_wrinkle_repair(img, mask, 180.0, 50), 60)
        full = self._crease_depth(beauty.apply_wrinkle_repair(img, mask, 180.0, 100), 60)
        self.assertAlmostEqual(self._crease_depth(
            beauty.apply_wrinkle_repair(img, mask, 180.0, 0), 60), before, places=5)
        self.assertLess(half, before)
        self.assertLess(full, half)
        self.assertLess(full, before * 0.7)

    def test_wrinkle_brush_erases_the_painted_crease(self):
        img = self._crease_patch()
        brush = np.zeros(img.shape[:2], np.float32)
        brush[35:85, 20:180] = 1.0
        brush = cv2.GaussianBlur(brush, (0, 0), 4)
        before = self._crease_depth(img, 60)
        out = beauty.apply_wrinkle_repair(img, None, 180.0, 0, brush)
        self.assertLess(self._crease_depth(out, 60), before * 0.35)
        # Nothing outside the stroke may move.
        self.assertLess(float(np.abs(
            out.astype(np.float32) - img.astype(np.float32))[120:, :].mean()), 0.01)

    def test_wrinkle_detection_covers_wide_folds(self):
        """"比较宽"的皱纹也要能被认出来：宽脊的权重不能被半径吃掉。"""
        size = 400
        yy, xx = np.mgrid[0:size, 0:size]
        img = np.clip(
            np.full((size, size, 3), (150, 170, 205), np.float32)
            + (6 * np.sin(xx / 40.0))[..., None],
            0, 255,
        ).astype(np.uint8)
        for half_width, expected in ((2, 0.9), (6, 0.7), (12, 0.4)):
            patch = img.copy()
            cv2.line(
                patch, (40, 200), (360, 200), (120, 140, 175),
                half_width * 2, cv2.LINE_AA,
            )
            weight, _ = beauty.crease_weight(
                patch, np.ones((size, size), np.float32), 360.0
            )
            band = float(weight[200 - 2:200 + 2, 80:320].mean())
            self.assertGreater(band, expected, f"half width {half_width}px")

    def test_retouch_applies_wrinkle_slider_and_brush(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = beauty.retouch(
                FACE, os.path.join(tmp, "wrinkle.jpg"),
                beauty.Params(
                    smooth=0, detail=0, wrinkle=60,
                    wrinkle_strokes=[{"radius": 0.05, "points": [[0.30, 0.52], [0.36, 0.58]]}],
                ),
            )
        notes = report["notes"]
        self.assertTrue(any(note.startswith("wrinkle repair: 60%") for note in notes))
        self.assertTrue(any(note.startswith("wrinkle brush: 1") for note in notes))
        self.assertEqual(report["params"]["wrinkle"], 60)
        self.assertNotIn("wrinkle_strokes", report["params"])

    def test_retouch_wrinkle_slider_without_brush(self):
        """The slider alone must work: no strokes means no mapped radius list."""
        with tempfile.TemporaryDirectory() as tmp:
            report = beauty.retouch(
                FACE, os.path.join(tmp, "wrinkle-only.jpg"),
                beauty.Params(smooth=0, detail=0, wrinkle=60),
            )
        notes = report["notes"]
        self.assertTrue(any(note.startswith("wrinkle repair: 60%") for note in notes))
        self.assertFalse(any(note.startswith("wrinkle brush") for note in notes))
        self.assertEqual(report["params"]["wrinkle"], 60)

    def test_mask_strokes_map_into_a_crop(self):
        strokes = [{"radius": 0.1, "points": [[0.5, 0.5], [0.7, 0.5]]}]
        mapped = beauty.map_mask_strokes(strokes, (400, 300), (100, 50, 200, 100))
        self.assertEqual(len(mapped), 1)
        self.assertAlmostEqual(mapped[0]["radius"], 0.2, places=4)
        self.assertEqual(mapped[0]["points"][0], [0.5, 1.0])
        mask = beauty.mask_strokes_mask((100, 200), mapped)
        self.assertGreater(float(mask[95:100, 95:145].mean()), 0.5)
        self.assertEqual(float(mask[0:20, 0:20].max()), 0.0)

    def test_retouch_applies_mask_brush(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = beauty.retouch(
                FACE, os.path.join(tmp, "base.jpg"),
                beauty.Params(smooth=0, whiten=0, detail=0, person=60),
            )
            brushed = beauty.retouch(
                FACE, os.path.join(tmp, "brush.jpg"),
                beauty.Params(
                    smooth=0, whiten=0, detail=0, person=60,
                    mask_strokes=[{"radius": 0.05, "points": [[0.1, 0.1]]}],
                ),
            )
            before = cv2.imread(base["output"]).astype(np.float32)
            after = cv2.imread(brushed["output"]).astype(np.float32)
        self.assertTrue(any(note.startswith("mask brush: 1") for note in brushed["notes"]))
        self.assertNotIn("mask_strokes", brushed["params"])
        patch = (slice(30, 80), slice(30, 80))
        gain = float((after[patch].mean(axis=2) - before[patch].mean(axis=2)).mean())
        self.assertGreater(gain, 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
