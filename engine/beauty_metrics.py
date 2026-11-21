"""Quantitative portrait metrics for the AI retouch flow.

The numbers here are deliberately boring and reproducible: MediaPipe gives the
landmarks, the multiclass segmenter gives skin/body/lip pixels, and OpenCV does
the colour and texture math.  The language model never has to guess ``L*`` or
``jaw_ratio``; it only translates these numbers into slider values.

Everything is a pure function over a BGR frame plus optional segmentation
carries, so the module can be unit-tested without the DSH host.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# MediaPipe FaceMesh groups.  Kept local so this module has no import cycle
# with beauty.py (which imports this module inside analyze_report()).
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
]
LIPS_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14,
    87, 178, 88, 95,
]

SEG_BODY_SKIN = 2
SEG_FACE_SKIN = 3
SEG_PERSON = (1, 2, 3, 4, 5)

# Symmetric landmark pairs used for the symmetry error.  The list only needs
# points that sit on the face itself; ears and the jaw hinge move too much with
# pose to be useful.
SYMMETRY_PAIRS = (
    (33, 263), (133, 362), (234, 454), (172, 397), (176, 400), (61, 291),
    (70, 300), (63, 293), (105, 334), (66, 296), (107, 336), (129, 358),
    (58, 288), (132, 361), (93, 323), (127, 356), (162, 389), (21, 251),
    (54, 284), (103, 332), (67, 297), (109, 338), (140, 369), (148, 377),
)


def _r(value: float | int | None, digits: int = 3):
    """Round for a compact, stable JSON payload."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _dist(pts: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(np.asarray(pts[a], np.float32) - np.asarray(pts[b], np.float32)))


def _polygon(shape: tuple[int, int], pts: np.ndarray, index: list[int], blur: float = 0.0) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(pts[index]).astype(np.int32)], 255)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    return mask.astype(np.float32) / 255.0


def _convex_hull(points: np.ndarray) -> np.ndarray:
    return cv2.convexHull(np.round(points).astype(np.int32)).reshape(-1, 2)


def _feature_mask(pts: np.ndarray, shape: tuple[int, int], fw: float) -> np.ndarray:
    """Brows, eyes and the lip opening; subtracted from the skin ROI."""
    mask = np.zeros(shape[:2], np.float32)
    for group in (LEFT_BROW, RIGHT_BROW, LEFT_EYE, RIGHT_EYE):
        mask = np.maximum(mask, _polygon(shape, pts, group, max(1.0, fw * 0.008)))
    outer = _convex_hull(pts[LIPS_OUTER])
    inner = _convex_hull(pts[LIPS_INNER])
    lips = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(lips, [outer.astype(np.int32)], 255)
    cv2.fillPoly(lips, [inner.astype(np.int32)], 0)
    mask = np.maximum(mask, lips.astype(np.float32) / 255.0)
    return np.clip(mask, 0.0, 1.0)


def _category_mask(seg: dict | None, shape: tuple[int, int], classes: tuple[int, ...]) -> np.ndarray | None:
    """Map a cached class-id crop back to source pixels."""
    if not isinstance(seg, dict):
        return None
    cat = seg.get("cat")
    box = seg.get("box")
    if cat is None or not getattr(cat, "size", 0) or not box:
        return None
    try:
        if len(box) == 4:
            bx, by, bw, bh = (float(v) for v in box)
        else:
            bx, by, side = (float(v) for v in box)
            bw = bh = side
    except (TypeError, ValueError):
        return None
    if bw <= 0 or bh <= 0:
        return None
    h, w = shape[:2]
    x0 = max(0, int(math.floor(bx)))
    y0 = max(0, int(math.floor(by)))
    x1 = min(w, int(math.ceil(bx + bw)))
    y1 = min(h, int(math.ceil(by + bh)))
    if x1 <= x0 or y1 <= y0:
        return None
    src = np.isin(cat, classes).astype(np.uint8) * 255
    src = cv2.resize(src, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = src
    return out


def _lab(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    return L, a, b


def _pixels(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return values[mask > 0.5]


def _skin_roi(pts: np.ndarray, bgr: np.ndarray, face_seg: dict | None) -> tuple[np.ndarray, dict, list[str]]:
    """Face-skin mask plus the segmenter's own face/body coverage stats."""
    h, w = bgr.shape[:2]
    fw = _dist(pts, 234, 454)
    notes: list[str] = []
    oval = _polygon((h, w), pts, FACE_OVAL, max(1.0, fw * 0.004))
    segmented = _category_mask(face_seg, (h, w), (SEG_FACE_SKIN,))
    if segmented is not None:
        skin = (segmented.astype(np.float32) / 255.0) * np.clip(oval * 2.0 + 0.35, 0.0, 1.0)
    else:
        notes.append("segmentation unavailable; skin metrics use the landmark oval")
        skin = oval
    skin = skin * (1.0 - _feature_mask(pts, (h, w), fw))
    # Pull the ROI a few pixels inside the visible skin so the blurred hairline
    # and the lip edge do not drag the colour stats around.
    k = max(3, int(round(fw * 0.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    skin = cv2.erode(skin, kernel)
    return np.clip(skin, 0.0, 1.0), {"face_width": fw, "segmented": segmented is not None}, notes


def _circle_mask(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.circle(mask, (int(round(centre[0])), int(round(centre[1]))), max(3, int(round(radius))), 1, -1)
    return mask.astype(np.float32)


def _region_median(L: np.ndarray, mask: np.ndarray) -> float | None:
    values = L[mask > 0.5]
    return float(np.median(values)) if values.size >= 30 else None


def analyze_metrics(
    bgr: np.ndarray,
    pts: np.ndarray,
    face_seg: dict | None = None,
    person_seg: dict | None = None,
    face_count: int = 1,
    face_index: int = 0,
) -> dict:
    """Compute the metric dictionary used by the prompt and the result card."""
    h, w = bgr.shape[:2]
    fw = max(1.0, _dist(pts, 234, 454))
    fh = max(1.0, _dist(pts, 10, 152))
    L, a, b = _lab(bgr)
    skin, meta, notes = _skin_roi(pts, bgr, face_seg)

    # --- pose ------------------------------------------------------------
    # X-only, as in the design: the nose tip relative to the face-width pair.
    yaw = (float(pts[1][0]) - (float(pts[234][0]) + float(pts[454][0])) / 2.0) / fw
    roll = math.degrees(math.atan2(float(pts[263][1] - pts[33][1]), float(pts[263][0] - pts[33][0])))
    reliable = abs(yaw) <= 0.08 and abs(roll) <= 6.0
    # Geometry degrades continuously with pose and face size; a single
    # reliable flag made every slightly-turned portrait lose slim/eyes.
    yaw_conf = max(0.0, min(1.0, 1.0 - abs(yaw) / 0.40))
    roll_conf = max(0.0, min(1.0, 1.0 - abs(roll) / 20.0))
    size_conf = max(0.0, min(1.0, fw / 180.0))
    geometry_confidence = yaw_conf * roll_conf * size_conf
    symmetry_reliable = abs(yaw) <= 0.10 and abs(roll) <= 6.0

    # --- geometry --------------------------------------------------------
    jaw = _dist(pts, 172, 397)
    chin = _dist(pts, 176, 400)
    eye_l = _dist(pts, 159, 145) / max(1e-6, _dist(pts, 33, 133))
    eye_r = _dist(pts, 386, 374) / max(1e-6, _dist(pts, 263, 362))
    eye_aspect = (eye_l + eye_r) / 2.0
    mouth = _dist(pts, 61, 291)
    axis = np.asarray(pts[152], np.float32) - np.asarray(pts[10], np.float32)
    norm = float(np.linalg.norm(axis)) or 1.0
    unit = axis / norm
    errors = []
    origin = np.asarray(pts[10], np.float32)
    for left, right in SYMMETRY_PAIRS:
        v = np.asarray(pts[left], np.float32) - origin
        reflected = 2.0 * float(np.dot(v, unit)) * unit - v + origin
        errors.append(float(np.linalg.norm(reflected - np.asarray(pts[right], np.float32))) / fw)
    symmetry = float(np.mean(errors)) if errors else None

    geometry = {
        "face_width": _r(fw, 1),
        "face_height": _r(fh, 1),
        "face_aspect": _r(fw / fh),
        "jaw_width": _r(jaw, 1),
        "jaw_ratio": _r(jaw / fw),
        "chin_width": _r(chin, 1),
        "chin_ratio": _r(chin / fw),
        "lower_face_ratio": _r(_dist(pts, 2, 152) / fh),
        "eye_aspect": _r(eye_aspect),
        "eye_aspect_l": _r(eye_l),
        "eye_aspect_r": _r(eye_r),
        "interocular_ratio": _r(_dist(pts, 133, 362) / fw),
        "mouth_width_ratio": _r(mouth / fw),
        "lip_fullness": _r(_dist(pts, 0, 17) / max(1e-6, mouth)),
        "symmetry": _r(symmetry),
        "nose_width_ratio": _r(_dist(pts, 129, 358) / fw),
    }

    # --- skin ------------------------------------------------------------
    skin_px = _pixels(L, skin)
    skin_available = skin_px.size >= 200
    skin_metrics = {
        "available": bool(skin_available),
        "l_median": None, "l_mean": None, "l_std": None, "l_p10": None, "l_p90": None,
        "a_median": None, "b_median": None, "hue_deg": None, "chroma": None,
        "ita_deg": None, "redness_p95": None, "shine_fraction": None,
        "blemish_density": None, "texture_energy": None, "under_eye_delta_l": None,
        "clipping": None,
    }
    if skin_available:
        a_px = _pixels(a, skin)
        b_px = _pixels(b, skin)
        l_med = float(np.median(skin_px))
        a_med = float(np.median(a_px))
        b_med = float(np.median(b_px))
        chroma = np.sqrt(a_px ** 2 + b_px ** 2)
        lab_b = b_med if abs(b_med) > 1e-3 else (1e-3 if b_med >= 0 else -1e-3)
        ita = math.degrees(math.atan((l_med - 50.0) / lab_b))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        med = cv2.medianBlur(gray, 15)
        diff = med.astype(np.float32) - gray.astype(np.float32)
        # Glasses frames, lashes and hair edges are dark with a steep local
        # gradient; blemishes are shallower, so reject high-gradient pixels.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gx, gy)
        blemish = (diff > 8) & (skin > 0.5) & (gradient < 90)
        # Texture is resolution-sensitive.  Compare a fixed-size crop so a
        # 124 px face and a 564 px face live on the same scale.
        ys, xs = np.where(skin > 0.5)
        texture_energy = None
        if ys.size >= 200:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            crop = gray[y0:y1, x0:x1]
            crop_mask = (skin[y0:y1, x0:x1] > 0.5).astype(np.uint8)
            if crop.size and crop_mask.any():
                scale = 256.0 / max(crop.shape[:2])
                size = (max(8, int(round(crop.shape[1] * scale))), max(8, int(round(crop.shape[0] * scale))))
                small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
                small_mask = cv2.resize(crop_mask, size, interpolation=cv2.INTER_NEAREST) > 0
                if small_mask.sum() >= 100:
                    texture_energy = float(np.var(cv2.Laplacian(small, cv2.CV_32F)[small_mask])) / (255.0 ** 2)
        cheek_r = max(8.0, fw * 0.18)
        cheek = np.maximum(
            _circle_mask((h, w), pts[50], cheek_r),
            _circle_mask((h, w), pts[280], cheek_r),
        ) * skin
        cheek_l = _region_median(L, cheek)
        under = np.maximum(
            _circle_mask((h, w), (float(pts[145][0]), float(pts[145][1] + fw * 0.03)), max(6.0, fw * 0.08)),
            _circle_mask((h, w), (float(pts[374][0]), float(pts[374][1] + fw * 0.03)), max(6.0, fw * 0.08)),
        ) * skin
        under_l = _region_median(L, under)
        skin_metrics.update({
            "l_median": _r(l_med, 2),
            "l_mean": _r(float(np.mean(skin_px)), 2),
            "l_std": _r(float(np.std(skin_px)), 2),
            "l_p10": _r(float(np.percentile(skin_px, 10)), 2),
            "l_p90": _r(float(np.percentile(skin_px, 90)), 2),
            "a_median": _r(a_med, 2),
            "b_median": _r(b_med, 2),
            "hue_deg": _r(math.degrees(math.atan2(b_med, a_med)) % 360.0, 1),
            "chroma": _r(float(np.median(chroma)), 2),
            "ita_deg": _r(ita, 1),
            "redness_p95": _r(float(np.percentile(a_px, 95)), 2),
            "shine_fraction": _r(float(np.mean((skin_px > 88) & (chroma < 20)))),
            "blemish_density": _r(float(np.mean(blemish[skin > 0.5]))),
            "texture_energy": _r(texture_energy, 5),
            "under_eye_delta_l": _r((cheek_l - under_l) if cheek_l is not None and under_l is not None else None, 2),
            "clipping": _r(float(np.mean((skin_px > 98) | (skin_px < 2)))),
        })

    # --- lips ------------------------------------------------------------
    lips = np.zeros((h, w), np.uint8)
    cv2.fillPoly(lips, [_convex_hull(pts[LIPS_OUTER]).astype(np.int32)], 255)
    cv2.fillPoly(lips, [_convex_hull(pts[LIPS_INNER]).astype(np.int32)], 0)
    lip_mask = lips.astype(np.float32) / 255.0
    lip_px = lip_mask > 0.5
    lips_metrics = {
        "available": bool(lip_px.sum() >= 50),
        "l_median": None, "a_median": None, "b_median": None,
        "chroma": None, "hue_deg": None,
    }
    if lip_px.sum() >= 50:
        lp = float(np.median(L[lip_px]))
        ap = float(np.median(a[lip_px]))
        bp = float(np.median(b[lip_px]))
        lips_metrics.update({
            "l_median": _r(lp, 2),
            "a_median": _r(ap, 2),
            "b_median": _r(bp, 2),
            "chroma": _r(math.hypot(ap, bp), 2),
            "hue_deg": _r(math.degrees(math.atan2(bp, ap)) % 360.0, 1),
        })

    # --- subject / body skin --------------------------------------------
    subject_ratio = 0.0
    face_ratio = 0.0
    person_mask = _category_mask(person_seg, (h, w), SEG_PERSON)
    if person_mask is not None:
        subject_ratio = float((person_mask > 0).mean())
    body_mask = _category_mask(person_seg, (h, w), (SEG_BODY_SKIN,))
    body_delta = None
    body_available = False
    body_l = None
    if body_mask is not None:
        body_roi = (body_mask.astype(np.float32) / 255.0) * (1.0 - _polygon((h, w), pts, FACE_OVAL, 2.0))
        body_px = _pixels(L, body_roi)
        body_available = body_px.size >= 200
        if body_available:
            body_l = float(np.median(body_px))
            if skin_metrics["l_median"] is not None:
                # Positive means the body is darker than the face, matching
                # the skill's "how much darker is the body" contract.
                body_delta = float(skin_metrics["l_median"]) - body_l

    if face_seg is not None:
        face_mask = _category_mask(face_seg, (h, w), (SEG_FACE_SKIN,))
        if face_mask is not None:
            face_ratio = float((face_mask > 0).mean())

    # --- background / light ---------------------------------------------
    border = max(2, min(h, w) // 12)
    bg = np.zeros((h, w), bool)
    bg[:border, :] = True
    bg[-border:, :] = True
    bg[:, :border] = True
    bg[:, -border:] = True
    bg_l = float(np.median(L[bg])) if bg.any() else float(np.median(L))
    clipping = skin_metrics["clipping"] or 0.0
    l_med = skin_metrics["l_median"]
    if l_med is None:
        light = "dim" if bg_l < 40 else "good"
    elif clipping > 0.02 and bg_l > l_med + 12:
        light = "backlit"
    elif l_med < 42 or bg_l < 38:
        light = "dim"
    elif l_med > 82 or clipping > 0.02:
        light = "bright"
    else:
        light = "good"

    if meta["face_width"] < 0.16 * min(w, h):
        notes.append("face is small in frame")
    if not reliable:
        notes.append("pose is off-axis")
    if clipping > 0.02:
        notes.append("skin is clipping")
    if person_mask is None:
        notes.append("subject segmentation unavailable")
    if lips_metrics["chroma"] is not None and lips_metrics["chroma"] > 45:
        notes.append("makeup may bias colour metrics")

    image = {
        "width": int(w),
        "height": int(h),
        "subject_ratio": _r(subject_ratio),
        "face_ratio": _r(face_ratio),
        "light": light,
    }
    pose = {
        "faces": int(face_count),
        "selected": int(face_index),
        "yaw": _r(yaw, 4),
        "roll_deg": _r(roll, 2),
        "reliable": bool(reliable),
        "geometry_confidence": _r(geometry_confidence),
        "symmetry_reliable": bool(symmetry_reliable),
    }
    body = {
        "available": bool(body_available),
        "l_median": _r(body_l, 2),
        "delta_l": _r(body_delta, 2),
    }
    quality = {
        "light": light,
        "clipping": _r(clipping),
        "segmentation": bool(face_seg is not None or person_seg is not None),
        "landmarks": bool(pts is not None and len(pts) >= 468),
        "notes": notes,
    }
    return {
        "image": image,
        "pose": pose,
        "geometry": geometry,
        "skin": skin_metrics,
        "body": body,
        "lips": lips_metrics,
        "quality": quality,
    }
