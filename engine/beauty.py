#!/usr/bin/env python3
"""Portrait beauty engine for the DSH photo plugin.

One photo in, one retouched photo out, with four tone/geometry controls plus
four landmark-painted makeup layers and a manual liquify brush:

    smooth   磨皮  — edge-preserving skin smoothing (guided filter + detail blend)
    whiten   提亮  — lifts the skin's L* channel and slightly calms saturation
    person   人物提亮 — gamma brightening 0-100 (Levels-style curve) on every
             skin part of the subject
    wrinkle  皱纹修复 — fades crease-like ridges 0-100 on the face skin, plus
             brush strokes (wrinkle_strokes) that erase what they cover
    slim     瘦脸  — pulls the jawline inward with a smooth displacement field
    eyes     大眼  — radial enlargement around both eye centres
    shadow   眼影  — a feathered wash between the lash line and the brow
    liner    眼线  — a tapered line along the upper lash line
    blush    腮红  — a soft Gaussian flush on the cheekbone
    lip      口红  — lipstick inside the vermilion border, teeth untouched
    liquify  液化  — replayable push / bloat / pucker brush strokes

Everything runs locally on CPU: MediaPipe FaceLandmarker gives 478 face
landmarks and the warp is a cheap RBF-style displacement field. The face skin
mask comes from the MediaPipe multiclass selfie segmenter (its ``face-skin``
class follows the hairline pixel by pixel); the same model on a downscaled full
frame gives the body-skin class, and the connected region around the chosen
face decides which person in a group photo is "the subject". The landmark oval
is only a fallback when the segmenter model is missing.

CLI (used by the DSH plugin host):

    python beauty.py --input in.jpg --output out.jpg --smooth 40 --whiten 20 \
        --slim 25 --eyes 15 --lip 45 --lip-color "#b4404f" --compare --json

It prints one JSON object on stdout. Failures exit non-zero with a JSON
``{"ok": false, "error": ...}`` payload.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# MediaPipe imports matplotlib for its drawing helpers. Without a writable
# config dir every short-lived render process rebuilds the font cache (~8 s),
# so pin it to the first candidate that actually accepts a write.
def _mpl_config_dir() -> Path | None:
    plugin = Path(__file__).resolve().parent.parent
    for base in (
        plugin / ".cache",
        Path(os.environ["DSH_BEAUTY_CACHE"]) if os.environ.get("DSH_BEAUTY_CACHE") else None,
        Path.home() / ".dsh" / "beauty-cache",
        Path(tempfile.gettempdir()) / "dsh-beauty-cache",
    ):
        if base is None:
            continue
        candidate = base / "mpl"
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    return None


_MPL_DIR = _mpl_config_dir()
if _MPL_DIR is not None:
    os.environ["MPLCONFIGDIR"] = str(_MPL_DIR)

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as error:  # friendly hint for a bare `python3` run
    print(
        json.dumps(
            {
                "ok": False,
                "error": {
                    "code": "E_DEPS",
                    "message": f"{error.name} is missing in {sys.executable}. "
                    "Run scripts/setup-engine.sh, or point DSH_BEAUTY_PYTHON at the plugin .venv.",
                },
            }
        ),
        file=sys.stdout,
    )
    raise SystemExit(2)

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "face_landmarker.task"
DEFAULT_SEG_MODEL = HERE / "models" / "selfie_multiclass_256x256.tflite"

# MediaPipe multiclass selfie segmenter labels.
SEG_BACKGROUND = 0
SEG_HAIR = 1
SEG_BODY_SKIN = 2
SEG_FACE_SKIN = 3
SEG_CLOTHES = 4
SEG_OTHERS = 5
SEG_PERSON = (SEG_HAIR, SEG_BODY_SKIN, SEG_FACE_SKIN, SEG_CLOTHES, SEG_OTHERS)
# The crop fed to the 256x256 segmenter is this many face-widths wide: enough
# hairline above the brows and enough hair either side of the temples.
SEG_CROP_SCALE = 1.9
# Full-frame subject segmentation runs on a frame with this long side. The model
# resizes its input to 256x256 anyway, and 640 keeps the cached class mask small
# enough to ride along in the landmark report.
PERSON_SEG_SIDE = 640

# ---------------------------------------------------------------------------
# Landmark index groups (MediaPipe FaceMesh, 468 points + 10 iris points)
# ---------------------------------------------------------------------------

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
# The vermilion border's inner edge. Filling the outer loop and cutting this one
# out keeps lipstick off the teeth and the dark mouth opening when it is open.
LIPS_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14,
    87, 178, 88, 95,
]

# Per side: eye and brow groups, the cheekbone point that anchors the blush,
# and the face-width pair whose direction gives the cheek its axis. "Left" here
# means image-left, matching LEFT_EYE/LEFT_JAW.
MAKEUP_SIDES = (
    {"eye": LEFT_EYE, "brow": LEFT_BROW, "cheek": 205, "outer": 234, "inner": 454},
    {"eye": RIGHT_EYE, "brow": RIGHT_BROW, "cheek": 425, "outer": 454, "inner": 234},
)

# Jaw control points, ordered from ear to chin on each side. The weight says
# how much of the total slim strength each point takes.
LEFT_JAW = [(172, 0.55), (136, 0.9), (150, 1.0), (149, 0.95), (176, 0.75), (148, 0.45)]
RIGHT_JAW = [(397, 0.55), (365, 0.9), (379, 1.0), (378, 0.95), (400, 0.75), (377, 0.45)]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class Params:
    smooth: float = 35.0
    whiten: float = 15.0
    # 人物提亮: the same L* lift over every skin part of the main subject —
    # face, neck, shoulders, arms and hands. 0 keeps the old face-only behaviour.
    person: float = 0.0
    # 皱纹修复: 0..100 fades crease-like ridges on the face skin (the same skin
    # mask as 磨皮, so brows/eyes/lips are skipped). Its brush strokes live in
    # `wrinkle_strokes` below and erase every crease they cover.
    wrinkle: float = 0.0
    slim: float = 15.0
    eyes: float = 10.0
    # extras that make the result look like a real retouch rather than a blur
    detail: float = 0.35      # crispness restored outside the skin mask
    # 整体肤色: temperature and tint for every patch of skin at once (face,
    # neck, shoulders, arms). Both stay inside the range real human skin covers:
    # ±50 is a visible but still believable shift, not a full white-balance range.
    temperature: float = 0.0  # -50 cool/blue .. +50 warm/yellow (Lab b*)
    tint: float = 0.0         # -50 green .. +50 magenta (Lab a*)
    # The old single 暖色 knob. Kept so existing tool calls and CLI scripts keep
    # working; the render pass folds it into `temperature`.
    warmth: float = 0.0
    # makeup: every layer is off by default, so a plain retouch is untouched
    shadow: float = 0.0       # 眼影 0..100 — feathered wash above the lash line
    shadow_color: str = "#a97b8c"
    liner: float = 0.0        # 眼线 0..100 — tapered upper lash line
    liner_color: str = "#2a211f"
    blush: float = 0.0        # 腮红 0..100 — soft flush on the cheekbone
    blush_color: str = "#e2827e"
    lip: float = 0.0          # 口红 0..100 — lipstick inside the lip border
    lip_color: str = "#b4404f"
    face: int = 0             # 0 = largest/most central, 1..n = that face
    compare: bool = False     # also write a before/after sheet
    max_size: int = 0         # 0 = full resolution, else cap the long side (previews)
    # 液化 brush strokes, in source-relative coordinates (0..1). Each stroke is
    # {"mode": "push"|"bloat"|"pucker", "points": [[x, y], ...],
    #  "radius": 0.01..0.5 (of the source width), "strength": 0..1}.
    liquify: list = field(default_factory=list)
    # 手动补皮肤画笔: {"radius": 0.01..0.2 (of the source width),
    #  "points": [[x, y], ...]} in source-relative coordinates. Painted areas
    # are added to the subject skin mask before 人物提亮 / 整体肤色.
    mask_strokes: list = field(default_factory=list)
    # 皱纹修复笔: the same stroke shape as mask_strokes, but painted areas are
    # flattened instead of joining the skin masks. Painted at any strength, so
    # the brush works even when the 皱纹修复 slider is at 0.
    wrinkle_strokes: list = field(default_factory=list)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Image IO
# ---------------------------------------------------------------------------


def load_image(path: str) -> np.ndarray:
    """Read an image as BGR uint8, honouring EXIF rotation and HEIC."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"input image not found: {path}")
    if p.suffix.lower() in {".heic", ".heif"}:
        tmp = Path(tempfile.mkdtemp(prefix="dsh-beauty-")) / "converted.jpg"
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(p), "--out", str(tmp)],
            check=True,
            capture_output=True,
        )
        p = tmp
    from PIL import Image, ImageOps

    img = Image.open(p)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def save_image(path: str, bgr: np.ndarray, quality: int = 95) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(path).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif ext == ".webp":
        cv2.imwrite(path, bgr, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    else:
        cv2.imwrite(path, bgr)
    if not Path(path).exists():
        raise IOError(f"failed to write {path}")


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------


class FaceModel:
    """Lazily-created FaceLandmarker; one instance serves a whole process."""

    _instance: "FaceModel | None" = None

    def __init__(self, model_path: str | None = None):
        self.model_path = str(model_path or os.environ.get("BEAUTY_MODEL", DEFAULT_MODEL))
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"face landmarker model missing at {self.model_path}; "
                "run scripts/setup-engine.sh or set BEAUTY_MODEL"
            )
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=4,
                min_face_detection_confidence=0.35,
            )
        )

    @classmethod
    def get(cls, model_path: str | None = None) -> "FaceModel":
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def detect(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Return landmark arrays shaped (478, 2) in pixel coordinates."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        result = self._landmarker.detect(
            self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
        )
        faces = []
        for marks in result.face_landmarks:
            pts = np.array([[m.x * w, m.y * h] for m in marks], dtype=np.float32)
            faces.append(pts)
        return faces


class SkinModel:
    """Lazily-created multiclass selfie segmenter (hair / face-skin / body).

    The landmark oval is fitted to *visible* skin: on a high forehead or under a
    fringe its top edge cuts across the forehead, and at the temples it runs
    over the sideburn. This model labels every pixel instead, so the hairline
    decides where the skin range stops.
    """

    _instance: "SkinModel | None" = None

    def __init__(self, model_path: str | None = None):
        self.model_path = str(
            model_path or os.environ.get("BEAUTY_SEG_MODEL") or DEFAULT_SEG_MODEL
        )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"skin segmentation model missing at {self.model_path}; "
                "run scripts/setup-engine.sh or set BEAUTY_SEG_MODEL"
            )
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True,
            )
        )

    @classmethod
    def get(cls, model_path: str | None = None) -> "SkinModel":
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def categories(self, crop: np.ndarray) -> np.ndarray:
        """Per-pixel class ids for a BGR crop (``SEG_*`` constants)."""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._segmenter.segment(
            self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
        )
        return np.squeeze(result.category_mask.numpy_view()).astype(np.uint8)


def _dedupe_faces(faces: list[np.ndarray]) -> list[np.ndarray]:
    """Drop the same face found in overlapping scan tiles, keeping the largest."""
    kept: list[np.ndarray] = []
    axes: list[tuple[float, float, float]] = []
    for pts in sorted(faces, key=lambda p: float(face_axes(p)[2]), reverse=True):
        cx, cy, fw = face_axes(pts)
        if any(math.hypot(cx - ox, cy - oy) < 0.6 * max(fw, ow) for ox, oy, ow in axes):
            continue
        kept.append(pts)
        axes.append((cx, cy, fw))
    return kept


def detect_faces(model: "FaceModel", bgr: np.ndarray) -> list[np.ndarray]:
    """Find faces, scanning overlapping regions when the subject is small.

    The FaceLandmarker resizes whatever it is given to a fixed detector input,
    so a face that fills a small part of a large photo turns into a handful of
    pixels and is missed. One full-frame pass is cheapest; when it finds
    nothing, re-scan the image in overlapping tiles where the subject is large.
    """
    faces = model.detect(bgr)
    if faces:
        return faces

    h, w = bgr.shape[:2]
    tile = max(512, int(min(h, w) * 0.45))
    if tile >= min(h, w) - 8:
        return faces

    stride = max(int(tile * 0.5), 1)
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    if xs[-1] != w - tile:
        xs.append(max(w - tile, 0))
    if ys[-1] != h - tile:
        ys.append(max(h - tile, 0))

    found: list[np.ndarray] = []
    for y in ys:
        for x in xs:
            sub = bgr[y:y + tile, x:x + tile]
            for pts in model.detect(sub):
                found.append(pts + np.array([x, y], dtype=np.float32))
    return _dedupe_faces(found)


def pick_face(faces: list[np.ndarray], index: int) -> np.ndarray | None:
    if not faces:
        return None
    if index >= 0 and index < len(faces):
        return faces[index]
    return faces[0]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def face_axes(pts: np.ndarray) -> tuple[float, float, float]:
    """(centre_x, centre_y, face_width) from the oval landmarks."""
    oval = pts[FACE_OVAL]
    cx, cy = float(oval[:, 0].mean()), float(oval[:, 1].mean())
    width = float(oval[:, 0].max() - oval[:, 0].min())
    return cx, cy, max(width, 1.0)


def eye_centres(pts: np.ndarray) -> list[np.ndarray]:
    left = pts[LEFT_EYE].mean(axis=0)
    right = pts[RIGHT_EYE].mean(axis=0)
    return [left, right]


def eye_width(pts: np.ndarray) -> float:
    widths = [float(np.linalg.norm(pts[LEFT_EYE[0]] - pts[LEFT_EYE[3]])),
              float(np.linalg.norm(pts[RIGHT_EYE[0]] - pts[RIGHT_EYE[3]]))]
    return max(float(np.mean(widths)), 1.0)


def eye_height(pts: np.ndarray) -> float:
    """Vertical eye opening (upper lid to lower lid) averaged over both eyes."""
    heights = [float(np.linalg.norm(pts[LEFT_EYE[1]] - pts[LEFT_EYE[5]])),
               float(np.linalg.norm(pts[RIGHT_EYE[1]] - pts[RIGHT_EYE[5]]))]
    return max(float(np.mean(heights)), 1.0)


def convex_hull(points: np.ndarray) -> np.ndarray:
    return cv2.convexHull(points.astype(np.int32)).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Warp: slim face + enlarge eyes
# ---------------------------------------------------------------------------


def warp_portrait(
    bgr: np.ndarray,
    pts: np.ndarray,
    slim: float,
    eyes: float,
    return_field: bool = False,
):
    """Apply a smooth displacement field and move the landmarks with it.

    ``slim`` (0..100) pulls each jaw control point horizontally toward the
    face axis; ``eyes`` (0..100) scales content around each eye centre.
    Only the face's bounding region is touched — everything else is copied.

    With ``return_field`` the sampled fields are returned as well, as
    ``(x0, y0, map_x, map_y)`` in image coordinates, so a mask painted before
    the warp can be moved with the exact same displacements.
    """
    h, w = bgr.shape[:2]
    if slim <= 0 and eyes <= 0:
        out, wpts = bgr.copy(), pts.copy()
        return (out, wpts, None) if return_field else (out, wpts)

    cx, cy, fw = face_axes(pts)
    eye_r = eye_width(pts) * 1.6

    # Region of interest: face bbox grown by the largest influence radius.
    oval = pts[FACE_OVAL]
    pad = int(max(fw * 0.9, eye_r * 2.5))
    x0 = max(int(oval[:, 0].min()) - pad, 0)
    x1 = min(int(oval[:, 0].max()) + pad, w)
    y0 = max(int(oval[:, 1].min()) - pad, 0)
    y1 = min(int(oval[:, 1].max()) + pad, h)
    roi_w, roi_h = x1 - x0, y1 - y0

    yy, xx = np.mgrid[0:roi_h, 0:roi_w].astype(np.float32)
    xx += x0
    yy += y0
    dx = np.zeros_like(xx)
    dy = np.zeros_like(yy)

    # --- jaw slimness -----------------------------------------------------
    if slim > 0:
        strength = (slim / 100.0) * fw * 0.075
        sigma = max(fw * 0.13, 6.0)
        accum = np.zeros_like(xx)
        norm = np.zeros_like(xx)
        for idx, weight in LEFT_JAW + RIGHT_JAW:
            px, py = float(pts[idx][0]), float(pts[idx][1])
            # positive for points right of the axis, negative left
            dir_x = 1.0 if px > cx else -1.0
            move = -dir_x * strength * weight
            wobble = np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma * sigma)))
            accum += wobble * move
            norm += wobble
        # Normalising by the weight sum keeps the field even across unevenly
        # spaced control points, but that sum decays to zero far from the jaw —
        # dividing by it there would leave the displacement at full strength
        # right up to the cut-off, i.e. a one-pixel step of up to `strength`
        # pixels: a visible tear across the cheek. Clamp the denominator so the
        # field fades out with the Gaussians instead of snapping to zero.
        floor = float(norm.max()) * 0.25
        dx = accum / np.maximum(norm, floor)

    # --- eye enlargement --------------------------------------------------
    # Two independent radial fields cancel at the nose bridge and leave a
    # crease, so fade each field out toward the face midline before summing.
    if eyes > 0:
        k_base = (eyes / 100.0) * 0.34
        centres = eye_centres(pts)
        ew = eye_width(pts)
        eh = eye_height(pts)
        rx = max(ew * 1.05, 6.0)
        ry = max(eh * 2.4, ew * 0.6)
        separation = max(
            float(np.linalg.norm(centres[1] - centres[0])) if len(centres) > 1 else ew * 2.6,
            ew * 1.2,
        )
        nose_fade = np.clip(
            (np.abs(xx - cx) - separation * 0.18) / (separation * 0.32), 0.0, 1.0
        )
        nose_fade = nose_fade * nose_fade * (3.0 - 2.0 * nose_fade)
        for centre in centres:
            ex, ey = float(centre[0]), float(centre[1])
            # Elliptical guard keeps the field on the eye instead of bending
            # the brow, the nose bridge or a hat brim above it.
            d_norm = ((xx - ex) / rx) ** 2 + ((yy - ey) / ry) ** 2
            guard = np.clip(1.0 - d_norm, 0.0, 1.0) ** 2
            wobble = np.exp(-d_norm / 2.0) * guard * nose_fade
            scale = k_base * wobble
            dx += (xx - ex) * scale / (1.0 + scale)
            dy += (yy - ey) * scale / (1.0 + scale)

    # sample the source from (x - dx, y - dy) → content appears to move by +d
    map_x = (xx - dx).astype(np.float32)
    map_y = (yy - dy).astype(np.float32)

    out = bgr.copy()
    out[y0:y1, x0:x1] = cv2.remap(
        bgr[y0:y1, x0:x1],
        map_x - x0,
        map_y - y0,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )

    # move the landmarks with the same field (bilinear lookup would be
    # overkill: the field is smooth, so nearest sample of the displacement is
    # accurate to a fraction of a pixel)
    warped = pts.copy()
    gx = np.clip(np.round(pts[:, 0] - x0).astype(int), 0, roi_w - 1)
    gy = np.clip(np.round(pts[:, 1] - y0).astype(int), 0, roi_h - 1)
    inside = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
    warped[inside, 0] += dx[gy[inside], gx[inside]]
    warped[inside, 1] += dy[gy[inside], gx[inside]]
    if return_field:
        return out, warped, (x0, y0, map_x, map_y)
    return out, warped


def warp_mask(mask: np.ndarray, field) -> np.ndarray:
    """Move a soft mask through the same field that warped the portrait."""
    if field is None:
        return mask
    x0, y0, map_x, map_y = field
    rh, rw = map_x.shape[:2]
    sub = mask[y0:y0 + rh, x0:x0 + rw]
    if sub.shape[:2] != (rh, rw):
        return mask
    moved = cv2.remap(
        sub,
        (map_x - x0).astype(np.float32),
        (map_y - y0).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    out = mask.copy()
    out[y0:y0 + rh, x0:x0 + rw] = moved
    return out


# ---------------------------------------------------------------------------
# Skin mask
# ---------------------------------------------------------------------------


def polygon_mask(shape: tuple[int, int], points: np.ndarray, blur: float = 0) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    return mask.astype(np.float32) / 255.0


def skin_mask(
    pts: np.ndarray, shape: tuple[int, int], protect_features: bool = True
) -> np.ndarray:
    """Fallback mask: face oval minus eyes, brows and lips, feathered.

    Used only when the segmentation model is unavailable. It inherits the
    landmark oval's weaknesses (a high forehead or a fringe is cut off, the
    temples reach into the sideburn), so prefer :func:`skin_mask_from_segment`.

    ``protect_features=False`` keeps the features in the mask — what tone
    adjustments want, so the eyes and lips brighten with the cheeks instead of
    staying dark holes in a lighter face.
    """
    _, _, fw = face_axes(pts)
    feather = max(2.0, fw * 0.012)
    mask = polygon_mask(shape, pts[FACE_OVAL], feather)
    if protect_features:
        mask = mask * (1.0 - feature_protect_mask(pts, shape, fw))
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _soften(poly: np.ndarray, grow: float, blur: float) -> np.ndarray:
    """Dilate a polygon mask, then feather it — keeps thin features covered."""
    k = max(int(round(grow)), 0)
    if k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        poly = cv2.dilate(poly, kernel)
    if blur > 0:
        poly = cv2.GaussianBlur(poly, (0, 0), blur)
    return poly


def feature_protect_mask(
    pts: np.ndarray, shape: tuple[int, int], fw: float, ox: float = 0.0, oy: float = 0.0
) -> np.ndarray:
    """1 where brows, eyes and mouth sit, so smoothing skips them.

    The old code feathered a thin polygon and subtracted it afterwards, which
    left 35-70 % of the eye and lip centres smoothed. Dilating first and
    subtracting an unblurred interior fixes that.
    """
    h, w = shape[:2]
    grow = max(1.0, fw * 0.012)
    blur = max(1.0, fw * 0.010)
    protect = np.zeros((h, w), np.float32)
    for group in (LEFT_BROW, RIGHT_BROW, LEFT_EYE, RIGHT_EYE):
        poly = polygon_mask((h, w), pts[group] - np.array([ox, oy], np.float32), 0)
        protect = np.maximum(protect, _soften(poly, grow, blur))
    # the outer-lip contour is not a simple polygon when the mouth is open, so
    # fill its hull: teeth and the dark opening are protected too
    lip = convex_hull(pts[LIPS_OUTER]) - np.array([ox, oy], np.float32)
    protect = np.maximum(protect, _soften(polygon_mask((h, w), lip, 0), grow, blur))
    return protect


# ---------------------------------------------------------------------------
# Segmentation mask
# ---------------------------------------------------------------------------


def segment_face(bgr: np.ndarray, pts: np.ndarray) -> dict:
    """Run the multiclass segmenter on a square crop around one face.

    Returns ``{"box": [x0, y0, side], "cat": uint8 class ids at crop size}`` in
    image pixels. The model runs on a 256x256 grid but hands its category mask
    back at the crop resolution.
    """
    h, w = bgr.shape[:2]
    oval = pts[FACE_OVAL]
    cx, cy, fw = face_axes(pts)
    fh = float(oval[:, 1].max() - oval[:, 1].min())
    side = int(round(max(fw, fh) * SEG_CROP_SCALE))
    side = max(min(side, w, h), 32)
    x0 = int(round(clamp(cx - side / 2.0, 0, max(w - side, 0))))
    y0 = int(round(clamp(cy - side / 2.0, 0, max(h - side, 0))))
    crop = bgr[y0:y0 + side, x0:x0 + side]
    return {"box": [x0, y0, side], "cat": SkinModel.get().categories(crop)}


def encode_segment(seg: dict) -> dict:
    """Compact JSON form of a segment: box + PNG-in-base64 class mask."""
    ok, buf = cv2.imencode(".png", seg["cat"], [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    if not ok:
        raise IOError("cannot encode skin segmentation mask")
    return {
        "box": [int(round(float(v))) for v in seg["box"]],
        "cat": base64.b64encode(buf.tobytes()).decode("ascii"),
    }


def decode_segment(data: dict) -> dict | None:
    try:
        raw = base64.b64decode(data.get("cat") or "")
        cat = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        box = [float(v) for v in data.get("box") or []]
    except Exception:  # noqa: BLE001 — a bad cache entry must never fail a render
        return None
    if cat is None or cat.ndim != 2 or len(box) not in (3, 4):
        return None
    return {"box": box, "cat": cat}


def skin_mask_from_segment(
    bgr: np.ndarray, pts: np.ndarray, seg: dict, scale: float = 1.0, protect_features: bool = True
) -> np.ndarray:
    """Skin range from the segmenter's face-skin class, refined to the hairline.

    ``seg`` comes from a cached landmark report and is expressed in the source
    photo's pixels; ``scale`` maps it to the resolution being rendered.

    ``protect_features=False`` keeps brows, eyes and mouth inside the mask, for
    the tone pass: brightening the cheeks while leaving dark brows, eyes and lips
    untouched reads as a stark patch on the face.
    """
    h, w = bgr.shape[:2]
    bx, by, side = (float(v) * scale for v in seg["box"])
    x0 = int(round(clamp(bx, 0, w - 2)))
    y0 = int(round(clamp(by, 0, h - 2)))
    x1 = int(round(clamp(bx + side, x0 + 1, w)))
    y1 = int(round(clamp(by + side, y0 + 1, h)))
    cw, ch = x1 - x0, y1 - y0

    face = cv2.resize(
        (seg["cat"] == SEG_FACE_SKIN).astype(np.float32),
        (cw, ch),
        interpolation=cv2.INTER_LINEAR,
    )

    # Keep the class mask inside the head: below the chin the neck is body-skin,
    # but a loose classifier should never whiten the clothes or the background.
    oval = pts[FACE_OVAL] - np.array([x0, y0], np.float32)
    fw = float(oval[:, 0].max() - oval[:, 0].min())
    fh = float(oval[:, 1].max() - oval[:, 1].min())
    left = int(round(clamp(oval[:, 0].min() - 0.18 * fw, 0, cw - 1)))
    right = int(round(clamp(oval[:, 0].max() + 0.18 * fw, left + 1, cw)))
    top = int(round(clamp(oval[:, 1].min() - 0.90 * fh, 0, ch - 1)))
    bottom = int(round(clamp(oval[:, 1].max() + 0.04 * fh, top + 1, ch)))
    band = np.zeros((ch, cw), np.float32)
    band[top:bottom, left:right] = 1.0
    face *= band

    protect = feature_protect_mask(pts, (ch, cw), fw, x0, y0)
    if protect_features:
        face *= 1.0 - protect
    else:
        # Tone keeps the features: the class mask drops dark eyelashes, lids and
        # teeth, so paint them back in with the soft feature shape. The result is
        # the whole face rather than a skin-shaped ring.
        face = np.maximum(face, protect * band)

    # Snap the soft class edge back to the real luminance edge so the mask
    # follows individual strands of hair instead of the model's 256px grid.
    if fw >= 24:
        gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        face = guided_filter(
            gray, face, max(3, int(round(fw * 0.015))), (0.045 * 255) ** 2
        )
    face = cv2.GaussianBlur(np.clip(face, 0.0, 1.0), (0, 0), max(1.0, fw * 0.008))

    mask = np.zeros((h, w), np.float32)
    mask[y0:y1, x0:x1] = np.clip(face, 0.0, 1.0)
    return mask


def segment_person(bgr: np.ndarray) -> dict:
    """Run the multiclass segmenter over the whole frame at model scale.

    The returned ``box`` is the full source frame; ``cat`` is a class-id map at
    a long side of at most :data:`PERSON_SEG_SIDE` pixels. The render pass
    resizes/crops that map with the same window as the photo, so a cached
    report works for previews, 1:1 crops and the full-resolution save.
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, PERSON_SEG_SIDE / max(h, w))
    small = bgr
    if scale < 1.0:
        small = cv2.resize(
            bgr,
            (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)),
            interpolation=cv2.INTER_AREA,
        )
    return {"box": [0, 0, w, h], "cat": SkinModel.get().categories(small)}


def subject_skin(
    shape: tuple[int, int],
    pts: np.ndarray,
    seg: dict | None,
    window: tuple[int, int, int, int],
    scale: float = 1.0,
    bgr: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Skin mask for the main subject, plus a small anatomy report.

    ``seg`` is the full-frame person segmentation in *source* pixels, ``window``
    is the source rectangle being rendered and ``scale`` maps source pixels to
    the render (``pts`` are already in render coordinates). The chosen face's
    connected person region wins: in a group photo another person's arm is
    body-skin too, but it is not this subject. When ``bgr`` (the render image)
    is given, the subject bounding box is re-segmented at a tighter crop so
    thin bare arms survive the model's 256 px input.

    Returns ``(mask, stats)`` where mask covers every skin class of that person
    (face and body) at render resolution. ``stats["body"]`` excludes the face
    oval so the panel can report "face 12% / neck+arms 4%" separately.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), np.float32)
    stats = {"face": 0.0, "body": 0.0, "parts": 0, "people": 0, "subject": 0.0}
    if seg is None:
        return mask, stats
    cat = seg.get("cat")
    if cat is None or getattr(cat, "size", 0) == 0:
        return mask, stats
    try:
        if len(seg["box"]) == 4:
            bx, by, bw, bh = (float(v) for v in seg["box"])
        else:
            bx, by, side = (float(v) for v in seg["box"])
            bw = bh = side
    except (KeyError, TypeError, ValueError):
        return mask, stats
    if bw <= 0 or bh <= 0:
        return mask, stats
    Hc, Wc = cat.shape[:2]
    wx, wy, ww, wh = (float(v) for v in window)
    if ww <= 0 or wh <= 0:
        return mask, stats

    # --- which person is the subject -------------------------------------
    person = np.isin(cat, SEG_PERSON).astype(np.uint8)
    count, labels, comp, centroids = cv2.connectedComponentsWithStats(person, connectivity=8)
    if count <= 1:
        return mask, stats
    cx, cy, _ = face_axes(pts)
    fx = (wx + cx / max(scale, 1e-6) - bx) / bw * Wc
    fy = (wy + cy / max(scale, 1e-6) - by) / bh * Hc
    fl = int(labels[int(clamp(fy, 0, Hc - 1)), int(clamp(fx, 0, Wc - 1))])
    area = comp[:, cv2.CC_STAT_AREA]
    min_person = max(1, int(round(0.01 * Hc * Wc)))
    if fl == 0 or int(area[fl]) < min_person:
        candidates = [i for i in range(1, count) if int(area[i]) >= min_person]
        if not candidates:
            return mask, stats
        fl = min(
            candidates,
            key=lambda i: (float(centroids[i][0]) - fx) ** 2 + (float(centroids[i][1]) - fy) ** 2,
        )
    subject = labels == fl
    stats["people"] = int(
        sum(1 for i in range(1, count) if int(area[i]) >= max(1, int(round(0.015 * Hc * Wc))))
    )

    # --- skin inside that person -----------------------------------------
    skin = subject & np.isin(cat, (SEG_BODY_SKIN, SEG_FACE_SKIN))

    # Window → class-map rectangle. A crop asks for a small part of the map;
    # resizing the whole map would smear it, so take the matching sub-rect.
    sx0 = int(np.floor((wx - bx) / bw * Wc))
    sx1 = int(np.ceil((wx + ww - bx) / bw * Wc))
    sy0 = int(np.floor((wy - by) / bh * Hc))
    sy1 = int(np.ceil((wy + wh - by) / bh * Hc))
    sx0, sx1 = max(0, min(sx0, Wc - 1)), max(1, min(sx1, Wc))
    sy0, sy1 = max(0, min(sy0, Hc - 1)), max(1, min(sy1, Hc))
    sub = skin[sy0:sy1, sx0:sx1].astype(np.float32)
    sub_person = subject[sy0:sy1, sx0:sx1]
    if sub.size == 0:
        return mask, stats
    skin_r = cv2.resize(sub, (w, h), interpolation=cv2.INTER_LINEAR)
    person_r = cv2.resize(
        sub_person.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(np.float32)

    # --- refine on a tight subject crop ----------------------------------
    # The selfie segmenter resizes everything to 256 px. A person who fills
    # only a few percent of a landscape frame turns a bare arm into a ~8 px
    # sliver, and the model labels it "clothes" (the arm touches the shirt).
    # Re-running the same model on the subject's bounding box makes the arm
    # 30-40 px wide, which it segments as body skin. We only add skin here,
    # never remove it, so a close-up (where no crop is needed) keeps the
    # full-frame result bit for bit.
    if bgr is not None and bgr.shape[0] == h and bgr.shape[1] == w:
        sbx, sby, sbw, sbh = (int(v) for v in comp[fl, :4])
        origin_x = bx + sbx / Wc * bw
        origin_y = by + sby / Hc * bh
        box_w = sbw / Wc * bw * scale
        box_h = sbh / Hc * bh * scale
        rx = (origin_x - wx) * scale
        ry = (origin_y - wy) * scale
        pad = max(box_w, box_h) * 0.18
        x0 = int(max(0, np.floor(rx - pad)))
        y0 = int(max(0, np.floor(ry - pad)))
        x1 = int(min(w, np.ceil(rx + box_w + pad)))
        y1 = int(min(h, np.ceil(ry + box_h + pad)))
        if (
            x1 - x0 >= 48
            and y1 - y0 >= 48
            and (x1 - x0) * (y1 - y0) < 0.92 * w * h
            and float(person_r.mean()) < 0.30
        ):
            crop = bgr[y0:y1, x0:x1]
            rcat = SkinModel.get().categories(crop)
            rc_h, rc_w = rcat.shape[:2]
            rperson = np.isin(rcat, SEG_PERSON).astype(np.uint8)
            n_ref, lab_ref, st_ref, cen_ref = cv2.connectedComponentsWithStats(
                rperson, connectivity=8
            )
            rfx = (cx - x0) * (rc_w / max(x1 - x0, 1))
            rfy = (cy - y0) * (rc_h / max(y1 - y0, 1))
            rfl = int(
                lab_ref[
                    int(clamp(rfy, 0, rc_h - 1)),
                    int(clamp(rfx, 0, rc_w - 1)),
                ]
            )
            if rfl == 0:
                areas = st_ref[1:, cv2.CC_STAT_AREA] if n_ref > 1 else np.array([])
                if areas.size:
                    rfl = int(
                        1
                        + int(np.argmin(
                            (cen_ref[1:, 0] - rfx) ** 2 + (cen_ref[1:, 1] - rfy) ** 2
                        ))
                    )
            if rfl > 0:
                ref = (lab_ref == rfl) & np.isin(rcat, (SEG_BODY_SKIN, SEG_FACE_SKIN))
                # At a strap or a fold the 256 px model can still merge a
                # sliver of lit skin into the clothing class. Recover only
                # small, skin-coloured patches that touch skin we already
                # have; the shirt itself is far away in Lab, so it stays out.
                core = ref & (lab_ref == rfl)
                if int(core.sum()) >= 64:
                    small = cv2.resize(
                        crop, (rc_w, rc_h), interpolation=cv2.INTER_AREA
                    )
                    lab_s = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
                    med = np.median(lab_s[core], axis=0)
                    dist = np.linalg.norm(lab_s - med, axis=2)
                    cand = (
                        (lab_ref == rfl)
                        & ~core
                        & (dist < 12.0)
                        & (lab_s[..., 0] > max(20.0, float(med[0]) - 60.0))
                    ).astype(np.uint8)
                    if cand.any():
                        n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(
                            cand, connectivity=8
                        )
                        touch = cv2.dilate(
                            core.astype(np.uint8), np.ones((3, 3), np.uint8)
                        )
                        limit = 0.02 * rc_w * rc_h
                        for i in range(1, n_c):
                            if (
                                st_c[i, cv2.CC_STAT_AREA] <= limit
                                and ((lab_c == i) & (touch > 0)).any()
                            ):
                                core |= lab_c == i
                    ref = core
                ref = cv2.resize(
                    ref.astype(np.float32),
                    (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_LINEAR,
                )
                patch = skin_r[y0:y1, x0:x1]
                # Only add refined skin that touches the skin we already trust:
                # the crop can also contain a bystander's hand or a skin-toned
                # wall, and those are not this subject's arm.
                if patch.any() and (ref > 0.5).any():
                    gate = cv2.dilate(
                        (patch > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)
                    )
                    n_add, lab_add = cv2.connectedComponents(
                        (ref > 0.5).astype(np.uint8), connectivity=8
                    )
                    keep = np.zeros_like(ref)
                    for i in range(1, n_add):
                        piece = lab_add == i
                        if (piece & (gate > 0)).any():
                            keep[piece] = 1.0
                    ref = ref * keep
                skin_r[y0:y1, x0:x1] = np.maximum(patch, ref)

    # Body-skin parts: the mask minus the landmark oval. The neck is right
    # below the chin, so the oval is grown a little before subtracting.
    face_poly = np.zeros((h, w), np.uint8)
    oval = np.round(pts[FACE_OVAL]).astype(np.int32)
    cv2.fillPoly(face_poly, [oval], 255)
    grow = max(1, int(round(0.02 * bw * scale)))
    face_poly = cv2.dilate(face_poly, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    solid = skin_r > 0.5
    body = solid & (face_poly < 128)
    if body.any():
        n_parts, _, part_stats, _ = cv2.connectedComponentsWithStats(
            body.astype(np.uint8), connectivity=8
        )
        min_part = max(3, int(round(0.0015 * w * h)))
        stats["parts"] = int(
            sum(1 for i in range(1, n_parts) if int(part_stats[i, cv2.CC_STAT_AREA]) >= min_part)
        )
        stats["body"] = round(float(body.mean()) * 100.0, 2)
    stats["face"] = round(float((solid & (face_poly >= 128)).mean()) * 100.0, 2)
    stats["subject"] = round(float(person_r.mean()) * 100.0, 2)

    # The segmenter works on a 256 px crop: one model pixel is a couple of
    # image pixels, so its boundary sits a few px inside the arm. Grow the
    # mask by that much before feathering, otherwise the last strip of skin
    # next to a strap or the background stays half-lit and looks unretouched.
    soft = np.clip(skin_r, 0.0, 1.0)
    grow_px = max(1, int(round(min(w, h) * 0.005)))
    if grow_px > 1:
        hard = (soft > 0.5).astype(np.uint8)
        if bgr is not None and bgr.shape[:2] == (h, w) and hard.any():
            # Grow into neighbouring skin-coloured pixels only. The strap
            # and the hair are close to skin in luminance but far in chroma,
            # so a hue gate lets the edge reach the real arm outline without
            # painting a light rim on the fabric.
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            med = np.median(lab[hard > 0], axis=0)
            chroma = np.hypot(lab[..., 1] - med[1], lab[..., 2] - med[2])
            ok = (chroma < 10.0) & (lab[..., 0] > max(30.0, float(med[0]) - 60.0))
            for _ in range(grow_px):
                grown = cv2.dilate(hard, np.ones((3, 3), np.uint8))
                hard = np.where((grown > 0) & ok, 1, hard).astype(np.uint8)
        else:
            hard = cv2.dilate(
                hard, np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8)
            )
        soft = hard.astype(np.float32)
    mask = cv2.GaussianBlur(soft, (0, 0), max(1.0, min(w, h) * 0.0015))
    return np.clip(mask, 0.0, 1.0), stats


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def guided_filter(gray: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Single-channel guided filter (fast edge-preserving smoothing)."""
    g = gray.astype(np.float32)
    p = src.astype(np.float32)
    r = max(int(radius), 1)
    size = (2 * r + 1, 2 * r + 1)
    mean_g = cv2.boxFilter(g, -1, size)
    mean_p = cv2.boxFilter(p, -1, size)
    corr_g = cv2.boxFilter(g * g, -1, size)
    corr_gp = cv2.boxFilter(g * p, -1, size)
    var_g = corr_g - mean_g * mean_g
    cov_gp = corr_gp - mean_g * mean_p
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    return cv2.boxFilter(a, -1, size) * g + cv2.boxFilter(b, -1, size)


def smooth_skin(bgr: np.ndarray, mask: np.ndarray, strength: float, fw: float) -> np.ndarray:
    """Blend a guided-filter base back in, keeping some of the fine detail."""
    if strength <= 0:
        return bgr
    radius = int(clamp(fw * 0.045, 3, 48))
    eps = (0.045 * 255) ** 2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    base = np.empty_like(bgr, dtype=np.float32)
    for c in range(3):
        base[..., c] = guided_filter(gray, bgr[..., c].astype(np.float32), radius, eps)

    img = bgr.astype(np.float32)
    detail = img - base
    blend = (strength / 100.0) * 1.45  # 0..100 maps to 0..1.45 of the base
    blend = min(blend, 1.0) * mask
    out = img - detail * blend[..., None] * 0.92
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 皱纹修复 (wrinkle repair)
# ---------------------------------------------------------------------------
#
# A wrinkle is a dark ridge — a few pixels wide on a young face, a couple of
# dozen on an older one — sitting inside otherwise continuous shading. No extra
# model is needed to find one: a guided filter run on a quarter-size copy follows
# the face's shading but not a ridge, so ``base - gray`` comes out positive on
# creases and near zero on flat skin (and on pores, which the downscale averages
# away). The radius is what decides how wide a ridge still counts: it is
# deliberately generous so deep folds are covered too. The skin mask already
# excludes brows, eyes and lips, so those stay crisp.
WRINKLE_RADIUS = 0.100  # detection radius, as a share of the face width
WRINKLE_LO = 1.0        # grey levels below the base where a crease starts
WRINKLE_HI = 6.0        # ...and where it counts as a full crease
WRINKLE_GAIN = 1.25     # slider over-drive, capped at full fill
WRINKLE_LIMIT = 72.0    # how far one pass may lift a pixel (deep shadows stay)
WRINKLE_DOWN = 4        # the base is estimated on a 1/4-size copy
BRUSH_RADIUS = 0.14     # the brush reads an even wider base than the slider


def _soft_base(gray: np.ndarray, radius: int, down: int = WRINKLE_DOWN) -> np.ndarray:
    """Low-pass that drops structures thinner than ~2*radius."""
    h, w = gray.shape[:2]
    down = max(1, min(down, max(1, min(h, w) // 96)))
    if down == 1:
        return guided_filter(gray, gray, max(4, radius), (0.06 * 255) ** 2)
    small = cv2.resize(gray, (w // down, h // down), interpolation=cv2.INTER_AREA)
    small = guided_filter(small, small, max(2, int(radius) // down), (0.06 * 255) ** 2)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def crease_weight(
    bgr: np.ndarray,
    mask: np.ndarray | None,
    fw: float,
    lo: float = WRINKLE_LO,
    hi: float = WRINKLE_HI,
    radius_k: float = WRINKLE_RADIUS,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """0..1 crease map inside *mask*, plus the low-pass base it was found on."""
    if mask is None or mask.size == 0 or float(mask.max()) <= 0:
        return None, None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    base = _soft_base(gray, int(clamp(fw * radius_k, 6, 220)))
    depth = base - gray
    weight = np.clip((depth - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
    weight = cv2.GaussianBlur(
        weight, (0, 0), max(0.8, min(gray.shape[:2]) * 0.0008)
    )
    return weight * np.clip(mask, 0.0, 1.0).astype(np.float32), base


def flatten_skin(
    bgr: np.ndarray,
    mask: np.ndarray,
    fw: float,
    radius_k: float = WRINKLE_RADIUS,
    limit: float = WRINKLE_LIMIT,
) -> np.ndarray:
    """Fill the dark grooves under *mask* from a deeper copy of the crease base.

    Only brightens: a wrinkle is always darker than the skin around it, so
    highlights, pores and colour stay put — that is what keeps a painted patch
    looking like repaired skin instead of a smudge. *limit* stops a deep shadow
    from being lifted all the way to the surrounding light.
    """
    if mask is None or float(mask.max()) <= 0:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    base = _soft_base(
        gray, int(clamp(fw * radius_k, 6, 220)), down=WRINKLE_DOWN * 2
    )
    fill = np.clip(base - gray, 0.0, limit)
    blend = np.clip(mask, 0.0, 1.0).astype(np.float32)[..., None]
    return np.clip(bgr.astype(np.float32) + fill[..., None] * blend, 0, 255).astype(np.uint8)


def apply_wrinkle_repair(
    bgr: np.ndarray,
    mask: np.ndarray | None,
    fw: float,
    amount: float,
    brush: np.ndarray | None = None,
) -> np.ndarray:
    """Fade creases by *amount* (0..100); erase the ones painted under *brush*.

    The slider lifts a ridge back towards the skin around it, so 100 still reads
    as skin with pores. The brush is the "彻底消除" tool: painted ridges get the
    same lift with a lower threshold plus a deeper base fill, which takes out the
    lines the slider leaves behind.
    """
    out = bgr
    fade = clamp(amount, 0.0, 100.0) / 100.0
    if fade > 0 and mask is not None:
        weight, base = crease_weight(out, mask, fw)
        if weight is not None and base is not None:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Cap the lift at the crease's own depth: over-driving must not
            # turn a ridge into a bright line.
            gain = np.clip(weight * fade * WRINKLE_GAIN, 0.0, 1.0)
            delta = (base - gray) * gain
            out = np.clip(out.astype(np.float32) + delta[..., None], 0, 255).astype(np.uint8)
    if brush is not None and float(brush.max()) > 0:
        weight, base = crease_weight(
            out, brush, fw, lo=0.6, hi=3.0, radius_k=BRUSH_RADIUS
        )
        if weight is not None and base is not None:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
            delta = (base - gray) * np.clip(weight, 0.0, 1.0)
            out = np.clip(out.astype(np.float32) + delta[..., None], 0, 255).astype(np.uint8)
        out = flatten_skin(out, brush, fw, radius_k=BRUSH_RADIUS)
    return out


# How far a full-strength skin-tone slider walks the two Lab chroma axes. These
# are deliberately small: 7.5 b* is roughly the gap between open shade and warm
# daylight, 4.5 a* is a subtle green→magenta undertone shift. A normal photo
# therefore cannot be pushed into a filter-coloured face.
SKIN_TEMP_SHIFT = 7.5
SKIN_TINT_SHIFT = 4.5


def tone_weighted(
    bgr: np.ndarray,
    lift: np.ndarray,
    temperature: np.ndarray | None = None,
    tint: np.ndarray | None = None,
    glow: np.ndarray | None = None,
    glow_gamma: float = 1.0,
) -> np.ndarray:
    """Lift L* and shift skin temperature/tint with per-pixel 0..1 weights.

    ``lift`` is the additive white-skin lift. ``glow`` is the person brightening
    mask: instead of another linear offset it bends the L* curve with a gamma,
    the way Levels/Curves does. Gamma grows with the slider, so the last part of
    the slider adds much more light while 0 and 255 stay put and highlights roll
    off instead of clipping.

    ``temperature`` and ``tint`` are normalised to -1..1: temperature moves Lab
    b* (blue ↔ yellow), tint moves Lab a* (green ↔ magenta). A single Lab round
    trip lets the face lift, the subject lift and the skin-tone shift combine
    without painting the cheek twice when several sliders are on.
    """
    amount = float(lift.max()) if lift.size else 0.0
    glow_amount = (
        float(glow.max()) if glow is not None and glow.size else 0.0
    )
    temp_amount = (
        float(np.abs(temperature).max())
        if temperature is not None and temperature.size
        else 0.0
    )
    tint_amount = float(np.abs(tint).max()) if tint is not None and tint.size else 0.0
    if (
        amount <= 1e-5
        and glow_amount <= 1e-5
        and temp_amount <= 1e-5
        and tint_amount <= 1e-5
    ):
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    weight = np.clip(lift, 0.0, 1.0)

    if amount > 1e-5:
        # roll the highlight range off so bright skin does not clip flat
        norm = np.clip(L / 255.0, 0.0, 1.0)
        L = L + 34.0 * weight * (1.0 - 0.55 * norm)
        # whiter skin reads as slightly less saturated
        a = a + (128.0 - a) * (0.20 * weight)
        b = b + (128.0 - b) * (0.12 * weight)

    if glow_amount > 1e-5 and glow is not None:
        strength = np.clip(glow, 0.0, 1.0)
        gamma = max(1.0, float(glow_gamma))
        base = np.clip(L / 255.0, 0.0, 1.0)
        curved = 255.0 * np.power(base, 1.0 / gamma)
        L = L * (1.0 - strength) + curved * strength
        a = a + (128.0 - a) * (0.12 * strength)
        b = b + (128.0 - b) * (0.08 * strength)

    # Small cross terms keep the shift looking like skin rather than a tinted
    # glass: warming also adds a touch of red, and magenta pulls a little yellow.
    if temp_amount > 1e-5 and temperature is not None:
        b = b + SKIN_TEMP_SHIFT * temperature
        a = a + 0.8 * temperature
    if tint_amount > 1e-5 and tint is not None:
        a = a + SKIN_TINT_SHIFT * tint
        b = b - 0.6 * tint

    lab[..., 0] = np.clip(L, 0, 255)
    lab[..., 1] = np.clip(a, 0, 255)
    lab[..., 2] = np.clip(b, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def adjust_skin_tone(
    bgr: np.ndarray, mask: np.ndarray, temperature: float, tint: float
) -> np.ndarray:
    """Warm/cool and green/magenta every masked skin pixel, leaving L* alone."""
    temp = clamp(temperature, -50.0, 50.0)
    tin = clamp(tint, -50.0, 50.0)
    if temp == 0 and tin == 0:
        return bgr
    weights = mask.astype(np.float32)
    return tone_weighted(
        bgr,
        np.zeros_like(weights),
        weights * (temp / 50.0),
        weights * (tin / 50.0),
    )


def crisp_details(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp the non-skin part so eyes, hair and clothing stay crisp."""
    if amount <= 0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.2)
    sharp = cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)
    m = (1.0 - mask)[..., None]
    out = bgr.astype(np.float32) * (1 - m) + sharp.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Makeup
# ---------------------------------------------------------------------------
#
# Landmarks only: no reference photo and no second model. Colour is moved in
# Lab by shifting the region's *mean* chroma onto the cosmetic while keeping the
# per-pixel variation, so lash shadow, lip lines and specular highlights survive
# — that is what separates a tint from a sticker. Soft features (blush) are an
# analytic Gaussian; hard ones (liner) are a painted, tapered polyline.

COLOR_FALLBACK = {
    "shadow": "#a97b8c",
    "liner": "#2a211f",
    "blush": "#e2827e",
    "lip": "#b4404f",
}


def parse_color(value: str | None, fallback: str) -> np.ndarray:
    """'#rrggbb', '#rgb' or bare hex -> BGR triplet in 0..255 floats."""
    for text in (value, fallback):
        digits = str(text or "").strip().lstrip("#")
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        if len(digits) != 6:
            continue
        try:
            r, g, b = (int(digits[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            continue
        return np.array([b, g, r], dtype=np.float32)
    return np.array([80, 60, 180], dtype=np.float32)


def up_vector(pts: np.ndarray) -> np.ndarray:
    """Image-space "up" for this head (chin -> forehead), unit length."""
    v = pts[10].astype(np.float32) - pts[152].astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-3 else np.array([0.0, -1.0], dtype=np.float32)


def _eye_frame(pts: np.ndarray, group: list[int], up: np.ndarray) -> dict:
    """Orthonormal frame for one eye: u inward->outward, v toward the brow."""
    lid = pts[group].astype(np.float32)
    inner, outer = lid[3], lid[0]
    u = outer - inner
    ew = float(np.linalg.norm(u))
    u = u / max(ew, 1e-3)
    v = up - u * float(np.dot(up, u))
    n = float(np.linalg.norm(v))
    v = v / n if n > 1e-3 else np.array([0.0, -1.0], dtype=np.float32)
    return {"lid": lid, "inner": inner, "outer": outer, "u": u, "v": v, "ew": ew}


def _roi(points: np.ndarray, shape: tuple[int, int], margin: float) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x0 = int(np.floor(max(0.0, float(points[:, 0].min()) - margin)))
    y0 = int(np.floor(max(0.0, float(points[:, 1].min()) - margin)))
    x1 = int(np.ceil(min(float(w), float(points[:, 0].max()) + margin)))
    y1 = int(np.ceil(min(float(h), float(points[:, 1].max()) + margin)))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _paste(shape: tuple[int, int], box: tuple[int, int, int, int], local: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros(shape[:2], np.float32)
    mask[y0:y1, x0:x1] = local
    return mask


def eyeshadow_mask(
    pts: np.ndarray, shape: tuple[int, int], group: list[int], brow: list[int], up: np.ndarray
) -> np.ndarray:
    """Feathered wash between the lash line and the brow, with a small wing."""
    frame = _eye_frame(pts, group, up)
    lid, ew, u, v = frame["lid"], frame["ew"], frame["u"], frame["v"]
    line = lid[[3, 2, 1, 0]] + v * (0.05 * ew)  # inner -> outer, just above the lashes
    gap = float(np.dot(pts[brow[:5]].astype(np.float32).mean(axis=0) - line.mean(axis=0), v))
    gap = max(gap, 0.34 * ew)  # the brow lower edge; fall back to eye size
    # the crease sits highest above the outer third, and the wing runs past the
    # corner — otherwise the wash reads as a bruise rather than makeup
    arc = np.array([
        line[0] + v * (0.40 * gap),
        line[1] + v * (0.62 * gap),
        line[2] + v * (0.82 * gap),
        line[3] + v * (0.90 * gap),
    ])
    tip = line[3] + u * (0.30 * ew) + v * (0.30 * ew)
    polygon = np.vstack([arc, tip[None], line[::-1]])
    box = _roi(polygon, shape, 0.15 * gap + 3)
    x0, y0, x1, y1 = box
    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(local, [np.round(polygon - [x0, y0]).astype(np.int32)], 255)
    # Fade with distance from the extended lash line: solid at the lashes,
    # transparent at the brow, and the wing melts outwards the same way.
    path = np.zeros_like(local)
    cv2.polylines(
        path,
        [np.round(np.vstack([line, tip[None]]) - [x0, y0]).astype(np.int32)],
        False,
        255,
        thickness=max(2, int(round(ew * 0.06))),
        lineType=cv2.LINE_AA,
    )
    sigma = max(0.46 * gap, 0.16 * ew)
    weight = np.exp(-(cv2.distanceTransform(255 - path, cv2.DIST_L2, 3) / sigma) ** 2)
    mask = local.astype(np.float32) / 255.0 * weight.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, ew * 0.05))
    return _paste(shape, box, mask)


def liner_mask(pts: np.ndarray, shape: tuple[int, int], group: list[int], up: np.ndarray) -> np.ndarray:
    """Tapered line hugging the upper lashes, following the eye's own curve.

    No wing: the stroke stops at the outer corner, so it reads as makeup sitting
    on the lashes instead of a line drawn past the eye.
    """
    frame = _eye_frame(pts, group, up)
    line = frame["lid"][[3, 2, 1, 0]] + frame["v"] * (0.02 * frame["ew"])
    ew, u = frame["ew"], frame["u"]
    # Start past the inner corner: the tear duct is not lid, and a line that
    # reaches it reads as a smudge rather than eyeliner.
    line[0] = line[0] + u * (0.12 * ew)
    thick = max(1.0, ew * 0.065)
    taper = (0.35, 0.62, 0.85)  # inner corner thin, firm over the outer lid

    box = _roi(line, shape, thick * 3 + 3)
    x0, y0, x1, y1 = box
    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    points = np.round(line - [x0, y0]).astype(np.int32)
    for i in range(len(points) - 1):
        cv2.line(local, (int(points[i][0]), int(points[i][1])),
                 (int(points[i + 1][0]), int(points[i + 1][1])), 255,
                 max(1, int(round(thick * taper[i]))), cv2.LINE_AA)
    # Round the outer end so the stroke does not finish on a squared-off corner.
    cv2.circle(local, (int(points[-1][0]), int(points[-1][1])), max(1, int(round(thick * 0.42))),
               255, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(local.astype(np.float32) / 255.0, (0, 0), max(0.8, ew * 0.02))
    return _paste(shape, box, mask)


def blush_mask(pts: np.ndarray, shape: tuple[int, int], side: dict, up: np.ndarray) -> np.ndarray:
    """Soft elliptical flush on the cheekbone, aligned with the face's width."""
    _, _, fw = face_axes(pts)
    axis = pts[side["outer"]].astype(np.float32) - pts[side["inner"]].astype(np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-3)
    centre = pts[side["cheek"]].astype(np.float32) + axis * (0.08 * fw) - up * (0.05 * fw)
    perp = np.array([-axis[1], axis[0]], dtype=np.float32)
    rx, ry = 0.19 * fw, 0.115 * fw
    box = _roi(centre[None], shape, 3.0 * rx)
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - centre[0], yy - centre[1]
    a = dx * axis[0] + dy * axis[1]
    b = dx * perp[0] + dy * perp[1]
    mask = np.exp(-((a / rx) ** 2 + (b / ry) ** 2) * 1.3).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, rx * 0.10))
    return _paste(shape, box, mask)


def lip_mask(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Vermilion border minus the mouth opening: teeth and gaps stay clean."""
    _, _, fw = face_axes(pts)
    h, w = shape[:2]
    outer = np.zeros((h, w), np.uint8)
    cv2.fillPoly(outer, [np.round(pts[LIPS_OUTER]).astype(np.int32)], 255)
    inner = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inner, [np.round(pts[LIPS_INNER]).astype(np.int32)], 255)
    band = ((outer > 127) & (inner < 128)).astype(np.float32)
    return np.clip(_soften(band, max(1.0, fw * 0.004), max(1.0, fw * 0.006)), 0.0, 1.0)


def tint_region(
    bgr: np.ndarray, mask: np.ndarray, color: np.ndarray, amount: float,
    lum: float = 0.30, sat: float = 1.0,
) -> np.ndarray:
    """Move a region's mean chroma onto ``color`` while keeping its shading."""
    amount = clamp(float(amount), 0.0, 1.0)
    weight = np.clip(mask, 0.0, 1.0) * amount
    ys, xs = np.where(weight > 0.004)
    if len(ys) == 0:
        return bgr
    # Work inside the mask's bounding box only: a full-frame Lab round trip
    # re-quantises pixels we never meant to touch.
    y0, y1 = max(int(ys.min()) - 2, 0), min(int(ys.max()) + 3, bgr.shape[0])
    x0, x1 = max(int(xs.min()) - 2, 0), min(int(xs.max()) + 3, bgr.shape[1])
    w = weight[y0:y1, x0:x1]
    total = float(w.sum())
    if total < 1.0:
        return bgr
    lab = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.array([[color]], np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    mean_l = float((lab[..., 0] * w).sum() / total)
    mean_a = float((lab[..., 1] * w).sum() / total)
    mean_b = float((lab[..., 2] * w).sum() / total)
    # A dark lipstick should read darker, a peach blush lighter — but only a
    # little: the shading is what makes it look like skin.
    lift = clamp(float(target[0]) / max(mean_l, 1.0) - 1.0, -0.45, 0.35) * lum
    da = (float(target[1]) - mean_a) * 0.95
    db = (float(target[2]) - mean_b) * 0.95
    tinted = lab.copy()
    tinted[..., 0] = np.clip(lab[..., 0] * (1.0 + lift), 0.0, 255.0)
    tinted[..., 1] = np.clip(128.0 + (lab[..., 1] + da - 128.0) * sat, 0.0, 255.0)
    tinted[..., 2] = np.clip(128.0 + (lab[..., 2] + db - 128.0) * sat, 0.0, 255.0)
    tinted_bgr = cv2.cvtColor(np.clip(tinted, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    original = bgr[y0:y1, x0:x1].astype(np.float32)
    out = bgr.copy()
    # Composite back through the mask: where it is 0 the pixels stay bit-exact.
    out[y0:y1, x0:x1] = np.clip(original + (tinted_bgr - original) * w[..., None], 0.0, 255.0).astype(np.uint8)
    return out


def paint_solid(bgr: np.ndarray, mask: np.ndarray, color: np.ndarray, amount: float) -> np.ndarray:
    """Straight alpha blend — for eyeliner, which really is opaque pigment."""
    weight = np.clip(mask, 0.0, 1.0) * clamp(float(amount), 0.0, 1.0)
    if float(weight.max()) < 0.004:
        return bgr
    layer = np.empty_like(bgr, dtype=np.float32)
    layer[...] = color
    out = bgr.astype(np.float32) * (1.0 - weight[..., None]) + layer * weight[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def apply_makeup(bgr: np.ndarray, pts: np.ndarray, params: "Params",
                 skin: np.ndarray | None = None) -> np.ndarray:
    """Paint the enabled layers, cheeks first and lips last."""
    shadow = clamp(params.shadow, 0, 100) / 100.0
    liner = clamp(params.liner, 0, 100) / 100.0
    blush = clamp(params.blush, 0, 100) / 100.0
    lip = clamp(params.lip, 0, 100) / 100.0
    if max(shadow, liner, blush, lip) <= 0.004:
        return bgr
    up = up_vector(pts)
    out = bgr
    if blush > 0:
        colour = parse_color(params.blush_color, COLOR_FALLBACK["blush"])
        for side in MAKEUP_SIDES:
            mask = blush_mask(pts, out.shape, side, up)
            if skin is not None:
                mask = mask * skin  # keep the flush off hair, brows and clothes
            out = tint_region(out, mask, colour, blush * 0.8, lum=0.42, sat=1.06)
    if shadow > 0:
        colour = parse_color(params.shadow_color, COLOR_FALLBACK["shadow"])
        for side in MAKEUP_SIDES:
            mask = eyeshadow_mask(pts, out.shape, side["eye"], side["brow"], up)
            if skin is not None:
                # Brow hair and a glasses rim are not skin; without this the
                # wash paints over them and the crease gets a hard edge.
                mask = mask * skin
            out = tint_region(out, mask, colour, shadow * 0.85, lum=0.34, sat=1.10)
    if liner > 0:
        colour = parse_color(params.liner_color, COLOR_FALLBACK["liner"])
        for side in MAKEUP_SIDES:
            mask = liner_mask(pts, out.shape, side["eye"], up)
            out = paint_solid(out, mask, colour, liner * 0.92)
    if lip > 0:
        colour = parse_color(params.lip_color, COLOR_FALLBACK["lip"])
        out = tint_region(out, lip_mask(pts, out.shape), colour, lip * 0.95, lum=0.45, sat=1.12)
    return out


# ---------------------------------------------------------------------------
# Liquify
# ---------------------------------------------------------------------------


def _stroke_path(points: np.ndarray, radius: float) -> list[tuple[float, float]]:
    """Resample a pointer path at roughly a quarter-brush spacing."""
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [(float(points[0][0]), float(points[0][1]))]
    spacing = max(radius * 0.22, 1.5)
    out: list[tuple[float, float]] = [(float(points[0][0]), float(points[0][1]))]
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        steps = max(1, int(math.ceil(length / spacing)))
        for i in range(1, steps + 1):
            p = start + delta * (i / steps)
            out.append((float(p[0]), float(p[1])))
    return out


def _liquify_stroke(bgr: np.ndarray, stroke: dict) -> np.ndarray:
    h, w = bgr.shape[:2]
    points = stroke.get("points") or []
    try:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return bgr
    if len(pts) == 0:
        return bgr
    # coordinates are normalised to the *source* frame; the caller maps a crop
    pts = pts * np.array([w, h], np.float32)
    radius = float(stroke.get("radius") or 0.05) * w
    radius = float(clamp(radius, 3.0, max(w, h) * 0.6))
    strength = float(clamp(float(stroke.get("strength") or 0.6), 0.0, 1.0))
    mode = str(stroke.get("mode") or "push")
    path = _stroke_path(pts, radius)

    pad = radius * 1.25
    x0 = int(max(min(p[0] for p in path) - pad, 0))
    x1 = int(min(max(p[0] for p in path) + pad, w))
    y0 = int(max(min(p[1] for p in path) - pad, 0))
    y1 = int(min(max(p[1] for p in path) + pad, h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return bgr

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = np.zeros_like(xx)
    dy = np.zeros_like(yy)

    if mode == "push" and len(path) >= 2:
        # Accumulate the drag direction with a smooth falloff; the centre of the
        # brush follows the pointer exactly and the rim follows it less.
        for (ax, ay), (bx, by) in zip(path[:-1], path[1:]):
            mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
            move_x = (bx - ax) * strength
            move_y = (by - ay) * strength
            d2 = ((xx - mx) ** 2 + (yy - my) ** 2) / (radius * radius)
            fall = np.clip(1.0 - d2, 0.0, 1.0) ** 2
            dx += move_x * fall
            dy += move_y * fall
    else:
        # bloat/pucker: radial scale around the pointer. A drag sums several
        # small applications, so normalise by sqrt(n) to keep it controllable.
        k = strength * 0.30 / max(1.0, math.sqrt(len(path)))
        if mode == "pucker":
            k = -k
        for px, py in path:
            d2 = ((xx - px) ** 2 + (yy - py) ** 2) / (radius * radius)
            fall = np.clip(1.0 - d2, 0.0, 1.0) ** 2
            scale = k * fall
            dx += (xx - px) * scale
            dy += (yy - py) * scale

    # A displacement larger than the brush folds the image over itself.
    limit = radius * 0.45
    mag = np.hypot(dx, dy)
    over = mag > limit
    if over.any():
        factor = np.ones_like(mag)
        factor[over] = limit / np.maximum(mag[over], 1e-6)
        dx *= factor
        dy *= factor
    if float(np.abs(dx).max()) < 0.25 and float(np.abs(dy).max()) < 0.25:
        return bgr

    map_x = (xx - dx).astype(np.float32)
    map_y = (yy - dy).astype(np.float32)
    out = bgr.copy()
    out[y0:y1, x0:x1] = cv2.remap(
        bgr[y0:y1, x0:x1],
        map_x - x0,
        map_y - y0,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    return out


def apply_liquify(bgr: np.ndarray, strokes: list) -> np.ndarray:
    """Replay brush strokes in order; each one warps the previous result."""
    out = bgr
    for stroke in strokes or []:
        if isinstance(stroke, dict) and stroke.get("points"):
            out = _liquify_stroke(out, stroke)
    return out


def map_liquify_strokes(
    strokes: list, source_size: tuple[int, int], window: tuple[int, int, int, int]
) -> list:
    """Re-base source-normalised strokes onto the frame being rendered.

    The panel records points in source-relative coordinates, so the same stroke
    lands on the same spot in a 2400px preview, a 1:1 crop and the full-res
    save. A crop needs both a translation and a radius scale: a brush that is
    4 % of the source width stays 4 % of the source width, not 4 % of the crop.
    """
    sw, sh = source_size
    wx, wy, ww, wh = window
    if ww <= 0 or wh <= 0:
        return []
    mapped = []
    for stroke in strokes or []:
        if not isinstance(stroke, dict):
            continue
        try:
            points = [[float(p[0]), float(p[1])] for p in stroke.get("points") or []]
        except (TypeError, ValueError, IndexError):
            continue
        if not points:
            continue
        mapped.append({
            "mode": str(stroke.get("mode") or "push"),
            "strength": float(clamp(float(stroke.get("strength") or 0.6), 0.0, 1.0)),
            "radius": float(clamp(float(stroke.get("radius") or 0.05), 0.004, 0.6)) * sw / ww,
            "points": [[(x * sw - wx) / ww, (y * sh - wy) / wh] for x, y in points],
        })
    return mapped


def map_mask_strokes(
    strokes: list, source_size: tuple[int, int], window: tuple[int, int, int, int]
) -> list:
    """Re-base source-normalised mask-brush strokes onto the rendered frame."""
    sw, sh = source_size
    wx, wy, ww, wh = window
    if ww <= 0 or wh <= 0:
        return []
    out = []
    for stroke in strokes or []:
        if not isinstance(stroke, dict):
            continue
        try:
            points = [[float(p[0]), float(p[1])] for p in stroke.get("points") or []]
        except (TypeError, ValueError, IndexError):
            continue
        if not points:
            continue
        out.append({
            "radius": float(clamp(float(stroke.get("radius") or 0.03), 0.0008, 0.4)) * sw / ww,
            "points": [[(x * sw - wx) / ww, (y * sh - wy) / wh] for x, y in points],
        })
    return out


def mask_strokes_mask(
    shape: tuple[int, int], strokes: list, feather: float | None = None
) -> np.ndarray:
    """Feathered mask painted from window-normalised brush strokes.

    *feather* overrides the edge blur (a sigma in pixels); pass 0 for a hard
    mask and soften it yourself when the softening must follow the tip size.
    """
    h, w = shape[:2]
    out = np.zeros((h, w), np.uint8)
    for stroke in strokes or []:
        if not isinstance(stroke, dict):
            continue
        r = max(1, int(round(float(stroke.get("radius") or 0.03) * w)))
        pts = []
        for point in stroke.get("points") or []:
            try:
                pts.append((
                    int(round(float(point[0]) * w)),
                    int(round(float(point[1]) * h)),
                ))
            except (TypeError, ValueError, IndexError):
                continue
        if not pts:
            continue
        step = max(2.0, r * 0.4)
        cv2.circle(out, pts[0], r, 255, -1, lineType=cv2.LINE_AA)
        for i in range(1, len(pts)):
            ax, ay = pts[i - 1]
            bx, by = pts[i]
            dist = math.hypot(bx - ax, by - ay)
            n = max(1, int(dist / step))
            for k in range(n + 1):
                t = k / n
                cv2.circle(
                    out,
                    (int(round(ax + (bx - ax) * t)), int(round(ay + (by - ay) * t))),
                    r,
                    255,
                    -1,
                    lineType=cv2.LINE_AA,
                )
    if not out.any():
        return np.zeros((h, w), np.float32)
    sigma = min(h, w) * 0.0015 if feather is None else float(feather)
    if sigma <= 0:
        return out.astype(np.float32) / 255.0
    return cv2.GaussianBlur(out.astype(np.float32) / 255.0, (0, 0), max(0.5, sigma))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def detect_report(input_path: str) -> dict:
    """Landmarks for an image, in native pixel coordinates.

    The host caches this once per source photo so slider previews never pay the
    model load: rendering with ``--landmarks`` skips MediaPipe entirely. The
    skin segmentation rides along in the same file (PNG-compressed class ids,
    ~13 KB per face), and one full-frame pass gives the subject/body-skin map,
    so a preview only has to resize it to the render size.
    """
    bgr = load_image(input_path)
    h, w = bgr.shape[:2]
    faces = detect_faces(FaceModel.get(), bgr)
    segs: list[dict | None] = []
    for pts in faces:
        try:
            segs.append(encode_segment(segment_face(bgr, pts)))
        except Exception:  # noqa: BLE001 — masks are an enhancement, not a hard dependency
            segs.append(None)
    person = None
    try:
        person = encode_segment(segment_person(bgr))
    except Exception:  # noqa: BLE001
        person = None
    return {
        "ok": True,
        "width": w,
        "height": h,
        "faces": [[[round(float(x), 2), round(float(y), 2)] for x, y in pts] for pts in faces],
        "seg": segs,
        "person": person,
    }


def load_landmarks(
    path: str, width: int, height: int
) -> tuple[list[np.ndarray], list[dict | None], dict | None]:
    data = json.loads(Path(path).read_text())
    scale_x = width / float(data.get("width") or width)
    scale_y = height / float(data.get("height") or height)
    faces: list[np.ndarray] = []
    segs: list[dict | None] = []
    raw_segs = data.get("seg") or []
    for i, pts in enumerate(data.get("faces") or []):
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 400:
            arr = arr * np.array([scale_x, scale_y], dtype=np.float32)
            faces.append(arr)
            entry = raw_segs[i] if i < len(raw_segs) else None
            segs.append(decode_segment(entry) if isinstance(entry, dict) else None)
    raw_person = data.get("person")
    person = decode_segment(raw_person) if isinstance(raw_person, dict) else None
    return faces, segs, person


def _vision_name(input_path: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(input_path).stem) or "photo"
    return f"{stem}-{suffix}.jpg"


def _jpeg_bytes(image: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("could not encode vision JPEG")
    return buffer.tobytes()


def _subject_box(person: dict | None, pts: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """A person-sized crop around the selected face.

    The model sees the full frame for composition and the face crop for
    features, but a small subject loses skin texture and body-skin tone in the
    full frame. This crop is only a visual aid: all metrics still come from the
    original image, so geometry confidence is not inflated by cropping.
    """
    if person and isinstance(person.get("cat"), np.ndarray):
        cat = person["cat"]
        mh, mw = cat.shape[:2]
        mask = np.isin(cat, SEG_PERSON).astype(np.uint8)
        if mask.any() and mw > 0 and mh > 0:
            cx = float(pts[:, 0].mean()) * mw / float(w)
            cy = float(pts[:, 1].mean()) * mh / float(h)
            count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            best = 0
            best_dist = float("inf")
            for i in range(1, count):
                x, y, bw, bh, area = (int(v) for v in stats[i])
                if area < max(1, int(0.004 * mw * mh)):
                    continue
                if x <= cx <= x + bw and y <= cy <= y + bh:
                    best = i
                    break
                dist = (float(centroids[i][0]) - cx) ** 2 + (float(centroids[i][1]) - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = i
            if best:
                x, y, bw, bh, _area = (int(v) for v in stats[best])
                sx, sy = w / float(mw), h / float(mh)
                return int(x * sx), int(y * sy), int((x + bw) * sx), int((y + bh) * sy)

    # Fallback: a generous portrait crop around the face oval.
    oval = pts[FACE_OVAL]
    x0, y0 = oval.min(axis=0)
    x1, y1 = oval.max(axis=0)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = side * 2.8, side * 3.6
    return int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)


def analyze_report(
    input_path: str,
    landmarks_path: str | None = None,
    face: int = 0,
    vision_out: str | None = None,
    vision_full: int = 1280,
    vision_face: int = 768,
    vision_person: int = 960,
) -> dict:
    """Compute the AI metrics and write the three small JPEGs sent to the model.

    This deliberately does not touch retouch(): a failed analysis must never
    change the normal preview/save path.  The host passes its cached landmark
    file, so a second click does not reload MediaPipe.
    """
    from beauty_metrics import analyze_metrics

    t0 = time.time()
    bgr = load_image(input_path)
    h, w = bgr.shape[:2]
    if landmarks_path:
        faces, segs, person = load_landmarks(landmarks_path, w, h)
    else:
        faces = detect_faces(FaceModel.get(), bgr)
        segs = []
        for pts in faces:
            try:
                segs.append(segment_face(bgr, pts))
            except Exception:  # noqa: BLE001
                segs.append(None)
        try:
            person = segment_person(bgr)
        except Exception:  # noqa: BLE001
            person = None

    count = len(faces)
    index = 0 if count == 0 else int(clamp(face, 0, count - 1))
    if count:
        pts = faces[index]
        metrics = analyze_metrics(bgr, pts, segs[index] if index < len(segs) else None, person, count, index)
    else:
        metrics = {
            "image": {"width": w, "height": h, "subject_ratio": 0.0, "face_ratio": 0.0, "light": "good"},
            "pose": {"faces": 0, "selected": 0, "yaw": None, "roll_deg": None, "reliable": False},
            "geometry": {}, "skin": {"available": False}, "body": {"available": False},
            "lips": {"available": False},
            "quality": {"light": "good", "clipping": 0.0, "segmentation": False, "landmarks": False,
                        "notes": ["no face detected"]},
        }

    vision = {"full": None, "person": None, "face": None}
    if vision_out:
        out_dir = Path(vision_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        full = bgr
        long_side = max(h, w)
        if long_side > vision_full:
            scale = vision_full / float(long_side)
            full = cv2.resize(bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                              interpolation=cv2.INTER_AREA)
        full_path = out_dir / _vision_name(input_path, "full")
        full_path.write_bytes(_jpeg_bytes(full, 85))
        vision["full"] = str(full_path)

        if count:
            # A person crop makes a small subject large enough for the model to
            # judge skin tone and body exposure.  The metrics above stay tied to
            # the original frame, so this cannot game geometry_confidence.
            bx0, by0, bx1, by1 = _subject_box(person, pts, w, h)
            pad_x = int((bx1 - bx0) * 0.12)
            pad_y = int((by1 - by0) * 0.08)
            bx0, by0 = max(0, bx0 - pad_x), max(0, by0 - pad_y)
            bx1, by1 = min(w, bx1 + pad_x), min(h, by1 + pad_y)
            if bx1 > bx0 and by1 > by0:
                person_crop = bgr[by0:by1, bx0:bx1]
                ch, cw = person_crop.shape[:2]
                scale = vision_person / float(max(ch, cw))
                person_crop = cv2.resize(
                    person_crop,
                    (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
                )
                person_path = out_dir / _vision_name(input_path, "person")
                person_path.write_bytes(_jpeg_bytes(person_crop, 86))
                vision["person"] = str(person_path)

            oval = pts[FACE_OVAL]
            x0, y0 = oval.min(axis=0)
            x1, y1 = oval.max(axis=0)
            side = max(x1 - x0, y1 - y0) * 1.5
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            left = int(round(cx - side / 2.0))
            top = int(round(cy - side / 2.0))
            side = max(8, int(round(side)))
            right = min(w, left + side)
            bottom = min(h, top + side)
            left = max(0, left)
            top = max(0, top)
            crop = bgr[top:bottom, left:right]
            if crop.size:
                crop = cv2.resize(crop, (vision_face, vision_face), interpolation=cv2.INTER_AREA)
                face_path = out_dir / _vision_name(input_path, "face")
                face_path.write_bytes(_jpeg_bytes(crop, 88))
                vision["face"] = str(face_path)

    return {
        "ok": True,
        "width": w,
        "height": h,
        "faces": count,
        "face_used": index,
        "metrics": metrics,
        "vision": vision,
        "notes": metrics.get("quality", {}).get("notes", []),
        "elapsed_ms": round((time.time() - t0) * 1000),
    }


def retouch(
    input_path: str,
    output_path: str,
    params: Params,
    landmarks_path: str | None = None,
    mask_out: str | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> dict:
    """Render one photo.

    ``crop`` renders only that rectangle of the source at native resolution —
    the panel uses it to show a 1:1 patch of whatever the user zoomed into.
    Landmarks and the cached segmentation are shifted into crop coordinates, so
    the filters see the same neighbours they would in a full-frame render.
    """
    t0 = time.time()
    bgr = load_image(input_path)
    source_w, source_h = bgr.shape[1], bgr.shape[0]
    original_w, original_h = source_w, source_h

    if crop is not None:
        x0 = int(clamp(crop[0], 0, max(source_w - 1, 0)))
        y0 = int(clamp(crop[1], 0, max(source_h - 1, 0)))
        cw = int(clamp(crop[2], 1, source_w - x0))
        ch = int(clamp(crop[3], 1, source_h - y0))
        bgr = bgr[y0:y0 + ch, x0:x0 + cw].copy()
        crop = (x0, y0, cw, ch)
    source_window = crop if crop is not None else (0, 0, source_w, source_h)

    person_seg: dict | None = None
    if landmarks_path:
        faces, segs, person_seg = load_landmarks(landmarks_path, source_w, source_h)
    else:
        # Detect at native resolution: shrinking first is what loses small subjects.
        faces = detect_faces(FaceModel.get(), bgr)
        segs = []
    if crop is not None:
        off = np.array([crop[0], crop[1]], dtype=np.float32)
        faces = [pts - off for pts in faces]
        segs = [
            dict(seg, box=[seg["box"][0] - crop[0], seg["box"][1] - crop[1], seg["box"][2]])
            if seg
            else None
            for seg in segs
        ]
    index = params.face if 0 <= params.face < len(faces) else 0
    chosen = faces[index] if faces else None
    seg = segs[index] if index < len(segs) else None

    scale = 1.0
    if params.max_size and max(bgr.shape[:2]) > params.max_size:
        # Resize the frame we are about to process, which is the crop when one
        # was requested — not the full source.
        cur_h, cur_w = bgr.shape[:2]
        scale = params.max_size / max(cur_h, cur_w)
        bgr = cv2.resize(bgr, (max(int(cur_w * scale), 1), max(int(cur_h * scale), 1)),
                         interpolation=cv2.INTER_AREA)
        if chosen is not None:
            chosen = chosen * scale
    h, w = bgr.shape[:2]

    notes: list[str] = []
    skin_stats = {"face": 0.0, "body": 0.0, "parts": 0, "people": 0, "subject": 0.0}
    if chosen is None:
        notes.append("no face detected — only global detail sharpening applied")
        out = bgr
    else:
        warped, wpts, field = warp_portrait(
            bgr, chosen, params.slim, params.eyes, return_field=True
        )
        if seg is None:
            # Cached report from before the segmenter shipped, or the host had
            # no cache yet: segment this render instead of degrading to the oval.
            try:
                seg = segment_face(bgr, chosen)
                scale = 1.0
                person_seg = None
            except FileNotFoundError:
                seg = None

        if seg is not None:
            mask = warp_mask(skin_mask_from_segment(bgr, chosen, seg, scale), field)
            notes.append("skin range: selfie segmentation (face-skin class)")
        else:
            mask = skin_mask(wpts, warped.shape)
            notes.append("skin range: landmark oval (segmentation model missing)")

        # Full-frame subject segmentation (cached with the landmarks). It is
        # what makes "人物提亮" reach the neck, shoulders and arms, and what
        # decides which person in a group is the subject.
        person_window = source_window
        person_scale = scale
        if person_seg is None:
            try:
                person_seg = segment_person(bgr)
                person_window = (0, 0, w, h)
                person_scale = 1.0
            except FileNotFoundError:
                person_seg = None
        if person_seg is not None:
            subject_mask, skin_stats = subject_skin(
                (h, w), chosen, person_seg, person_window, person_scale, warped
            )
        else:
            subject_mask = np.zeros((h, w), np.float32)
            notes.append("subject skin: body map unavailable")

        # 手动补皮肤: painted areas join the subject mask, so they follow the
        # same 人物提亮 / 整体肤色 pass as the automatic segmentation.
        brush_mask = np.zeros((h, w), np.float32)
        if params.mask_strokes:
            mapped_masks = map_mask_strokes(
                params.mask_strokes, (source_w, source_h), source_window
            )
            brush_mask = mask_strokes_mask((h, w), mapped_masks)
            subject_mask = np.maximum(subject_mask, brush_mask)
            notes.append(f"mask brush: {len(mapped_masks)} stroke(s)")

        # Smoothing skips brows/eyes/lips; tone does not. Punching them out of a
        # brightened face leaves dark features floating in a pale patch.
        temperature = clamp(params.temperature + params.warmth, -50.0, 50.0)
        tint = clamp(params.tint, -50.0, 50.0)
        needs_tone = params.whiten > 0 or temperature != 0 or tint != 0 or params.person > 0
        tone = mask
        if needs_tone:
            if seg is not None:
                tone = warp_mask(
                    skin_mask_from_segment(bgr, chosen, seg, scale, protect_features=False),
                    field,
                )
            else:
                tone = skin_mask(wpts, warped.shape, protect_features=False)
            notes.append("tone range: whole face (features included)")

        out = smooth_skin(warped, mask, params.smooth, face_axes(wpts)[2])
        # 皱纹修复 rides on the smoothed frame: creases are measured after the
        # 磨皮 pass, so the two sliders stack instead of re-measuring each
        # other's output. The brush erases whatever it is painted over, which is
        # why it works even with the slider at 0.
        wrinkle_brush = np.zeros((h, w), np.float32)
        mapped_wrinkle = []
        if params.wrinkle_strokes:
            mapped_wrinkle = map_mask_strokes(
                params.wrinkle_strokes, (source_w, source_h), source_window
            )
            wrinkle_brush = mask_strokes_mask((h, w), mapped_wrinkle, feather=0.0)
            # The edge softening follows the tip, not the frame: a frame-sized
            # feather turns a 3 px tip into a 20 px blob, which is exactly why
            # the small sizes used to feel huge.
            tip = (w * max(float(st.get("radius") or 0.0) for st in mapped_wrinkle))
            wrinkle_brush = cv2.GaussianBlur(
                wrinkle_brush, (0, 0), float(clamp(tip * 0.35, 0.5, 14.0))
            )
            # Painting across an eyelid or a lip must not erase the feature.
            wrinkle_brush = wrinkle_brush * (
                1.0 - feature_protect_mask(wpts, warped.shape, face_axes(wpts)[2])
            )
            notes.append(f"wrinkle brush: {len(mapped_wrinkle)} stroke(s)")
        if params.wrinkle > 0 or wrinkle_brush.any():
            out = apply_wrinkle_repair(
                out, mask, face_axes(wpts)[2], params.wrinkle, wrinkle_brush
            )
            if params.wrinkle > 0:
                notes.append(f"wrinkle repair: {params.wrinkle:.0f}%")
        lift = tone * (clamp(params.whiten, 0.0, 100.0) / 100.0)
        # 人物提亮 is a gamma curve rather than another additive lift: the
        # slider drives both the blend amount and the curve's gamma, and gamma
        # grows faster than the slider, so the last third brightens much more
        # than the first third. At 100 the skin is pushed right up toward the
        # top of the range without a hard clip.
        person_glow = None
        person_gamma = 1.0
        if params.person > 0 and (person_seg is not None or brush_mask.any()):
            t = clamp(params.person, 0.0, 100.0) / 100.0
            person_glow = subject_mask * t
            person_gamma = 1.0 + 2.4 * (t ** 1.2)
            notes.append(
                f"person glow: {params.person:.0f}% over face + body skin "
                f"(gamma {person_gamma:.2f})"
            )
        if temperature != 0 or tint != 0:
            # Only actual skin, like the smoothing mask: the whole-face tone
            # mask exists so whitening does not punch holes in the brows, but
            # warming the lips and eyelashes would read as a filter.
            overall_skin = np.maximum(mask, subject_mask)
            temp = overall_skin * (temperature / 50.0)
            tin = overall_skin * (tint / 50.0)
            notes.append(
                "skin tone: face + body skin"
                if person_seg is not None
                else "skin tone: face skin"
            )
        else:
            temp = tin = None
        out = tone_weighted(out, lift, temp, tin, person_glow, person_gamma)
        out = crisp_details(out, mask, params.detail)

        skin_stats["face"] = round(float(tone.mean()) * 100.0, 2)
        if person_seg is not None:
            notes.append(
                f"subject skin: {int(skin_stats['people'])} person, "
                f"face {skin_stats['face']:.1f}%, body {skin_stats['body']:.1f}% "
                f"in {int(skin_stats['parts'])} parts"
            )

        makeup = [
            (name, getattr(params, key))
            for name, key in (("eyes", "shadow"), ("liner", "liner"),
                              ("blush", "blush"), ("lip", "lip"))
            if getattr(params, key) > 0
        ]
        if makeup:
            # Painted last, on the warped landmarks, so the tint lands on the
            # finished skin instead of being smeared by the smoothing pass.
            out = apply_makeup(out, wpts, params, mask)
            notes.append("makeup: " + ", ".join(f"{k} {v:.0f}%" for k, v in makeup))

        notes.append(f"image {w}x{h}px, skin-mask coverage {float(mask.mean()) * 100:.1f}%")
        if mask_out:
            Path(mask_out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(mask_out, np.clip(mask * 255.0, 0, 255).astype(np.uint8))
            notes.append(f"mask written to {mask_out}")

    if params.liquify:
        strokes = map_liquify_strokes(params.liquify, (source_w, source_h), source_window)
        out = apply_liquify(out, strokes)
        notes.append(f"liquify: {len(strokes)} stroke(s)")

    save_image(output_path, out)

    result = {
        "ok": True,
        "output": str(Path(output_path).resolve()),
        "input": str(Path(input_path).resolve()),
        "width": w,
        "height": h,
        "original_width": original_w,
        "original_height": original_h,
        "faces": len(faces),
        "face_used": params.face if chosen is not None else None,
        "params": {
            k: v for k, v in asdict(params).items()
            if k not in ("compare", "face", "liquify", "mask_strokes", "wrinkle_strokes")
        },
        "elapsed_ms": int((time.time() - t0) * 1000),
        "skin": skin_stats,
        "notes": notes,
    }
    if crop is not None:
        result["crop"] = [int(v) for v in crop]

    if params.compare:
        sheet = compare_sheet(bgr, out)
        cmp_path = str(Path(output_path).with_name(Path(output_path).stem + "-compare.jpg"))
        save_image(cmp_path, sheet)
        result["compare"] = cmp_path

    return result


def compare_sheet(before: np.ndarray, after: np.ndarray, width: int = 1600) -> np.ndarray:
    h, w = after.shape[:2]
    scale = min(1.0, width / (w * 2))
    bw, bh = max(int(w * scale), 1), max(int(h * scale), 1)
    left = cv2.resize(before, (bw, bh), interpolation=cv2.INTER_AREA)
    right = cv2.resize(after, (bw, bh), interpolation=cv2.INTER_AREA)
    gap = 8
    sheet = np.full((bh, bw * 2 + gap, 3), 255, dtype=np.uint8)
    sheet[:, :bw] = left
    sheet[:, bw + gap:] = right
    for label, x in (("BEFORE", 12), ("AFTER", bw + gap + 12)):
        cv2.putText(sheet, label, (x, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                    (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(sheet, label, (x, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                    (30, 30, 30), 1, cv2.LINE_AA)
    return sheet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local portrait beauty retouch")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None, help="required unless --detect/--analyze")
    p.add_argument("--smooth", type=float, default=0)
    p.add_argument("--whiten", type=float, default=0)
    p.add_argument("--person", type=float, default=0,
                   help="subject-wide brightening 0-100 (gamma curve; face + neck/shoulders/arms)")
    p.add_argument("--wrinkle", type=float, default=0,
                   help="wrinkle repair 0-100 on the face skin (fades crease-like ridges)")
    p.add_argument("--slim", type=float, default=0)
    p.add_argument("--eyes", type=float, default=0)
    p.add_argument("--detail", type=float, default=0.35)
    p.add_argument("--temperature", type=float, default=0,
                   help="overall skin temperature -50 cool .. +50 warm (Lab b*)")
    p.add_argument("--tint", type=float, default=0,
                   help="overall skin tint -50 green .. +50 magenta (Lab a*)")
    p.add_argument("--warmth", type=float, default=0,
                   help="deprecated alias for --temperature")
    p.add_argument("--shadow", type=float, default=0, help="eyeshadow strength 0-100")
    p.add_argument("--shadow-color", default=None, help="#rrggbb eyeshadow colour")
    p.add_argument("--liner", type=float, default=0, help="eyeliner strength 0-100")
    p.add_argument("--liner-color", default=None, help="#rrggbb eyeliner colour")
    p.add_argument("--blush", type=float, default=0, help="blush strength 0-100")
    p.add_argument("--blush-color", default=None, help="#rrggbb blush colour")
    p.add_argument("--lip", type=float, default=0, help="lipstick strength 0-100")
    p.add_argument("--lip-color", default=None, help="#rrggbb lipstick colour")
    p.add_argument("--face", type=int, default=0)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--max-size", type=int, default=0, help="cap the long side (0 = full)")
    p.add_argument("--mask-strokes", default=None,
                   help="manual skin mask strokes as JSON: [{points, radius}]")
    p.add_argument("--wrinkle-strokes", default=None,
                   help="wrinkle brush strokes as JSON: [{points, radius}] in source-normalised coords")
    p.add_argument("--liquify", default=None,
                   help='JSON strokes: [{"mode":"push|bloat|pucker","radius":0.05,'
                        '"strength":0.6,"points":[[x,y],...]}] in source-normalised coords')
    p.add_argument("--model", default=None)
    p.add_argument("--landmarks", default=None, help="cached landmarks JSON (skips the face model)")
    p.add_argument("--mask-out", default=None, help="also write the skin mask as PNG (debug)")
    p.add_argument("--crop", default=None, help="render only X,Y,W,H of the source at native size")
    p.add_argument("--seg-model", default=None, help="override the multiclass segmentation model")
    p.add_argument("--detect", action="store_true", help="print face landmarks instead of rendering")
    p.add_argument("--analyze", action="store_true", help="print AI metrics + vision JPEG paths")
    p.add_argument("--vision-out", default=None, help="directory for the three AI vision JPEGs")
    p.add_argument("--vision-full", type=int, default=1280, help="long side of the full vision JPEG")
    p.add_argument("--vision-face", type=int, default=768, help="side of the square face vision JPEG")
    p.add_argument("--vision-person", type=int, default=960, help="long side of the person vision JPEG")
    p.add_argument("--json", action="store_true", help="(default) print JSON on stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model:
        os.environ["BEAUTY_MODEL"] = args.model
    if args.seg_model:
        os.environ["BEAUTY_SEG_MODEL"] = args.seg_model
    params = Params(
        smooth=clamp(args.smooth, 0, 100),
        whiten=clamp(args.whiten, 0, 100),
        person=clamp(args.person, 0, 100),
        wrinkle=clamp(args.wrinkle, 0, 100),
        slim=clamp(args.slim, 0, 100),
        eyes=clamp(args.eyes, 0, 100),
        detail=clamp(args.detail, 0, 1.5),
        temperature=clamp(args.temperature, -50, 50),
        tint=clamp(args.tint, -50, 50),
        warmth=clamp(args.warmth, -100, 100),
        shadow=clamp(args.shadow, 0, 100),
        shadow_color=args.shadow_color or "#a97b8c",
        liner=clamp(args.liner, 0, 100),
        liner_color=args.liner_color or "#2a211f",
        blush=clamp(args.blush, 0, 100),
        blush_color=args.blush_color or "#e2827e",
        lip=clamp(args.lip, 0, 100),
        lip_color=args.lip_color or "#b4404f",
        face=max(args.face, 0),
        compare=args.compare,
        max_size=max(args.max_size, 0),
    )
    try:
        if args.detect:
            print(json.dumps(detect_report(args.input), ensure_ascii=False))
            return 0
        if args.analyze:
            print(json.dumps(
                analyze_report(
                    args.input, args.landmarks, args.face, args.vision_out,
                    max(256, args.vision_full), max(256, args.vision_face), max(256, args.vision_person),
                ),
                ensure_ascii=False,
            ))
            return 0
        if args.mask_strokes:
            strokes = json.loads(args.mask_strokes)
            if not isinstance(strokes, list):
                raise ValueError("--mask-strokes wants a JSON array of strokes")
            params.mask_strokes = strokes
        if args.wrinkle_strokes:
            strokes = json.loads(args.wrinkle_strokes)
            if not isinstance(strokes, list):
                raise ValueError("--wrinkle-strokes wants a JSON array of strokes")
            params.wrinkle_strokes = strokes
        if args.liquify:
            strokes = json.loads(args.liquify)
            if not isinstance(strokes, list):
                raise ValueError("--liquify wants a JSON array of strokes")
            params.liquify = strokes
        if not args.output:
            print(json.dumps({"ok": False, "error": "--output is required unless --detect/--analyze"}))
            return 2
        crop = None
        if args.crop:
            crop = tuple(int(v) for v in args.crop.replace(" ", "").split(","))
            if len(crop) != 4:
                raise ValueError("--crop wants X,Y,W,H")
        result = retouch(args.input, args.output, params, args.landmarks, args.mask_out, crop)
    except Exception as exc:  # noqa: BLE001 — the host consumes this JSON
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
