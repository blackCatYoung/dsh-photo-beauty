/**
 * dsh-photo-beauty — Host half.
 *
 * Two surfaces share one local engine (`engine/beauty.py`, MediaPipe FaceMesh +
 * OpenCV):
 *   1. the `beauty_retouch` Tool, for chat-driven retouching;
 *   2. the `/api/dsh-beauty/*` Fetch routes that back the Web workbench panel.
 *
 * The engine is a local Python process. Images only leave the Host through the
 * explicit "AI 一键修图" flow, after the user consents to sending two compressed
 * previews to the configured model.
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { randomUUID, createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, stat, readdir, rm, rename } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const name = 'dsh-photo-beauty';
export const inject = ['connection', 'tools', 'workingDirectory'];

const PLUGIN_ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const ENGINE = path.join(PLUGIN_ROOT, 'engine', 'beauty.py');
const MODEL = path.join(PLUGIN_ROOT, 'engine', 'models', 'face_landmarker.task');
const WORKBENCH = path.join(PLUGIN_ROOT, 'lib', 'workbench.html');
const SKILL_FILE = path.join(PLUGIN_ROOT, 'skills', 'portrait-beauty', 'SKILL.md');

const ROUTE = {
  workbench: '/api/dsh-beauty/workbench',
  upload: '/api/dsh-beauty/upload',
  source: '/api/dsh-beauty/source',
  thumb: '/api/dsh-beauty/thumb',
  render: '/api/dsh-beauty/render',
  renderFile: '/api/dsh-beauty/render-file',
  detail: '/api/dsh-beauty/detail',
  save: '/api/dsh-beauty/save',
  pickDir: '/api/dsh-beauty/pick-dir',
  reveal: '/api/dsh-beauty/reveal',
  aiSuggest: '/api/dsh-beauty/ai-suggest',
  aiSkill: '/api/dsh-beauty/ai-skill',
};

const ID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{3,80}$/;
const MAX_UPLOAD_BYTES = 40 * 1024 * 1024;
const ENGINE_TIMEOUT_MS = 10 * 60 * 1000;
// 2400 keeps the preview at >=1 preview px per CSS px at 100% zoom on a 2x
// display (the pane is ~1080 CSS px wide), so only real zoom-outs need the
// detail patch below.
const PREVIEW_LONG_SIDE = 2400;
// Zoomed-in detail patches: native pixels up to this long side, downscaled past it.
const DETAIL_LONG_SIDE = 2600;
const CACHE_KEEP = 240;
const AI_TIMEOUT_MS = 120_000;
const AI_DEFAULT_LONG_SIDE = { full: 1280, person: 960, face: 768 };
// Bump when the metric schema or formula changes; analysis caches are keyed by it.
const ANALYSIS_VERSION = 4;

const JSON_HEADERS = {
  'content-type': 'application/json; charset=utf-8',
  'cache-control': 'no-store',
};

const IMAGE_TYPES = new Map([
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.png', 'image/png'],
  ['.webp', 'image/webp'],
  ['.heic', 'image/heic'],
]);

// ---------------------------------------------------------------------------
// helpers

function json(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });
}

function fail(code, message, status = 400, hint) {
  const error = { code, message };
  if (typeof hint === 'string' && hint.length > 0) error.hint = hint;
  return json({ ok: false, error }, status);
}

function log(ctx, message, error) {
  try {
    const logger = ctx.logger;
    if (logger && typeof logger.warn === 'function') logger.warn(`dsh-photo-beauty: ${message}`, error ?? '');
  } catch {
    /* diagnostics never break a request */
  }
}

function clamp(value, min, max, fallback) {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function localStamp(date = new Date()) {
  const pad = (value) => String(value).padStart(2, '0');
  return (
    `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}` +
    `-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`
  );
}

function safeBase(name) {
  const base = path.basename(String(name || ''), path.extname(String(name || '')));
  return base.replace(/[\\/:*?"<>|\u0000-\u001f]/g, '-').replace(/\s+/g, ' ').trim().slice(0, 60);
}

function numberOr(value, fallback) {
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) ? n : fallback;
}

// MediaPipe / TFLite print routine diagnostics to stderr on every model load.
// They are never the failure, so keep them out of user-facing error text.
const STDERR_NOISE =
  /^(?:[WIEF]\d{4} \d|INFO: Created TensorFlow Lite|WARNING: All log messages before absl::InitializeLog)/;

/**
 * Best error text for a failed engine run.
 *
 * The engine reports real failures as {"ok":false,"error":...} on stdout and
 * still exits non-zero, so that JSON wins over stderr; without this the UI
 * showed the harmless XNNPACK banner instead of the actual exception.
 */
function engineFailure(stdout, stderr, code) {
  const line = stdout
    .split('\n')
    .reverse()
    .find((entry) => entry.trim().startsWith('{'));
  if (line) {
    try {
      const value = JSON.parse(line);
      if (value && value.ok === false && value.error) return String(value.error);
    } catch {
      /* not JSON after all — fall back to stderr */
    }
  }
  const lines = stderr
    .split('\n')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0 && !STDERR_NOISE.test(entry));
  const tail = lines.slice(-6).join('\n');
  if (tail.length > 0) return tail;
  return code === null ? 'Beauty engine was killed before it finished' : `Beauty engine exited with code ${code}`;
}

/** '#rgb' / '#rrggbb' / bare hex -> '#rrggbb', or the fallback. */
function colourOr(value, fallback) {
  const text = String(value ?? '').trim().replace(/^#/, '');
  const hex = text.length === 3 ? text.split('').map((c) => c + c).join('') : text;
  return /^[0-9a-f]{6}$/i.test(hex) ? '#' + hex.toLowerCase() : fallback;
}

/** One liquify stroke: source-normalised points, a brush radius, a mode. */
function normaliseStroke(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const mode = raw.mode === 'bloat' || raw.mode === 'pucker' ? raw.mode : 'push';
  const points = Array.isArray(raw.points)
    ? raw.points
        .slice(0, 512)
        .map((p) => {
          if (!Array.isArray(p) || p.length < 2) return null;
          const x = clamp(p[0], -0.5, 1.5, NaN);
          const y = clamp(p[1], -0.5, 1.5, NaN);
          return Number.isFinite(x) && Number.isFinite(y) ? [Number(x.toFixed(5)), Number(y.toFixed(5))] : null;
        })
        .filter(Boolean)
    : [];
  if (points.length === 0) return null;
  return {
    mode,
    radius: Number(clamp(raw.radius, 0.004, 0.6, 0.05).toFixed(4)),
    strength: Number(clamp(raw.strength, 0.02, 1, 0.6).toFixed(3)),
    points,
  };
}

/** One manual mask-brush stroke: source-normalised points and a radius. */
function normaliseMaskStroke(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const points = Array.isArray(raw.points)
    ? raw.points
        .slice(0, 512)
        .map((p) => {
          if (!Array.isArray(p) || p.length < 2) return null;
          const x = clamp(p[0], -0.2, 1.2, NaN);
          const y = clamp(p[1], -0.2, 1.2, NaN);
          return Number.isFinite(x) && Number.isFinite(y) ? [Number(x.toFixed(5)), Number(y.toFixed(5))] : null;
        })
        .filter(Boolean)
    : [];
  if (points.length === 0) return null;
  return {
    radius: Number(clamp(raw.radius, 0.0008, 0.4, 0.03).toFixed(4)),
    points,
  };
}

/** Accept only the knobs the engine understands, already range-checked. */
function normaliseParams(raw) {
  const source = raw && typeof raw === 'object' ? raw : {};
  // `warmth` is the pre-2D-pad name for temperature. Accept it as an alias so
  // saved tool calls keep working; the engine still understands it too.
  const legacyWarmth = clamp(source.warmth, -50, 50, 0);
  return {
    smooth: clamp(source.smooth, 0, 100, 35),
    whiten: clamp(source.whiten, 0, 100, 15),
    person: clamp(source.person, 0, 100, 0),
    // 皱纹修复: 0-100 fade on the face skin; its brush strokes are render state.
    wrinkle: clamp(source.wrinkle, 0, 100, 0),
    slim: clamp(source.slim, 0, 100, 15),
    eyes: clamp(source.eyes, 0, 100, 10),
    detail: clamp(source.detail, 0, 1, 0.35),
    temperature: clamp(
      source.temperature === undefined ? legacyWarmth : source.temperature,
      -50,
      50,
      0,
    ),
    tint: clamp(source.tint, -50, 50, 0),
    shadow: clamp(source.shadow, 0, 100, 0),
    shadowColor: colourOr(source.shadowColor, '#a97b8c'),
    liner: clamp(source.liner, 0, 100, 0),
    linerColor: colourOr(source.linerColor, '#2a211f'),
    blush: clamp(source.blush, 0, 100, 0),
    blushColor: colourOr(source.blushColor, '#e2827e'),
    lip: clamp(source.lip, 0, 100, 0),
    lipColor: colourOr(source.lipColor, '#b4404f'),
    face: Math.trunc(clamp(source.face, 0, 9, 0)),
    compare: source.compare === true,
    max_size: Math.trunc(clamp(source.max_size, 0, 12000, 0)),
    // Liquify is replayable state, not a slider: the panel sends the strokes in
    // order, and the host forwards them to the engine as one JSON argument.
    liquify: Array.isArray(source.liquify)
      ? source.liquify.slice(0, 240).map(normaliseStroke).filter(Boolean)
      : [],
    // 手动补皮肤: painted strokes are part of the render state, like liquify.
    maskStrokes: Array.isArray(source.maskStrokes)
      ? source.maskStrokes.slice(0, 120).map(normaliseMaskStroke).filter(Boolean)
      : [],
    // 皱纹修复笔: same stroke shape as maskStrokes, painted areas are flattened.
    wrinkleStrokes: Array.isArray(source.wrinkleStrokes)
      ? source.wrinkleStrokes.slice(0, 120).map(normaliseMaskStroke).filter(Boolean)
      : [],
  };
}

// ---------------------------------------------------------------------------
// AI suggester: parameter contract, skill loading, prompt + output validation

const AI_PARAM_SPEC = [
  { key: 'smooth', kind: 'int', min: 0, max: 100, default: 35, label: '磨皮', desc: '强度越高皮肤越平滑，过高会失去纹理' },
  { key: 'whiten', kind: 'int', min: 0, max: 100, default: 15, label: '提亮', desc: '提亮肤色，过高会发灰、发假' },
  { key: 'person', kind: 'int', min: 0, max: 100, default: 0, label: '人物提亮', desc: '提亮全身皮肤，不只是脸' },
  { key: 'detail', kind: 'float', min: 0, max: 1, default: 0.35, label: '细节', desc: '磨皮后恢复纹理，建议 0.30–0.50' },
  { key: 'slim', kind: 'int', min: 0, max: 100, default: 15, label: '瘦脸', desc: '收窄下颌与脸宽' },
  { key: 'eyes', kind: 'int', min: 0, max: 100, default: 10, label: '大眼', desc: '放大眼裂' },
  { key: 'temperature', kind: 'int', min: -50, max: 50, default: 0, label: '色温', desc: '正=暖，负=冷' },
  { key: 'tint', kind: 'int', min: -50, max: 50, default: 0, label: '色调', desc: '正=洋红，负=绿' },
  { key: 'shadow', kind: 'int', min: 0, max: 100, default: 0, label: '眼影', desc: '眼影强度' },
  { key: 'shadowColor', kind: 'color', default: '#a97b8c', label: '眼影色', desc: '#rrggbb' },
  { key: 'liner', kind: 'int', min: 0, max: 100, default: 0, label: '眼线', desc: '眼线强度' },
  { key: 'linerColor', kind: 'color', default: '#2a211f', label: '眼线色', desc: '#rrggbb' },
  { key: 'blush', kind: 'int', min: 0, max: 100, default: 0, label: '腮红', desc: '腮红强度' },
  { key: 'blushColor', kind: 'color', default: '#e2827e', label: '腮红色', desc: '#rrggbb' },
  { key: 'lip', kind: 'int', min: 0, max: 100, default: 0, label: '口红', desc: '口红强度' },
  { key: 'lipColor', kind: 'color', default: '#b4404f', label: '口红色', desc: '#rrggbb' },
];

// Host-side safety rail. Deliberately wider than the skill's aesthetic range:
// it only exists so a hallucinating model cannot turn a face into plastic.
const AI_LIMITS = {
  smooth: [0, 60], whiten: [0, 70], person: [0, 80], detail: [0.2, 0.7],
  slim: [0, 30], eyes: [0, 50], temperature: [-10, 8], tint: [-12, 22],
  shadow: [0, 60], liner: [0, 50], blush: [0, 50], lip: [0, 60],
};

const AI_DEFAULTS = Object.fromEntries(AI_PARAM_SPEC.map((spec) => [spec.key, spec.default]));

/** Only the 16 visual knobs; never touches liquify/maskStrokes. */
function normaliseAiParams(raw, current) {
  const source = raw && typeof raw === 'object' ? raw : {};
  const base = current && typeof current === 'object' ? current : AI_DEFAULTS;
  const out = {};
  for (const spec of AI_PARAM_SPEC) {
    if (spec.kind === 'color') {
      out[spec.key] = colourOr(source[spec.key], colourOr(base[spec.key], spec.default));
      continue;
    }
    const [lo, hi] = AI_LIMITS[spec.key] ?? [spec.min, spec.max];
    const fallback = numberOr(base[spec.key], spec.default);
    const value = clamp(source[spec.key] === undefined ? fallback : source[spec.key], lo, hi, fallback);
    out[spec.key] = spec.kind === 'int' ? Math.round(value) : Number(value.toFixed(2));
  }
  return out;
}

function aiError(code, message, hint) {
  const error = new Error(message);
  error.code = code;
  if (hint) error.hint = hint;
  return error;
}

/** Parse the tiny YAML frontmatter the skill catalog uses (name, description, metadata.version). */
function splitFrontmatter(raw) {
  const match = String(raw).match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!match) return { meta: {}, body: String(raw) };
  const meta = {};
  let nested = null;
  for (const line of match[1].split(/\r?\n/)) {
    if (/^\s+/.test(line) && nested) {
      const child = line.trim().match(/^([\w-]+):\s*(.*)$/);
      if (child) meta[nested][child[1]] = child[2].trim();
      continue;
    }
    const entry = line.match(/^([\w-]+):\s*(.*)$/);
    if (!entry) continue;
    if (entry[2].length === 0) {
      nested = entry[1];
      meta[nested] = {};
    } else {
      meta[entry[1]] = entry[2].trim();
      nested = null;
    }
  }
  return { meta, body: match[2] };
}

async function loadSkill() {
  let raw;
  try {
    raw = await readFile(SKILL_FILE, 'utf8');
  } catch {
    throw aiError('E_SKILL_MISSING', '内置的美颜标准文件缺失，请重新安装插件。');
  }
  const { meta, body } = splitFrontmatter(raw);
  const metadata = meta.metadata && typeof meta.metadata === 'object' ? meta.metadata : {};
  const version = Number(metadata.version) || 1;
  return {
    name: typeof meta.name === 'string' && meta.name.length > 0 ? meta.name : 'portrait-beauty',
    version,
    hash: createHash('sha256').update(body).digest('hex').slice(0, 16),
    description: typeof meta.description === 'string' ? meta.description : '',
    whenToUse: typeof meta.whenToUse === 'string' ? meta.whenToUse : '',
    body: body.trim(),
  };
}

function changedList(current, next) {
  const changed = [];
  for (const spec of AI_PARAM_SPEC) {
    if (spec.kind === 'color') {
      const layer = { shadowColor: 'shadow', linerColor: 'liner', blushColor: 'blush', lipColor: 'lip' }[spec.key];
      // A colour under a zero-strength layer changes nothing visible; keep the
      // diff focused on knobs the user can actually see.
      if (layer && Number(next[layer]) <= 0) continue;
      if (String(current[spec.key]).toLowerCase() !== String(next[spec.key]).toLowerCase()) changed.push(spec.key);
    } else if (Math.abs(Number(current[spec.key]) - Number(next[spec.key])) > 0.01) {
      changed.push(spec.key);
    }
  }
  return changed;
}

function expectedTone(metrics) {
  const a = Number(metrics?.skin?.a_median);
  const b = Number(metrics?.skin?.b_median);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  const redness = Number(metrics?.skin?.redness_p95);
  return {
    temperature: clamp(Math.round(-4 - (b - 12) * 1.1), -10, 8, -4),
    tint: clamp(
      Math.round(18 - (a - 14) * 1.1 - Math.max(0, (Number.isFinite(redness) ? redness : 0) - 18) * 0.4),
      -12, 22, 20,
    ),
  };
}

function metricNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

const AI_PALETTES = {
  cool: { shadowColor: '#a87f92', linerColor: '#2a211f', blushColor: '#d98292', lipColor: '#a83e55' },
  neutral: { shadowColor: '#a97b8c', linerColor: '#2a211f', blushColor: '#e2827e', lipColor: '#b4404f' },
  warm: { shadowColor: '#b08a72', linerColor: '#3a2a24', blushColor: '#e08a72', lipColor: '#b4513f' },
};

function makeupPalette(metrics) {
  const a = metricNumber(metrics?.skin?.a_median, 14);
  const b = metricNumber(metrics?.skin?.b_median, 12);
  if (b > a + 1) return AI_PALETTES.warm;
  if (a > b + 1) return AI_PALETTES.cool;
  return AI_PALETTES.neutral;
}

/**
 * The deterministic replacement for the LLM call.
 *
 * Every value here is a pure function of the local metrics, so the same photo
 * produces the same plan and nothing leaves this Mac. The skill remains the
 * human-readable specification of these formulas.
 */
export function buildLocalSuggestion(metrics, current = AI_DEFAULTS, context = {}) {
  const skin = metrics?.skin ?? {};
  const geo = metrics?.geometry ?? {};
  const pose = metrics?.pose ?? {};
  const image = metrics?.image ?? {};
  const body = metrics?.body ?? {};
  const lips = metrics?.lips ?? {};
  const quality = metrics?.quality ?? {};

  const lMedian = metricNumber(skin.l_median, 60);
  const lStd = metricNumber(skin.l_std, 8);
  const lP10 = metricNumber(skin.l_p10, 45);
  const aMedian = metricNumber(skin.a_median, 14);
  const bMedian = metricNumber(skin.b_median, 12);
  const redness = metricNumber(skin.redness_p95, 18);
  const blemish = metricNumber(skin.blemish_density, 0.05);
  const clipping = metricNumber(skin.clipping, 0);
  const subject = metricNumber(image.subject_ratio, 0.2);
  const deltaL = metricNumber(body.delta_l, 0);
  const jaw = metricNumber(geo.jaw_ratio, 0.84);
  const faceAspect = metricNumber(geo.face_aspect, 0.86);
  const chin = metricNumber(geo.chin_ratio, 0.35);
  const eyeAspect = metricNumber(geo.eye_aspect, 0.28);
  const symmetry = metricNumber(geo.symmetry, 0);
  const faceWidth = metricNumber(geo.face_width, 200);
  const lipChroma = metricNumber(lips.chroma, 22);
  const light = quality.light ?? image.light ?? 'good';
  const confidence = geometryConfidence(metrics);

  const notes = [];
  let smooth = Math.round(clamp(24 + lStd * 0.25 + blemish * 50, 0, 40, 24));
  let whiten = Math.round(clamp(8 + Math.max(0, 62 - lMedian) * 0.5, 0, 70, 8));
  let person = 0;

  const strongBacklit = (light === 'dim' || light === 'backlit') && lP10 < 45 && subject < 0.10 && clipping <= 0.01;
  if (strongBacklit) {
    whiten = Math.max(whiten, 40);
    person = 55;
    notes.push('逆光小主体，提亮按可见亮度基线');
  } else if (light === 'dim' || light === 'backlit' || lP10 < 45) {
    whiten = Math.max(whiten, 20);
    person = 30;
    notes.push('欠曝场景，脸和身体一起补光');
  } else if (subject < 0.20 || deltaL > 8) {
    person = 15;
  }
  if (clipping > 0.02) {
    whiten = Math.min(whiten, 6);
    person = Math.min(person, 15);
    notes.push('皮肤高光接近溢出，提亮已收回');
  }

  // A small subject needs a little more of the same slider to read at all on
  // screen. The compensation is continuous, capped, and zero for subjects
  // that already fill a reasonable part of the frame.
  const framingBoost = Math.min(15, Math.max(0, 0.060 - subject) * 480);
  let slim = Math.round(
    clamp(
      (jaw - 0.780) / 0.0032
        + Math.max(0, faceAspect - 0.880) * 30
        + Math.max(0, chin - 0.380) * 20
        + framingBoost,
      0, 30, 0,
    ),
  );
  if (jaw < 0.76 || faceAspect < 0.80) slim = Math.min(18, Math.max(0, slim - 4));
  let eyes = Math.round(clamp((0.360 - eyeAspect) / 0.00145, 0, 50, 0));

  if (confidence < 0.08 || faceWidth < 100) {
    if (slim > 0 || eyes > 0) notes.push('脸太小或几何置信度太低，瘦脸和大眼已归零');
    slim = 0;
    eyes = 0;
  } else if (confidence < 0.12) {
    slim = Math.min(slim, 12);
    eyes = Math.min(eyes, 24);
    notes.push('几何置信度很低，瘦脸和大眼已减半');
  }
  if (pose.symmetry_reliable === true && symmetry > 0.10 && (slim > 0 || eyes > 0)) {
    slim = Math.round(slim * 0.9);
    eyes = Math.round(eyes * 0.9);
    notes.push('左右对称度偏低，瘦脸和大眼已轻微收敛');
  }

  const detail = Math.round(clamp(0.40 + (smooth - 20) * 0.005, 0.30, 0.55, 0.40) * 100) / 100;
  const shadow = Math.round(clamp(Math.max(0, eyeAspect - 0.34) * 25, 0, 4, 0));
  const liner = Math.round(clamp(20.4 + (0.34 - Math.min(eyeAspect, 0.34)) * 25, 0, 30, 20));
  const blush = Math.round(clamp(20 - redness * 0.4, 0, 25, 15));
  const lip = Math.round(clamp(30 - lipChroma * 0.35, 0, 45, 22));
  const tone = expectedTone(metrics) ?? { temperature: 0, tint: 0 };
  const palette = makeupPalette(metrics);

  const raw = {
    smooth, whiten, person, detail, slim, eyes,
    temperature: tone.temperature, tint: tone.tint,
    shadow, liner, blush, lip,
    ...palette,
  };
  const next = normaliseAiParams(raw, current);

  const summaryBits = [];
  if (strongBacklit) summaryBits.push('逆光小主体，先补足主体亮度');
  else if (light === 'dim' || light === 'backlit') summaryBits.push('欠曝场景，脸和身体一起补光');
  else summaryBits.push('按肤色与比例做轻度修正');
  if (next.slim > 0 || next.eyes > 0) summaryBits.push('脸型按下颌与眼裂比例收放');
  summaryBits.push(`肤色按 a*${Math.round(aMedian)}/b*${Math.round(bMedian)} 校正`);
  const summary = summaryBits.join('；').slice(0, 120);

  const changed = changedList(current, next);
  const why = {
    smooth: `皮肤均匀度 σL=${lStd.toFixed(1)}，瑕疵占比 ${(blemish * 100).toFixed(1)}%`,
    whiten: `肤色 L*=${lMedian.toFixed(1)}，暗部 p10=${lP10.toFixed(1)}`,
    person: `主体占画面 ${(subject * 100).toFixed(1)}%，身体与脸亮度差 ${deltaL.toFixed(1)}`,
    detail: `磨皮强度 ${smooth}，需要保留毛孔纹理`,
    slim: `下颌/脸宽 ${jaw.toFixed(3)}，脸宽/脸高 ${faceAspect.toFixed(3)}`
      + (framingBoost > 0.5 ? `，小主体补偿 +${Math.round(framingBoost)}` : ''),
    eyes: `眼裂高/宽 ${eyeAspect.toFixed(3)}`,
    temperature: `b*=${bMedian.toFixed(1)}，相对中性 12 的偏移`,
    tint: `a*=${aMedian.toFixed(1)}，泛红 p95=${redness.toFixed(1)}`,
    shadow: `眼裂比 ${eyeAspect.toFixed(3)}`,
    liner: `眼裂比 ${eyeAspect.toFixed(3)}，贴睫毛根部`,
    blush: `泛红 p95=${redness.toFixed(1)}`,
    lip: `唇色饱和度 ${lipChroma.toFixed(1)}`,
  };
  const reasons = changed
    .filter((key) => why[key])
    .slice(0, 6)
    .map((key) => ({ param: key, why: why[key] }));

  const warnings = [...new Set([...notes, ...(Array.isArray(quality.notes) ? quality.notes : [])])].slice(0, 5);
  const ruleConfidence = clamp(0.55 + 0.45 * confidence - (warnings.length > 1 ? 0.08 : 0), 0.3, 1, 0.6);

  return {
    params: next,
    summary,
    reasons,
    changed,
    confidence: Number(ruleConfidence.toFixed(2)),
    warnings,
  };
}

function geometryConfidence(metrics) {
  const value = Number(metrics?.pose?.geometry_confidence);
  if (Number.isFinite(value)) return clamp(value, 0, 1, 0);
  // Older cached analyses only have the strict reliable flag.
  return metrics?.pose?.reliable === false ? 0 : 1;
}

/** Clamp + consistency checks + presentation fields. */
export function validateSuggestion(parsed, current, metrics) {
  const warnings = [];
  const rawParams = parsed && parsed.params && typeof parsed.params === 'object' ? parsed.params : {};
  const next = normaliseAiParams(rawParams, current);
  let clamped = 0;
  for (const [key, [lo, hi]] of Object.entries(AI_LIMITS)) {
    const value = Number(rawParams[key]);
    if (Number.isFinite(value) && (value < lo || value > hi)) clamped += 1;
  }
  if (clamped > 0) warnings.push(`已把 ${clamped} 个超出安全范围的值收回到上限`);

  const geoConfidence = geometryConfidence(metrics);
  const faceWidth = Number(metrics?.geometry?.face_width);
  if (geoConfidence < 0.08 || (Number.isFinite(faceWidth) && faceWidth < 100)) {
    if (next.slim > 0 || next.eyes > 0) warnings.push('脸太小或姿态置信度过低，瘦脸和大眼已归零');
    next.slim = 0;
    next.eyes = 0;
  } else if (geoConfidence < 0.12) {
    // Below 0.12 there is still enough geometry for a gentle change, but not
    // for the full signature style.
    next.slim = Math.min(next.slim, 12);
    next.eyes = Math.min(next.eyes, 24);
    warnings.push('姿态置信度很低，瘦脸和大眼已减半');
  }
  // Symmetry is only meaningful on a near-frontal face; on a 3/4 view it is
  // mostly pose error.  Never zero or halve geometry for it.
  const symmetry = Number(metrics?.geometry?.symmetry);
  if (metrics?.pose?.symmetry_reliable === true && Number.isFinite(symmetry) && symmetry > 0.10 && (next.slim > 0 || next.eyes > 0)) {
    next.slim = Math.round(next.slim * 0.9);
    next.eyes = Math.round(next.eyes * 0.9);
    warnings.push('左右对称度偏低，瘦脸和大眼已轻微收敛');
  }
  // A strongly backlit, small-subject portrait is the one case where the
  // median face brightness lies: face and body are equally underexposed. The
  // model tends to stay at "natural" levels, so guarantee a visible baseline.
  const light = metrics?.quality?.light ?? metrics?.image?.light;
  const p10 = Number(metrics?.skin?.l_p10);
  const subject = Number(metrics?.image?.subject_ratio);
  const clipping = Number(metrics?.skin?.clipping);
  if (
    (light === 'dim' || light === 'backlit') &&
    Number.isFinite(p10) && p10 < 45 &&
    Number.isFinite(subject) && subject < 0.10 &&
    (!Number.isFinite(clipping) || clipping <= 0.01)
  ) {
    if (next.whiten < 40) next.whiten = 40;
    if (next.person < 55) next.person = 55;
    warnings.push('强逆光小主体，提亮已按可见亮度基线兜底');
  }
  if (next.smooth > 25 && next.detail < 0.30) {
    next.detail = 0.30;
    warnings.push('磨皮较高，细节已提到 0.30');
  }
  // White balance is a measurable correction, not a taste knob. Reconcile the
  // model's answer with the a*/b* formula so two photos with different casts
  // never collapse to the same tone just because the model defaults to the
  // signature value.
  const toneExpect = expectedTone(metrics);
  if (toneExpect) {
    const drift = Math.abs(next.temperature - toneExpect.temperature) + Math.abs(next.tint - toneExpect.tint);
    if (drift >= 1) {
      next.temperature = toneExpect.temperature;
      next.tint = toneExpect.tint;
      warnings.push('肤色已按 a*/b* 白平衡公式校正');
    }
  }
  const toneSum = Math.abs(next.temperature) + Math.abs(next.tint);
  if (toneSum > 30) {
    const scale = 30 / toneSum;
    next.temperature = Math.round(next.temperature * scale);
    next.tint = Math.round(next.tint * scale);
    warnings.push('色温+色调超出自然范围，已按比例收回');
  }

  const summary = typeof parsed?.summary === 'string' ? parsed.summary.trim().slice(0, 120) : '';
  const reasons = Array.isArray(parsed?.reasons)
    ? parsed.reasons
        .filter((item) => item && typeof item === 'object' && typeof item.param === 'string')
        .slice(0, 6)
        .map((item) => ({ param: item.param, why: String(item.why ?? '').slice(0, 80) }))
    : [];
  const modelWarnings = Array.isArray(parsed?.warnings)
    ? parsed.warnings.filter((item) => typeof item === 'string' && item.trim().length > 0).map((item) => item.slice(0, 80))
    : [];
  const allWarnings = [...modelWarnings, ...warnings].slice(0, 5);
  const confidence = clamp(parsed?.confidence, 0, 1, 0.5);

  return {
    params: next,
    summary,
    reasons,
    changed: changedList(current, next),
    confidence: Number(confidence.toFixed(2)),
    warnings: allWarnings,
  };
}

/**
 * Bumped whenever a change to the engine changes the pixels for the same
 * parameters (a retuned look), so cached 1:1 patches are not served stale.
 */
const RENDER_VERSION = 9;

function paramsKey(params) {
  return createHash('sha1').update(JSON.stringify([RENDER_VERSION, params])).digest('hex').slice(0, 12);
}

function mediaTypeFor(file) {
  return IMAGE_TYPES.get(path.extname(file).toLowerCase()) ?? 'image/jpeg';
}

function sniffImageExt(buffer) {
  if (buffer.length > 3 && buffer[0] === 0xff && buffer[1] === 0xd8) return '.jpg';
  if (buffer.length > 8 && buffer.slice(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return '.png';
  if (buffer.length > 12 && buffer.slice(8, 12).toString('latin1') === 'WEBP') return '.webp';
  if (buffer.length > 12 && buffer.slice(4, 8).toString('latin1') === 'ftyp') return '.heic';
  return null;
}

/** Uniform result shape for both the Tool and the workbench routes. */
function summarise(result, extra) {
  return {
    ok: true,
    faces: result.faces,
    face: result.face_used,
    width: result.width,
    height: result.height,
    original_width: result.original_width,
    original_height: result.original_height,
    elapsed_ms: result.elapsed_ms,
    params: result.params,
    skin: result.skin ?? null,
    notes: Array.isArray(result.notes) ? result.notes : [],
    ...extra,
  };
}

// ---------------------------------------------------------------------------
// engine process

class Engine {
  constructor(ctx, config) {
    this.ctx = ctx;
    this.config = config && typeof config === 'object' ? config : {};
    this.python = null;
    this.probe = null;
    this.cacheDir =
      typeof this.config.cacheDir === 'string' && this.config.cacheDir.length > 0
        ? this.config.cacheDir
        : path.join(os.homedir(), '.dsh', 'beauty-cache');
    this.outputDir =
      typeof this.config.outputDir === 'string' && this.config.outputDir.length > 0
        ? this.config.outputDir
        : path.join(os.homedir(), 'Pictures', 'DSH Beauty');
    this.ready = null;
    /** landmarks per source photo, so slider previews never reload the model */
    this.landmarks = new Map();
    this.detecting = new Map();
    /** AI analysis reports, keyed by the same inputs as the vision files. */
    this.metrics = new Map();
  }

  get landmarksDir() {
    return path.join(this.cacheDir, 'landmarks');
  }

  get sourceDir() {
    return path.join(this.cacheDir, 'source');
  }

  get renderDir() {
    return path.join(this.cacheDir, 'render');
  }

  get thumbDir() {
    return path.join(this.cacheDir, 'thumbs');
  }

  candidates() {
    return [
      typeof this.config.python === 'string' ? this.config.python : null,
      typeof process.env.DSH_BEAUTY_PYTHON === 'string' && process.env.DSH_BEAUTY_PYTHON.length > 0
        ? process.env.DSH_BEAUTY_PYTHON
        : null,
      path.join(PLUGIN_ROOT, '.venv', 'bin', 'python'),
      'python3',
    ].filter((value) => typeof value === 'string' && value.length > 0);
  }

  async resolve() {
    if (this.ready) return this.ready;
    this.ready = (async () => {
      for (const candidate of this.candidates()) {
        const ok = await this.canImport(candidate);
        if (ok) {
          this.python = candidate;
          return candidate;
        }
      }
      throw Object.assign(
        new Error(
          'Beauty engine is not ready. Run `scripts/setup-engine.sh` (or install numpy, opencv-python-headless, mediapipe and pillow into the configured Python).',
        ),
        { code: 'E_ENGINE_MISSING' },
      );
    })();
    try {
      return await this.ready;
    } catch (error) {
      this.ready = null;
      throw error;
    }
  }

  canImport(candidate) {
    return new Promise((resolve) => {
      let settled = false;
      let child;
      const done = (value) => {
        if (settled) return;
        settled = true;
        resolve(value);
      };
      try {
        // `import mediapipe` costs ~8 s when matplotlib's font cache is cold, so
        // only import the cheap modules and look mediapipe up on sys.path.
        child = spawn(
          candidate,
          [
            '-c',
            "import cv2, numpy, PIL, importlib.util as u, sys; sys.exit(0 if u.find_spec('mediapipe') else 1)",
          ],
          {
            stdio: 'ignore',
            timeout: 25000,
          },
        );
      } catch {
        done(false);
        return;
      }
      child.on('error', () => done(false));
      child.on('exit', (code) => done(code === 0));
    });
  }

  async ensureDirs() {
    await mkdir(this.sourceDir, { recursive: true });
    await mkdir(this.renderDir, { recursive: true });
  }

  /** Run the face model once for a photo and print its landmarks. */
  async detect({ input, signal }) {
    const python = await this.resolve();
    const args = [ENGINE, '--input', input, '--detect', '--model', MODEL, '--json'];
    return this.run(python, args, { signal, timeout: ENGINE_TIMEOUT_MS });
  }

  /**
   * Landmarks for a photo, cached by path + mtime + size.
   *
   * The model costs seconds to load (MediaPipe imports matplotlib), so pay it
   * once per source image instead of on every slider tick.
   */
  async landmarksFor(input, signal) {
    const info = await stat(input);
    // v2 carries the skin-segmentation masks; older entries only have points.
    // v3 also carries the full-frame person mask (人物提亮 / body skin).
    const key = `${input}:${info.mtimeMs}:${info.size}:v3`;
    const memo = this.landmarks.get(key);
    if (memo) return memo;
    // Upload warms the cache in the background; a slider tick may arrive first,
    // so share one in-flight detection instead of starting a second model load.
    const running = this.detecting.get(key);
    if (running) return running;

    const job = (async () => {
      const report = await this.detect({ input, signal });
      if (!report.ok) throw new Error(report.error || 'face detection failed');
      const file = await this.storeLandmarks(key, report);
      this.landmarks.set(key, file);
      if (this.landmarks.size > 64) {
        const oldest = this.landmarks.keys().next().value;
        this.landmarks.delete(oldest);
      }
      return file;
    })();
    this.detecting.set(key, job);
    try {
      return await job;
    } finally {
      this.detecting.delete(key);
    }
  }

  /**
   * Metrics + two compact JPEGs for the AI suggester.
   *
   * Cached by source fingerprint + face + output sizes, so pressing the button
   * twice reuses the analysis instead of spawning a second model load.
   */
  async analyze({ input, face = 0, signal, vision = true, visionFull = AI_DEFAULT_LONG_SIDE.full, visionPerson = AI_DEFAULT_LONG_SIDE.person, visionFace = AI_DEFAULT_LONG_SIDE.face }) {
    const python = await this.resolve();
    const landmarks = await this.landmarksFor(input, signal);
    const info = await stat(input);
    const key = `${ANALYSIS_VERSION}:${input}:${info.mtimeMs}:${info.size}:${Number(face) || 0}:${vision ? `${visionFull}:${visionPerson}:${visionFace}` : 'metrics'}`;
    const memo = this.metrics.get(key);
    if (memo && (!vision || ['full', 'person', 'face'].every((kind) => !memo.vision?.[kind] || existsSync(memo.vision[kind])))) {
      return memo;
    }
    let dir = null;
    if (vision) {
      dir = path.join(this.cacheDir, 'vision', createHash('sha1').update(key).digest('hex').slice(0, 16));
      await mkdir(dir, { recursive: true });
    }
    const args = [
      ENGINE,
      '--input', input,
      '--analyze',
      '--landmarks', landmarks,
      '--face', String(Math.max(0, Math.trunc(Number(face) || 0))),
    ];
    if (dir) {
      args.push(
        '--vision-out', dir,
        '--vision-full', String(visionFull),
        '--vision-person', String(visionPerson),
        '--vision-face', String(visionFace),
      );
    }
    args.push('--model', MODEL, '--json');
    const report = await this.run(python, args, { signal });
    this.metrics.set(key, report);
    if (this.metrics.size > 64) this.metrics.delete(this.metrics.keys().next().value);
    return report;
  }

  /** Small JSON file next to the caches: remembers the last save folder. */
  get settingsFile() {
    return path.join(this.cacheDir, 'settings.json');
  }

  async readSettings() {
    try {
      const parsed = JSON.parse(await readFile(this.settingsFile, 'utf8'));
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
      return {};
    }
  }

  async rememberSaveDir(dir) {
    try {
      const settings = await this.readSettings();
      settings.saveDir = dir;
      await mkdir(this.cacheDir, { recursive: true });
      await writeFile(this.settingsFile, JSON.stringify(settings, null, 2), 'utf8');
    } catch {
      /* remembering the folder is a nicety, never a failure */
    }
  }

  /**
   * Ask the user where to save, with a native macOS panel.
   *
   * Resolves with a path, `null` when the platform has no such dialog, or the
   * string 'cancelled' when the user dismissed it. The panel opens in the
   * folder used last time, so repeated saves stay in one place.
   */
  async pickSavePath(defaultName) {
    if (process.platform !== 'darwin') return null;
    const settings = await this.readSettings();
    const startDir =
      typeof settings.saveDir === 'string' && settings.saveDir.length > 0
        ? settings.saveDir
        : this.outputDir;
    const script = [
      'on run argv',
      '  set nm to item 1 of argv',
      '  set dr to item 2 of argv',
      '  try',
      '    set f to choose file name with prompt "保存美颜照片" default name nm default location (POSIX file dr)',
      '    return POSIX path of f',
      '  on error errMsg number errNum',
      '    if errNum is -128 then return "__CANCELLED__"',
      '    error errMsg number errNum',
      '  end try',
      'end run',
    ].join('\n');
    const output = await new Promise((resolve, reject) => {
      let stdout = '';
      let stderr = '';
      const child = spawn('osascript', ['-e', script, defaultName, startDir], { stdio: ['ignore', 'pipe', 'pipe'] });
      child.stdout.setEncoding('utf8');
      child.stderr.setEncoding('utf8');
      child.stdout.on('data', (chunk) => {
        stdout += chunk;
      });
      child.stderr.on('data', (chunk) => {
        stderr += chunk;
      });
      child.on('error', reject);
      child.on('close', (code) => {
        if (code === 0) resolve(stdout.trim());
        else reject(new Error(stderr.trim() || `save dialog exited with code ${code}`));
      });
    });
    if (output === '__CANCELLED__') return 'cancelled';
    return output.length > 0 ? output : null;
  }

  /**
   * Ask the user for an output folder, with a native macOS panel.
   *
   * Same contract as pickSavePath: a path, `null` when the platform has no such
   * dialog, or 'cancelled'. The trailing slash `POSIX path of` adds is removed.
   */
  async pickFolder() {
    if (process.platform !== 'darwin') return null;
    const settings = await this.readSettings();
    const startDir =
      typeof settings.saveDir === 'string' && settings.saveDir.length > 0
        ? settings.saveDir
        : this.outputDir;
    // `default location` only accepts an existing folder.
    await mkdir(startDir, { recursive: true }).catch(() => {});
    const script = [
      'on run argv',
      '  set dr to item 1 of argv',
      '  try',
      '    set f to choose folder with prompt "选择导出文件夹" default location (POSIX file dr)',
      '    return POSIX path of f',
      '  on error errMsg number errNum',
      '    if errNum is -128 then return "__CANCELLED__"',
      '    error errMsg number errNum',
      '  end try',
      'end run',
    ].join('\n');
    const output = await new Promise((resolve, reject) => {
      let stdout = '';
      let stderr = '';
      const child = spawn('osascript', ['-e', script, startDir], { stdio: ['ignore', 'pipe', 'pipe'] });
      child.stdout.setEncoding('utf8');
      child.stderr.setEncoding('utf8');
      child.stdout.on('data', (chunk) => {
        stdout += chunk;
      });
      child.stderr.on('data', (chunk) => {
        stderr += chunk;
      });
      child.on('error', reject);
      child.on('close', (code) => {
        if (code === 0) resolve(stdout.trim());
        else reject(new Error(stderr.trim() || `folder dialog exited with code ${code}`));
      });
    });
    if (output === '__CANCELLED__') return 'cancelled';
    return output.length > 0 ? path.resolve(output) : null;
  }

  /**
   * Cached JPEG thumbnail for the filmstrip.
   *
   * Chrome cannot decode HEIC, and sending 40 MB originals to an `<img>` for a
   * 96 px tile is wasteful, so `sips` (macOS, already present) makes a small
   * JPEG once per source. Throws when sips is unavailable or fails; the route
   * then falls back to the original bytes.
   */
  async thumbnail(input, width) {
    const key = createHash('sha1').update(`${input}:${width}:v1`).digest('hex').slice(0, 16);
    const file = path.join(this.thumbDir, `thm_${key}.jpg`);
    try {
      const info = await stat(file);
      if (info.size > 0) return file;
    } catch {
      /* render below */
    }
    await mkdir(this.thumbDir, { recursive: true });
    const tmp = `${file}.${process.pid}.${Date.now().toString(36)}.tmp.jpg`;
    try {
      await new Promise((resolve, reject) => {
        let stderr = '';
        const child = spawn('sips', ['-s', 'format', 'jpeg', '-Z', String(width), input, '--out', tmp], {
          stdio: ['ignore', 'ignore', 'pipe'],
        });
        child.stderr.setEncoding('utf8');
        child.stderr.on('data', (chunk) => {
          stderr += chunk;
        });
        child.on('error', reject);
        child.on('close', (code) => {
          if (code === 0) resolve();
          else reject(new Error(stderr.trim() || `sips exited with code ${code}`));
        });
      });
      await rename(tmp, file);
      return file;
    } catch (error) {
      await rm(tmp, { force: true }).catch(() => {});
      throw error;
    }
  }

  /** Persist a landmark report, falling back to the OS temp dir when the cache is read-only. */
  async storeLandmarks(key, report) {
    const name = `${createHash('sha1').update(key).digest('hex').slice(0, 16)}.json`;
    for (const dir of [this.landmarksDir, path.join(os.tmpdir(), 'dsh-beauty-landmarks')]) {
      try {
        await mkdir(dir, { recursive: true });
        const file = path.join(dir, name);
        await writeFile(file, JSON.stringify(report), 'utf8');
        return file;
      } catch {
        /* try the next location */
      }
    }
    throw Object.assign(new Error('Cannot cache face landmarks'), { code: 'E_ENGINE_FAILED' });
  }

  /** Run one retouch. Resolves with the parsed engine report. */
  async retouch({ input, output, params, signal, crop = null }) {
    const python = await this.resolve();
    await this.ensureDirs();
    const landmarks = await this.landmarksFor(input, signal);
    const args = [
      ENGINE,
      '--input',
      input,
      '--landmarks',
      landmarks,
      '--output',
      output,
      '--smooth',
      String(params.smooth),
      '--whiten',
      String(params.whiten),
      '--person',
      String(params.person),
      '--wrinkle',
      String(params.wrinkle),
      '--slim',
      String(params.slim),
      '--eyes',
      String(params.eyes),
      '--detail',
      String(params.detail),
      '--temperature',
      String(params.temperature),
      '--tint',
      String(params.tint),
      '--shadow',
      String(params.shadow),
      '--shadow-color',
      params.shadowColor,
      '--liner',
      String(params.liner),
      '--liner-color',
      params.linerColor,
      '--blush',
      String(params.blush),
      '--blush-color',
      params.blushColor,
      '--lip',
      String(params.lip),
      '--lip-color',
      params.lipColor,
      '--face',
      String(params.face),
      '--max-size',
      String(params.max_size),
      '--model',
      MODEL,
      '--json',
    ];
    if (params.liquify && params.liquify.length > 0) {
      args.push('--liquify', JSON.stringify(params.liquify));
    }
    if (params.wrinkleStrokes && params.wrinkleStrokes.length > 0) {
      args.push('--wrinkle-strokes', JSON.stringify(params.wrinkleStrokes));
    }
    if (params.maskStrokes && params.maskStrokes.length > 0) {
      args.push('--mask-strokes', JSON.stringify(params.maskStrokes));
    }
    if (crop) args.push('--crop', crop.map((v) => String(Math.round(v))).join(','));
    if (params.compare) args.push('--compare');

    return this.run(python, args, { signal });
  }

  /**
   * Spawn the engine and parse its JSON report.
   *
   * Every call is a fresh process: the engine is stateless by design, and the
   * expensive model load is amortised through the landmark cache instead.
   */
  run(python, args, { signal, timeout = ENGINE_TIMEOUT_MS } = {}) {
    return new Promise((resolve, reject) => {
      let stdout = '';
      let stderr = '';
      let settled = false;
      const child = spawn(python, args, {
        cwd: path.dirname(ENGINE),
        env: { ...process.env, PYTHONUNBUFFERED: '1', DSH_BEAUTY_CACHE: this.cacheDir },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      const timer = setTimeout(() => {
        try {
          child.kill('SIGKILL');
        } catch {
          /* already gone */
        }
        finish(new Error('Beauty engine timed out'));
      }, timeout);
      timer.unref?.();

      function finish(error, value) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (error) reject(error);
        else resolve(value);
      }

      child.stdout.setEncoding('utf8');
      child.stdout.on('data', (chunk) => {
        stdout += chunk;
        if (stdout.length > 4 * 1024 * 1024) stdout = stdout.slice(-1024 * 1024);
      });
      child.stderr.setEncoding('utf8');
      child.stderr.on('data', (chunk) => {
        stderr += chunk;
        if (stderr.length > 256 * 1024) stderr = stderr.slice(-64 * 1024);
      });
      child.on('error', (error) => finish(error));
      child.on('close', (code) => {
        if (code !== 0) {
          finish(
            Object.assign(new Error(engineFailure(stdout, stderr, code)), {
              code: 'E_ENGINE_FAILED',
            }),
          );
          return;
        }
        const line = stdout
          .split('\n')
          .reverse()
          .find((entry) => entry.trim().startsWith('{'));
        if (!line) {
          finish(Object.assign(new Error('Beauty engine produced no report'), { code: 'E_ENGINE_FAILED' }));
          return;
        }
        try {
          const value = JSON.parse(line);
          if (value && value.ok === false) {
            finish(Object.assign(new Error(String(value.error ?? 'Beauty engine failed')), { code: 'E_ENGINE_FAILED' }));
            return;
          }
          finish(undefined, value);
        } catch (error) {
          finish(Object.assign(error, { code: 'E_ENGINE_FAILED' }));
        }
      });
      if (signal) {
        const abort = () => {
          try {
            child.kill('SIGKILL');
          } catch {
            /* already gone */
          }
          finish(Object.assign(new Error('Cancelled'), { code: 'E_CANCELLED' }));
        };
        if (signal.aborted) abort();
        else signal.addEventListener('abort', abort, { once: true });
      }
    });
  }

  async saveSource(buffer, name) {
    await this.ensureDirs();
    const ext = sniffImageExt(buffer) ?? path.extname(typeof name === 'string' ? name : '').toLowerCase() ?? '.jpg';
    const safeExt = IMAGE_TYPES.has(ext) ? ext : '.jpg';
    const id = `src_${randomUUID().replace(/-/g, '').slice(0, 20)}`;
    const file = path.join(this.sourceDir, `${id}${safeExt}`);
    await writeFile(file, buffer);
    const displayName =
      typeof name === 'string' && name.length > 0 ? path.basename(name) : `${id}${safeExt}`;
    // Remember the original file name so a saved result can reuse it.
    await writeFile(path.join(this.sourceDir, `${id}.json`), JSON.stringify({ name: displayName })).catch(
      () => {},
    );
    return { id, file, name: displayName };
  }

  async sourceName(id) {
    try {
      const raw = await readFile(path.join(this.sourceDir, `${id}.json`), 'utf8');
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.name === 'string' && parsed.name.length > 0) return parsed.name;
    } catch {
      /* no sidecar: fall back to the generated id */
    }
    return `${id}.jpg`;
  }

  sourcePath(id) {
    if (!ID_RE.test(String(id))) return null;
    for (const ext of IMAGE_TYPES.keys()) {
      const candidate = path.join(this.sourceDir, `${id}${ext}`);
      if (candidate.startsWith(this.sourceDir + path.sep)) return candidate;
    }
    return null;
  }

  async findSource(id) {
    if (!ID_RE.test(String(id))) return null;
    for (const ext of IMAGE_TYPES.keys()) {
      const candidate = path.join(this.sourceDir, `${id}${ext}`);
      try {
        const info = await stat(candidate);
        if (info.isFile()) return candidate;
      } catch {
        /* next extension */
      }
    }
    return null;
  }

  async prune() {
    try {
      const entries = await readdir(this.renderDir);
      if (entries.length <= CACHE_KEEP) return;
      const stamps = await Promise.all(
        entries.map(async (entry) => {
          const file = path.join(this.renderDir, entry);
          try {
            return { file, time: (await stat(file)).mtimeMs };
          } catch {
            return null;
          }
        }),
      );
      const ordered = stamps.filter(Boolean).sort((a, b) => a.time - b.time);
      for (const entry of ordered.slice(0, Math.max(0, ordered.length - CACHE_KEEP))) {
        await rm(entry.file, { force: true });
      }
    } catch {
      /* pruning is best-effort */
    }
  }
}

// ---------------------------------------------------------------------------
// AI suggester route

const AI_STATUS = {
  E_BODY: 400,
  E_NOT_FOUND: 404,
  E_ENGINE_MISSING: 503,
  E_NO_FACE: 422,
  E_ANALYZE_FAILED: 500,
  E_LLM_UNAVAILABLE: 503,
  E_NO_MODEL: 503,
  E_MODEL_NO_VISION: 422,
  E_AI_TIMEOUT: 504,
  E_AI_FORMAT: 502,
  E_AI_EMPTY: 502,
  E_PROVIDER: 502,
  E_CANCELLED: 499,
  E_ATTACHMENT: 413,
  E_SKILL_MISSING: 500,
};

function aiContext(raw) {
  const source = raw && typeof raw === 'object' ? raw : {};
  const strokes = Array.isArray(source.liquify) ? source.liquify : [];
  const masks = Array.isArray(source.maskStrokes) ? source.maskStrokes : [];
  const wrinkles = Array.isArray(source.wrinkleStrokes) ? source.wrinkleStrokes : [];
  const modes = [...new Set(strokes.map((stroke) => (stroke && stroke.mode) || 'push'))];
  return {
    face: Math.trunc(clamp(source.face, 0, 9, 0)),
    compare: source.compare === true,
    max_size: Math.trunc(clamp(source.max_size, 0, 12000, 0)),
    liquify: { count: strokes.length, modes },
    mask_strokes: { count: masks.length },
    wrinkle_strokes: { count: wrinkles.length },
  };
}

async function aiSuggest(ctx, engine, config, request) {
  const startedAt = Date.now();
  let body;
  try {
    body = await request.json();
  } catch {
    return fail('E_BODY', 'Expected a JSON body', 400);
  }
  if (!body || typeof body !== 'object') return fail('E_BODY', 'Expected a JSON object', 400);
  if (request.signal?.aborted) return fail('E_CANCELLED', '已取消计算', 499);
  const input = await engine.findSource(body.id);
  if (input === null) return fail('E_NOT_FOUND', '图片不存在，请重新打开图片', 404);

  const raw = body.params && typeof body.params === 'object' ? body.params : {};
  const current = normaliseAiParams(raw, AI_DEFAULTS);
  const context = aiContext(raw);

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, AI_TIMEOUT_MS);
  timer.unref?.();
  const external = request.signal;
  const onAbort = () => controller.abort();
  if (external) {
    if (external.aborted) controller.abort();
    else external.addEventListener('abort', onAbort, { once: true });
  }

  try {
    let analysis;
    try {
      // Pure local: metrics only, no vision JPEGs, no attachments, no LLM.
      analysis = await engine.analyze({ input, face: context.face, signal: controller.signal, vision: false });
    } catch (error) {
      if (external?.aborted) return fail('E_CANCELLED', '已取消计算', 499);
      if (timedOut || controller.signal.aborted) return fail('E_AI_TIMEOUT', '本机分析超时，请重试。', 504);
      log(ctx, 'local-analyze-failed', error);
      return fail('E_ANALYZE_FAILED', `本机指标分析失败：${String(error?.message ?? error).slice(0, 500)}`, 500);
    }
    if (!analysis || Number(analysis.faces) === 0) {
      return fail('E_NO_FACE', '没有检测到人脸，换一张正面人像试试', 422);
    }

    let skill = { name: 'portrait-beauty', version: 4, hash: 'local' };
    try {
      const loaded = await loadSkill();
      skill = { name: loaded.name, version: loaded.version, hash: loaded.hash };
    } catch {
      /* The formulas ship with the plugin; a missing spec must not block them. */
    }

    const suggestion = buildLocalSuggestion(analysis.metrics, current, context);
    return json({
      ok: true,
      mode: 'local',
      suggestion,
      metrics: analysis.metrics,
      model: { provider: 'local', model: 'portrait-beauty-formula', name: '本地量化引擎 v4' },
      skill,
      sentImages: [],
      elapsedMs: Date.now() - startedAt,
      cached: false,
    });
  } catch (error) {
    if (timedOut) return fail('E_AI_TIMEOUT', '本机分析超时，请重试。', 504);
    if (external?.aborted) return fail('E_CANCELLED', '已取消计算', 499);
    const code = typeof error?.code === 'string' ? error.code : 'E_PROVIDER';
    const status = AI_STATUS[code] ?? 500;
    log(ctx, 'local-suggest-failed', error);
    return fail(code, String(error?.message ?? error).slice(0, 600), status, error?.hint);
  } finally {
    clearTimeout(timer);
    if (external) external.removeEventListener('abort', onAbort);
  }
}

// ---------------------------------------------------------------------------
// plugin body

export function apply(ctx, config) {
  const engine = new Engine(ctx, config);
  const disposers = [];

  function register(route) {
    disposers.push(ctx.connection.fetch.register(route));
  }

  function startResources() {
    try {
      register({
        path: ROUTE.workbench,
        methods: ['GET'],
        requestBody: 'buffered',
        fetch: async () => {
          const html = await readFile(WORKBENCH, 'utf8');
          return new Response(html, {
            headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' },
          });
        },
      });

      register({
        path: ROUTE.upload,
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          let body;
          try {
            body = await request.json();
          } catch {
            return fail('E_BODY', 'Expected a JSON body', 400);
          }
          if (!body || typeof body !== 'object') return fail('E_BODY', 'Expected a JSON object');
          const raw = typeof body.data === 'string' ? body.data : '';
          const base64 = raw.startsWith('data:') ? raw.slice(raw.indexOf(',') + 1) : raw;
          if (base64.length === 0) return fail('E_BODY', 'No image data');
          if (base64.length > MAX_UPLOAD_BYTES * 1.4) return fail('E_TOO_LARGE', 'Image is larger than 40 MB', 413);
          let buffer;
          try {
            buffer = Buffer.from(base64, 'base64');
          } catch {
            return fail('E_BODY', 'Image data is not valid base64');
          }
          if (buffer.length === 0) return fail('E_BODY', 'Image data is empty');
          if (buffer.length > MAX_UPLOAD_BYTES) return fail('E_TOO_LARGE', 'Image is larger than 40 MB', 413);
          if (sniffImageExt(buffer) === null) return fail('E_FORMAT', 'Only JPEG, PNG, WebP and HEIC images are supported', 415);
          let saved;
          try {
            saved = await engine.saveSource(buffer, body.name);
          } catch (error) {
            log(ctx, 'upload-save-failed', error);
            return fail('E_SAVE', `保存上传图片失败：${String(error?.message ?? error).slice(0, 300)}`, 500);
          }
          // Warm the landmark cache while the UI is still uploading: the first
          // slider move then renders in ~0.3 s instead of loading the model.
          engine
            .landmarksFor(saved.file)
            .catch((error) => log(ctx, 'beauty: landmark warm-up failed', error));
          return json({ ok: true, id: saved.id, name: saved.name, bytes: buffer.length });
        },
      });

      register({
        path: ROUTE.source,
        methods: ['GET'],
        requestBody: 'buffered',
        fetch: async (request) => {
          const url = new URL(request.url);
          const file = await engine.findSource(url.searchParams.get('id'));
          if (file === null) return fail('E_NOT_FOUND', 'Unknown image', 404);
          const data = await readFile(file);
          return new Response(data, {
            headers: {
              'content-type': mediaTypeFor(file),
              'cache-control': 'private, max-age=3600',
            },
          });
        },
      });

      register({
        path: ROUTE.thumb,
        methods: ['GET'],
        requestBody: 'buffered',
        fetch: async (request) => {
          const url = new URL(request.url);
          const width = Math.min(1024, Math.max(48, Math.trunc(Number(url.searchParams.get('w')) || 360)));
          // `id=` is an imported original (HEIC included); `ren=` is a rendered
          // preview, so the filmstrip can show the edited look without decoding
          // a 2400 px JPEG for a 96 px tile.
          const ren = url.searchParams.get('ren');
          let file = null;
          if (ren) {
            if (!/^ren_[a-f0-9]{12}_[a-z0-9]+\.jpg$/.test(ren)) return fail('E_BAD_ID', 'Unknown render id', 400);
            const candidate = path.join(engine.renderDir, ren);
            if (existsSync(candidate)) file = candidate;
          } else {
            file = await engine.findSource(url.searchParams.get('id'));
          }
          if (file === null) return fail('E_NOT_FOUND', 'Unknown image', 404);
          try {
            const thumb = await engine.thumbnail(file, width);
            const data = await readFile(thumb);
            return new Response(data, {
              headers: { 'content-type': 'image/jpeg', 'cache-control': 'private, max-age=86400' },
            });
          } catch (error) {
            // No sips (or HEIC it cannot read): the browser may still decode the
            // original, so serve that instead of an empty tile.
            log(ctx, 'thumb-failed', error);
            const data = await readFile(file);
            return new Response(data, {
              headers: { 'content-type': mediaTypeFor(file), 'cache-control': 'private, max-age=3600' },
            });
          }
        },
      });

      register({
        path: ROUTE.render,
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          let body;
          try {
            body = await request.json();
          } catch {
            return fail('E_BODY', 'Expected a JSON body');
          }
          const params = normaliseParams(body?.params);
          const preview = body?.preview !== false;
          if (preview) params.max_size = params.max_size > 0 ? Math.min(params.max_size, PREVIEW_LONG_SIDE) : PREVIEW_LONG_SIDE;

          let input;
          if (typeof body?.id === 'string' && body.id.length > 0) {
            input = await engine.findSource(body.id);
            if (input === null) return fail('E_NOT_FOUND', 'Unknown image', 404);
          } else if (typeof body?.path === 'string' && body.path.length > 0) {
            input = path.resolve(body.path);
            try {
              const info = await stat(input);
              if (!info.isFile()) throw new Error('not a file');
            } catch {
              return fail('E_NOT_FOUND', 'Image path not found', 404);
            }
          } else {
            return fail('E_BODY', 'Provide an image id or path');
          }

          await engine.ensureDirs();
          const key = paramsKey(params);
          const output = path.join(engine.renderDir, `ren_${key}_${Date.now().toString(36)}.jpg`);
          let report;
          try {
            report = await engine.retouch({ input, output, params, signal: request.signal });
          } catch (error) {
            if (error && error.code === 'E_CANCELLED') return fail('E_CANCELLED', 'Cancelled', 499);
            if (error && error.code === 'E_ENGINE_MISSING') return fail('E_ENGINE_MISSING', error.message, 503);
            const message = String(error && error.message ? error.message : error);
            if (/no face/i.test(message)) return fail('E_NO_FACE', '没有检测到人脸，换一张正面人像试试', 422);
            log(ctx, 'render-failed', error);
            return fail('E_ENGINE_FAILED', message.slice(0, 600), 500);
          }
          if (report.faces === 0) {
            await rm(output, { force: true });
            return fail('E_NO_FACE', '没有检测到人脸，换一张正面人像试试', 422);
          }
          engine.prune().catch(() => {});
          const renderId = path.basename(output);
          return json(
            summarise(report, {
              url: `${ROUTE.renderFile}?id=${encodeURIComponent(renderId)}&v=${key}${Date.now().toString(36)}`,
              beforeUrl: `${ROUTE.source}?id=${encodeURIComponent(body.id ?? '')}`,
            }),
          );
        },
      });

      register({
        path: ROUTE.renderFile,
        methods: ['GET'],
        requestBody: 'buffered',
        fetch: async (request) => {
          const url = new URL(request.url);
          const id = url.searchParams.get('id');
          if (typeof id !== 'string' || !/^ren_[a-f0-9]{12}_[a-z0-9]+\.jpg$/.test(id)) {
            return fail('E_NOT_FOUND', 'Unknown render', 404);
          }
          const file = path.join(engine.renderDir, id);
          if (!file.startsWith(engine.renderDir + path.sep)) return fail('E_NOT_FOUND', 'Unknown render', 404);
          let data;
          try {
            data = await readFile(file);
          } catch {
            return fail('E_NOT_FOUND', 'Unknown render', 404);
          }
          return new Response(data, {
            headers: { 'content-type': 'image/jpeg', 'cache-control': 'private, no-store' },
          });
        },
      });

      // A 1:1 patch of the region the panel is currently showing, for the
      // zoomed-in view where even the 2400px preview is no longer enough.
      register({
        path: ROUTE.detail,
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          let body;
          try {
            body = await request.json();
          } catch {
            return fail('E_BODY', 'Expected a JSON body');
          }
          const input = await engine.findSource(body?.id);
          if (input === null) return fail('E_NOT_FOUND', 'Unknown image', 404);
          const rect = Array.isArray(body?.rect) ? body.rect.map((v) => Math.round(Number(v))) : null;
          if (rect === null || rect.length !== 4 || rect.some((v) => !Number.isFinite(v)) || rect[2] < 8 || rect[3] < 8) {
            return fail('E_BODY', 'rect wants [x, y, w, h]');
          }
          // mode:'original' returns the untouched pixels for the hold-to-compare
          // view; zeroed params make the engine an identity pass (crop only).
          const mode = body?.mode === 'original' ? 'original' : 'patched';
          // Every knob at zero is an identity pass in the engine, so the crop
          // comes back as untouched source pixels.
          const params =
            mode === 'original'
              ? normaliseParams({
                  smooth: 0, whiten: 0, slim: 0, eyes: 0, detail: 0,
                  temperature: 0, tint: 0,
                })
              : normaliseParams(body?.params);
          params.max_size = DETAIL_LONG_SIDE;
          await engine.ensureDirs();
          const key = createHash('sha1')
            .update(JSON.stringify({ id: body.id, rect, params, mode }))
            .digest('hex')
            .slice(0, 16);
          const file = path.join(engine.renderDir, `dtl_${key}.jpg`);
          let data = null;
          try {
            data = await readFile(file);
          } catch {
            /* render below */
          }
          if (data === null) {
            const output = path.join(engine.renderDir, `dtl_${key}_${Date.now().toString(36)}.jpg`);
            try {
              await engine.retouch({ input, output, params, crop: rect, signal: request.signal });
            } catch (error) {
              if (error && error.code === 'E_CANCELLED') return fail('E_CANCELLED', 'Cancelled', 499);
              const message = String(error && error.message ? error.message : error);
              if (/no face/i.test(message)) return fail('E_NO_FACE', '没有检测到人脸，换一张正面人像试试', 422);
              log(ctx, 'detail-failed', error);
              return fail('E_ENGINE_FAILED', message.slice(0, 600), 500);
            }
            try {
              data = await readFile(output);
              await rename(output, file).catch(() => {});
            } catch {
              return fail('E_ENGINE_FAILED', 'Detail render produced no file', 500);
            }
            engine.prune().catch(() => {});
          }
          return new Response(data, {
            headers: { 'content-type': 'image/jpeg', 'cache-control': 'private, max-age=600' },
          });
        },
      });

      register({
        path: ROUTE.save,
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          let body;
          try {
            body = await request.json();
          } catch {
            return fail('E_BODY', 'Expected a JSON body');
          }
          const input = await engine.findSource(body?.id);
          if (input === null) return fail('E_NOT_FOUND', 'Unknown image', 404);
          const params = normaliseParams(body?.params);
          // Saving always renders at full resolution.
          params.max_size = 0;
          const base =
            safeBase(await engine.sourceName(body?.id)) || path.basename(input, path.extname(input));
          const stamp = localStamp();
          const suggested = `${base}-beauty-${stamp}.jpg`;
          // Batch export: the panel picked one folder up front, so every image
          // lands there under its own name instead of opening N save panels.
          const batchDir =
            typeof body?.dir === 'string' && body.dir.length > 0 ? path.resolve(body.dir) : null;

          // Ask where to put it. The panel remembers the folder between runs.
          let picked = null;
          if (batchDir === null) {
            try {
              picked = await engine.pickSavePath(suggested);
            } catch (error) {
              log(ctx, 'save-dialog-failed', error);
              return fail('E_SAVE_DIALOG', `无法打开保存对话框：${String(error && error.message ? error.message : error)}`, 500);
            }
            if (picked === 'cancelled') return fail('E_CANCELLED', '已取消保存', 409);
          }

          let target;
          if (batchDir !== null) {
            try {
              await mkdir(batchDir, { recursive: true });
            } catch (error) {
              return fail('E_SAVE', `无法创建导出文件夹：${String(error && error.message ? error.message : error).slice(0, 300)}`, 500);
            }
            target = path.join(batchDir, suggested);
          } else {
            await mkdir(engine.outputDir, { recursive: true });
            target =
              typeof picked === 'string' && picked.length > 0
                ? picked
                : path.join(engine.outputDir, suggested);
          }
          if (!/\.jpe?g$/i.test(target)) target += '.jpg';
          await mkdir(path.dirname(target), { recursive: true });
          for (let n = 2; existsSync(target) && n < 100; n += 1) {
            const ext = path.extname(target);
            target = path.join(path.dirname(target), `${path.basename(target, ext)}-${n}${ext}`);
          }
          let report;
          try {
            report = await engine.retouch({ input, output: target, params, signal: request.signal });
          } catch (error) {
            const message = String(error && error.message ? error.message : error);
            if (/no face/i.test(message)) return fail('E_NO_FACE', '没有检测到人脸，换一张正面人像试试', 422);
            log(ctx, 'save-failed', error);
            return fail('E_ENGINE_FAILED', message.slice(0, 600), 500);
          }
          await engine.rememberSaveDir(path.dirname(target));
          return json(summarise(report, { path: target, dir: path.dirname(target) }));
        },
      });

      // One native folder panel for a whole batch, before the renders start.
      register({
        path: ROUTE.pickDir,
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async () => {
          let picked = null;
          try {
            picked = await engine.pickFolder();
          } catch (error) {
            log(ctx, 'folder-dialog-failed', error);
            return fail('E_SAVE_DIALOG', `无法打开文件夹选择框：${String(error && error.message ? error.message : error)}`, 500);
          }
          if (picked === 'cancelled') return fail('E_CANCELLED', '已取消选择导出位置', 409);
          // No native folder dialog on this platform: the configured output
          // folder is the sensible default and the panel still shows it.
          const dir = typeof picked === 'string' && picked.length > 0 ? picked : engine.outputDir;
          await mkdir(dir, { recursive: true }).catch(() => {});
          return json({ ok: true, dir });
        },
      });

      register({
        path: ROUTE.reveal,
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          let dir = engine.outputDir;
          try {
            const body = await request.json();
            if (body && typeof body.dir === 'string' && body.dir.length > 0) dir = path.resolve(body.dir);
          } catch {
            /* empty body: reveal the configured output folder */
          }
          await mkdir(dir, { recursive: true }).catch(() => {});
          try {
            const child = spawn('open', [dir], { stdio: 'ignore' });
            child.on('error', () => {});
            return json({ ok: true, dir });
          } catch (error) {
            return fail('E_REVEAL', String(error), 500);
          }
        },
      });

      register({
        path: ROUTE.aiSuggest,
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: (request) => aiSuggest(ctx, engine, config, request),
      });

      register({
        path: ROUTE.aiSkill,
        methods: ['GET'],
        requestBody: 'buffered',
        fetch: async () => {
          try {
            const skill = await loadSkill();
            return json({ ok: true, ...skill });
          } catch (error) {
            log(ctx, 'ai-skill-failed', error);
            return fail('E_SKILL_MISSING', String(error?.message ?? error), 500);
          }
        },
      });
    } catch (error) {
      log(ctx, 'route-registration-failed', error);
    }

    try {
      disposers.push(ctx.tools.register(createTool(engine, ctx)));
    } catch (error) {
      log(ctx, 'tool-registration-failed', error);
    }

    return async function disposeBeautyResources() {
      for (const disposer of disposers.splice(0).reverse()) {
        try {
          await disposer();
        } catch (error) {
          log(ctx, 'dispose-failed', error);
        }
      }
    };
  }

  ctx.effect(startResources, 'dsh-photo-beauty: engine, routes and tool');
}

// ---------------------------------------------------------------------------
// Tool

const TOOL_NAME = 'beauty_retouch';

/** Relative tool paths follow the Session's current directory, like the fs tools. */
async function sessionDirectory(ctx, exec) {
  try {
    const service = ctx.workingDirectory;
    if (service && typeof service.ensure === 'function' && exec && exec.agent) {
      const dir = await service.ensure(exec.agent, exec.signal);
      if (typeof dir === 'string' && dir.length > 0) return dir;
    }
  } catch {
    /* fall back to the Host process directory */
  }
  return exec && typeof exec.cwd === 'string' && exec.cwd.length > 0 ? exec.cwd : process.cwd();
}

function createTool(engine, ctx) {
  return {
    name: TOOL_NAME,
    description:
      '本地人像美颜：磨皮(smooth)、提亮肤色(whiten)、瘦脸(slim)、大眼(eyes)、细节(detail)、皱纹修复(wrinkle 0-100 淡化面部皱纹，wrinkleStrokes 画笔把涂到的皱纹彻底抹平)、整体肤色(temperature 色温 + tint 色调，一块 2D 盘调冷暖/绿洋红，只动皮肤)、化妆(shadow眼影/liner眼线/blush腮红/lip口红，各带十六进制颜色)。Engine stays on this Mac (MediaPipe FaceMesh + OpenCV); never uploads the photo. ' +
      'Makeup is painted from face landmarks, so it needs a clear, front-facing face; colours are "#rrggbb". ' +
      'Pick a conservative preset first (natural ≈ smooth 18/whiten 8/slim 6/eyes 4), then adjust from the user\'s feedback. ' +
      '`person` lifts all skin of the main subject (人物提亮), not just the face; `maskStrokes` paints extra areas into that subject mask. ' +
      '`liquify` replays push/bloat/pucker brush strokes given in source-normalised coordinates; strokes are applied after makeup. ' +
      'When the photo has several faces, set `face` to the 0-based index. ' +
      'Returns the written file path plus a rendered preview image; the user can open the 美颜工坊 panel for slider-based editing.',
    parameters: {
      type: 'object',
      properties: {
        input: { type: 'string', description: 'Image path on this machine (JPEG/PNG/WebP/HEIC).' },
        output: { type: 'string', description: 'Optional output path; defaults to a cache render.' },
        smooth: { type: 'number', minimum: 0, maximum: 100, description: 'Skin smoothing strength (磨皮).' },
        whiten: { type: 'number', minimum: 0, maximum: 100, description: 'Skin brightening (提亮肤色), face only.' },
        person: { type: 'number', minimum: 0, maximum: 100, description: 'Subject-wide brightening (人物提亮): brightens the whole main person\'s skin — face, neck, shoulders, arms — with a gamma curve that grows faster toward 100.' },
        wrinkle: { type: 'number', minimum: 0, maximum: 100, description: 'Wrinkle repair (皱纹修复) 0-100: fades crease-like ridges on the face skin only (brows, eyes and lips are skipped). 100 is a strong fade that still reads as skin.' },
        slim: { type: 'number', minimum: 0, maximum: 100, description: 'Face slimming (瘦脸).' },
        eyes: { type: 'number', minimum: 0, maximum: 100, description: 'Eye enlargement (大眼).' },
        detail: { type: 'number', minimum: 0, maximum: 1, description: 'Detail restoration after smoothing (细节), default 0.35.' },
        temperature: { type: 'number', minimum: -50, maximum: 50, description: 'Overall skin temperature (色温) over every patch of skin: positive warms toward yellow, negative cools toward blue. ±50 stays inside natural human skin; 0 is neutral.' },
        tint: { type: 'number', minimum: -50, maximum: 50, description: 'Overall skin tint (色调): positive pushes magenta/rosy, negative pushes green/olive. ±50 stays inside natural human skin; 0 is neutral.' },
        warmth: { type: 'number', minimum: -50, maximum: 50, description: 'Deprecated alias for `temperature` (暖色), kept for old calls.' },
        shadow: { type: 'number', minimum: 0, maximum: 100, description: 'Eyeshadow strength (眼影), 0 = off.' },
        shadowColor: { type: 'string', description: 'Eyeshadow colour as #rrggbb, default #a97b8c.' },
        liner: { type: 'number', minimum: 0, maximum: 100, description: 'Eyeliner strength (眼线), 0 = off.' },
        linerColor: { type: 'string', description: 'Eyeliner colour as #rrggbb, default #2a211f.' },
        blush: { type: 'number', minimum: 0, maximum: 100, description: 'Blush strength (腮红), 0 = off.' },
        blushColor: { type: 'string', description: 'Blush colour as #rrggbb, default #e2827e.' },
        lip: { type: 'number', minimum: 0, maximum: 100, description: 'Lipstick strength (口红), 0 = off.' },
        lipColor: { type: 'string', description: 'Lipstick colour as #rrggbb, default #b4404f.' },
        maskStrokes: {
          type: 'array',
          description:
            'Manual skin-mask brush strokes in source-normalised coordinates (0–1), used with `person`. Each stroke: radius as a fraction of the source width (0.002–0.4) and up to 512 [x, y] points.',
          items: {
            type: 'object',
            properties: {
              radius: { type: 'number' },
              points: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
            },
            required: ['points'],
          },
        },
        wrinkleStrokes: {
          type: 'array',
          description:
            'Wrinkle brush strokes (皱纹修复笔) in source-normalised coordinates (0–1): every crease the stroke covers is erased (lower threshold and a wider detection radius than the slider, so bigger folds count too). Same shape as maskStrokes and independent of the `wrinkle` slider.',
          items: {
            type: 'object',
            properties: {
              radius: { type: 'number' },
              points: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
            },
            required: ['points'],
          },
        },
        liquify: {
          type: 'array',
          description:
            'Liquify brush strokes in source-normalised coordinates (0–1). Each stroke: mode push/bloat/pucker, radius as a fraction of the source width (0.004–0.6), strength 0.02–1, and up to 512 [x, y] points.',
          items: {
            type: 'object',
            properties: {
              mode: { type: 'string', enum: ['push', 'bloat', 'pucker'] },
              radius: { type: 'number' },
              strength: { type: 'number' },
              points: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
            },
            required: ['mode', 'points'],
          },
        },
        face: { type: 'integer', minimum: 0, maximum: 9, description: '0-based index of the face to edit when several are detected.' },
        compare: { type: 'boolean', description: 'Also write a before/after sheet next to the output.' },
        max_size: { type: 'integer', minimum: 0, maximum: 12000, description: 'Cap the long side in pixels (0 = full resolution).' },
      },
      required: ['input'],
    },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render(_args, value) {
        const blocks = [
          {
            type: 'text',
            text: `<output>${value.output}</output>\n<type>image</type>\n<content>\n${value.image ? `${value.image.mediaType} image, ${value.image.width}x${value.image.height} px, ${value.image.bytes} bytes` : 'no preview'}\n</content>\n<beauty>faces=${value.faces} used=${value.face} ${JSON.stringify(value.params)} ${value.elapsed_ms}ms</beauty>`,
          },
        ];
        if (value.image) blocks.push({ type: 'image', attachment: value.image });
        return blocks;
      },
    },
    async execute(args, exec) {
      const input = args && typeof args.input === 'string' ? args.input : '';
      if (input.length === 0) throw new Error('`input` must be an image path');
      const params = normaliseParams(args);
      const baseDir = await sessionDirectory(ctx, exec);
      const inputPath = path.resolve(baseDir, input);
      try {
        const info = await stat(inputPath);
        if (!info.isFile()) throw new Error('not a file');
      } catch {
        throw new Error(`Image not found: ${inputPath}`);
      }

      let target = typeof args.output === 'string' && args.output.length > 0
        ? path.resolve(baseDir, args.output)
        : null;
      if (target === null) {
        await engine.ensureDirs();
        target = path.join(engine.renderDir, `tool_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}.jpg`);
      } else {
        await mkdir(path.dirname(target), { recursive: true });
      }

      const report = await engine.retouch({
        input: inputPath,
        output: target,
        params,
        signal: exec && exec.signal,
      });
      if (report.faces === 0) {
        throw new Error('没有检测到人脸：请换一张正面、五官清晰的照片。');
      }

      const value = {
        ok: true,
        output: target,
        faces: report.faces,
        face: report.face_used,
        width: report.width,
        height: report.height,
        original_width: report.original_width,
        original_height: report.original_height,
        params: report.params,
        skin: report.skin ?? null,
        elapsed_ms: report.elapsed_ms,
        notes: report.notes ?? [],
        image: null,
      };

      try {
        const attachments = ctx.get('attachments');
        const data = await readFile(target);
        const ref = await attachments.saveImage({ data, mediaType: 'image/jpeg', name: path.basename(target) });
        value.image = {
          attachmentId: ref.attachmentId,
          mediaType: ref.mediaType,
          bytes: ref.bytes,
          width: ref.width,
          height: ref.height,
          name: ref.name,
        };
      } catch (error) {
        log(ctx, 'attachment-save-failed', error);
      }

      return value;
    },
    presentCall(args) {
      const knobs = ['smooth', 'whiten', 'person', 'wrinkle', 'slim', 'eyes', 'shadow', 'liner', 'blush', 'lip']
        .filter((key) => typeof args?.[key] === 'number' && args[key] > 0)
        .map((key) => `${key} ${args[key]}`);
      const temperature = typeof args?.temperature === 'number' ? args.temperature : args?.warmth;
      if (typeof temperature === 'number' && temperature !== 0) {
        const value = Math.max(-50, Math.min(50, temperature));
        knobs.push(`肤色 ${value > 0 ? '+' : ''}${value}`);
      }
      if (typeof args?.tint === 'number' && args.tint !== 0) {
        const value = Math.max(-50, Math.min(50, args.tint));
        knobs.push(`色调 ${value > 0 ? '+' : ''}${value}`);
      }
      const text = knobs.join(' · ');
      return {
        card: 'generic',
        title: text.length > 0 ? `美颜 · ${text}` : '美颜',
        subtitle: typeof args?.input === 'string' ? path.basename(args.input) : undefined,
        kind: 'other',
        rawInput: args,
      };
    },
  };
}
