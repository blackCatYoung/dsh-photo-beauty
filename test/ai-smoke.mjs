// Host smoke test for the pure-local planner: real metrics engine, no LLM,
// no attachments, no image uploads.
import { apply, buildLocalSuggestion, validateSuggestion } from '../lib/index.js';
import { mkdtemp, readFile, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';

const routes = new Map();
const tools = new Map();
let disposed = null;

const ctx = {
  logger: { warn() {} },
  effect(fn) { disposed = fn(); },
  connection: { fetch: { register(route) { routes.set(route.path, route); return () => routes.delete(route.path); } } },
  tools: { register(tool) { tools.set(tool.name, tool); return () => tools.delete(tool.name); } },
  get() { return undefined; },
};

const scratch = await mkdtemp(path.join(tmpdir(), 'dsh-ai-smoke-'));
apply(ctx, { cacheDir: scratch, outputDir: path.join(scratch, 'out') });
assert.equal(typeof disposed, 'function');
assert.ok(routes.has('/api/dsh-beauty/ai-suggest'), 'ai-suggest route');
assert.ok(routes.has('/api/dsh-beauty/ai-skill'), 'ai-skill route');

const skillResponse = await routes.get('/api/dsh-beauty/ai-skill').fetch(
  new Request('http://local/api/dsh-beauty/ai-skill', { method: 'GET' }),
);
const skill = await skillResponse.json();
assert.equal(skill.ok, true);
assert.equal(skill.name, 'portrait-beauty');
assert.ok(skill.body.includes('量化标准'), 'skill body is shipped');
assert.match(skill.hash, /^[0-9a-f]{16}$/);

const bytes = await readFile('test-assets/face1.jpg');
const upload = await routes.get('/api/dsh-beauty/upload').fetch(
  new Request('http://local/api/dsh-beauty/upload', {
    method: 'POST',
    body: JSON.stringify({ name: 'face1.jpg', data: bytes.toString('base64') }),
  }),
);
const uploaded = await upload.json();
assert.equal(uploaded.ok, true);

const call = async (params = {}, signal) => {
  const response = await routes.get('/api/dsh-beauty/ai-suggest').fetch(
    new Request('http://local/api/dsh-beauty/ai-suggest', {
      method: 'POST',
      body: JSON.stringify({ id: uploaded.id, params: { smooth: 20, whiten: 10, ...params } }),
      ...(signal ? { signal } : {}),
    }),
  );
  return { status: response.status, body: await response.json() };
};

// Happy path: local formula engine, no model, no attachments, no sent images.
const happy = await call();
assert.equal(happy.status, 200, JSON.stringify(happy.body));
assert.equal(happy.body.ok, true);
assert.equal(happy.body.mode, 'local');
assert.equal(happy.body.model.provider, 'local');
assert.deepEqual(happy.body.sentImages, []);
assert.equal(happy.body.suggestion.params.smooth > 0, true);
assert.equal(happy.body.suggestion.params.slim, 0, 'face1 geometry confidence is too low for slim');
assert.equal(happy.body.suggestion.params.eyes, 0, 'face1 geometry confidence is too low for eyes');
assert.equal(happy.body.suggestion.params.temperature, -10, 'face1 warm cast is cooled');
assert.equal(happy.body.suggestion.params.tint, -12, 'face1 red cast is neutralised');
assert.ok(happy.body.suggestion.summary.length > 0);
assert.ok(happy.body.suggestion.reasons.length > 0);
assert.ok('liquify' in happy.body.suggestion.params === false);
assert.ok('maskStrokes' in happy.body.suggestion.params === false);

// No vision JPEG directory was created: pure local never writes previews.
await assert.rejects(stat(path.join(scratch, 'vision')), /ENOENT/);

// Deterministic: same source + same params -> byte-identical suggestion.
const again = await call();
assert.deepEqual(again.body.suggestion.params, happy.body.suggestion.params);

// White balance must change the tone even when the current value is the
// signature -4/+20.
const metricsFor = (a, b, redness) => ({
  image: { subject_ratio: 0.5, face_ratio: 0.2, light: 'good' },
  pose: { geometry_confidence: 0.9, symmetry_reliable: true, reliable: true },
  geometry: { face_width: 220, jaw_ratio: 0.84, face_aspect: 0.86, chin_ratio: 0.35, eye_aspect: 0.26, symmetry: 0.01 },
  skin: { available: true, l_median: 62, l_std: 8, l_p10: 55, a_median: a, b_median: b, redness_p95: redness, blemish_density: 0.05, clipping: 0 },
  body: { available: true, delta_l: 1 },
  lips: { available: true, chroma: 22 },
  quality: { light: 'good', notes: [] },
});
const neutral = buildLocalSuggestion(metricsFor(12, 12, 15), {}, {});
const cool = buildLocalSuggestion(metricsFor(10, 11, 15), {}, {});
assert.equal(neutral.params.temperature, -4);
assert.equal(neutral.params.tint, 20);
assert.equal(cool.params.temperature, -3);
assert.equal(cool.params.tint, 22);
assert.notEqual(`${neutral.params.temperature}/${neutral.params.tint}`, `${cool.params.temperature}/${cool.params.tint}`);

// Different face proportions must produce different slim/eyes; the formula is
// not a fixed style value.
const wide = buildLocalSuggestion(metricsFor(12, 12, 15), {}, {});
const narrowMetrics = metricsFor(12, 12, 15);
narrowMetrics.geometry.jaw_ratio = 0.80;
narrowMetrics.geometry.eye_aspect = 0.33;
const narrow = buildLocalSuggestion(narrowMetrics, {}, {});
assert.ok(wide.params.slim > narrow.params.slim, 'wider jaw → more slim');
assert.ok(wide.params.eyes > narrow.params.eyes, 'smaller eye aspect → more eyes');

// A small subject in a wide frame gets a continuous framing boost; this is
// the calibration that lifts the 3.2%-of-frame portrait from ~12 to 25.
const framedMetrics = metricsFor(9, 9, 15);
framedMetrics.image.subject_ratio = 0.032;
framedMetrics.image.face_ratio = 0.004;
framedMetrics.geometry.face_width = 132.3;
framedMetrics.geometry.jaw_ratio = 0.814;
framedMetrics.geometry.face_aspect = 0.921;
framedMetrics.geometry.chin_ratio = 0.334;
framedMetrics.geometry.eye_aspect = 0.251;
framedMetrics.geometry.symmetry = 0.018;
framedMetrics.pose.geometry_confidence = 0.646;
framedMetrics.body.delta_l = -14.5;
const framed = buildLocalSuggestion(framedMetrics, {}, {});
const sameFaceLarger = structuredClone(framedMetrics);
sameFaceLarger.image.subject_ratio = 0.5;
assert.equal(framed.params.slim, 25, 'small-subject framing boost');
assert.ok(
  framed.params.slim > buildLocalSuggestion(sameFaceLarger, {}, {}).params.slim,
  'framing boost only applies to small subjects',
);

// A backlit small subject still gets the visible baseline.
const backlit = metricsFor(12, 12, 15);
backlit.image.subject_ratio = 0.06;
backlit.quality.light = 'dim';
backlit.skin.l_median = 40;
backlit.skin.l_p10 = 24;
const lifted = buildLocalSuggestion(backlit, {}, {});
assert.ok(lifted.params.whiten >= 40, 'backlit whiten baseline');
assert.ok(lifted.params.person >= 55, 'backlit person baseline');

// The old validator remains exported and still clamps a hostile suggestion.
const baseParams = { ...happy.body.suggestion.params };
const clamped = validateSuggestion(
  { params: { ...baseParams, smooth: 99, whiten: 99, slim: 99, eyes: 99, temperature: 40, tint: -40, liquify: [] } },
  baseParams,
  { pose: { geometry_confidence: 0.05, symmetry_reliable: false }, geometry: { face_width: 220, symmetry: 0.2 }, skin: { a_median: 41, b_median: 18, redness_p95: 50 } },
);
assert.equal(clamped.params.smooth, 60);
assert.equal(clamped.params.whiten, 70);
assert.equal(clamped.params.slim, 0);
assert.equal(clamped.params.eyes, 0);
assert.equal(clamped.params.temperature, -10);
assert.equal(clamped.params.tint, -12);
assert.ok(!('liquify' in clamped.params));

// A cancelled request never reports a timeout or a provider failure.
const cancelled = await call({}, AbortSignal.abort());
assert.equal(cancelled.status, 499);
assert.equal(cancelled.body.error.code, 'E_CANCELLED');

console.log('ai smoke: ok');
