// Host smoke test: no DSH runtime, just the plugin's own contract.
// Usage: node test/host-smoke.mjs
import { apply } from '../lib/index.js';
import { mkdtemp, readFile, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';

const routes = new Map();
const tools = new Map();
let disposed = null;

const ctx = {
  logger: { warn() {} },
  effect(fn) {
    disposed = fn();
  },
  connection: {
    fetch: {
      register(route) {
        routes.set(route.path, route);
        return () => routes.delete(route.path);
      },
    },
  },
  tools: {
    register(tool) {
      tools.set(tool.name, tool);
      return () => tools.delete(tool.name);
    },
  },
  get(name) {
    if (name !== 'attachments') return undefined;
    return {
      async saveImage({ data, mediaType, name }) {
        return { attachmentId: 'att_test', mediaType, bytes: data.length, width: 100, height: 100, name };
      },
    };
  },
};

const scratch = await mkdtemp(path.join(path.dirname(new URL(import.meta.url).pathname), '..', '.cache', 'smoke-'));
apply(ctx, { cacheDir: scratch, outputDir: path.join(scratch, 'out') });
assert.equal(typeof disposed, 'function', 'effect returns a disposer');
assert.ok(tools.has('beauty_retouch'), 'tool registered');
for (const key of ['workbench', 'upload', 'source', 'render', 'render-file', 'detail', 'save', 'reveal']) {
  assert.ok(routes.has('/api/dsh-beauty/' + key), 'route ' + key);
}

// Tool renders an image preview.
const tool = tools.get('beauty_retouch');
const value = await tool.execute(
  {
    input: 'test-assets/face1.jpg', smooth: 40, whiten: 16, slim: 22, eyes: 14,
    shadow: 50, shadowColor: '#7a5c78', liner: 40, blush: 35, lip: 60, lipColor: 'b4404f',
  },
  { signal: undefined },
);
assert.equal(value.ok, true);
assert.ok(value.faces >= 1);
assert.ok((await stat(value.output)).isFile(), 'output written');
// Makeup reaches the engine, colours are normalised, and the report says so.
assert.equal(value.params.shadow, 50);
assert.equal(value.params.shadow_color, '#7a5c78');
assert.equal(value.params.lip_color, '#b4404f', 'bare hex gets a leading #');
assert.equal(value.params.liner_color, '#2a211f', 'default while the layer is off');
assert.ok(value.notes.some((note) => note.startsWith('makeup:')), 'makeup note: ' + value.notes.join(' | '));

// A bogus colour must not reach the CLI.
const bogus = await tool.execute(
  { input: 'test-assets/face1.jpg', shadow: 20, shadowColor: 'rebeccapurple', max_size: 320 },
  { signal: undefined },
);
assert.equal(bogus.params.shadow_color, '#a97b8c');
// Person brightening + liquify strokes reach the engine and are reported back.
const body = await tool.execute(
  {
    input: 'test-assets/face1.jpg', smooth: 20, whiten: 8, person: 55, max_size: 480,
    temperature: 12, tint: -6,
    liquify: [{ mode: 'push', radius: 0.2, strength: 0.8, points: [[0.4, 0.55], [0.46, 0.58]] }],
    maskStrokes: [{ radius: 0.05, points: [[0.1, 0.1], [0.12, 0.12]] }],
    wrinkle: 62,
    wrinkleStrokes: [{ radius: 0.04, points: [[0.42, 0.52], [0.46, 0.56]] }],
  },
  { signal: undefined },
);
assert.equal(body.params.person, 55);
assert.equal(body.params.temperature, 12);
assert.equal(body.params.tint, -6);
assert.ok(body.skin && body.skin.subject > 0, 'subject skin stats: ' + JSON.stringify(body.skin));
assert.ok(body.notes.some((note) => note.startsWith('liquify: 1')), 'liquify note: ' + body.notes.join(' | '));
assert.ok(body.notes.some((note) => note.startsWith('mask brush: 1')), 'mask brush note: ' + body.notes.join(' | '));
assert.ok(!('maskStrokes' in body.params), 'mask strokes stay out of echoed params');
assert.ok(body.notes.some((note) => note.startsWith('person glow')), 'person glow note');
assert.equal(body.params.wrinkle, 62);
assert.ok(body.notes.some((note) => note.startsWith('wrinkle repair: 62%')), 'wrinkle note: ' + body.notes.join(' | '));
assert.ok(body.notes.some((note) => note.startsWith('wrinkle brush: 1')), 'wrinkle brush note: ' + body.notes.join(' | '));
assert.ok(!('wrinkleStrokes' in body.params), 'wrinkle strokes stay out of echoed params');

// The old single 暖色 knob still works: it becomes temperature, clamped to ±50.
const legacy = await tool.execute(
  { input: 'test-assets/face1.jpg', warmth: 70, max_size: 240 },
  { signal: undefined },
);
assert.equal(legacy.params.temperature, 50, 'warmth maps to temperature and clamps');
assert.equal(legacy.params.tint, 0);

const blocks = tool.output.render({}, value);
assert.ok(blocks.some((block) => block.type === 'image'), 'image block');

// Routes: upload -> render -> render-file.
const bytes = await readFile('test-assets/face1.jpg');
const upload = await routes.get('/api/dsh-beauty/upload').fetch(
  new Request('http://local/api/dsh-beauty/upload', {
    method: 'POST',
    body: JSON.stringify({ name: 'face1.jpg', data: bytes.toString('base64') }),
  }),
);
const uploaded = await upload.json();
assert.equal(uploaded.ok, true);

const rendered = await (
  await routes.get('/api/dsh-beauty/render').fetch(
    new Request('http://local/api/dsh-beauty/render', {
      method: 'POST',
      body: JSON.stringify({ id: uploaded.id, params: { smooth: 30, whiten: 12, slim: 15, eyes: 10, person: 160 } }),
    }),
  )
).json();
assert.equal(rendered.ok, true, JSON.stringify(rendered));
assert.equal(rendered.params.person, 100, 'person clamps at 100');
assert.ok(rendered.elapsed_ms > 0);

const fileUrl = new URL(rendered.url, 'http://local');
const served = await routes.get('/api/dsh-beauty/render-file').fetch(new Request(fileUrl));
assert.equal(served.status, 200);
assert.equal(served.headers.get('content-type'), 'image/jpeg');

/** Read a JPEG's pixel size from its SOF marker. */
function jpegSize(buffer) {
  for (let i = 2; i < buffer.length - 9; ) {
    if (buffer[i] !== 0xff) {
      i += 1;
      continue;
    }
    const marker = buffer[i + 1];
    const length = buffer.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xc3) {
      return { height: buffer.readUInt16BE(i + 5), width: buffer.readUInt16BE(i + 7) };
    }
    i += 2 + length;
  }
  return null;
}

// The zoomed-in detail patch: a crop of the source, capped at the long side.
const detailResponse = await routes.get('/api/dsh-beauty/detail').fetch(
  new Request('http://local/api/dsh-beauty/detail', {
    method: 'POST',
    body: JSON.stringify({
      id: uploaded.id,
      rect: [40, 40, 400, 300],
      params: { smooth: 20, whiten: 10, detail: 0.4 },
    }),
  }),
);
assert.equal(detailResponse.status, 200, 'detail render');
assert.equal(detailResponse.headers.get('content-type'), 'image/jpeg');
const detailBytes = Buffer.from(await detailResponse.arrayBuffer());
const detailSize = jpegSize(detailBytes);
assert.ok(detailSize, 'detail jpeg has a size');
assert.equal(detailSize.width, 400);
assert.equal(detailSize.height, 300);
// Same request again is served from the patch cache.
const cached = await routes.get('/api/dsh-beauty/detail').fetch(
  new Request('http://local/api/dsh-beauty/detail', {
    method: 'POST',
    body: JSON.stringify({
      id: uploaded.id,
      rect: [40, 40, 400, 300],
      params: { smooth: 20, whiten: 10, detail: 0.4 },
    }),
  }),
);
assert.equal((await cached.arrayBuffer()).byteLength, detailBytes.byteLength, 'cached patch');

const badRect = await routes.get('/api/dsh-beauty/detail').fetch(
  new Request('http://local/api/dsh-beauty/detail', {
    method: 'POST',
    body: JSON.stringify({ id: uploaded.id, rect: [1, 2, 3] }),
  }),
);
assert.equal(badRect.status, 400);
assert.equal((await badRect.json()).error.code, 'E_BODY');

// Unknown image is a typed 404, not a crash.
const missing = await routes.get('/api/dsh-beauty/source').fetch(
  new Request('http://local/api/dsh-beauty/source?id=src_nope'),
);
assert.equal(missing.status, 404);

await disposed();
console.log('host smoke: ok');
