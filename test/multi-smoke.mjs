// Exercises the multi-image host surface without DSH: upload -> thumbnail ->
// preview render -> render thumbnail -> export into an explicit directory.
// Usage: node test/multi-smoke.mjs
import { apply } from '../lib/index.js';
import { readFile, mkdtemp, mkdir, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

const routes = new Map();
const ctx = {
  logger: { warn() {}, info() {} },
  effect(fn) {
    fn();
  },
  connection: {
    fetch: {
      register(route) {
        routes.set(route.path, route);
        return () => routes.delete(route.path);
      },
    },
  },
  tools: { register: () => () => {} },
  get: () => undefined,
};

const scratch = await mkdtemp(path.join(tmpdir(), 'dsh-beauty-multi-'));
const outputDir = path.join(scratch, 'out');
await apply(ctx, { cacheDir: scratch, outputDir });

const base = 'http://127.0.0.1';
const route = (name) => routes.get('/api/dsh-beauty/' + name);
const call = async (pathAndQuery, init) => {
  const name = pathAndQuery.split('?')[0];
  return route(name).fetch(new Request(base + '/api/dsh-beauty/' + pathAndQuery, init));
};

const assert = (condition, message) => {
  if (!condition) {
    console.error('multi smoke: FAIL -', message);
    process.exit(1);
  }
};

const upload = async (file) => {
  const bytes = await readFile(file);
  const response = await call('upload', {
    method: 'POST',
    body: JSON.stringify({ name: path.basename(file), data: bytes.toString('base64') }),
  });
  const body = await response.json();
  assert(body.ok, `upload ${file} -> ${JSON.stringify(body)}`);
  return body;
};

const first = await upload('test-assets/face1.jpg');
const second = await upload('test-assets/face1.jpg');

const thumbResponse = await call(`thumb?id=${first.id}&w=240`);
assert(thumbResponse.status === 200, 'thumb status');
assert((thumbResponse.headers.get('content-type') || '').includes('image/jpeg'), 'thumb content-type');
const thumbBytes = Buffer.from(await thumbResponse.arrayBuffer());
assert(thumbBytes[0] === 0xff && thumbBytes[1] === 0xd8, 'thumb is a JPEG');
assert(thumbBytes.length < 91814, 'thumb is smaller than the source');

const params = { smooth: 20, whiten: 8, detail: 0.35 };
const rendered = await (
  await call('render', {
    method: 'POST',
    body: JSON.stringify({ id: first.id, params, preview: true }),
  })
).json();
assert(rendered.ok, `render -> ${JSON.stringify(rendered)}`);
const renderName = new URL(rendered.url, base).searchParams.get('id');
assert(/^ren_[a-f0-9]{12}_[a-z0-9]+\.jpg$/.test(renderName), 'render file name');

const renderThumb = await call(`thumb?ren=${encodeURIComponent(renderName)}&w=240`);
assert(renderThumb.status === 200, 'render thumb status');
const renderThumbBytes = Buffer.from(await renderThumb.arrayBuffer());
assert(renderThumbBytes[0] === 0xff && renderThumbBytes[1] === 0xd8, 'render thumb is a JPEG');

const exportDir = path.join(scratch, 'exported');
const saved = await (
  await call('save', {
    method: 'POST',
    body: JSON.stringify({ id: second.id, params, dir: exportDir }),
  })
).json();
assert(saved.ok, `save -> ${JSON.stringify(saved)}`);
assert(saved.dir === exportDir || path.resolve(saved.dir) === path.resolve(exportDir), 'save used the chosen dir');
const savedStat = await stat(saved.path);
assert(savedStat.size > 0, 'exported file exists');

// A second export into the same directory must not overwrite the first one.
const again = await (
  await call('save', {
    method: 'POST',
    body: JSON.stringify({ id: second.id, params, dir: exportDir }),
  })
).json();
assert(again.ok && again.path !== saved.path, 'second export avoids overwriting');

const listing = await import('node:fs/promises').then((fs) => fs.readdir(exportDir));
console.log('multi smoke: ok', { thumb: thumbBytes.length, rendered: renderName, files: listing });
