// Renders the workbench UI against the plugin's real routes, without DSH:
// the fake ctx collects the registered routes and a tiny http server serves them,
// so a headless browser (or you) can look at the panel.
//
// Usage: node test/workbench-preview.mjs [image] [preset]
//   node test/workbench-preview.mjs test-assets/face1.jpg refined
import { apply } from '../lib/index.js';
import { readFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createServer } from 'node:http';

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

const scratch = await mkdtemp(path.join(tmpdir(), 'dsh-beauty-preview-'));
apply(ctx, { cacheDir: scratch, outputDir: path.join(scratch, 'out') });

const image = process.argv[2] || 'test-assets/face1.jpg';
const preset = process.argv[3] || 'refined';

/** Dispatch one node request through a registered DSH route. */
async function dispatch(route, req, body) {
  const url = new URL(req.url, 'http://127.0.0.1');
  if (req.method === 'GET' && route.path.endsWith('/workbench')) {
    return new Response(await readFile(new URL('../lib/workbench.html', import.meta.url)), {
      status: 200,
      headers: { 'content-type': 'text/html; charset=utf-8' },
    });
  }
  const headers = {};
  for (const [key, value] of Object.entries(req.headers)) {
    if (typeof value === 'string') headers[key] = value;
  }
  const hasBody = req.method !== 'GET' && req.method !== 'HEAD' && body.length > 0;
  return route.fetch(
    new Request(url, {
      method: req.method,
      headers,
      body: hasBody ? body : undefined,
    }),
  );
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://127.0.0.1');
    const body = await new Promise((resolve, reject) => {
      const chunks = [];
      req.on('data', (chunk) => chunks.push(chunk));
      req.on('end', () => resolve(Buffer.concat(chunks)));
      req.on('error', reject);
    });
    const route = routes.get(url.pathname);
    if (!route) {
      res.writeHead(404, { 'content-type': 'text/plain' });
      res.end('not found');
      return;
    }
    const response = await dispatch(route, req, body);
    const buffer = Buffer.from(await response.arrayBuffer());
    const headers = {};
    response.headers.forEach((value, key) => {
      headers[key] = value;
    });
    res.writeHead(response.status, headers);
    res.end(buffer);
  } catch (error) {
    res.writeHead(500, { 'content-type': 'text/plain' });
    res.end(String((error && error.stack) || error));
  }
});

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const { port } = server.address();

const bytes = await readFile(image);
const uploaded = await (
  await routes.get('/api/dsh-beauty/upload').fetch(
    new Request('http://127.0.0.1/api/dsh-beauty/upload', {
      method: 'POST',
      body: JSON.stringify({ name: image.split('/').pop(), data: bytes.toString('base64') }),
    }),
  )
).json();
if (!uploaded.ok) {
  console.error('upload failed', uploaded);
  process.exit(1);
}

console.log(
  `workbench: http://127.0.0.1:${port}/api/dsh-beauty/workbench?src=${uploaded.id}&preset=${preset}`,
);
console.log('Ctrl-C to stop.');
