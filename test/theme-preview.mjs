// Screenshots the workbench embedded in a fake DSH host page, so the panel's
// palette can be checked against the Host theme instead of the OS appearance.
//
// Usage: node test/theme-preview.mjs [image] [preset]
//   node test/theme-preview.mjs test-assets/face1.jpg transformed
// Writes test-assets/out/theme-dark.png and theme-light.png.
import { apply } from '../lib/index.js';
import { mkdir, readFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { createServer } from 'node:http';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

// The values DSH's own theme package puts on <body> / <body data-ds-dark-theme>
// (see @deepseek-ai/dsh-client-ui-theme). Keep them in sync with the real ones:
// this page is only a stand-in for the Host chrome around the iframe.
const PALETTES = {
  dark: {
    '--dsw-alias-bg-base': '#151517',
    '--dsw-alias-bg-layer-1': '#232324',
    '--dsw-alias-bg-layer-2': '#2c2c2e',
    '--dsw-alias-border-l1': '#ffffff0f',
    '--dsw-alias-border-l2': '#ffffff1f',
    '--dsw-alias-label-primary': '#f9fafb',
    '--dsw-alias-label-secondary': '#cfd3d6',
    '--dsw-alias-label-primary-foreground': '#0f1115',
    '--dsw-alias-button-info-fill': '#7aaaff',
    '--dsw-alias-state-error-primary': '#f25a5a',
    '--dsw-alias-state-success-primary': '#22c55e',
    '--dsw-alias-interactive-bg-hover-solid': '#353638',
    '--dsw-alias-switch-thumb': '#adb2b8',
    '--dsw-alias-scrollbar-bg-l1': '#3c3c3d',
  },
  light: {
    '--dsw-alias-bg-base': '#ffffff',
    '--dsw-alias-bg-layer-1': '#ffffff',
    '--dsw-alias-bg-layer-2': '#ffffff',
    '--dsw-alias-border-l1': '#0000000a',
    '--dsw-alias-border-l2': '#0000001a',
    '--dsw-alias-label-primary': '#0f1115',
    '--dsw-alias-label-secondary': '#61666b',
    '--dsw-alias-label-primary-foreground': '#ffffff',
    '--dsw-alias-button-info-fill': '#4176e6',
    '--dsw-alias-state-error-primary': '#ec1313',
    '--dsw-alias-state-success-primary': '#22c55e',
    '--dsw-alias-interactive-bg-hover-solid': '#f1f3f5',
    '--dsw-alias-switch-thumb': '#ffffff',
    '--dsw-alias-scrollbar-bg-l1': '#e5e5e5',
  },
};

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

const scratch = await mkdtemp(path.join(tmpdir(), 'dsh-beauty-theme-'));
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

/** Host stand-in: DSH's palette on <body>, workbench in the main panel iframe. */
function hostPage(theme, src, flip) {
  const palette = PALETTES[theme];
  const other = PALETTES[theme === 'dark' ? 'light' : 'dark'];
  const declarations = Object.entries(palette)
    .map(([name, value]) => `${name}:${value}`)
    .join(';');
  const flipped = Object.entries(other)
    .map(([name, value]) => `"${name}":"${value}"`)
    .join(',');
  return `<!doctype html>
<html lang="zh-CN" style="color-scheme:${theme}">
<head><meta charset="utf-8"><title>fake dsh ${theme}</title>
<style>
  html, body { height: 100%; }
  body { margin: 0; padding: 0; background: var(--dsw-alias-bg-base); }
  main { height: 100vh; }
  iframe { display: block; width: 100%; height: 100%; border: 0; }
</style></head>
<body${theme === 'dark' ? ' data-ds-dark-theme' : ''} style="${declarations}">
<main><iframe src="/api/dsh-beauty/workbench?src=${src}&preset=${preset}" title="美颜工坊"></iframe></main>
<script>
  // flip=1: switch the Host theme after mount, to prove the panel follows live.
  if (${flip ? 'true' : 'false'}) setTimeout(function () {
    var next = {${flipped}};
    for (var name in next) document.body.style.setProperty(name, next[name]);
    document.body.toggleAttribute('data-ds-dark-theme', ${theme === 'light' ? 'true' : 'false'});
    document.documentElement.style.colorScheme = '${theme === 'dark' ? 'light' : 'dark'}';
  }, 800);
</script>
</body></html>`;
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://127.0.0.1');
    if (url.pathname === '/host') {
      const theme = url.searchParams.get('theme') === 'light' ? 'light' : 'dark';
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      res.end(
        hostPage(theme, url.searchParams.get('src') || '', url.searchParams.get('flip') === '1'),
      );
      return;
    }
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

const verbose = !!process.env.PREVIEW_VERBOSE;

function shoot(url, file) {
  return new Promise((resolve, reject) => {
    const chrome = spawn(
      CHROME,
      [
        '--headless=old',
        '--no-sandbox',
        '--no-first-run',
        '--disable-gpu',
        ...(verbose ? ['--enable-logging=stderr'] : []),
        `--user-data-dir=${path.join(scratch, 'chrome-' + Math.random().toString(36).slice(2))}`,
        '--hide-scrollbars',
        '--force-device-scale-factor=1',
        '--window-size=1440,1400',
        `--screenshot=${file}`,
        '--virtual-time-budget=4000',
        url,
      ],
      { stdio: ['ignore', 'ignore', 'pipe'] },
    );
    if (verbose) {
      chrome.stderr.on('data', (chunk) => {
        for (const line of String(chunk).split('\n')) {
          if (line.includes('CONSOLE') || line.includes('Uncaught')) console.log('  ' + line.trim());
        }
      });
    }
    chrome.on('error', reject);
    chrome.on('exit', (code) => (code === 0 ? resolve() : reject(new Error('chrome exit ' + code))));
  });
}

const shots = [
  ['theme-dark', 'theme=dark'],
  ['theme-light', 'theme=light'],
  ['theme-flip', 'theme=dark&flip=1'],
];
for (const [name, query] of shots) {
  const file = path.resolve('test-assets/out', `${name}.png`);
  await mkdir(path.dirname(file), { recursive: true });
  await shoot(`http://127.0.0.1:${port}/host?${query}&src=${uploaded.id}`, file);
  console.log(`${name}: ${file}`);
}
server.close();
