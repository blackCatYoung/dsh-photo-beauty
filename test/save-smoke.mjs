// Exercises the save route (native panel + remembered folder) without opening
// a real dialog: a fake `osascript` on PATH stands in for the macOS panel.
import { apply } from '../lib/index.js';
import { mkdtemp, mkdir, writeFile, readFile, chmod, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';

const scratch = await mkdtemp(path.join(tmpdir(), 'dsh-beauty-save-'));
const fakeBin = path.join(scratch, 'bin');
await mkdir(fakeBin, { recursive: true });
const replies = path.join(scratch, 'reply.txt');
await writeFile(replies, path.join(scratch, 'picked.jpg'));
const fake = path.join(fakeBin, 'osascript');
await writeFile(
  fake,
  `#!/bin/sh\ncat "${replies}"\n`,
);
await chmod(fake, 0o755);

const originalPath = process.env.PATH;
process.env.PATH = `${fakeBin}:${originalPath}`;

const routes = new Map();
const ctx = {
  logger: { warn() {}, info() {} },
  effect(fn) { fn(); },
  connection: { fetch: { register(route) { routes.set(route.path, route); return () => routes.delete(route.path); } } },
  tools: { register: () => () => {} },
  get: () => undefined,
};

apply(ctx, { cacheDir: path.join(scratch, 'cache'), outputDir: path.join(scratch, 'out') });

const bytes = await readFile(new URL('../test-assets/face1.jpg', import.meta.url));
const upload = await (
  await routes.get('/api/dsh-beauty/upload').fetch(
    new Request('http://x/api/dsh-beauty/upload', {
      method: 'POST',
      body: JSON.stringify({ name: 'portrait.jpg', data: bytes.toString('base64') }),
    }),
  )
).json();
assert.equal(upload.ok, true, 'upload works');

const saveUrl = 'http://x/api/dsh-beauty/save';
const saveBody = (id) => JSON.stringify({ id, params: { smooth: 20, whiten: 10 } });

const saved = await (
  await routes.get('/api/dsh-beauty/save').fetch(new Request(saveUrl, { method: 'POST', body: saveBody(upload.id) }))
).json();
assert.equal(saved.ok, true, `save ok: ${JSON.stringify(saved).slice(0, 300)}`);
assert.equal(saved.path, path.join(scratch, 'picked.jpg'));
assert.ok((await stat(saved.path)).size > 1000, 'file written to the chosen path');

const settings = JSON.parse(await readFile(path.join(scratch, 'cache', 'settings.json'), 'utf8'));
assert.equal(settings.saveDir, scratch, 'chosen folder remembered');

// Second save reuses the remembered folder as the panel's start location.
const calls = path.join(scratch, 'calls.txt');
await writeFile(
  fake,
  `#!/bin/sh\nprintf '%s\\n' "$*" >> "${calls}"\ncat "${replies}"\n`,
);
await chmod(fake, 0o755);
await routes.get('/api/dsh-beauty/save').fetch(new Request(saveUrl, { method: 'POST', body: saveBody(upload.id) }));
const recorded = await readFile(calls, 'utf8');
assert.ok(recorded.includes(scratch), 'panel opens in the remembered folder');

// Cancelling must not write anything and must report E_CANCELLED.
await writeFile(fake, `#!/bin/sh\necho __CANCELLED__\n`);
await chmod(fake, 0o755);
const cancelled = await routes.get('/api/dsh-beauty/save').fetch(
  new Request(saveUrl, { method: 'POST', body: saveBody(upload.id) }),
);
assert.equal(cancelled.status, 409);
const payload = await cancelled.json();
assert.equal(payload.error.code, 'E_CANCELLED');

process.env.PATH = originalPath;
await rm(scratch, { recursive: true, force: true });
console.log('save smoke: ok');
