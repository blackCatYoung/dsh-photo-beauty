// Client smoke test: loads the web half the way the DSH module loader does,
// with a fake React and a fake slot registry, and checks both seats register.
// Usage: node test/client-smoke.mjs
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const source = await readFile(new URL('../lib/client.js', import.meta.url), 'utf8');

let definition = null;
const window = {
  __ModuleLoader__: {
    load(value) {
      definition = value;
    },
  },
};
new Function('window', source)(window);
assert.ok(definition, 'client bundle calls __ModuleLoader__.load');
assert.equal(definition.id, 'dsh-photo-beauty');

const React = {
  createElement(type, props, ...children) {
    return { type, props, children };
  },
};
const module = definition.factory((name) => {
  if (name === 'react') return React;
  throw new Error('unexpected module ' + name);
});
assert.equal(typeof module.apply, 'function');
// cordis refuses ctx.slots access unless the module declares the service.
assert.deepEqual(module.inject, ['slots']);

const registered = [];
const injected = [];
const ctx = {
  effect(fn) {
    fn();
  },
  slots: {
    inject(name, occupy) {
      injected.push(name);
      const dispose = occupy();
      return () => {
        if (typeof dispose === 'function') dispose();
      };
    },
    register(options, component) {
      registered.push({ options, component });
      return () => registered.pop();
    },
  },
};

module.apply(ctx);
assert.deepEqual(injected, ['main', 'sidebar.panellist']);
assert.equal(registered.length, 2);
const [main, sidebar] = registered;
assert.equal(main.options.name, 'main');
assert.equal(main.options.key, 'dsh-photo-beauty');
assert.equal(typeof main.component, 'function');
const element = main.component({});
assert.equal(element.type, 'iframe');
assert.equal(element.props.src, '/api/dsh-beauty/workbench');
assert.equal(sidebar.options.name, 'sidebar.panellist');
assert.equal(sidebar.options.id, 'dsh-photo-beauty');
assert.equal(sidebar.options.order, 34);
assert.equal(sidebar.options.label(), '美颜工坊');
assert.equal(typeof sidebar.component, 'function');

// A missing seat must not poison the other one.
const attempted = [];
const partialCtx = {
  effect(fn) {
    fn();
  },
  slots: {
    inject(name, occupy) {
      attempted.push(name);
      if (name === 'main') throw new Error('no main seat');
      occupy();
      return () => {};
    },
    register() {
      return () => {};
    },
  },
};
module.apply(partialCtx);
assert.deepEqual(attempted, ['main', 'sidebar.panellist']);

console.log('client smoke: ok');
