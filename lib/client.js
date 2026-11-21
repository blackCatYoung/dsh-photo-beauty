/**
 * dsh-photo-beauty — Web half.
 *
 * Registers one global main panel (the workbench iframe) plus its sidebar row.
 * The workbench itself is served by the Host at /api/dsh-beauty/workbench, so
 * this file stays tiny: no bundler, no third-party imports, only React from the
 * Host module loader.
 */

window.__ModuleLoader__.load({
  id: 'dsh-photo-beauty',

  factory(require) {
    const React = require('react');
    const h = React.createElement;

    const PANEL_ID = 'dsh-photo-beauty';
    const SIDEBAR_ORDER = 34;
    const WORKBENCH_URL = '/api/dsh-beauty/workbench';

    /** Camera-with-sparkle outline; follows the row's currentColor only. */
    function SidebarBeautyIcon(props) {
      let size = props && typeof props.size === 'number' && isFinite(props.size) ? props.size : 16;
      if (size < 12) size = 12;
      if (size > 24) size = 24;
      return h(
        'svg',
        {
          width: size,
          height: size,
          viewBox: '0 0 16 16',
          fill: 'none',
          xmlns: 'http://www.w3.org/2000/svg',
          'aria-hidden': 'true',
          focusable: 'false',
          stroke: 'currentColor',
          strokeWidth: 1,
          strokeLinecap: 'round',
          strokeLinejoin: 'round',
        },
        h('rect', { x: 1.4, y: 4.2, width: 10.2, height: 8.4, rx: 2.2 }),
        h('circle', { cx: 6.5, cy: 8.4, r: 2.1 }),
        h('path', { d: 'M12.6 6.3l1.4-.9c.4-.25.9.03.9.5v5c0 .47-.5.75-.9.5l-1.4-.9' }),
        h('path', { d: 'M13.3 1.6l.45 1.05 1.05.45-1.05.45-.45 1.05-.45-1.05L11.8 3.1l1.05-.45z' })
      );
    }

    /**
     * The workbench page reads the Host palette itself (same origin), so the
     * iframe must not pin a color scheme: leaving it out also lets the embedded
     * document inherit the Host's preferred scheme before it paints.
     */
    function WorkbenchPanel() {
      return h('iframe', {
        src: WORKBENCH_URL,
        title: '美颜工坊',
        style: {
          width: '100%',
          height: '100%',
          border: '0',
          display: 'block',
          background: 'transparent',
        },
        allow: 'clipboard-read; clipboard-write',
      });
    }

    function report(code, error) {
      try {
        if (typeof console !== 'undefined' && console.warn) console.warn('[dsh-photo-beauty] ' + code, error);
      } catch {
        /* diagnostics never break registration */
      }
    }

    function apply(ctx) {
      try {
        ctx.effect(
          function beautyPanelResources() {
            const disposers = [];

            /** Register once the host declares the seat, not before. */
            function seat(slotName, register) {
              try {
                const dispose = ctx.slots.inject(slotName, function occupy() {
                  try {
                    return register();
                  } catch (error) {
                    report(slotName + '-register-failed', error);
                    return function noopDispose() {};
                  }
                });
                if (typeof dispose === 'function') disposers.push(dispose);
              } catch (error) {
                report(slotName + '-inject-failed', error);
              }
            }

            seat('main', function () {
              return ctx.slots.register({ name: 'main', key: PANEL_ID }, WorkbenchPanel);
            });

            seat('sidebar.panellist', function () {
              return ctx.slots.register(
                {
                  name: 'sidebar.panellist',
                  id: PANEL_ID,
                  order: SIDEBAR_ORDER,
                  label: function () {
                    return '美颜工坊';
                  },
                },
                SidebarBeautyIcon
              );
            });

            return function disposeBeautyPanel() {
              for (const dispose of disposers.splice(0).reverse()) {
                try {
                  dispose();
                } catch (error) {
                  report('panel-cleanup-failed', error);
                }
              }
            };
          },
          'dsh-photo-beauty: workbench panel'
        );
      } catch (error) {
        report('effect-failed', error);
      }
    }

    return { inject: ['slots'], apply };
  },
});
