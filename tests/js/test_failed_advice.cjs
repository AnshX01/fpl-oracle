const fs = require('fs'), vm = require('vm'), assert = require('assert');
let config;
const previous = { decisionCard: { transfers: {}, chip: {}, captain: { web_name: 'Saka' }, vice_captain: {} }, squadData: {} };
const failed = { id: 'job', status: 'failed', error: 'FPL is not answering', previous_result: previous, previous_published_at: '2026-10-08T12:00:00Z' };
async function fetch(path) {
  if (path === '/api/team/confirmation') return { ok: true, json: async () => ({ locked: false }) };
  if (path === '/api/advice/start') return { ok: true, json: async () => failed };
  if (path.startsWith('/api/game-state')) return { ok: true, json: async () => ({}) };
  return { ok: true, json: async () => ({}) };
}
vm.runInNewContext(fs.readFileSync('web/static/js/app.js', 'utf8'), {
  Vue: { createApp(c) { config = c; return { mount() { return {}; } }; } },
  window: {}, console, setTimeout, clearTimeout, AbortController, fetch, location: { hash: '' },
});
(async () => {
  const app = config.data();
  Object.assign(app, config.methods);
  await app.loadAdvice();
  assert.equal(app.advice.decisionCard.captain.web_name, 'Saka', 'earlier advice stays on screen');
  assert.equal(app.adviceError, 'FPL is not answering');
  const view = new Proxy(app, { get: (t, k) => (k in t ? t[k] : config.computed[k] && config.computed[k].call(view)) });
  assert.equal(config.computed.adviceCurrent.call(view), false, 'stale advice cannot be confirmed');
  app.adviceError = '';
  app.adviceFrom = null;
  assert.equal(config.computed.adviceCurrent.call(view), true, 'the same advice without a failure can be confirmed');
  console.log('Failed update keeps the earlier advice, shows the reason and blocks Done: passed');
})().catch(e => { console.error(e); process.exitCode = 1; });
