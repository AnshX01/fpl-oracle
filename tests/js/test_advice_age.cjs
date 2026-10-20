const fs = require('fs'), vm = require('vm'), assert = require('assert');
let config;
vm.runInNewContext(fs.readFileSync('web/static/js/app.js', 'utf8'), {
  Vue: { createApp(c) { config = c; return { mount() { return {}; } }; } },
  window: {}, console, setTimeout, clearTimeout, location: { hash: '' },
});
const app = config.data();
Object.assign(app, config.methods);
const view = new Proxy(app, { get: (t, k) => (k in t ? t[k] : config.computed[k] && config.computed[k].call(view)) });
const minutes = m => new Date(Date.now() - m * 60000).toISOString();
const card = { gameweek: 8, transfers: {}, chip: {}, captain: {}, vice_captain: {} };

// A run published 61 minutes after its data time, published just now: no age warning.
Object.assign(app, { now: Date.now(), advice: { decisionCard: card, data_as_of: minutes(61) }, advicePublishedAt: minutes(0) });
assert.equal(view.banners.some(b => b.key === 'age'), false, 'fresh publish shows no age warning');
// Published 4 hours ago: warning appears (soft limit is 3 hours).
app.advicePublishedAt = minutes(240);
assert.equal(view.banners.some(b => b.key === 'age'), true, 'old publish shows the age warning');
// No published_at: falls back to the data time.
app.advicePublishedAt = null; app.advice.data_as_of = minutes(240);
assert.equal(view.banners.some(b => b.key === 'age'), true, 'falls back to data_as_of');
// Advice being replaced cannot be confirmed.
app.advice.data_as_of = minutes(1); app.adviceFrom = minutes(5);
assert.equal(view.adviceCurrent, false);
// Number helpers and meta never print a bare separator.
assert.equal(app.pts('7.25'), '7.3'); assert.equal(app.pts(5), '5.0');
assert.equal(app.joinMeta(null, 'ARS', ''), 'ARS');
console.log('Advice age from publish time, confirm blocked while updating, pts and meta helpers: passed');
