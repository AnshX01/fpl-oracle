/**
 * FPL Oracle: single page app (Vue 3). Everything on screen comes from ONE published advice result
 * (POST /api/advice/start, then GET /api/advice/status/{id}): the decision card, the squad and the data basis.
 * No screen recalculates a plan on its own.
 */

const { createApp } = Vue;

const POSITIONS = [['GKP', 'Goalkeepers'], ['DEF', 'Defenders'], ['MID', 'Midfielders'], ['FWD', 'Forwards']];
const STATUS_TEXT = { d: 'Doubtful', i: 'Injured', s: 'Suspended', u: 'Unavailable', n: 'Not available' };
const MAX_ADVICE_HOURS = 6;
const SOFT_ADVICE_MINUTES = 60;

const app = createApp({
  data() {
    return {
      // Draft of keep / don't-buy calls: edited freely, applied (and recalculated) only on request.
      draftChoices: [],
      theme: 'dark',
      activeTab: 'week',
      mainTabs: [
        { key: 'week', label: 'This week' },
        { key: 'team', label: 'My team' },
        { key: 'league', label: 'League' }
      ],
      now: Date.now(),
      ticker: null,

      // The one published advice result and how it is going
      advice: null,
      adviceBusy: false,
      adviceStage: '',
      adviceStartedAt: null,
      adviceError: '',
      adviceFixable: null,
      adviceFrom: null,
      adviceNeedsRefresh: false,
      generation: 0,
      refreshPromise: null,
      gameState: {},

      // Overlays and feedback
      showChatDrawer: false,
      showPanicModal: false,
      showConfirmTeam: false,
      toasts: [],
      toastSeq: 0,
      busy: {},

      // "Did something different" / done flow
      followedLocked: false,
      deadlineTimer: null,
      confirmTeamData: null,
      confirmForm: {},
      confirmRows: [],
      confirmDelta: { out: '', in: '' },
      confirmNeedsPrice: false,
      confirmSellPrice: '',
      confirmTeamError: '',
      confirmTeamSaving: false,

      // A player is ruled out
      panicQuery: '',
      panicDuration: '',
      panicResolving: false,
      panicResult: null,

      // Chat
      chatMessages: [],
      userInput: '',
      isChatLoading: false,
      suggestedPrompts: [
        'Why this captain?',
        'Why this chip decision?',
        'What if one of my starters is injured?',
        'How do I catch the leader of my league?'
      ]
    };
  },

  computed: {
    isDark() { return this.theme === 'dark'; },
    savedChoices() {
      const c = (this.card && this.card.your_choices) || {};
      return [...(c.keep || []).map(p => ({ ...p, kind: 'keep' })), ...(c.avoid || []).map(p => ({ ...p, kind: 'avoid' }))];
    },
    choicesChanged() {
      const key = list => list.map(p => p.kind + p.element).sort().join(',');
      return key(this.draftChoices) !== key(this.savedChoices);
    },
    card() {
      const c = this.advice && this.advice.decisionCard;
      return c && c.transfers && c.chip && c.captain && c.vice_captain ? c : null;
    },
    squad() { return (this.advice && this.advice.squadData) || {}; },
    standing() { return (this.card && this.card.rivals && this.card.rivals.standing) || null; },

    // ---- time and freshness ----
    deadlineTime() { return (this.card && this.card.deadline && this.card.deadline.deadline_time) || this.gameState.deadline_time || null; },
    deadlineGw() { return this.card ? this.card.gameweek : this.gameState.next_gw; },
    secondsToDeadline() {
      const t = Date.parse(this.deadlineTime || '');
      return Number.isFinite(t) ? (t - this.now) / 1000 : null;
    },
    deadlinePassed() { return this.secondsToDeadline !== null && this.secondsToDeadline <= 0; },
    countdownText() {
      const s = this.secondsToDeadline;
      if (s === null) return '';
      if (s <= 0) return 'Deadline passed';
      const when = new Date(this.deadlineTime).toLocaleString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
      return `Deadline in ${this.formatDuration(s)} (${when})`;
    },
    deadlineLine() {
      const s = this.secondsToDeadline;
      const gw = this.deadlineGw ? `GW${this.deadlineGw} ` : '';
      if (s === null) return gw ? gw.trim() : '';
      return s <= 0 ? `${gw}deadline passed` : `${gw}deadline in ${this.formatDuration(s)}`;
    },
    dataAsOf() { return this.advice && Date.parse(this.advice.data_as_of || '') ? Date.parse(this.advice.data_as_of) : null; },
    adviceAgeMinutes() { return this.dataAsOf ? Math.max(0, (this.now - this.dataAsOf) / 60000) : 0; },
    basis() { return (this.advice && this.advice.dataBasis) || {}; },
    dataStale() {
      return Boolean(this.card && (this.card.is_stale || this.squad.is_stale || this.basis.warning));
    },
    formBehind() {
      const b = this.basis;
      return Number.isInteger(b.latest_finalized_gameweek) && Number.isInteger(b.form_through_gameweek) &&
        b.latest_finalized_gameweek > b.form_through_gameweek;
    },
    adviceCurrent() {
      return Boolean(this.card) && !this.adviceError && !this.deadlinePassed && !this.dataStale && this.adviceAgeMinutes < MAX_ADVICE_HOURS * 60;
    },
    adviceFromText() {
      return this.adviceFrom ? new Date(this.adviceFrom).toLocaleString('en-GB', { weekday: 'short', hour: '2-digit', minute: '2-digit' }) : '';
    },
    basisText() {
      const b = this.basis, f = b.form_through_gameweek, m = b.model_trained_through_gameweek;
      let text = Number.isInteger(f) && Number.isInteger(m)
        ? `Based on form through gameweek ${f} and a prediction model trained through gameweek ${m}.`
        : Number.isInteger(f) ? `Based on form through gameweek ${f}.` : 'Based on the latest data available.';
      if (this.dataAsOf) {
        text += ` Data from ${new Date(this.dataAsOf).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}.`;
      }
      return text;
    },
    liveNote() {
      const cur = this.gameState.current_gw, done = this.basis.latest_finalized_gameweek;
      return this.gameState.is_live && Number.isInteger(cur) && Number.isInteger(done) && cur > done
        ? `Gameweek ${cur} is being played now. Its results count once it finishes.` : '';
    },
    leagueAsOf() {
      const gw = this.card && this.card.rivals.standings_after_gameweek;
      return gw ? `Points after gameweek ${gw}${this.liveNote ? ` (gameweek ${this.gameState.current_gw} still being played)` : ''}` : '';
    },
    elapsedText() {
      return this.adviceStartedAt ? this.formatDuration((this.now - this.adviceStartedAt) / 1000, true) : '';
    },
    banners() {
      const out = [];
      if (this.followedLocked || !this.card) return out;
      const gw = this.card.gameweek;
      if (this.deadlinePassed) {
        out.push({ key: 'deadline', tone: 'danger', icon: 'clock', refresh: true, title: `The gameweek ${gw} deadline has passed`,
          text: 'This advice is out of date. Refresh to get advice for the next gameweek.' });
        return out;
      }
      if (this.dataStale) {
        out.push({ key: 'stale', tone: 'warn', icon: 'triangle-alert', refresh: true, title: 'Some data could not be refreshed',
          text: 'These numbers may be out of date. Refresh to try again before you act.' });
      } else if (this.adviceAgeMinutes >= MAX_ADVICE_HOURS * 60) {
        out.push({ key: 'old', tone: 'danger', icon: 'clock', refresh: true, title: 'This advice is out of date',
          text: `It was prepared ${this.formatDuration(this.adviceAgeMinutes * 60)} ago. Refresh before you act.` });
      } else if (this.adviceAgeMinutes >= SOFT_ADVICE_MINUTES) {
        out.push({ key: 'age', tone: 'warn', icon: 'clock', refresh: true, title: `This advice is ${this.formatDuration(this.adviceAgeMinutes * 60)} old`,
          text: 'Injuries and team news may have changed. Refresh to check.' });
      }
      if (this.formBehind) {
        out.push({ key: 'form', tone: 'warn', icon: 'info', refresh: true, title: 'Newest results are missing',
          text: `Gameweek ${this.basis.latest_finalized_gameweek} results are not in the form data yet. Refresh to include them.` });
      }
      return out;
    },

    // ---- the decision card ----
    isChipSquadChange() {
      const c = this.card;
      return Boolean(c) && ['wildcard', 'freehit'].includes(c.chip.chip_name) && c.transfers.in.length > 0;
    },
    chipSquadNote() {
      const c = this.card, n = c.transfers.in.length;
      const base = `${n} of your 15 players change; ${15 - n} stay.`;
      return c.chip.chip_name === 'freehit' ? `${base} This team is for this gameweek only; your current team comes back next week.` : base;
    },
    squadGroups() {
      const c = this.card;
      return POSITIONS.map(([pos, label]) => ({
        pos, label,
        players: c.transfers.in.filter(p => p.position === pos).map(p => ({ ...p, price: p.cost })),
        out: c.transfers.out.filter(p => p.position === pos)
      })).filter(g => g.players.length || g.out.length);
    },
    xiLines() {
      return POSITIONS.map(([pos]) => ({ pos, players: this.card.xi.filter(p => p.position === pos) })).filter(l => l.players.length);
    },
    benchLines() {
      let sub = 0;
      return [...this.card.bench].sort((a, b) => a.bench_order - b.bench_order)
        .map(p => ({ ...p, label: p.position === 'GKP' ? 'Goalkeeper' : `Sub ${++sub}` }));
    },

    // ---- the My team screen ----
    detailById() {
      const details = {};
      [...(this.squad.starters || []), ...(this.squad.bench || [])].forEach(p => { details[p.element] = p; });
      return details;
    },
    teamRows() {
      const row = p => {
        const d = this.detailById[p.element] || {};
        return { ...p, price: d.now_cost, fixtures: (d.next_fixtures || []).slice(0, 3), ...this.availability(d) };
      };
      const order = POSITIONS.map(([pos]) => pos);
      const starters = [...this.card.xi].sort((a, b) => order.indexOf(a.position) - order.indexOf(b.position));
      return { starters: starters.map(row), bench: this.benchLines.map(row) };
    },
    flaggedPlayers() {
      return [...this.teamRows.starters, ...this.teamRows.bench].filter(p => p.flag);
    }
  },

  methods: {
    // ---- helpers ----
    formatDuration(seconds, withSeconds = false) {
      const s = Math.max(0, Math.floor(seconds));
      const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
      if (d > 0) return `${d}d ${h}h`;
      if (h > 0) return `${h}h ${m}m`;
      return withSeconds ? `${m}m ${s % 60}s` : `${m}m`;
    },
    signed(value) { return `${value >= 0 ? '+' : ''}${Number(value).toFixed(1)}`; },
    priceText(p) { return p.price != null ? `£${Number(p.price).toFixed(1)}m` : ''; },
    playerName(id) {
      const p = ((this.confirmTeamData && this.confirmTeamData.players) || []).find(x => x.element === Number(id));
      return p ? p.name : 'Unknown player';
    },
    availability(player) {
      const chance = player.chance_of_playing;
      const known = Number.isFinite(chance);
      const flagged = player.status && player.status !== 'a' || (known && chance < 100) || player.news_quote;
      if (!flagged) return { flag: '', tone: 'ok', news: '' };
      const text = known ? `${chance}% chance` : (STATUS_TEXT[player.status] || 'Check news');
      return { flag: text, tone: 'warn', news: player.news || player.news_quote || '' };
    },
    renderMarkdown(text) {
      if (!text) return '';
      const escapeText = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      if (!window.DOMPurify) return '<pre>' + escapeText(text) + '</pre>';
      try {
        return DOMPurify.sanitize(window.marked ? marked.parse(text) : '<pre>' + escapeText(text) + '</pre>');
      } catch (e) {
        return '<pre>' + escapeText(text) + '</pre>';
      }
    },
    triggerToast(msg, tone = 'ok') {
      const id = ++this.toastSeq;
      this.toasts = [...this.toasts.slice(-2), { id, message: msg, tone }];
      setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, 3800);
    },
    async guard(key, fn) {
      if (this.busy[key]) return;
      this.busy = { ...this.busy, [key]: true };
      try { return await fn(); } finally { this.busy = { ...this.busy, [key]: false }; }
    },

    // ---- theme and route ----
    initTheme() {
      let saved = null;
      try { saved = localStorage.getItem('fpl_oracle_theme'); } catch (e) { /* storage may be blocked */ }
      this.theme = saved === 'light' ? 'light' : 'dark';
      this.applyTheme();
    },
    toggleTheme() {
      this.theme = this.isDark ? 'light' : 'dark';
      try { localStorage.setItem('fpl_oracle_theme', this.theme); } catch (e) { /* storage may be blocked */ }
      this.applyTheme();
    },
    applyTheme() {
      document.documentElement.classList.toggle('light', this.theme === 'light');
      document.documentElement.classList.toggle('dark', this.theme === 'dark');
    },
    restoreRoute() {
      const key = location.hash.slice(1);
      this.activeTab = this.mainTabs.some(t => t.key === key) ? key : 'week';
    },

    // ---- the one published advice ----
    async refreshAll() {
      if (this.refreshPromise) return this.refreshPromise;
      this.refreshPromise = this.loadAdvice();
      try { await this.refreshPromise; } finally { this.refreshPromise = null; }
    },
    async loadAdvice() {
      const generation = ++this.generation;
      let job = null;
      Object.assign(this, { advice: null, adviceError: '', adviceFixable: null, adviceFrom: null, adviceNeedsRefresh: false, panicResult: null, adviceBusy: true,
        adviceStage: 'Getting started', adviceStartedAt: Date.now() });
      try {
        // Ask FPL for the latest picks, bank and transfers before the advice reads them.
        await fetch('/api/squad/basic?refresh=true').catch(() => null);
        const game = await fetch('/api/game-state').then(r => (r.ok ? r.json() : {})).catch(() => ({}));
        if (generation !== this.generation) return;
        this.gameState = game;
        const confirmation = await fetch('/api/team/confirmation');
        if (confirmation.ok) {
          const data = await confirmation.json();
          this.confirmTeamData = data;
          this.followedLocked = Boolean(data.locked);
          if (this.followedLocked) {
            this.adviceBusy = false;
            this.scheduleDeadlineRefresh(data.state && data.state.deadline_utc);
            return;
          }
        }
        job = await (await fetch('/api/advice/start', { method: 'POST' })).json();
        if (!job.id) throw new Error(job.reason || 'The advice could not start');
        while (job.status === 'calculating') {
          this.adviceStage = job.stage || this.adviceStage;
          // The last advice for this team (until its deadline) stays visible, labelled, while it updates.
          if (!this.advice && job.previous_result && job.previous_result.decisionCard && job.previous_result.squadData) {
            this.advice = job.previous_result;
            this.adviceFrom = job.previous_published_at || null;
          }
          await new Promise(resolve => setTimeout(resolve, 1500));
          if (generation !== this.generation) return;
          const response = await fetch(`/api/advice/status/${job.id}`);
          if (!response.ok) throw new Error('The advice could not be read');
          job = await response.json();
        }
        if (generation !== this.generation) return;
        if (job.status === 'refresh_required') { this.adviceNeedsRefresh = true; return; }
        const result = job.result;
        if (job.status !== 'ready' || !result || !result.decisionCard || !result.squadData) throw new Error(job.error || 'The advice stopped early');
        this.advice = result;
        this.adviceFrom = null;
      } catch (error) {
        if (generation !== this.generation) return;
        // A failed update must not blank the screen while an earlier valid advice exists: keep showing it, labelled.
        const earlier = job && job.previous_result && job.previous_result.decisionCard && job.previous_result.squadData;
        this.advice = earlier ? job.previous_result : null;
        this.adviceFrom = earlier ? job.previous_published_at || null : null;
        this.adviceFixable = (job && job.fixable) || null;
        this.adviceError = (job && job.status === 'failed' && job.error && !job.error.includes('application log'))
          ? job.error
          : 'We could not prepare your advice. Press Refresh to try again.';
      } finally {
        if (generation === this.generation) this.adviceBusy = false;
      }
    },
    choiceOf(player) {
      const found = this.draftChoices.find(p => p.element === player.element);
      return found ? found.kind : null;
    },
    toggleChoice(player, kind) {
      const rest = this.draftChoices.filter(p => p.element !== player.element);
      this.draftChoices = this.choiceOf(player) === kind ? rest : [...rest, { element: player.element, web_name: player.web_name, kind }];
    },
    discardChoices() { this.draftChoices = this.savedChoices.map(p => ({ ...p })); },
    async applyChoices() {
      const ids = kind => this.draftChoices.filter(p => p.kind === kind).map(p => p.element);
      await fetch('/api/profile', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keep_ids: ids('keep'), avoid_ids: ids('avoid') }) });
      this.refreshAll(true);
    },
    async useOfficialNumbers() {
      // Drop saved bank / free-transfer overrides so the advice uses FPL's own figures.
      await fetch('/api/profile', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bank: null, free_transfers: null }) });
      this.refreshAll(true);
    },
    scheduleDeadlineRefresh(deadlineUtc) {
      if (this.deadlineTimer) clearTimeout(this.deadlineTimer);
      const deadline = Date.parse(deadlineUtc || '');
      if (Number.isFinite(deadline) && deadline > Date.now()) {
        this.deadlineTimer = setTimeout(() => this.refreshAll(), Math.min(deadline - Date.now() + 1000, 2147483647));
      }
    },

    // ---- telling the app what you did ----
    async openConfirmTeam() {
      this.showConfirmTeam = true;
      this.confirmTeamError = '';
      try {
        const r = await fetch('/api/team/confirmation');
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'Your current team is unavailable');
        this.confirmTeamData = data;
        const state = data.state, saved = data.saved, rec = data.recommendation;
        // Start from the advice, so the form agrees with the This week screen. Once moves are saved for this
        // gameweek, start from the saved team instead.
        const advised = rec && rec.gameweek === state.target_gw && !(saved && saved.gameweek === state.target_gw) ? rec : null;
        const squad = state.squad.map(p => ({ element: p.element, purchase: p.purchase_price / 10, selling: p.selling_price / 10 }));
        const kept = advised ? squad.filter(r => !advised.outs.includes(r.element)) : squad;
        this.confirmRows = advised
          ? [...kept, ...advised.ins.map(id => { const price = advised.buy_prices[id] / 10; return { element: id, purchase: price, selling: price }; })]
          : squad;
        const current = state.squad.find(p => p.is_captain), vice = state.squad.find(p => p.is_vice_captain);
        this.confirmForm = {
          gameweek: state.target_gw, bank: (advised ? advised.remaining_bank : state.bank_tenths) / 10,
          free_transfers: advised && !['wildcard', 'freehit'].includes(advised.chip) ? Math.max(0, state.free_transfers - advised.ins.length) : state.free_transfers,
          available_chips: state.target_gw <= 19 ? state.chips_remaining_set_1 : state.chips_remaining_set_2,
          active_chip: (advised ? advised.chip : state.active_chip) || '',
          hit_cost: advised ? advised.hit_cost : saved && saved.gameweek === state.target_gw ? saved.hit_cost : 0,
          captain: (advised ? advised.captain : current && current.element) || '',
          vice_captain: (advised ? advised.vice_captain : vice && vice.element) || '',
          bench: advised ? advised.bench : state.squad.filter(p => !p.is_starter).sort((a, b) => a.bench_order - b.bench_order).map(p => p.element)
        };
      } catch (e) { this.confirmTeamError = e.message; }
    },
    async undoFollowedAdvice() {
      try {
        const r = await fetch('/api/team/undo', { method: 'POST' });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'Undo failed');
        this.followedLocked = false;
        await this.refreshAll();
      } catch (e) { this.triggerToast(e.message, 'warn'); }
    },
    async confirmFollowedAdvice() {
      if (!this.adviceCurrent || this.confirmTeamSaving) return;
      this.confirmTeamSaving = true;
      try {
        const r = await fetch('/api/team/followed', { method: 'POST' });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'Could not save your moves');
        this.triggerToast('Saved');
        await this.refreshAll();
      } catch (e) { this.triggerToast(e.message, 'warn'); } finally { this.confirmTeamSaving = false; }
    },
    addConfirmedMove() {
      const out = Number(this.confirmDelta.out), incoming = Number(this.confirmDelta.in);
      const row = this.confirmRows.find(r => Number(r.element) === out);
      const player = ((this.confirmTeamData && this.confirmTeamData.players) || []).find(p => p.element === incoming);
      if (!row || !player || this.confirmRows.some(r => Number(r.element) === incoming)) {
        this.confirmTeamError = 'Choose a player you own to go out and a different player to come in.';
        return;
      }
      const original = this.confirmTeamData.state.squad.find(p => p.element === out);
      if (original && original.price_provenance === 'market_estimate') {
        const price = Number(this.confirmSellPrice);
        if (!(price > 0)) {
          this.confirmNeedsPrice = true;
          this.confirmTeamError = 'We are not sure what you would get for that player. Enter his selling price from FPL below.';
          return;
        }
        row.selling = price;
      }
      const form = this.confirmForm;
      form.bank = Number((Number(form.bank) + Number(row.selling) - player.price / 10).toFixed(1));
      // Swapping out a player the advice added is not an extra transfer.
      if (original && !['wildcard', 'freehit'].includes(form.active_chip)) {
        if (Number(form.free_transfers) > 0) form.free_transfers--; else form.hit_cost = Number(form.hit_cost) + 4;
      }
      form.bench = form.bench.map(id => (id === out ? incoming : id));
      if (Number(form.captain) === out) form.captain = incoming;
      if (Number(form.vice_captain) === out) form.vice_captain = incoming;
      row.element = incoming; row.purchase = player.price / 10; row.selling = player.price / 10;
      this.confirmDelta = { out: '', in: '' };
      this.confirmNeedsPrice = false;
      this.confirmSellPrice = '';
      this.confirmTeamError = '';
    },
    async saveConfirmTeam() {
      if (this.confirmTeamSaving) return;
      this.confirmTeamSaving = true;
      this.confirmTeamError = '';
      try {
        const f = this.confirmForm, rows = this.confirmRows;
        const payload = {
          gameweek: f.gameweek, player_ids: rows.map(p => Number(p.element)),
          bank_tenths: Math.round(Number(f.bank) * 10), free_transfers: Number(f.free_transfers),
          available_chips: f.available_chips.filter(c => c !== f.active_chip), active_chip: f.active_chip || null,
          captain: Number(f.captain) || null, vice_captain: Number(f.vice_captain) || null, hit_cost: Number(f.hit_cost),
          bench: (f.bench || []).filter(id => rows.some(p => Number(p.element) === id)),
          selling_prices: Object.fromEntries(rows.map(p => [p.element, Math.round(Number(p.selling) * 10)])),
          purchase_prices: Object.fromEntries(rows.map(p => [p.element, Math.round(Number(p.purchase) * 10)]))
        };
        const r = await fetch('/api/team/confirmation', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'Could not save');
        this.showConfirmTeam = false;
        this.triggerToast('Saved');
        await this.refreshAll();
      } catch (e) { this.confirmTeamError = e.message; } finally { this.confirmTeamSaving = false; }
    },
    resetFinancialOverrides() {
      return this.guard('reset', async () => {
        const res = await fetch('/api/profile', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bank: null, free_transfers: null }) });
        if (!res.ok) { this.triggerToast('Could not reset your bank and transfers.', 'warn'); return; }
        this.triggerToast('Using the official figures');
        await this.refreshAll();
      });
    },

    // ---- a player is ruled out ----
    openPanic() {
      this.showPanicModal = true;
      this.panicResult = null;
    },
    async executePanicReoptimize() {
      if (!this.panicQuery.trim() || this.panicResolving) return;
      this.panicResolving = true;
      try {
        const res = await fetch('/api/contingency/panic', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: this.panicQuery, duration: this.panicDuration })
        });
        const result = await res.json();
        if (!res.ok) throw new Error(result.detail || 'The check failed');
        this.panicResult = result;
      } catch (e) {
        this.panicResult = { status: 'error', recommendation: e.message || 'The check failed. Try again.' };
      } finally {
        this.panicResolving = false;
      }
    },

    // ---- chat ----
    async sendMessage() {
      if (!this.userInput.trim() || this.isChatLoading) return;
      const q = this.userInput.trim();
      this.chatMessages.push({ role: 'user', content: q });
      this.userInput = '';
      this.isChatLoading = true;
      try {
        const res = await fetch('/api/chat', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          // One server-side conversation per gameweek so old advice never leaks into new turns.
          body: JSON.stringify({ message: q, session_id: this.card ? `gw${this.card.gameweek}` : 'default' })
        });
        const data = await res.json();
        const body = res.ok && typeof data.response === 'string' && data.response.trim()
          ? data.response : (data.error || 'No answer came back. Try again.');
        this.chatMessages.push({ role: 'assistant', content: body });
      } catch (e) {
        this.chatMessages.push({ role: 'assistant', content: 'Could not get an answer. Try again.' });
      } finally {
        this.isChatLoading = false;
      }
    },
    sendPrompt(p) {
      this.userInput = p;
      this.sendMessage();
    }
  },

  watch: {
    // A new or saved advice brings its own choices; start the draft from them.
    savedChoices(value) { if (!this.choicesChanged || !this.draftChoices.length) this.draftChoices = value.map(p => ({ ...p })); },
    activeTab(value) { if (location.hash !== '#' + value) location.hash = value; },
    showChatDrawer(v) { if (v) { this.showPanicModal = false; this.showConfirmTeam = false; } },
    showPanicModal(v) { if (v) { this.showChatDrawer = false; this.showConfirmTeam = false; } },
    showConfirmTeam(v) { if (v) { this.showChatDrawer = false; this.showPanicModal = false; } }
  },

  beforeUnmount() {
    if (this.deadlineTimer) clearTimeout(this.deadlineTimer);
    if (this.ticker) clearInterval(this.ticker);
    if (this.routeListener) window.removeEventListener('hashchange', this.routeListener);
  },
  mounted() {
    this.initTheme();
    this.restoreRoute();
    this.routeListener = () => this.restoreRoute();
    window.addEventListener('hashchange', this.routeListener);
    this.ticker = setInterval(() => { this.now = Date.now(); }, 1000);
    this.refreshAll();
  }
});

if (window.FPLUI) window.FPLUI.register(app);
window.__fpl_vm__ = app.mount('#app');
