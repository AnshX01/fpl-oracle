/**
 * FPL Oracle 2026/27 — Atlas / Council Decision Dashboard Application
 * Single Page Application powered by Vue 3, DOMPurify, Marked.js.
 * Operates 100% locally with zero external network requirement.
 */

const { createApp } = Vue;

const app = createApp({
  data() {
    return {
      // Theme
      theme: 'dark',
      servedRevisions: [],
      snapshotBuffer: null,
      refreshPromise: null,

      // Navigation: 4 Everyday Destinations
      activeTab: 'overview', // 'overview' | 'team' | 'transfers' | 'league'

      // Collapsible Expected Points Breakdown
      showXpBreakdown: false,

      // Secondary Drawers and Modals
      showSettingsModal: false,
      showChatDrawer: false,
      showPanicModal: false,
      showChecklistModal: false,
      showSquadModal: false,
      showBriefingModal: false,
      showSystemModal: false,
      showToast: false,
      toastMessage: '',

      // Telemetry & Game State
      health: {},
      gameState: {},
      systemStatus: {},

      // Profile & Overrides
      profile: {
        manager_id: null,
        target_league_id: null,
        bank: null,
        free_transfers: null,
        risk_preference: 'balanced',
        llm_provider: 'gemini'
      },

      // Core Squad & Lineup
      squadData: {},
      basicSquadData: {},
      basicSquadLoading: false,
      basicSquadError: null,
      squadLoading: false,
      squadError: null,
      simulatedOutIds: [],

      // Transfers & Contingency Plans
      contingencyPlans: {},
      plansLoading: false,
      plansError: null,
      selectedPlanKey: 'plan_a',
      contingencyMatrix: [],
      priceChanges: { rises: [], falls: [] },

      // Unified Decision Card (D1, D2, D3)
      decisionCard: null,
      decisionCardLoading: false,
      decisionCardError: null,

      // Chip Strategy & Roadmaps
      chipData: {},

      // Mini-League Intelligence
      leagueData: {},

      // Briefing & Reviews
      briefingData: {},
      reviewData: {},
      briefingSubTab: 'preview',

      // Pre-deadline Audit Checklist
      checklistItems: [],
      checklistSummary: { status: 'PENDING', passed_count: 0 },

      // Emergency Panic Re-Optimizer
      panicQuery: '',
      panicResolving: false,
      panicResult: null,

      // Manual Squad Entry
      manualSquadText: '',
      manualSquadBank: 0.0,
      manualSquadFT: 1,
      matchingInProgress: false,
      matchResult: null,
      savingSquad: false,

      // Chat Agent
      chatMessages: [],
      userInput: '',
      isChatLoading: false,
      suggestedPrompts: [
        "Who is the safest captain pick this week?",
        "Should I roll my free transfer or buy Gabriel?",
        "When is the best week for Bench Boost?",
        "What is my probability of winning my mini-league?"
      ],

      // Background Sync Pipeline
      pipelineRunning: false,
      pipelineProgress: 0,
      pipelineCurrentStage: 'Ready',
      eventSource: null
    };
  },

  computed: {
    isDark() {
      return this.theme === 'dark';
    },

    hasLoadedSquad() {
      if (this.squadLoading || this.squadError || this.squadData?.is_stale) return false;
      const s = this.squadData;
      if (!Array.isArray(s?.starters) || s.starters.length !== 11 || !Array.isArray(s?.bench) || s.bench.length !== 4) return false;
      const ids = [...s.starters, ...s.bench].map(p => p.element);
      return ids.every(Number.isInteger) && new Set(ids).size === 15 &&
        s.starters.some(p => p.element === s.captain?.element);
    },

    hasBasicSquad() {
      const s = this.basicSquadData;
      return Array.isArray(s?.starters) && s.starters.length === 11 && Array.isArray(s?.bench) && s.bench.length === 4;
    },

    flaggedSquadPlayers() {
      if (!this.hasLoadedSquad) return [];
      return [...this.squadData.starters, ...this.squadData.bench].filter(p =>
        p.status !== 'a' || (Number.isFinite(p.chance_of_playing) && p.chance_of_playing < 100) || p.news_quote);
    },

    unknownAvailabilityPlayers() {
      if (!this.hasLoadedSquad) return [];
      return [...this.squadData.starters, ...this.squadData.bench].filter(p => !Number.isFinite(p.chance_of_playing));
    },

    allSquadConfirmedAvailable() {
      return this.hasLoadedSquad && this.flaggedSquadPlayers.length === 0 && this.unknownAvailabilityPlayers.length === 0;
    },

    bankDisplay() {
      const s = this.hasLoadedSquad ? this.squadData : this.basicSquadData;
      return Number.isFinite(s?.bank_millions) && s.bank_source !== 'default' ? `£${s.bank_millions.toFixed(1)}m` : 'Unavailable';
    },

    ftDisplay() {
      const s = this.hasLoadedSquad ? this.squadData : this.basicSquadData;
      return Number.isInteger(s?.free_transfers) && s.ft_source !== 'default' ? String(s.free_transfers) : 'Unavailable';
    },

    activePlan() {
      if (this.plansLoading || this.plansError || this.contingencyPlans?.is_stale) return null;
      const p = this.contingencyPlans?.[this.selectedPlanKey] || this.contingencyPlans?.plan_a;
      if (!p || !Array.isArray(p.transfers_in) || !Array.isArray(p.transfers_out) ||
          !['ROLL_TRANSFER', '1_TRANSFER', '2_TRANSFERS', 'FREE_HIT', 'WILDCARD'].includes(p.plan_type)) return null;
      const isRoll = p.plan_type === 'ROLL_TRANSFER';
      if (isRoll ? (p.transfers_in.length !== 0 || p.transfers_out.length !== 0) :
          (p.transfers_in.length === 0 || p.transfers_in.length !== p.transfers_out.length)) return null;
      return p;
    },

    nextDecision() {
      if (this.decisionCardLoading || this.plansLoading || this.decisionCardError || this.plansError || this.squadError || this.decisionCard?.is_stale) {
        return {title: (this.decisionCardLoading || this.plansLoading) ? "Calculating recommendations..." : "Transfer recommendations unavailable",
          badge: "Unavailable", badgeClass: "text-zinc-400", gainText: "Unavailable", hitText: "Unavailable",
          bankText: this.bankDisplay, ftText: this.ftDisplay,
          reasons: [this.decisionCardError || this.plansError || this.squadError || "Waiting for current analysis."],
          caveat: "No current advice until analysis succeeds.", card: null, isUnavailable: true};
      }
      if (!this.hasLoadedSquad) {
        if (this.decisionCardLoading || this.plansLoading || this.squadLoading) {
          return {
            title: "Loading recommendations...",
            badge: "Loading",
            badgeClass: "bg-zinc-800 text-zinc-400 border border-zinc-700",
            gainText: "Evaluating",
            hitText: "--",
            bankText: "Bank unavailable",
            ftText: "FT unavailable",
            reasons: [
              "Evaluating gameweek projections and multi-GW trajectories.",
              "Solving optimal transfer and chip combinations."
            ],
            caveat: "Please wait while optimization runs.",
            card: null,
            isUnavailable: true
          };
        }
        return {
          title: "Transfer recommendations unavailable",
          badge: "Unavailable",
          badgeClass: "bg-zinc-900 text-zinc-400 border border-zinc-700",
          gainText: "Unavailable",
          hitText: "Unavailable",
          bankText: "Bank unavailable",
          ftText: "FT unavailable",
          reasons: [
            this.decisionCardError || this.plansError || this.squadError || "No verified squad data loaded.",
            "Cannot provide transfer recommendations without verified squad and fixture data."
          ],
          caveat: "Configure manager ID or upload a squad in settings to generate recommendations.",
          card: null,
          isUnavailable: true
        };
      }

      if (this.decisionCard?.status !== 'unavailable' && this.decisionCard?.transfers &&
          typeof this.decisionCard.transfers.is_roll === 'boolean' &&
          Array.isArray(this.decisionCard.transfers.in) && Array.isArray(this.decisionCard.transfers.out)) {
        const card = this.decisionCard;
        const chip = card.chip || {};
        const t = card.transfers || {};
        const isRoll = t.is_roll;


        let title = "";
        let badge = "";
        let badgeClass = "";

        if (chip.recommend) {
          title = `Deploy Chip: ${chip.chip_display_name || 'Active Chip'}`;
          badge = "Chip Deployment";
          badgeClass = "bg-amber-500/20 text-amber-400 border border-amber-500/40";
        } else if (isRoll) {
          title = `Roll Free Transfer (Bank to ${t.ft_next_gw ?? "unavailable"} FTs)`;
          badge = "Hold & Roll";
          badgeClass = "bg-blue-500/20 text-blue-400 border border-blue-500/40";
        } else {
          const inNames = (t.in || []).map(p => p.web_name).join(', ') || 'Target';
          const outNames = (t.out || []).map(p => p.web_name).join(', ') || 'Outgoing';
          title = `Transfer ${outNames} → ${inNames}`;
          badge = t.no_regret_flag ? "No-Regret Move" : "Recommended Move";
          badgeClass = "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";
        }

        const reasons = [
          card.two_line_reasoning || "Optimized trajectory across 5-GW horizon based on empirical team and player form.",
          chip.recommend ? chip.reason : `Starting XI led by captain ${card.captain?.web_name || 'Captain'} (${card.captain?.expected_points || 0.0} xP) in a ${card.formation || '3-5-2'} shape.`
        ];

        return {
          title: title,
          badge: badge,
          badgeClass: badgeClass,
          gainText: `+${(t.net_gain_vs_roll || 0.0).toFixed(1)} pts (5-GW)`,
          hitText: `${t.hit_cost ? '-' + t.hit_cost : '0'} hit pts`,
          bankText: Number.isFinite(t.bank_after) ? `£${t.bank_after.toFixed(1)}m in bank` : "Bank unavailable",
          ftText: `${t.ft_remaining ?? "unavailable"} FT left`,
          reasons: reasons,
          caveat: (card.caveats && card.caveats.length > 0) ? card.caveats[0] : "Check Friday press conference updates for confirmed starter status.",
          card: card,
          isUnavailable: false
        };
      }

      const plan = this.activePlan;
      if (!plan) {
        if (this.decisionCardLoading || this.plansLoading || this.squadLoading) {
          return {
            title: "Loading transfer recommendations...",
            badge: "Loading",
            badgeClass: "bg-zinc-800 text-zinc-400 border border-zinc-700",
            gainText: "Evaluating",
            hitText: "--",
            bankText: (this.squadData && this.squadData.bank_millions !== undefined) ? `£${this.squadData.bank_millions}m in bank` : "Bank unavailable",
            ftText: (this.squadData && this.squadData.free_transfers !== undefined) ? `${this.squadData.free_transfers} FT left` : "FT unavailable",
            reasons: [
              "Evaluating gameweek projections and multi-GW trajectories.",
              "Solving optimal transfer and chip combinations."
            ],
            caveat: "Please wait while optimization runs.",
            isUnavailable: true
          };
        }

        const errDetail = this.decisionCardError || this.plansError || this.squadError;
        return {
          title: "Transfer recommendations unavailable",
          badge: "Unavailable",
          badgeClass: "bg-zinc-900 text-zinc-400 border border-zinc-700",
          gainText: "Unavailable",
          hitText: "Unavailable",
          bankText: (this.squadData && this.squadData.bank_millions !== undefined) ? `£${this.squadData.bank_millions}m in bank` : "Bank unavailable",
          ftText: (this.squadData && this.squadData.free_transfers !== undefined) ? `${this.squadData.free_transfers} FT left` : "FT unavailable",
          reasons: [
            errDetail || "No transfer plan available. Verify squad configuration and network connection.",
            "Cannot provide transfer recommendations without verified squad and fixture data."
          ],
          caveat: "Configure manager ID or upload a squad in settings to generate recommendations.",
          isUnavailable: true
        };
      }

      const transfersOut = plan.transfers_out || [];
      const transfersIn = plan.transfers_in || [];

      if (transfersOut.length > 0 && transfersIn.length > 0) {
        const outNames = transfersOut.map(p => p.web_name).join(', ');
        const inNames = transfersIn.map(p => p.web_name).join(', ');
        const delta = plan.delta_vs_plan_a !== undefined ? plan.delta_vs_plan_a : (plan.net_expected_points || 0);

        return {
          title: `Transfer ${outNames} → ${inNames}`,
          badge: "Recommended Move",
          badgeClass: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40",
          gainText: `+${Math.abs(delta).toFixed(1)} pts projected gain`,
          hitText: `${plan.hits ? '-' + (plan.hits * 4) : '0'} hit pts`,
          bankText: Number.isFinite(plan.bank_after) ? `£${(plan.bank_after / 10).toFixed(1)}m in bank` : "Bank unavailable",
          ftText: `${plan.free_transfers_remaining || 0} FT left`,
          reasons: [
            plan.action_summary || `Targeting favorable fixture run and superior attacking form.`,
            `Maintains optimal budget allocation without requiring unnecessary point deductions.`
          ],
          caveat: `Subject to Friday press conference clearance and expected starting lineups.`
        };
      } else {
        if (!this.hasLoadedSquad) {
          return {
            title: "Transfer recommendations unavailable",
            badge: "Unavailable",
            badgeClass: "bg-zinc-900 text-zinc-400 border border-zinc-700",
            gainText: "Unavailable",
            hitText: "Unavailable",
            bankText: "Bank unavailable",
            ftText: "FT unavailable",
            reasons: [
              "Cannot recommend rolling without a verified loaded squad.",
              "Please configure your squad to generate valid recommendations."
            ],
            caveat: "No squad loaded.",
            isUnavailable: true
          };
        }
        return {
          title: "Save your free transfer",
          badge: "Roll Free Transfer",
          badgeClass: "bg-blue-500/20 text-blue-400 border border-blue-500/40",
          gainText: "0.0 pts (Roll)",
          hitText: "0 hit pts",
          bankText: `£${(this.squadData.bank_millions !== undefined ? this.squadData.bank_millions : 0.0).toFixed(1)}m in bank`,
          ftText: Number.isInteger(plan.next_banked_ft) ? `${plan.next_banked_ft} FTs next week` : "FT unavailable",
          reasons: [
            "Starting XI holds high expected output across all fixtures.",
            "Accumulating a second free transfer provides greater pivot leverage next week."
          ],
          caveat: "Ensure vice-captain is locked on an early kickoff starter as backup.",
          isUnavailable: false
        };
      }
    },

    startersExpectedPoints() {
      if (!this.hasLoadedSquad || typeof this.squadData?.starters_expected_points !== 'number') return '—';
      return (this.squadData.starters_expected_points).toFixed(1);
    },

    captainBonusPoints() {
      if (!this.hasLoadedSquad || typeof this.squadData?.captain_bonus_expected_points !== 'number') return '—';
      return (this.squadData.captain_bonus_expected_points).toFixed(1);
    },

    totalGameweekPoints() {
      if (!this.hasLoadedSquad || typeof this.squadData?.total_expected_points !== 'number') return '—';
      return (this.squadData.total_expected_points).toFixed(1);
    },

    deadlineDisplay() {
      if (!this.gameState || (!this.gameState.next_gw && !this.gameState.deadline_time && (this.gameState.seconds_to_deadline === null || this.gameState.seconds_to_deadline === undefined))) {
        return 'Deadline: Unknown';
      }
      const formatted = this.formatDeadline(this.gameState.seconds_to_deadline, this.gameState.deadline_time);
      if (formatted === 'Deadline unknown') {
        return this.gameState.next_gw ? `GW${this.gameState.next_gw} Deadline: Unknown` : 'Deadline: Unknown';
      }
      return this.gameState.next_gw ? `GW${this.gameState.next_gw} Deadline: ${formatted}` : `Deadline: ${formatted}`;
    }
  },

  watch: {
    activeTab() { this.saveSessionHistory(); },
    chatMessages: { deep: true, handler() { this.saveSessionHistory(); } }
  },

  methods: {
    publishSnapshot(field, value) {
      if (this.snapshotBuffer) this.snapshotBuffer[field] = value;
      else this[field] = value;
    },
    async fetchSnapshot(path) {
      const response = await fetch(path);
      const revision = response.headers.get('X-FPL-Revision');
      if (revision !== null) this.servedRevisions.push(revision);
      return response;
    },
    restoreSessionHistory() {
      try {
        const saved = JSON.parse(localStorage.getItem('fpl_oracle_session_v1') || 'null');
        if (!saved || saved.version !== 1) return;
        if (['overview', 'team', 'transfers', 'league'].includes(saved.activeTab)) this.activeTab = saved.activeTab;
        if (Array.isArray(saved.chatMessages)) this.chatMessages = saved.chatMessages.slice(-50).filter(m =>
          m && ['user', 'assistant'].includes(m.role) && typeof m.content === 'string').map(m =>
          ({ role: m.role, content: m.content, historical: true }));
      } catch (e) { /* Storage may be unavailable or invalid; never restore advice as current. */ }
    },
    saveSessionHistory() {
      try {
        localStorage.setItem('fpl_oracle_session_v1', JSON.stringify({ version: 1, activeTab: this.activeTab,
          chatMessages: this.chatMessages.slice(-50), savedAt: new Date().toISOString() }));
      } catch (e) { /* Read-only/private storage does not block live data. */ }
    },
    // Theme Management
    initTheme() {
      const saved = localStorage.getItem('fpl_oracle_theme');
      if (saved === 'light' || saved === 'dark') {
        this.theme = saved;
      } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
        this.theme = 'light';
      } else {
        this.theme = 'dark';
      }
      this.applyTheme();
    },

    toggleTheme() {
      this.theme = this.theme === 'dark' ? 'light' : 'dark';
      localStorage.setItem('fpl_oracle_theme', this.theme);
      this.applyTheme();
    },

    applyTheme() {
      const root = document.documentElement;
      if (this.theme === 'light') {
        root.classList.remove('dark');
        root.classList.add('light');
      } else {
        root.classList.remove('light');
        root.classList.add('dark');
      }
    },

    triggerToast(msg) {
      this.toastMessage = msg;
      this.showToast = true;
      setTimeout(() => {
        this.showToast = false;
      }, 3500);
    },

    formatDeadline(seconds, deadlineTime) {
      if ((seconds === null || seconds === undefined) && !deadlineTime) {
        return 'Deadline unknown';
      }

      let sec = seconds;
      if (deadlineTime) {
        const dt = new Date(deadlineTime).getTime();
        if (!isNaN(dt)) {
          sec = (dt - Date.now()) / 1000;
        }
      }

      if (sec === null || sec === undefined || isNaN(sec)) {
        return 'Deadline unknown';
      }

      if (sec <= 0) {
        return 'Passed';
      }

      const days = Math.floor(sec / 86400);
      const hours = Math.floor((sec % 86400) / 3600);
      const mins = Math.floor((sec % 3600) / 60);

      if (days > 0) {
        return `${days}d ${hours}h`;
      } else if (hours > 0) {
        return `${hours}h ${mins}m`;
      } else {
        return `${mins}m`;
      }
    },

    renderMarkdown(text) {
      if (!text) return '';
      try {
        const rawHtml = marked.parse(text);
        return DOMPurify.sanitize(rawHtml);
      } catch (e) {
        return DOMPurify.sanitize(text);
      }
    },

    getPositionPlayers(pos) {
      if (!this.squadData.starters) return [];
      return this.squadData.starters.filter(p => p.position === pos);
    },

    toggleSimulateOut(elemId) {
      const idx = this.simulatedOutIds.indexOf(elemId);
      if (idx >= 0) {
        this.simulatedOutIds.splice(idx, 1);
      } else {
        this.simulatedOutIds.push(elemId);
      }
    },

    // Background Synchronization Pipeline
    async triggerSyncPipeline() {
      this.squadData = {}; this.decisionCard = null; this.contingencyPlans = {};
      this.chipData = {}; this.leagueData = {}; this.briefingData = {};
      this.pipelineRunning = true;
      this.pipelineProgress = 0;
      this.pipelineCurrentStage = "Triggering data refresh...";

      try {
        const res = await fetch('/api/sync/trigger', { method: 'POST' });
        const init = await res.json();
        this.triggerToast(`Updating data (Run #${init.run_id})`);
      } catch (e) {}

      if (this.eventSource) this.eventSource.close();
      this.eventSource = new EventSource('/api/sync/stream');

      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.pipelineProgress = data.progress_pct || 0;
          this.pipelineCurrentStage = `${data.step}: ${data.message || ''}`;
          if (data.done) {
            this.pipelineRunning = false;
            this.eventSource.close();
            this.refreshAll();
            this.triggerToast(data.error ? "Analysis failed. Current advice unavailable." : "Data updated successfully!");
          }
        } catch (e) {}
      };

      this.eventSource.onerror = () => {
        this.pipelineRunning = false;
        if (this.eventSource) this.eventSource.close();
      };
    },

    // Emergency Crisis Solver (Panic Button)
    openPanicModal() {
      this.showPanicModal = true;
      this.panicResult = null;
    },

    async executePanicReoptimize() {
      if (!this.panicQuery.trim()) return;
      this.panicResolving = true;
      try {
        const res = await fetch('/api/contingency/panic', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: this.panicQuery })
        });
        this.panicResult = await res.json();
      } catch (e) {
        this.panicResult = { recommendation: "Failed to generate emergency re-optimization." };
      } finally {
        this.panicResolving = false;
      }
    },

    // Pre-deadline Checklist
    async openChecklistModal() {
      this.showChecklistModal = true;
      await this.loadChecklist();
    },

    async loadChecklist() {
      try {
        const res = await fetch('/api/contingency/checklist');
        if (res.ok) {
          const data = await res.json();
          this.checklistItems = data.checklist || [];
          const passed = this.checklistItems.filter(i => i.status === 'PASS').length;
          this.checklistSummary = {
            status: passed === this.checklistItems.length ? 'PASS' : 'WARN',
            passed_count: passed
          };
        }
      } catch (e) {}
    },

    // Data Loaders
    async loadGameState() {
      try {
        const res = await fetch('/api/game-state');
        if (res.ok) this.gameState = await res.json();
      } catch (e) {}
    },

    async loadHealth() {
      try {
        const res = await fetch('/api/health');
        if (res.ok) this.health = await res.json();
      } catch (e) {}
    },

    async loadSystemStatus() {
      try {
        const res = await fetch('/api/system/status');
        if (res.ok) this.systemStatus = await res.json();
      } catch (e) {}
    },

    async loadProfile() {
      try {
        const res = await fetch('/api/profile');
        if (res.ok) this.profile = await res.json();
      } catch (e) {}
    },

    async resetFinancialOverrides() {
      const res = await fetch('/api/profile', { method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({bank: null, free_transfers: null}) });
      if (!res.ok) { this.triggerToast('Could not reset overrides.'); return; }
      await this.refreshAll(true);
      this.triggerToast('Official bank and free transfers restored.');
    },

    async saveProfile() {
      try {
        const res = await fetch('/api/profile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({risk_preference: this.profile.risk_preference, llm_provider: this.profile.llm_provider})
        });
        if (!res.ok) throw new Error(`Settings HTTP ${res.status}`);
        this.showSettingsModal = false;
        this.triggerToast("Settings saved. Refreshing analysis...");
        await this.refreshAll();
      } catch (e) {
        alert("Failed to save profile settings.");
      }
    },

    async loadSquad() {
      this.squadLoading = true;
      this.squadError = null;
      try {
        const res = await this.fetchSnapshot('/api/squad');
        if (res.ok) {
          this.publishSnapshot('squadData', await res.json());
          const loaded = this.snapshotBuffer?.squadData || this.squadData;
          if (!Array.isArray(loaded.starters) || !Array.isArray(loaded.bench)) {
            this.squadData = {};
            this.squadError = "Malformed squad response.";
          }
        } else {
          this.squadData = {};
          this.squadError = `Squad unavailable (HTTP ${res.status})`;
        }
      } catch (e) {
        this.squadData = {};
        this.squadError = "Network error loading squad.";
      } finally {
        this.squadLoading = false;
      }
    },

    async loadContingencyPlans() {
      this.plansLoading = true;
      this.plansError = null;
      try {
        const res = await this.fetchSnapshot('/api/contingency/plans');
        if (res.ok) {
          this.publishSnapshot('contingencyPlans', await res.json());
        } else {
          this.contingencyPlans = {};
          this.plansError = `Plans unavailable (HTTP ${res.status})`;
        }
      } catch (e) {
        this.contingencyPlans = {};
        this.plansError = "Network error loading transfer plans.";
      } finally {
        this.plansLoading = false;
      }
    },

    async loadContingencyMatrix() {
      try {
        const res = await fetch('/api/contingency/matrix');
        if (res.ok) {
          const data = await res.json();
          this.contingencyMatrix = data.contingency_matrix || [];
        }
      } catch (e) {}
    },

    async loadPriceChanges() {
      try {
        const res = await fetch('/api/price-changes');
        if (res.ok) this.priceChanges = await res.json();
      } catch (e) {}
    },

    async loadChips() {
      try {
        const res = await this.fetchSnapshot('/api/chips');
        if (res.ok) this.publishSnapshot('chipData', await res.json());
      } catch (e) {}
    },

    async loadLeague() {
      try {
        const res = await this.fetchSnapshot('/api/league');
        if (res.ok) this.publishSnapshot('leagueData', await res.json());
      } catch (e) {}
    },

    async loadDecisionCard() {
      this.decisionCardLoading = true;
      this.decisionCardError = null;
      try {
        const res = await this.fetchSnapshot('/api/decision-card');
        if (res.ok) {
          this.publishSnapshot('decisionCard', await res.json());
        } else {
          this.decisionCard = null;
          this.decisionCardError = `Decision card unavailable (HTTP ${res.status})`;
        }
      } catch (e) {
        console.warn('Failed to fetch decision card:', e);
        this.decisionCard = null;
        this.decisionCardError = "Network error loading decision card.";
      } finally {
        this.decisionCardLoading = false;
      }
    },

    async exportDecisionCard() {
      try {
        const res = await fetch('/api/decision-card/export');
        if (res.ok) {
          const text = await res.text();
          await navigator.clipboard.writeText(text);
          this.triggerToast("Decision Card (Markdown) copied to clipboard!");
        } else {
          alert("Failed to export decision card.");
        }
      } catch (e) {
        console.error("Export error:", e);
        alert("Could not copy decision card to clipboard.");
      }
    },

    async loadBriefing() {
      try {
        const res = await this.fetchSnapshot('/api/briefing');
        if (res.ok) this.publishSnapshot('briefingData', await res.json());
      } catch (e) {}
    },

    async loadReview() {
      try {
        const res = await fetch('/api/review');
        if (res.ok) this.reviewData = await res.json();
      } catch (e) {}
    },

    // Chat Agent
    async sendMessage() {
      if (!this.userInput.trim() || this.isChatLoading) return;
      const q = this.userInput.trim();
      this.chatMessages.push({ role: 'user', content: q });
      this.userInput = '';
      this.isChatLoading = true;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: q })
        });
        const data = await res.json();
        this.chatMessages.push({ role: 'assistant', content: data.response });
      } catch (e) {
        this.chatMessages.push({ role: 'assistant', content: "Error communicating with local expert agent." });
      } finally {
        this.isChatLoading = false;
      }
    },

    sendPrompt(p) {
      this.userInput = p;
      this.sendMessage();
    },

    copyBriefingMarkdown() {
      const text = this.briefingSubTab === 'preview' ? (this.briefingData.markdown || '') : (this.reviewData.review_markdown || '');
      navigator.clipboard.writeText(text);
      this.triggerToast("Weekly briefing markdown copied!");
    },

    // Manual Squad
    openSquadModal() {
      this.showSquadModal = true;
      this.matchResult = null;
    },

    handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        this.manualSquadText = e.target.result;
        this.matchPastedSquad();
      };
      reader.readAsText(file);
    },

    async matchPastedSquad() {
      if (!this.manualSquadText.trim()) return;
      this.matchingInProgress = true;
      try {
        const res = await fetch('/api/squad/match', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ raw_text: this.manualSquadText })
        });
        this.matchResult = await res.json();
      } finally {
        this.matchingInProgress = false;
      }
    },

    async saveManualSquad() {
      if (!this.matchResult || !this.matchResult.is_valid_15) return;
      this.savingSquad = true;
      try {
        const playerIds = this.matchResult.matches.map(m => m.element);
        const res = await fetch('/api/squad/manual', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            player_ids: playerIds,
            bank: this.manualSquadBank,
            free_transfers: this.manualSquadFT
          })
        });
        if (res.ok) {
          this.showSquadModal = false;
          this.triggerToast("Custom squad confirmed and saved!");
          await this.refreshAll();
        }
      } finally {
        this.savingSquad = false;
      }
    },

    async loadBasicSquad(refresh = false) {
      this.basicSquadLoading = true;
      this.basicSquadError = null;
      try {
        const res = await fetch(`/api/squad/basic${refresh ? '?refresh=true' : ''}`);
        if (!res.ok) throw new Error(`Basic squad HTTP ${res.status}`);
        this.basicSquadData = await res.json();
      } catch (e) {
        this.basicSquadData = {};
        this.basicSquadError = String(e);
      } finally { this.basicSquadLoading = false; }
    },

    async refreshAll(refresh = false) {
      if (this.refreshPromise) return this.refreshPromise;
      this.refreshPromise = this.refreshSnapshot(refresh);
      try { await this.refreshPromise; } finally { this.refreshPromise = null; }
    },
    async refreshSnapshot(refresh = false) {
      this.servedRevisions = [];
      this.snapshotBuffer = {};
      this.squadData = {};
      this.decisionCard = null;
      this.contingencyPlans = {};
      this.chipData = {};
      this.leagueData = {};
      this.briefingData = {};
      // A forced reopen refresh completes before advice requests use cached upstream data.
      // Render basic data before expensive requests enter the event loop.
      await Promise.all([this.loadBasicSquad(refresh), this.loadGameState(), this.loadProfile()]);
      await this.$nextTick();
      await Promise.all([
        this.loadHealth(),
        this.loadGameState(),
        this.loadProfile(),
        this.loadDecisionCard(),
        this.loadSquad(),
        this.loadContingencyPlans(),
        this.loadContingencyMatrix(),
        this.loadPriceChanges(),
        this.loadChips(),
        this.loadLeague(),
        this.loadBriefing(),
        this.loadChecklist(),
        this.loadSystemStatus()
      ]);
      if (new Set(this.servedRevisions).size > 1) {
        this.squadData = {}; this.decisionCard = null; this.contingencyPlans = {};
        this.chipData = {}; this.leagueData = {}; this.briefingData = {};
        this.decisionCardError = 'Data changed during refresh. Refresh again for one coherent snapshot.';
      } else {
        // Publish all advice in one synchronous Vue update only after validation.
        Object.assign(this, this.snapshotBuffer);
      }
      this.snapshotBuffer = null;
    }
  },

  mounted() {
    this.initTheme();
    this.restoreSessionHistory();
    this.refreshAll(true);
  }
});

const vm = app.mount('#app');
window.__fpl_vm__ = vm;


