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
        target_league_id: 314,
        bank: 1.5,
        free_transfers: 3,
        risk_preference: 'balanced',
        llm_provider: 'no-key'
      },

      // Core Squad & Lineup
      squadData: {},
      simulatedOutIds: [],

      // Transfers & Contingency Plans
      contingencyPlans: {},
      selectedPlanKey: 'plan_a',
      contingencyMatrix: [],
      priceChanges: { rises: [], falls: [] },

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
      checklistSummary: { status: 'PASS', passed_count: 5 },

      // Emergency Panic Re-Optimizer
      panicQuery: '',
      panicResolving: false,
      panicResult: null,

      // Manual Squad Entry
      manualSquadText: '',
      manualSquadBank: 1.5,
      manualSquadFT: 3,
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

    activePlan() {
      if (!this.contingencyPlans) return null;
      return this.contingencyPlans[this.selectedPlanKey] || this.contingencyPlans.plan_a || null;
    },

    nextDecision() {
      const plan = this.activePlan;
      if (!plan) {
        return {
          title: "Save your free transfer",
          badge: "Hold Transfer",
          badgeClass: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40",
          gainText: "+0.0 pts gain",
          hitText: "0 hit pts",
          bankText: `£${this.squadData.bank_millions || '0.0'}m in bank`,
          ftText: `${this.squadData.free_transfers || 1} FT left`,
          reasons: [
            "Your starting XI projected points baseline is solid with no flagged absences.",
            "Banking a free transfer builds tactical flexibility for upcoming fixture swings."
          ],
          caveat: "Check team news before the deadline in case late rotation emerges."
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
          bankText: `£${plan.bank_after !== undefined ? (plan.bank_after / 10).toFixed(1) : this.squadData.bank_millions}m in bank`,
          ftText: `${plan.free_transfers_remaining || 0} FT left`,
          reasons: [
            plan.action_summary || `Targeting favorable fixture run and superior attacking form.`,
            `Maintains optimal budget allocation without requiring unnecessary point deductions.`
          ],
          caveat: `Subject to Friday press conference clearance and expected starting lineups.`
        };
      } else {
        return {
          title: "Save your free transfer",
          badge: "Roll Free Transfer",
          badgeClass: "bg-blue-500/20 text-blue-400 border border-blue-500/40",
          gainText: "0.0 pts (Roll)",
          hitText: "0 hit pts",
          bankText: `£${this.squadData.bank_millions || '0.0'}m in bank`,
          ftText: `${(this.squadData.free_transfers || 1) + 1} FTs next week`,
          reasons: [
            "Starting XI holds high expected output across all fixtures.",
            "Accumulating a second free transfer provides greater pivot leverage next week."
          ],
          caveat: "Ensure vice-captain is locked on an early kickoff starter as backup."
        };
      }
    },

    startersExpectedPoints() {
      return (this.squadData.starters_expected_points || 0).toFixed(1);
    },

    captainBonusPoints() {
      return (this.squadData.captain_bonus_expected_points || 0).toFixed(1);
    },

    totalGameweekPoints() {
      return (this.squadData.total_expected_points || 0).toFixed(1);
    }
  },

  methods: {
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

    formatDeadline(seconds) {
      if (!seconds || seconds <= 0) return 'Passed';
      const days = Math.floor(seconds / 86400);
      const hours = Math.floor((seconds % 86400) / 3600);
      const mins = Math.floor((seconds % 3600) / 60);
      if (days > 0) return `${days}d ${hours}h`;
      return `${hours}h ${mins}m`;
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
          this.pipelineCurrentStage = `${data.step_label || data.step}: ${data.detail || ''}`;
          if (data.is_running === false && data.progress_pct === 100) {
            this.pipelineRunning = false;
            this.eventSource.close();
            this.refreshAll();
            this.triggerToast("Data updated successfully!");
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

    async saveProfile() {
      try {
        await fetch('/api/profile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.profile)
        });
        this.showSettingsModal = false;
        this.triggerToast("Settings saved. Refreshing analysis...");
        await this.refreshAll();
      } catch (e) {
        alert("Failed to save profile settings.");
      }
    },

    async loadSquad() {
      try {
        const res = await fetch('/api/squad');
        if (res.ok) this.squadData = await res.json();
      } catch (e) {}
    },

    async loadContingencyPlans() {
      try {
        const res = await fetch('/api/contingency/plans');
        if (res.ok) this.contingencyPlans = await res.json();
      } catch (e) {}
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
        const res = await fetch('/api/chips');
        if (res.ok) this.chipData = await res.json();
      } catch (e) {}
    },

    async loadLeague() {
      try {
        const res = await fetch('/api/league');
        if (res.ok) this.leagueData = await res.json();
      } catch (e) {}
    },

    async loadBriefing() {
      try {
        const res = await fetch('/api/briefing');
        if (res.ok) this.briefingData = await res.json();
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

    async refreshAll() {
      await Promise.all([
        this.loadHealth(),
        this.loadGameState(),
        this.loadProfile(),
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
    }
  },

  mounted() {
    this.initTheme();
    this.refreshAll();
  }
});

app.mount('#app');
