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
      showNavigation: false,
      servedRevisions: [],
      snapshotBuffer: null,
      snapshotGeneration: 0,
      snapshotAbort: null,
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
      toasts: [],
      toastSeq: 0,
      busy: {},
      aux: Object.fromEntries(['matrix','prices','league','briefing','review'].map(k => [k, {loading:false,error:null}])),
      mainTabs: [
        { key: 'overview', label: 'Overview', icon: 'layout-dashboard' },
        { key: 'team', label: 'My Team', icon: 'users' },
        { key: 'transfers', label: 'Transfers', icon: 'arrow-left-right' },
        { key: 'league', label: 'My League', icon: 'trophy' }
      ],
      planTabs: [
        { key: 'plan_a', label: 'Plan A' },
        { key: 'plan_b', label: 'Plan B' },
        { key: 'plan_c', label: 'Plan C' }
      ],
      briefingTabs: [
        { key: 'preview', label: 'Weekly briefing' },
        { key: 'review', label: 'Post-match review' }
      ],

      // Telemetry & Game State
      health: {},
      gameState: {},
      systemStatus: {},
      rivalStress: null,
      rivalStressLoading: false,
      rivalStressError: null,
      expiryResearch: null,
      expiryResearchLoading: false,
      expiryResearchError: null,
      leagueValidation: {},

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
      detailedPlansLoading: false,
      detailedPlansError: null,
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
      panicDuration: "",
      showConfirmTeam: false,
      confirmTeamData: null,
      confirmTeamText: "",
      confirmForm: {},
      confirmRows: [],
      confirmMode: 'simple',
      followedLocked: false,
      deadlineTimer: null,
      confirmDelta: {out:'',in:''},
      confirmTeamError: "",
      confirmTeamSaving: false,

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
        "Should I make a transfer?",
        "When is the best week for Bench Boost?",
        "How can I catch my league leader?"
      ],

      // Background Sync Pipeline
      adviceStage: '',
      pipelineRunning: false,
      pipelineProgress: 0,
      pipelineCurrentStage: 'Ready',
      eventSource: null
    };
  },

  computed: {
    overallRankLabel() {
      const rank = this.leagueData.overall_rank;
      return Number.isInteger(rank) && rank > 0 ? '#' + rank.toLocaleString('en-GB') : (this.aux.league.loading ? 'Loading overall rank...' : 'Overall rank unavailable');
    },
    currentLeagueRankLabel() {
      if (this.aux.league.loading) return 'Loading league rank...';
      if (this.aux.league.error) return 'League rank unavailable';
      const rank = this.leagueData.current_league_rank;
      if (Number.isInteger(rank) && rank > 0) return '#' + rank;
      return {not_listed:'Not listed in this league',manager_unconfigured:'Set your manager ID',league_unconfigured:'Set your mini-league ID'}[this.leagueData.league_rank_status] || 'League rank unavailable';
    },
    moreItems() {
      return [
        { key: 'panic', label: 'Emergency check', icon: 'siren' },
        { key: 'audit', label: 'Pre-deadline checklist', icon: 'list-checks',
          badge: `${this.checklistSummary.passed_count ?? 0}/5`, tone: this.checklistSummary.status === 'PASS' ? 'ok' : 'warn' },
        { key: 'report', label: 'Weekly report', icon: 'file-text' },
        { key: 'squad', label: 'Import squad', icon: 'clipboard-paste' },
        { key: 'settings', label: 'Settings', icon: 'settings-2' },

      ];
    },

    simulatedSubs() {
      const starters=this.displaySquad.starters || [], bench=this.displaySquad.bench || [];
      const absent=starters.filter(p=>this.simulatedOutIds.includes(p.element));
      let best=[];
      const legal=players=>{
        const n={GKP:0,DEF:0,MID:0,FWD:0};players.forEach(p=>n[p.position]++);
        return n.GKP===1 && n.DEF>=3 && n.DEF<=5 && n.MID>=2 && n.MID<=5 && n.FWD>=1 && n.FWD<=3;
      };
      function search(i,players,used,moves){
        if(i===absent.length){if(legal(players) && moves.length>best.length) best=moves;return;}
        const out=absent[i];
        for(const b of bench){
          if(used.has(b.element) || (out.position==='GKP')!==(b.position==='GKP')) continue;
          search(i+1,players.map(p=>p.element===out.element?b:p),new Set([...used,b.element]),[...moves,{out:out.web_name,in:b.web_name}]);
        }
        search(i+1,players,used,moves);
      }
      search(0,starters,new Set(),[]);
      return {moves:best,unfilled:absent.length-best.length};
    },
    transferPairs() {
      const out=this.activePlan?.transfers_out || [], ins=this.activePlan?.transfers_in || [];
      return Array.from({length:Math.max(out.length,ins.length)},(_,i)=>({out:out[i],in:ins[i]}));
    },
    chipCheck() {
      const chip = this.decisionCard && this.decisionCard.chip;
      if (!chip || !chip.recommend || this.decisionCardLoading || this.plansLoading) return null;
      const r = this.expiryResearch;
      if (!r) return {state: 'idle',tone:'muted',label:'Check timing',text:'Check timing'};
      if(r.status==='pending' || this.expiryResearchLoading) return {state: 'pending',tone:'info',label:'Checking timing',text:'Checking...'};
      if(r.status==='model_expiry_sensitivity' && r.short && r.extended){
        if(!r.first_chip_changes) return {state: 'holds',tone:'info',label:'Timing checked',text:'Timing unchanged'};
        return {state: 'changed',tone:'warn',label:'Timing changes',text:`The longer comparison starts with ${String(r.extended.first_chip || 'saving chips').replace(/_/g,' ')}. Review before using a chip.`};
      }
      return {state: 'unavailable',tone:'muted',label:'Timing unavailable',text:r.reason || 'Update data and check again.'};
    },

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

    displaySquad() { return this.hasLoadedSquad ? this.squadData : (this.hasBasicSquad ? this.basicSquadData : {}); },
    usingPublishedTeam() { return this.hasBasicSquad && !this.confirmTeamData?.state?.team_confirmed; },
    publishedTeamNote() {
      const gw = this.basicSquadData.published_gameweek;
      return `Based on your team at ${Number.isInteger(gw) ? 'the GW' + gw + ' deadline' : 'the last deadline'}.`;
    },

    flaggedSquadPlayers() {
      if (!this.hasLoadedSquad && !this.hasBasicSquad) return [];
      return [...this.displaySquad.starters, ...this.displaySquad.bench].filter(p =>
        p.status !== 'a' || (Number.isFinite(p.chance_of_playing) && p.chance_of_playing < 100) || p.news_quote);
    },

    unknownAvailabilityPlayers() {
      if (!this.hasLoadedSquad && !this.hasBasicSquad) return [];
      return [...this.displaySquad.starters, ...this.displaySquad.bench].filter(p => !Number.isFinite(p.chance_of_playing));
    },

    allSquadConfirmedAvailable() {
      return (this.hasLoadedSquad || this.hasBasicSquad) && this.flaggedSquadPlayers.length === 0 && this.unknownAvailabilityPlayers.length === 0;
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
        return {title: (this.decisionCardLoading || this.plansLoading) ? "Checking..." : "Advice unavailable",
          badge: "Unavailable", tone: "muted", gainText: "Unavailable", hitText: "Unavailable",
          bankText: this.bankDisplay, ftText: this.ftDisplay,
          reasons: [this.decisionCardError || this.plansError || this.squadError || "Checking..."],
          caveat: "Refresh needed", card: null, isUnavailable: true};
      }
      if (!this.hasLoadedSquad) {
        if (this.decisionCardLoading || this.plansLoading || this.squadLoading) {
          return {
            title: "Loading recommendations...",
            badge: "Loading",
            tone: "muted",
            gainText: "Evaluating",
            hitText: "--",
            bankText: "Bank unavailable",
            ftText: "Free transfers unknown",
            reasons: [
              "Checking...",
              "Solving optimal transfer and chip combinations."
            ],
            caveat: "Please wait while optimization runs.",
            card: null,
            isUnavailable: true
          };
        }
        return {
          title: "Advice unavailable",
          badge: "Unavailable",
          tone: "muted",
          gainText: "Unavailable",
          hitText: "Unavailable",
          bankText: "Bank unavailable",
          ftText: "Free transfers unknown",
          reasons: [
            this.decisionCardError || this.plansError || this.squadError || "Team not confirmed",
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
        let tone = "muted";

        if (chip.recommend) {
          title = `Use chip: ${chip.chip_display_name || 'Active chip'}`;
          badge = "Chip";
          tone = "warn";
        } else if (isRoll) {
          title = "Save your transfer";
          badge = "Hold & Roll";
          tone = "info";
        } else {
          const inNames = (t.in || []).map(p => p.web_name).join(', ') || 'Target';
          const outNames = (t.out || []).map(p => p.web_name).join(', ') || 'Outgoing';
          title = `Transfer ${outNames} → ${inNames}`;
          badge = t.no_regret_flag ? "No-Regret Move" : "Recommended Move";
          tone = "ok";
        }

        const reasons = [
          card.two_line_reasoning || "Plan for the upcoming weeks.",
          chip.recommend ? chip.reason : `Starting XI led by captain ${card.captain?.web_name || 'Captain'} (${card.captain?.expected_points || 0.0} xP) in a ${card.formation || '3-5-2'} shape.`
        ];

        return {
          title: title,
          badge: badge,
          tone: tone,
          gainText: `+${(t.net_gain_vs_roll || 0.0).toFixed(1)} pts (${card.decision_scope?.horizon_gameweeks?.length ?? "unknown"}-GW)`,
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
            tone: "muted",
            gainText: "Evaluating",
            hitText: "--",
            bankText: (this.squadData && this.squadData.bank_millions !== undefined) ? `£${this.squadData.bank_millions}m in bank` : "Bank unavailable",
            ftText: (this.squadData && this.squadData.free_transfers !== undefined) ? `${this.squadData.free_transfers} FT left` : "Free transfers unknown",
            reasons: [
              "Checking...",
              "Solving optimal transfer and chip combinations."
            ],
            caveat: "Please wait while optimization runs.",
            isUnavailable: true
          };
        }

        const errDetail = this.decisionCardError || this.plansError || this.squadError;
        return {
          title: "Advice unavailable",
          badge: "Unavailable",
          tone: "muted",
          gainText: "Unavailable",
          hitText: "Unavailable",
          bankText: (this.squadData && this.squadData.bank_millions !== undefined) ? `£${this.squadData.bank_millions}m in bank` : "Bank unavailable",
          ftText: (this.squadData && this.squadData.free_transfers !== undefined) ? `${this.squadData.free_transfers} FT left` : "Free transfers unknown",
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
          tone: "ok",
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
            title: "Advice unavailable",
            badge: "Unavailable",
            tone: "muted",
            gainText: "Unavailable",
            hitText: "Unavailable",
            bankText: "Bank unavailable",
            ftText: "Free transfers unknown",
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
          tone: "info",
          gainText: "0.0 pts (Roll)",
          hitText: "0 hit pts",
          bankText: `£${(this.squadData.bank_millions !== undefined ? this.squadData.bank_millions : 0.0).toFixed(1)}m in bank`,
          ftText: Number.isInteger(plan.next_banked_ft) ? `${plan.next_banked_ft} FTs next week` : "Free transfers unknown",
          reasons: [
            "Starting XI holds high expected output across all fixtures.",
            "Save the transfer for next week."
          ],
          caveat: "Check team news before the deadline.",
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
    showNavigation(v) { if (v) this.closeOverlaysExcept('showNavigation'); },
    showChatDrawer(v) { if (v) this.closeOverlaysExcept('showChatDrawer'); },
    showSettingsModal(v) { if (v) this.closeOverlaysExcept('showSettingsModal'); },
    showPanicModal(v) { if (v) this.closeOverlaysExcept('showPanicModal'); },
    showChecklistModal(v) { if (v) this.closeOverlaysExcept('showChecklistModal'); },
    showSquadModal(v) { if (v) this.closeOverlaysExcept('showSquadModal'); },
    showBriefingModal(v) { if (v) this.closeOverlaysExcept('showBriefingModal'); },
    showSystemModal(v) { if (v) this.closeOverlaysExcept('showSystemModal'); },
    briefingSubTab(value) { if (value === 'review') this.loadReview(); },
    activeTab(value) { this.saveSessionHistory(); if (location.hash !== '#' + value) location.hash = value; },
    chatMessages: { deep: true, handler() { this.saveSessionHistory(); } }
  },

  methods: {
    closeOverlaysExcept(keep) {
      for (const key of ['showNavigation', 'showChatDrawer', 'showSettingsModal', 'showPanicModal', 'showChecklistModal',
        'showSquadModal', 'showBriefingModal', 'showSystemModal']) {
        if (key !== keep && this[key]) this[key] = false;
      }
    },
    restoreRoute() {
      const key = location.hash.slice(1);
      if (['overview','team','transfers','league'].includes(key)) this.activeTab = key;
      else if (key) { this.activeTab = 'overview'; history.replaceState(null, '', '#overview'); }
    },
    humanLabel(value, fallback = 'Not available') {
      if (value == null || value === '') return fallback;
      const names = {ROLL_TRANSFER:'Save free transfer',WILDCARD:'Wildcard',FREE_HIT:'Free Hit',BENCH_BOOST:'Bench Boost',TRIPLE_CAPTAIN:'Triple Captain',PASS:'Passed',WARN:'Needs review',FAIL:'Failed',PENDING:'Pending',api_only:'Official data only',CHECK_TRANSFER:'Check transfer'};
      if (names[value] || names[String(value).toUpperCase()]) return names[value] || names[String(value).toUpperCase()];
      return String(value).replace(/[_-]+/g,' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
    },
    priceLabel(direction) {
      return {RISE_IMMINENT:'Possible rise',LIKELY_RISE:'Rising transfer demand',FALL_IMMINENT:'Possible fall',LIKELY_FALL:'Falling transfer demand'}[direction] || 'Price change unconfirmed';
    },
    overlapLabel(team) {
      if (team.squad_overlap_pct == null) return 'Not measured';
      return `${Number(team.squad_overlap_pct).toFixed(0)}% (${team.shared_players}/${team.compared_squad_size} players, GW${team.overlap_gameweek})`;
    },
    navigate(key) { this.activeTab = key; this.showNavigation = false; },
    onMenu(key) {
      this.showNavigation = false;
      if (key === 'panic') this.openPanicModal();
      else if (key === 'audit') this.openChecklistModal();
      else if (key === 'report') this.showBriefingModal = true;
      else if (key === 'squad') this.openSquadModal();
      else if (key === 'settings') this.showSettingsModal = true;
      else if (key === 'system') this.showSystemModal = true;
    },
    async guard(key, fn) {
      if (this.busy[key]) return;
      this.busy = { ...this.busy, [key]: true };
      try { return await fn(); } finally { this.busy = { ...this.busy, [key]: false }; }
    },
    async watchExpiry(jobId, generation) {
      // Core advice is already visible; the long-horizon check is attached later.
      for (let i = 0; i < 240; i++) {
        if (generation !== this.snapshotGeneration) return;
        if (!this.expiryResearch || this.expiryResearch.status !== 'pending') return;
        await new Promise(resolve => setTimeout(resolve, 4000));
        if (generation !== this.snapshotGeneration) return;
        try {
          const response = await fetch(`/api/advice/status/${jobId}`);
          if (!response.ok) return;
          const job = await response.json();
          if (generation !== this.snapshotGeneration) return;
          if (job.result && job.result.expiryResearch) this.expiryResearch = job.result.expiryResearch;
        } catch (error) { return; }
      }
    },
    async loadDetailedPlans() {
      if(this.detailedPlansLoading)return;
      const generation=this.snapshotGeneration;
      this.detailedPlansLoading=true;this.detailedPlansError=null;
      try {
        const response=await fetch('/api/contingency/plans?detailed=true');
        if(!response.ok)throw new Error(`Detailed plans HTTP ${response.status}`);
        const plans=await response.json();
        if(generation!==this.snapshotGeneration)return;
        const primary=this.contingencyPlans?.plan_a;
        if(!primary || JSON.stringify(primary.transfers_in.map(p=>p.element))!==JSON.stringify(plans.plan_a?.transfers_in?.map(p=>p.element)) ||
           JSON.stringify(primary.transfers_out.map(p=>p.element))!==JSON.stringify(plans.plan_a?.transfers_out?.map(p=>p.element)) || primary.plan_type!==plans.plan_a.plan_type)
          throw new Error('Primary plan changed. Refresh core advice before loading alternatives.');
        this.contingencyPlans={...plans,plan_a:primary};
      } catch(error){this.detailedPlansError=String(error);}
      finally{this.detailedPlansLoading=false;}
    },
    async loadRivalStress() {
      if(this.rivalStressLoading)return;
      this.rivalStressLoading=true;this.rivalStressError=null;
      try {
        const response=await fetch('/api/league/scenarios');
        if(!response.ok)throw new Error(`Rival stress HTTP ${response.status}`);
        this.rivalStress=await response.json();
      } catch(error){this.rivalStressError=String(error);}
      finally{this.rivalStressLoading=false;}
    },
    async loadExpiryResearch() {
      if (this.expiryResearchLoading) return;
      this.expiryResearchLoading=true; this.expiryResearchError=null;
      try {
        const response=await fetch('/api/research/chip-expiry');
        if (!response.ok) throw new Error(`Expiry research HTTP ${response.status}`);
        this.expiryResearch=await response.json();
      } catch (error) { this.expiryResearchError=String(error); }
      finally { this.expiryResearchLoading=false; }
    },
    async loadLeagueValidation() {
      try {
        const response=await fetch('/api/research/league-validation');
        if(response.ok)this.leagueValidation=await response.json();
      } catch(error) { this.leagueValidation={status:'unavailable'}; }
    },
    publishSnapshot(field, value, generation = this.snapshotGeneration) {
      if (generation !== this.snapshotGeneration) return;
      if (this.snapshotBuffer) this.snapshotBuffer[field] = value;
      else this[field] = value;
    },
    async fetchSnapshot(path, generation = this.snapshotGeneration) {
      const response = await fetch(path, this.snapshotAbort ? {signal: this.snapshotAbort.signal} : {});
      if (generation !== this.snapshotGeneration) throw new Error('Superseded snapshot');
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

    triggerToast(msg, tone = 'ok') {
      const id = ++this.toastSeq;
      this.toasts = [...this.toasts.slice(-2), { id, message: msg, tone }];
      setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, 3800);
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
      const escapeText = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      if (!window.DOMPurify) return '<pre>' + escapeText(text) + '</pre>';
      try {
        const rawHtml = window.marked ? marked.parse(text) : '<pre>' + escapeText(text) + '</pre>';
        return DOMPurify.sanitize(rawHtml);
      } catch (e) {
        return '<pre>' + escapeText(text) + '</pre>';
      }
    },

    getPositionPlayers(pos) {
      if (!this.displaySquad.starters) return [];
      return this.displaySquad.starters.filter(p => p.position === pos);
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
      if (this.pipelineRunning) return;
      if (this.followedLocked) {
        await this.refreshAll(true);
        this.triggerToast(this.followedLocked ? 'Done. Next advice after the deadline.' : 'Data updated.');
        return;
      }
      this.squadData = {}; this.decisionCard = null; this.contingencyPlans = {};
      this.chipData = {}; this.leagueData = {}; this.briefingData = {};
      this.pipelineRunning = true;
      this.pipelineProgress = 0;
      this.pipelineCurrentStage = "Triggering data refresh...";

      try {
        const res = await fetch('/api/sync/trigger', { method: 'POST' });
        if (!res.ok) throw new Error('Update could not start');
        await res.json();
        this.triggerToast('Updating your data...');
      } catch (e) { this.pipelineRunning = false; this.triggerToast('Update could not start. Check your connection and try again.', 'danger'); return; }

      if (this.eventSource) this.eventSource.close();
      this.eventSource = new EventSource('/api/sync/stream');

      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.pipelineProgress = data.progress_pct || 0;
          this.pipelineCurrentStage = `${this.humanLabel(data.step)}: ${data.message || ''}`;
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
        this.triggerToast('Update connection lost. Try again to refresh your data.', 'warn');
      };
    },

    async openConfirmTeam() {
      this.showConfirmTeam=true;this.confirmTeamError="";this.confirmMode="simple";
      try {
        const r=await fetch('/api/team/confirmation');const data=await r.json();
        if(!r.ok)throw new Error(data.detail||'Current team unavailable');
        this.confirmTeamData=data;
        this.confirmTeamData.players = data.players || data.state.players || [];
        const state=data.state, saved=data.saved;
        this.confirmForm={gameweek:state.target_gw,bank:state.bank_tenths/10,free_transfers:state.free_transfers,
          available_chips:state.target_gw<=19?state.chips_remaining_set_1:state.chips_remaining_set_2,
          active_chip:state.active_chip||"",hit_cost:saved?.gameweek===state.target_gw?saved.hit_cost:0,
          captain:state.squad.find(p=>p.is_captain)?.element || null,
          vice_captain:state.squad.find(p=>p.is_vice_captain)?.element || null,
          bench:state.squad.filter(p=>!p.is_starter).sort((a,b)=>a.bench_order-b.bench_order).map(p=>p.element)};
        this.confirmRows=state.squad.map(p=>({element:p.element,purchase:p.purchase_price/10,selling:p.selling_price/10}));
      } catch(e){this.confirmTeamError=e.message;}
    },
    async undoFollowedAdvice() {
      try {const r=await fetch('/api/team/undo',{method:'POST'});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Undo failed');this.followedLocked=false;await this.refreshAll();}
      catch(e){this.triggerToast(e.message,'warn');}
    },
    async confirmFollowedAdvice() {
      if(this.confirmTeamSaving)return;this.confirmTeamSaving=true;this.confirmTeamError='';
      try {const r=await fetch('/api/team/followed',{method:'POST'});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Could not confirm advice');this.showConfirmTeam=false;this.triggerToast('Done');await this.refreshAll();}
      catch(e){this.confirmTeamError=e.message;this.triggerToast(e.message,'warn');}finally{this.confirmTeamSaving=false;}
    },
    addConfirmedMove() {
      const out=Number(this.confirmDelta.out), incoming=Number(this.confirmDelta.in), row=this.confirmRows.find(r=>Number(r.element)===out), player=this.confirmTeamData?.players.find(p=>p.element===incoming);
      if(!row||!player||this.confirmRows.some(r=>Number(r.element)===incoming)){this.confirmTeamError='Choose an owned player out and a different player in.';return;}
      const original=this.confirmTeamData.state.squad.find(p=>p.element===out);
      if(original?.price_provenance==='market_estimate'){this.confirmTeamError='Check selling price in the full editor.';return;}
      this.confirmForm.bank=Number((Number(this.confirmForm.bank)+Number(row.selling)-player.price/10).toFixed(1));
      if(!['wildcard','freehit'].includes(this.confirmForm.active_chip)){if(Number(this.confirmForm.free_transfers)>0)this.confirmForm.free_transfers--;else this.confirmForm.hit_cost=Number(this.confirmForm.hit_cost)+4;}
      this.confirmForm.bench=this.confirmForm.bench.map(id=>id===out?incoming:id);
      if(this.confirmForm.captain===out)this.confirmForm.captain=incoming;
      if(this.confirmForm.vice_captain===out)this.confirmForm.vice_captain=incoming;
      row.element=incoming;row.purchase=player.price/10;row.selling=player.price/10;this.confirmDelta={out:'',in:''};this.confirmTeamError='';
    },
    async saveConfirmTeam() {
      if(this.confirmTeamSaving)return;this.confirmTeamSaving=true;this.confirmTeamError="";
      try {
        const payload={gameweek:this.confirmForm.gameweek,player_ids:this.confirmRows.map(p=>Number(p.element)),
          bank_tenths:Math.round(Number(this.confirmForm.bank)*10),free_transfers:Number(this.confirmForm.free_transfers),
          available_chips:this.confirmForm.available_chips.filter(c=>c!==this.confirmForm.active_chip),active_chip:this.confirmForm.active_chip||null,
          captain:Number(this.confirmForm.captain)||null,vice_captain:Number(this.confirmForm.vice_captain)||null,
          hit_cost:Number(this.confirmForm.hit_cost),
          bench:(this.confirmForm.bench || []).filter(id=>this.confirmRows.some(p=>Number(p.element)===id)),
          selling_prices:Object.fromEntries(this.confirmRows.map(p=>[p.element,Math.round(Number(p.selling)*10)])),
          purchase_prices:Object.fromEntries(this.confirmRows.map(p=>[p.element,Math.round(Number(p.purchase)*10)]))};
        const r=await fetch('/api/team/confirmation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        const data=await r.json();if(!r.ok)throw new Error(data.detail||'Confirmation failed');
        this.showConfirmTeam=false;this.triggerToast('Saved: '+data.comparison);
        await this.refreshAll();
      } catch(e){this.confirmTeamError=e.message;this.triggerToast(e.message,'warn');}finally{this.confirmTeamSaving=false;}
    },

    // Emergency Crisis Solver (Panic Button)
    openPanicModal() {
      this.showPanicModal = true;
      this.panicResult = null;
    },

    async executePanicReoptimize() {
      if (!this.panicQuery.trim() || this.panicResolving) return;
      this.panicResolving = true;
      try {
        const res = await fetch('/api/contingency/panic', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: this.panicQuery, duration:this.panicDuration })
        });
        const result = await res.json();
        if(!res.ok) throw new Error(result.detail || 'Emergency check failed');
        this.panicResult = result;
      } catch (e) {
        this.panicResult = { status: "error", recommendation: e.message || "Emergency check failed. Try again." };
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
        const generation = this.snapshotGeneration;
        const res = await this.fetchSnapshot('/api/contingency/checklist', generation);
        if (res.ok) {
          const data = await res.json();
          this.checklistItems = data.checklist || [];
          const passed = this.checklistItems.filter(i => i.status === 'PASS').length;
          this.checklistSummary = {
            status: this.checklistItems.length === 5 && passed === 5 ? 'PASS' : 'WARN',
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
      return this.guard('reset', () => this._resetFinancialOverrides());
    },
    async _resetFinancialOverrides() {
      const res = await fetch('/api/profile', { method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({bank: null, free_transfers: null}) });
      if (!res.ok) { this.triggerToast('Could not reset overrides.'); return; }
      await this.refreshAll(true);
      this.triggerToast('Bank and transfers reset');
    },

    async saveProfile() {
      return this.guard('save', () => this._saveProfile());
    },
    async _saveProfile() {
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
        this.triggerToast("Failed to save profile settings.", "danger");
      }
    },

    async loadSquad() {
      const generation = this.snapshotGeneration;
      this.squadLoading = true;
      this.squadError = null;
      try {
        const res = await this.fetchSnapshot('/api/squad', generation);
        if (res.ok) {
          this.publishSnapshot('squadData', await res.json(), generation);
          if (generation !== this.snapshotGeneration) return;
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
        if (generation !== this.snapshotGeneration) return;
        this.squadData = {};
        this.squadError = "Network error loading squad.";
      } finally {
        if (generation !== this.snapshotGeneration) return;
        this.squadLoading = false;
      }
    },

    async loadContingencyPlans() {
      const generation = this.snapshotGeneration;
      this.plansLoading = true;
      this.plansError = null;
      try {
        const res = await this.fetchSnapshot('/api/contingency/plans', generation);
        if (res.ok) {
          this.publishSnapshot('contingencyPlans', await res.json(), generation);
        } else {
          this.contingencyPlans = {};
          this.plansError = `Plans unavailable (HTTP ${res.status})`;
        }
      } catch (e) {
        if (generation !== this.snapshotGeneration) return;
        this.contingencyPlans = {};
        this.plansError = "Network error loading transfer plans.";
      } finally {
        if (generation !== this.snapshotGeneration) return;
        this.plansLoading = false;
      }
    },

    async loadAux(key, path, field, transform = x => x) {
      const generation = this.snapshotGeneration;
      const state = this.aux[key];
      state.loading = true; state.error = null;
      try {
        const res = await fetch(path, {signal: AbortSignal.timeout(180000)});
        if (!res.ok) throw new Error('Source request failed');
        const data = await res.json();
        if (generation !== this.snapshotGeneration) return;
        this.publishSnapshot(field, transform(data), generation);
      } catch (error) {
        if (generation !== this.snapshotGeneration) return;
        this[field] = field === 'contingencyMatrix' ? [] : {};
        state.error = 'Refresh failed. Try again.';
      } finally {
        if (generation === this.snapshotGeneration) state.loading = false;
      }
    },
    retryAux(key) {
      const routes = {matrix:'loadContingencyMatrix',prices:'loadPriceChanges',league:'loadLeague',briefing:'loadBriefing',review:'loadReview'};
      if (routes[key] && !this.aux[key].loading) return this[routes[key]]();
    },
    loadContingencyMatrix() { return this.loadAux('matrix','/api/contingency/matrix','contingencyMatrix', x => x.contingency_matrix || []); },
    loadPriceChanges() { return this.loadAux('prices','/api/price-changes','priceChanges'); },

    async loadChips() {
      const generation = this.snapshotGeneration;
      try {
        const res = await this.fetchSnapshot('/api/chips', generation);
        if (res.ok) this.publishSnapshot('chipData', await res.json(), generation);
      } catch (e) {
        if (generation !== this.snapshotGeneration) return;}
    },

    loadLeague() { return this.loadAux('league','/api/league','leagueData'); },

    async loadDecisionCard() {
      const generation = this.snapshotGeneration;
      this.decisionCardLoading = true;
      this.decisionCardError = null;
      try {
        const res = await this.fetchSnapshot('/api/decision-card', generation);
        if (res.ok) {
          this.publishSnapshot('decisionCard', await res.json(), generation);
        } else {
          this.decisionCard = null;
          this.decisionCardError = `Decision card unavailable (HTTP ${res.status})`;
        }
      } catch (e) {
        if (generation !== this.snapshotGeneration) return;
        console.warn('Failed to fetch decision card:', e);
        this.decisionCard = null;
        this.decisionCardError = "Network error loading decision card.";
      } finally {
        if (generation !== this.snapshotGeneration) return;
        this.decisionCardLoading = false;
      }
    },

    async exportDecisionCard() {
      return this.guard('export', () => this._exportDecisionCard());
    },
    async _exportDecisionCard() {
      try {
        const res = await fetch('/api/decision-card/export');
        if (res.ok) {
          const text = await res.text();
          await navigator.clipboard.writeText(text);
          this.triggerToast("Decision Card (Markdown) copied to clipboard!");
        } else {
          this.triggerToast("Failed to export decision card.", "danger");
        }
      } catch (e) {
        console.error("Export error:", e);
        this.triggerToast("Could not copy decision card to clipboard.", "danger");
      }
    },

    loadBriefing() { return this.loadAux('briefing','/api/briefing','briefingData'); },
    loadReview() { return this.loadAux('review','/api/review','reviewData'); },

    // Chat Agent
    async sendMessage() {
      if (!this.userInput.trim() || this.isChatLoading) return;
      const q = this.userInput.trim();
      this.chatMessages.push({ role: 'user', content: q });
      this.userInput = '';
      this.isChatLoading = true;

      try {
        const res = await fetch('/api/chat', {
          signal: AbortSignal.timeout(170000),
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: q })
        });
        const data = await res.json();
        const body = res.ok && typeof data.response === 'string' && data.response.trim()
          ? data.response : (data.error || 'No answer returned. Check the server error and try again.');
        this.chatMessages.push({ role: 'assistant', content: body });
      } catch (e) {
        this.chatMessages.push({ role: 'assistant', content: "Could not load an answer. Try again." });
      } finally {
        this.isChatLoading = false;
      }
    },

    sendPrompt(p) {
      this.userInput = p;
      this.sendMessage();
    },

    copyBriefingMarkdown() {
      return this.guard('copy', async () => {
        await this._copyBriefingMarkdown();
      });
    },
    async _copyBriefingMarkdown() {
      const text = this.briefingSubTab === 'preview' ? (this.briefingData.markdown || '') : (this.reviewData.review_markdown || '');
      try { await navigator.clipboard.writeText(text); this.triggerToast('Report markdown copied.'); }
      catch (e) { this.triggerToast('Could not copy to clipboard.', 'danger'); }
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
      if (!this.manualSquadText.trim() || this.matchingInProgress) return;
      this.matchingInProgress = true;
      try {
        const res = await fetch('/api/squad/match', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ raw_text: this.manualSquadText })
        });
        if (!res.ok) throw new Error('Matching request failed');
        this.matchResult = await res.json();
      } catch (error) { this.triggerToast('Could not match players. Check your connection and try again.', 'danger');
      } finally {
        this.matchingInProgress = false;
      }
    },

    async saveManualSquad() {
      if (!this.matchResult || !this.matchResult.is_valid_15 || this.savingSquad) return;
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
        } else throw new Error('Squad save failed');
      } catch (error) { this.triggerToast('Could not save the squad. Nothing was confirmed. Try again.', 'danger');
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
      this.snapshotGeneration++;
      this.servedRevisions = [];
      this.snapshotBuffer = {};
      this.snapshotAbort = new AbortController();
      this.squadData = {};
      this.decisionCard = null;
      this.contingencyPlans = {};
      this.expiryResearch = null;
      this.chipData = {};
      this.leagueData = {};
      this.briefingData = {};
      this.checklistItems = []; this.contingencyMatrix = [];
      this.decisionCardError = null; this.squadError = null; this.plansError = null;
      // A forced reopen refresh completes before advice requests use cached upstream data.
      // Render basic data before expensive requests enter the event loop.
      await Promise.all([this.loadBasicSquad(refresh), this.loadGameState(), this.loadProfile()]);
      await this.$nextTick();
      const confirmationResponse=await fetch('/api/team/confirmation');
      if(confirmationResponse.ok){const confirmation=await confirmationResponse.json();this.confirmTeamData=confirmation;this.followedLocked=Boolean(confirmation.locked);
        if(this.followedLocked){this.decisionCardLoading=false;this.squadLoading=false;this.plansLoading=false;
          if(this.deadlineTimer)clearTimeout(this.deadlineTimer);
          const deadline=Date.parse(confirmation.state?.deadline_utc||'');
          if(Number.isFinite(deadline)&&deadline>Date.now())this.deadlineTimer=setTimeout(()=>this.refreshAll(true),Math.min(deadline-Date.now()+1000,2147483647));
          return;}}

      // One durable calculation publishes the core atomically. Browser latency
      // cannot discard a completed squad merely because league/report is slower.
      this.snapshotBuffer = null;
      this.decisionCardLoading = true; this.squadLoading = true; this.plansLoading = true;
      this.adviceStage = "Checking...";
      this.decisionCardError = null; this.squadError = null; this.plansError = null;
      const generation = this.snapshotGeneration;
      try {
        const start = await fetch('/api/advice/start', {method:'POST'});
        if (!start.ok) throw new Error(`Advice start HTTP ${start.status}`);
        let job = await start.json();
        if (!job.id) throw new Error(job.reason || 'Advice job could not start');
        while (job.status === 'calculating') {
          this.adviceStage = job.stage || 'Calculating shared advice';
          await new Promise(resolve => setTimeout(resolve, 1500));
          if (generation !== this.snapshotGeneration) return;
          const response = await fetch(`/api/advice/status/${job.id}`);
          if (!response.ok) throw new Error(`Advice status HTTP ${response.status}`);
          job = await response.json();
        }
        if (generation !== this.snapshotGeneration) return;
        if (job.status !== 'ready' || !job.result) throw new Error(job.error || 'Advice calculation stopped');
        Object.assign(this, {squadData:job.result.squadData, decisionCard:job.result.decisionCard,
          contingencyPlans:job.result.contingencyPlans});
        this.expiryResearch = job.result.expiryResearch || null;
        if (this.expiryResearch && this.expiryResearch.status === 'pending') void this.watchExpiry(job.id, generation);
      } catch (error) {
        if (generation !== this.snapshotGeneration) return;
        this.squadError = 'Advice unavailable';
        this.decisionCardError = 'Could not prepare advice. Try Update again.';
      } finally {
        if (generation === this.snapshotGeneration) {
          this.decisionCardLoading=false; this.squadLoading=false; this.plansLoading=false;
        }
      }
      // Optional surfaces never veto or overwrite coherent ready core.
      void Promise.all([this.loadHealth(), this.loadContingencyMatrix(), this.loadPriceChanges(),
        this.loadChecklist(), this.loadSystemStatus(), this.loadLeagueValidation(), this.loadChips(), this.loadLeague(), this.loadBriefing()]);
    }
  },

  beforeUnmount() { if(this.deadlineTimer)clearTimeout(this.deadlineTimer); if (this.routeListener) window.removeEventListener('hashchange', this.routeListener); },
  mounted() {
    this.initTheme();
    this.restoreSessionHistory();
    this.restoreRoute();
    this.routeListener = () => { this.closeOverlaysExcept(''); this.restoreRoute(); };
    window.addEventListener('hashchange', this.routeListener);
    this.refreshAll(true);
  }
});

if (window.FPLUI) window.FPLUI.register(app);
const vm = app.mount('#app');
window.__fpl_vm__ = vm;


