"""
Adversarial Game-Theoretic Minimax Solver for Head-to-Head & Mini-League Duels.
Directly optimizes probability of winning against primary rival(s) under adversarial response.

Formulates decision as:
1. Pure Minimax: max_u min_a P(Win | u, a)  (Guaranteed worst-case robustness)
2. Bayesian Nash / Softmax Response: max_u sum_a P(a) * P(Win | u, a)
3. Minimax Regret: min_u max_a (Best_u'(a) - Payoff(u, a))
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RivalAction:
    action_id: str
    description: str
    transfers_in: list[int]
    transfers_out: list[int]
    captain_id: int | None
    expected_gain: float
    prior_probability: float = 0.0
    rival_elements: list[int] = field(default_factory=list)
    rival_starters: list[int] = field(default_factory=list)


@dataclass
class RivalProfile:
    rival_id: str | int
    player_name: str
    entry_name: str
    total_points: float
    base_xp: float
    elements: list[int]
    bank_itb: float = 0.0
    captain_id: int | None = None
    actions: list[RivalAction] = field(default_factory=list)
    starters: list[int] = field(default_factory=list)


@dataclass
class MultiRivalMinimaxResult:
    best_minimax_plan: dict[str, Any]
    best_bayesian_rank_plan: dict[str, Any]
    joint_pareto_plan: dict[str, Any]
    pareto_frontier_plans: list[dict[str, Any]]
    global_worst_case_guarantees: list[float]  # per user plan
    expected_ranks: list[float]  # per user plan (1.0 = 1st)
    pack_win_rates: list[float]  # per user plan (mean pairwise win prob)
    bottleneck_rivals: list[str]  # per user plan: identity of most dangerous rival
    rival_worst_threats: list[str]  # per user plan: action description of worst threat
    per_rival_payoffs: dict[str, list[list[float]]]  # rival_id -> [user_idx][action_idx]
    user_plan_labels: list[str]
    summary: str


@dataclass
class MinimaxEvaluationResult:
    best_minimax_plan: dict[str, Any]
    best_bayesian_plan: dict[str, Any]
    best_regret_plan: dict[str, Any]
    payoff_matrix: list[list[float]]  # [user_idx][rival_idx] = win_prob
    user_plan_labels: list[str]
    rival_action_labels: list[str]
    worst_case_guarantees: list[float]
    expected_payoffs: list[float]
    max_regrets: list[float]
    summary: str


class AdversarialMinimaxSolver:
    """
    Computes game-theoretic minimax and Bayesian response against primary league rivals.
    """

    def __init__(self, rationality_beta: float = 1.5):
        """
        rationality_beta: Softmax temperature parameter for rival decision model.
        Higher beta means rival is assumed to pick closer to the optimal move.
        """
        self.beta = rationality_beta

    VALID_FORMATIONS = [
        (3, 5, 2),
        (3, 4, 3),
        (4, 4, 2),
        (4, 3, 3),
        (4, 5, 1),
        (5, 3, 2),
        (5, 4, 1),
        (5, 2, 3),
    ]

    def _predict_starting_xi(
        self,
        elements: set[int] | list[int],
        pool: dict[int, dict[str, Any]] | None = None,
        captain_id: int | None = None,
    ) -> list[int]:
        """
        Predicts rival starting XI (11 players), strictly excluding the bench.
        Selects highest-xP valid formation when pool is provided.
        Guarantees captain inclusion in starters.
        """
        elem_list = list(elements)
        if len(elem_list) <= 11:
            return elem_list

        if pool:
            gks: list[tuple[float, int]] = []
            defs: list[tuple[float, int]] = []
            mids: list[tuple[float, int]] = []
            fwds: list[tuple[float, int]] = []

            for eid in elem_list:
                p = pool.get(eid)
                if not p:
                    continue
                pos = p.get("position", "MID")
                xp = float(p.get("expected_points", 0.0) or 0.0)
                if pos == "GKP":
                    gks.append((xp, eid))
                elif pos == "DEF":
                    defs.append((xp, eid))
                elif pos == "MID":
                    mids.append((xp, eid))
                elif pos == "FWD":
                    fwds.append((xp, eid))

            gks.sort(key=lambda x: x[0], reverse=True)
            defs.sort(key=lambda x: x[0], reverse=True)
            mids.sort(key=lambda x: x[0], reverse=True)
            fwds.sort(key=lambda x: x[0], reverse=True)

            if gks and len(defs) >= 3 and len(mids) >= 2 and len(fwds) >= 1:
                best_xp = -1e9
                best_starters: list[int] = []
                for n_def, n_mid, n_fwd in self.VALID_FORMATIONS:
                    if n_def <= len(defs) and n_mid <= len(mids) and n_fwd <= len(fwds):
                        cand = (
                            [gks[0][1]]
                            + [x[1] for x in defs[:n_def]]
                            + [x[1] for x in mids[:n_mid]]
                            + [x[1] for x in fwds[:n_fwd]]
                        )
                        tot_xp = sum(float(pool.get(e, {}).get("expected_points", 0.0) or 0.0) for e in cand)
                        if tot_xp > best_xp:
                            best_xp = tot_xp
                            best_starters = cand
                if len(best_starters) == 11:
                    return best_starters

        # Fallback when pool data is missing or incomplete
        starters: list[int] = []
        if captain_id is not None and captain_id in elem_list:
            starters.append(captain_id)
        for e in elem_list:
            if e not in starters:
                starters.append(e)
            if len(starters) == 11:
                break
        return starters

    def generate_rival_action_space(
        self,
        rival: dict[str, Any],
        pool: dict[int, dict[str, Any]],
        current_gw: int,
        max_actions: int = 5,
    ) -> list[RivalAction]:
        """
        Enumerates candidate actions available to the rival:
        - Action 0: Roll transfer (status quo)
        - Action 1..k: Top single transfers affordable with rival's bank ITB
        """
        rival_elements = set(int(e) for e in rival.get("elements", []))
        if not rival_elements and "squad" in rival:
            rival_elements = set(int(p["element"] if isinstance(p, dict) else p) for p in rival["squad"])
        bank = float(rival.get("bank_itb", 0.0) or 0.0)
        actions: list[RivalAction] = []

        # Action 0: Roll (no transfers)
        curr_cap = rival.get("captain_element") or (
            rival.get("captain") if isinstance(rival.get("captain"), int) else None
        )
        base_starters = [
            int(p["element"] if isinstance(p, dict) else p)
            for p in (rival.get("starters") or rival.get("starting_xi") or [])
        ]
        if not base_starters and "squad" in rival:
            base_starters = [
                int(p["element"]) for p in rival["squad"] if isinstance(p, dict) and p.get("is_starter", True)
            ][:11]
        if not base_starters:
            base_starters = self._predict_starting_xi(rival_elements, pool, captain_id=curr_cap)

        actions.append(
            RivalAction(
                action_id="rival_roll",
                description="Roll Free Transfer (preserve bank and FTs)",
                transfers_in=[],
                transfers_out=[],
                captain_id=curr_cap,
                expected_gain=0.0,
                rival_elements=list(rival_elements),
                rival_starters=list(base_starters),
            )
        )

        if len(rival_elements) != 15 or not pool:
            actions[0].prior_probability = 1.0
            return actions

        # Find best single transfers (respecting position, budget, club limits)
        candidate_moves = []
        for out_id in rival_elements:
            if out_id not in pool:
                continue
            out_row = pool[out_id]
            out_pos = out_row.get("position")

            def _to_tenths(v: float) -> float:
                return float(v) * 10.0 if float(v) <= 25.0 else float(v)

            out_val = _to_tenths(out_row.get("value", 50.0))
            out_xp = float(out_row.get("expected_points", 0.0))
            bank_tenths = _to_tenths(bank)
            max_afford = out_val + bank_tenths

            # Club counts without out_id
            club_counts: dict[int, int] = {}
            for e in rival_elements - {out_id}:
                if e in pool:
                    c_id = pool[e].get("team")
                    if c_id is not None:
                        club_counts[c_id] = club_counts.get(c_id, 0) + 1

            for in_id, in_row in pool.items():
                if in_id in rival_elements or in_row.get("position") != out_pos:
                    continue
                in_val = _to_tenths(in_row.get("value", 50.0))
                if in_val > max_afford:
                    continue
                in_team = in_row.get("team")
                if in_team is not None and club_counts.get(in_team, 0) >= 3:
                    continue

                in_xp = float(in_row.get("expected_points", 0.0))
                gain = in_xp - out_xp
                if gain > 0.3:  # Viable improvement hurdle
                    candidate_moves.append((gain, in_id, out_id, in_row.get("web_name", f"Player {in_id}")))

        # Sort by gain descending
        candidate_moves.sort(key=lambda x: x[0], reverse=True)
        for idx, (gain, in_id, out_id, in_name) in enumerate(candidate_moves[: max_actions - 1], start=1):
            new_elems = list((rival_elements - {out_id}) | {in_id})
            new_starters = self._predict_starting_xi(new_elems, pool, captain_id=curr_cap)
            actions.append(
                RivalAction(
                    action_id=f"rival_move_{idx}",
                    description=f"Transfer: {pool[out_id].get('web_name', out_id)} -> {in_name} (+{gain:.2f} xP)",
                    transfers_in=[in_id],
                    transfers_out=[out_id],
                    captain_id=curr_cap,
                    expected_gain=gain,
                    rival_elements=new_elems,
                    rival_starters=new_starters,
                )
            )

        # Compute prior probabilities using softmax over expected gains
        gains = np.array([a.expected_gain for a in actions], dtype=np.float64)
        exp_gains = np.exp(self.beta * (gains - np.max(gains)))
        probs = exp_gains / np.sum(exp_gains)
        for i, a in enumerate(actions):
            a.prior_probability = float(probs[i])

        return actions

    def evaluate_game(
        self,
        user_candidates: list[dict[str, Any]],
        rival_actions: list[RivalAction],
        user_base_points: float,
        rival_base_points: float,
        user_base_xp: float,
        rival_base_xp: float,
        var_user: float = 120.0,
        var_rival: float = 120.0,
        covariance: float = 35.0,
    ) -> MinimaxEvaluationResult:
        """
        Builds the payoff matrix P(User beats Rival | u, a) and computes:
        1. Pure Minimax choice: max_u min_a Payoff[u, a]
        2. Bayesian choice: max_u sum_a P(a) * Payoff[u, a]
        3. Minimax Regret choice: min_u max_a (Best_u'(a) - Payoff[u, a])
        """
        from statistics import NormalDist

        if not user_candidates:
            dummy_plan = {"label": "No Action Available", "expected_gain": 0.0, "hit_cost": 0}
            return MinimaxEvaluationResult(
                best_minimax_plan=dummy_plan,
                best_bayesian_plan=dummy_plan,
                best_regret_plan=dummy_plan,
                payoff_matrix=[[0.5]],
                user_plan_labels=["No Action"],
                rival_action_labels=["Status Quo"],
                worst_case_guarantees=[0.5],
                expected_payoffs=[0.5],
                max_regrets=[0.0],
                summary="No user candidate plans provided for minimax evaluation.",
            )

        if not rival_actions:
            rival_actions = [
                RivalAction(
                    action_id="rival_roll",
                    description="Roll Free Transfer (preserve bank and FTs)",
                    transfers_in=[],
                    transfers_out=[],
                    captain_id=None,
                    expected_gain=0.0,
                    prior_probability=1.0,
                )
            ]

        n_u = len(user_candidates)
        n_a = len(rival_actions)
        payoff = np.zeros((n_u, n_a), dtype=np.float64)

        current_margin = user_base_points - rival_base_points

        for i, u_cand in enumerate(user_candidates):
            u_gain = float(u_cand.get("expected_gain", 0.0) or 0.0)
            u_hit = float(u_cand.get("hit_cost", 0.0) or 0.0)
            u_net_gain = u_gain - u_hit

            u_starters: set[int] = set()
            u_cap: int | None = None
            if "trajectory" in u_cand and len(u_cand["trajectory"]) > 0:
                t0 = u_cand["trajectory"][0]
                u_starters = {int(p) for p in (t0.get("starters") or t0.get("starting_xi") or [])}
                u_cap = int(t0.get("captain") or t0.get("captain_id") or 0) or None
            elif "starting_xi" in u_cand and u_cand["starting_xi"]:
                u_starters = {int(p.get("element", p) if isinstance(p, dict) else p) for p in u_cand["starting_xi"]}
                u_cap = int(u_cand.get("captain_id") or u_cand.get("captain") or 0) or None
            elif "starters" in u_cand:
                u_starters = {int(p.get("element", p) if isinstance(p, dict) else p) for p in u_cand["starters"]}
                u_cap = int(u_cand.get("captain_id") or u_cand.get("captain") or 0) or None

            for j, r_act in enumerate(rival_actions):
                r_net_gain = r_act.expected_gain

                # Net projected gap
                projected_margin = current_margin + (user_base_xp + u_net_gain) - (rival_base_xp + r_net_gain)

                # Dynamic squad-overlap covariance
                cell_cov = covariance
                r_starters = getattr(r_act, "rival_starters", None)
                if not r_starters:
                    raw_elems = getattr(r_act, "rival_elements", None)
                    if raw_elems:
                        r_starters = self._predict_starting_xi(raw_elems, captain_id=getattr(r_act, "captain_id", None))
                    else:
                        r_starters = []
                r_elems = r_starters

                if u_starters and r_elems:
                    r_cap = getattr(r_act, "captain_id", None)
                    # Bilinear covariance expansion:
                    # Cov(w_U * X_k, w_R * X_k) = w_U * w_R * sigma^2 with sigma^2 = 6.5:
                    # - Both captain same player: 2 * 2 * 6.5 = 26.0
                    # - User captains, rival starts: 2 * 1 * 6.5 = 13.0
                    # - Rival captains, user starts: 1 * 2 * 6.5 = 13.0
                    # - Both start (neither captains): 1 * 1 * 6.5 = 6.5
                    cell_cov = float(
                        sum(
                            (2.0 if k == u_cap else 1.0) * (2.0 if k == r_cap else 1.0) * 6.5
                            for k in (u_starters & set(r_elems))
                        )
                    )

                # Variance of differential: Var(U - R) = Var(U) + Var(R) - 2*Cov(U, R)
                diff_var = max(10.0, var_user + var_rival - 2.0 * cell_cov)
                diff_sd = np.sqrt(diff_var)

                # Probability margin > 0: Phi(projected_margin / diff_sd)
                z = projected_margin / diff_sd
                win_p = float(NormalDist().cdf(z))
                payoff[i, j] = round(win_p, 4)

        # 1. Pure Minimax: worst-case payoff for each user plan across all rival counter-moves
        worst_cases = np.min(payoff, axis=1)
        best_minimax_idx = int(np.argmax(worst_cases))

        # 2. Bayesian Expected Payoff: weighted by rival prior action probability
        rival_priors = np.array([a.prior_probability for a in rival_actions], dtype=np.float64)
        expected_payoffs = np.dot(payoff, rival_priors)
        best_bayesian_idx = int(np.argmax(expected_payoffs))

        # 3. Minimax Regret
        # Column maxima (best possible user move against each specific rival action)
        col_max = np.max(payoff, axis=0)  # Shape (n_a,)
        regret_matrix = col_max[None, :] - payoff  # Shape (n_u, n_a)
        max_regrets = np.max(regret_matrix, axis=1)  # Shape (n_u,)
        best_regret_idx = int(np.argmin(max_regrets))

        u_labels = [c.get("label", f"Plan {idx}") for idx, c in enumerate(user_candidates)]
        r_labels = [a.description for a in rival_actions]

        summary = (
            f"Adversarial Minimax Solution: Best robust plan '{u_labels[best_minimax_idx]}' "
            f"guarantees >= {worst_cases[best_minimax_idx] * 100:.1f}% win rate against all rival counter-moves. "
            f"Bayesian optimal plan '{u_labels[best_bayesian_idx]}' yields {expected_payoffs[best_bayesian_idx] * 100:.1f}% "
            f"expected win rate."
        )

        return MinimaxEvaluationResult(
            best_minimax_plan=user_candidates[best_minimax_idx],
            best_bayesian_plan=user_candidates[best_bayesian_idx],
            best_regret_plan=user_candidates[best_regret_idx],
            payoff_matrix=payoff.tolist(),
            user_plan_labels=u_labels,
            rival_action_labels=r_labels,
            worst_case_guarantees=worst_cases.tolist(),
            expected_payoffs=expected_payoffs.tolist(),
            max_regrets=max_regrets.tolist(),
            summary=summary,
        )

    def evaluate_multi_rival_game(
        self,
        user_candidates: list[dict[str, Any]],
        rival_profiles: list[dict[str, Any]],
        player_pool: dict[int, dict[str, Any]] | Any,
        target_gw: int,
        user_base_points: float,
        user_base_xp: float,
        var_user: float = 120.0,
        var_rival: float = 120.0,
        hurdle: float = 1.20,
    ) -> MultiRivalMinimaxResult:
        """
        Evaluates user candidate plans against a pack of rivals (e.g. top-3 to top-5).
        Computes:
        1. Pairwise dynamic covariance & win probabilities per rival action.
        2. Guaranteed worst-case win probability across all rivals: min_{r} worst_case(u, r).
        3. Average expected rank across the rival pack: 1 + sum_r (1 - P(u > r)).
        4. Joint Pareto-optimal action recommendation.
        """
        from statistics import NormalDist

        if not user_candidates:
            dummy = {"label": "No Action Available", "total_objective": 0.0}
            return MultiRivalMinimaxResult(
                best_minimax_plan=dummy,
                best_bayesian_rank_plan=dummy,
                joint_pareto_plan=dummy,
                pareto_frontier_plans=[dummy],
                global_worst_case_guarantees=[0.5],
                expected_ranks=[1.0],
                pack_win_rates=[0.5],
                bottleneck_rivals=["None"],
                rival_worst_threats=["None"],
                per_rival_payoffs={},
                user_plan_labels=["No Action"],
                summary="No user candidate plans provided.",
            )

        # Standardize player pool to dict[int, dict]
        pool_dict: dict[int, dict[str, Any]] = {}
        if hasattr(player_pool, "iterrows"):
            pool_dict = {int(r["element"]): dict(r) for _, r in player_pool.iterrows()}
        elif isinstance(player_pool, dict):
            pool_dict = player_pool

        # Prepare Rival Profiles and Actions
        rivals: list[RivalProfile] = []
        for idx, r in enumerate(rival_profiles or []):
            r_id = str(r.get("entry_id", r.get("entry", r.get("rival_id", f"rival_{idx}"))))
            p_name = str(r.get("player_name", f"Rival {idx + 1}"))
            e_name = str(r.get("entry_name", f"Team {idx + 1}"))
            pts = float(r.get("total_points", r.get("total", r.get("points", user_base_points))))
            elems = [
                int(p["element"] if isinstance(p, dict) else p) for p in (r.get("elements") or r.get("squad") or [])
            ]
            r_squad_starters = [
                int(p["element"] if isinstance(p, dict) else p)
                for p in (r.get("starters") or r.get("starting_xi") or [])
            ]
            if not r_squad_starters and "squad" in r:
                r_squad_starters = [
                    int(p["element"]) for p in r["squad"] if isinstance(p, dict) and p.get("is_starter", True)
                ][:11]
            bank = float(r.get("bank_itb", 0.0) or 0.0)
            cap = (
                int(r.get("captain_element") or (r.get("captain") if isinstance(r.get("captain"), int) else 0) or 0)
                or None
            )
            rxp = float(r.get("base_xp", user_base_xp))

            if not r_squad_starters and elems:
                r_squad_starters = self._predict_starting_xi(elems, pool=pool_dict, captain_id=cap)

            actions = self.generate_rival_action_space(
                rival={"elements": elems, "bank_itb": bank, "captain": cap, "starters": r_squad_starters},
                pool=pool_dict,
                current_gw=target_gw,
            )
            rivals.append(
                RivalProfile(
                    rival_id=r_id,
                    player_name=p_name,
                    entry_name=e_name,
                    total_points=pts,
                    base_xp=rxp,
                    elements=elems,
                    bank_itb=bank,
                    captain_id=cap,
                    actions=actions,
                    starters=r_squad_starters,
                )
            )

        if not rivals:
            # Fallback to single neutral rival
            rivals.append(
                RivalProfile(
                    rival_id="neutral_rival",
                    player_name="Neutral Competitor",
                    entry_name="Neutral Squad",
                    total_points=user_base_points,
                    base_xp=user_base_xp,
                    elements=[],
                    bank_itb=0.0,
                    captain_id=None,
                    actions=[
                        RivalAction(
                            action_id="neutral_roll",
                            description="Status Quo",
                            transfers_in=[],
                            transfers_out=[],
                            captain_id=None,
                            expected_gain=0.0,
                            prior_probability=1.0,
                        )
                    ],
                )
            )

        n_u = len(user_candidates)
        n_r = len(rivals)

        # Store payoffs: per_rival_payoffs[r_id][u_idx][a_idx]
        per_rival_payoffs: dict[str, list[list[float]]] = {}
        # rival_bayesian_win[u_idx][r_idx] = P(User beats Rival r_k)
        rival_bayesian_win = np.zeros((n_u, n_r), dtype=np.float64)
        # rival_worst_case[u_idx][r_idx] = min_a P(User beats Rival r_k | a)
        rival_worst_case = np.zeros((n_u, n_r), dtype=np.float64)
        rival_worst_action_desc: list[list[str]] = [["" for _ in range(n_r)] for _ in range(n_u)]

        for i, u_cand in enumerate(user_candidates):
            u_gain = float(u_cand.get("expected_gain", 0.0) or 0.0)
            u_hit = float(u_cand.get("hit_cost", 0.0) or 0.0)
            u_net_gain = u_gain - u_hit

            u_starters: set[int] = set()
            u_cap: int | None = None
            if "trajectory" in u_cand and len(u_cand["trajectory"]) > 0:
                t0 = u_cand["trajectory"][0]
                u_starters = {int(p) for p in (t0.get("starters") or t0.get("starting_xi") or [])}
                u_cap = int(t0.get("captain") or t0.get("captain_id") or 0) or None
            elif "starting_xi" in u_cand and u_cand["starting_xi"]:
                u_starters = {int(p.get("element", p) if isinstance(p, dict) else p) for p in u_cand["starting_xi"]}
                u_cap = int(u_cand.get("captain_id") or u_cand.get("captain") or 0) or None
            elif "starters" in u_cand:
                u_starters = {int(p.get("element", p) if isinstance(p, dict) else p) for p in u_cand["starters"]}
                u_cap = int(u_cand.get("captain_id") or u_cand.get("captain") or 0) or None

            for k, riv in enumerate(rivals):
                n_a = len(riv.actions)
                if riv.rival_id not in per_rival_payoffs:
                    per_rival_payoffs[str(riv.rival_id)] = [[0.0] * n_a for _ in range(n_u)]

                action_probs = np.array([a.prior_probability for a in riv.actions], dtype=np.float64)
                act_payoffs = np.zeros(n_a, dtype=np.float64)

                for j, r_act in enumerate(riv.actions):
                    proj_margin = (
                        (user_base_points - riv.total_points)
                        + (user_base_xp + u_net_gain)
                        - (riv.base_xp + r_act.expected_gain)
                    )

                    # Dynamic squad-overlap covariance
                    r_starters = getattr(r_act, "rival_starters", None)
                    if not r_starters:
                        raw_elems = getattr(r_act, "rival_elements", riv.starters or riv.elements)
                        if raw_elems:
                            r_starters = self._predict_starting_xi(
                                raw_elems, pool=pool_dict, captain_id=getattr(r_act, "captain_id", None)
                            )
                        else:
                            r_starters = []
                    r_elems = r_starters

                    if u_starters and r_elems:
                        r_cap = getattr(r_act, "captain_id", None)
                        # Bilinear covariance expansion:
                        # Cov(w_U * X_k, w_R * X_k) = w_U * w_R * sigma^2 with sigma^2 = 6.5:
                        # - Both captain same player: 2 * 2 * 6.5 = 26.0
                        # - User captains, rival starts: 2 * 1 * 6.5 = 13.0
                        # - Rival captains, user starts: 1 * 2 * 6.5 = 13.0
                        # - Both start (neither captains): 1 * 1 * 6.5 = 6.5
                        cell_cov = float(
                            sum(
                                (2.0 if k == u_cap else 1.0) * (2.0 if k == r_cap else 1.0) * 6.5
                                for k in (u_starters & set(r_elems))
                            )
                        )
                    else:
                        cell_cov = 0.0

                    diff_var = max(10.0, var_user + var_rival - 2.0 * cell_cov)
                    diff_sd = np.sqrt(diff_var)
                    win_p = float(NormalDist().cdf(proj_margin / diff_sd))

                    per_rival_payoffs[str(riv.rival_id)][i][j] = round(win_p, 4)
                    act_payoffs[j] = win_p

                min_j = int(np.argmin(act_payoffs))
                rival_worst_case[i, k] = act_payoffs[min_j]
                rival_worst_action_desc[i][k] = riv.actions[min_j].description
                rival_bayesian_win[i, k] = float(np.dot(act_payoffs, action_probs))

        # 1. Global Worst-Case Win Probability across ALL rivals
        global_worst_case = np.min(rival_worst_case, axis=1)  # shape (n_u,)
        bottleneck_rival_idx = np.argmin(rival_worst_case, axis=1)
        bottleneck_rival_names = [rivals[bottleneck_rival_idx[i]].player_name for i in range(n_u)]
        worst_threat_descs = [rival_worst_action_desc[i][bottleneck_rival_idx[i]] for i in range(n_u)]

        # 2. Average Expected Rank across the pack: E[Rank] = 1 + sum_k (1 - P(u > r_k))
        expected_ranks = 1.0 + np.sum(1.0 - rival_bayesian_win, axis=1)
        pack_win_rates = np.mean(rival_bayesian_win, axis=1)

        # 3. Identify Single-Criterion Optima
        best_minimax_idx = int(np.argmax(global_worst_case))
        best_rank_idx = int(np.argmin(expected_ranks))

        # 4. Joint Pareto Frontier & Recommendation
        objectives = []
        for i, c in enumerate(user_candidates):
            obj = float(c.get("total_objective", c.get("expected_gain", 0.0)))
            objectives.append((obj, global_worst_case[i], -expected_ranks[i]))
        objs = np.array(objectives)

        is_pareto = np.ones(n_u, dtype=bool)
        for i in range(n_u):
            for j in range(n_u):
                if i != j:
                    if np.all(objs[j] >= objs[i]) and np.any(objs[j] > objs[i]):
                        is_pareto[i] = False
                        break

        pareto_indices = [i for i in range(n_u) if is_pareto[i]]
        pareto_plans = [user_candidates[i] for i in pareto_indices]

        # Hurdle-Gated Pareto Choice starting from highest objective plan
        top_obj_idx = int(np.argmax(objs[:, 0]))
        rec_idx = top_obj_idx
        incumbent_obj = objs[top_obj_idx, 0]
        incumbent_wc = global_worst_case[top_obj_idx]
        incumbent_rank = expected_ranks[top_obj_idx]

        for p_idx in pareto_indices:
            if p_idx == top_obj_idx:
                continue
            cand_obj = objs[p_idx, 0]
            cand_wc = global_worst_case[p_idx]
            cand_rank = expected_ranks[p_idx]

            if (incumbent_obj - cand_obj) <= hurdle:
                if (cand_wc >= incumbent_wc + 0.05) or (cand_rank <= incumbent_rank - 0.25):
                    rec_idx = p_idx
                    incumbent_obj = cand_obj
                    incumbent_wc = cand_wc
                    incumbent_rank = cand_rank

        u_labels = [c.get("label", c.get("source", f"Plan {i}")) for i, c in enumerate(user_candidates)]

        summary = (
            f"Multi-Rival Adversarial Duel ({n_r} rivals): "
            f"Joint Pareto Plan '{u_labels[rec_idx]}' guarantees >= {global_worst_case[rec_idx] * 100:.1f}% "
            f"win prob against worst rival threat ({bottleneck_rival_names[rec_idx]}), "
            f"with expected pack rank {expected_ranks[rec_idx]:.2f} / {n_r + 1}."
        )

        return MultiRivalMinimaxResult(
            best_minimax_plan=user_candidates[best_minimax_idx],
            best_bayesian_rank_plan=user_candidates[best_rank_idx],
            joint_pareto_plan=user_candidates[rec_idx],
            pareto_frontier_plans=pareto_plans,
            global_worst_case_guarantees=global_worst_case.tolist(),
            expected_ranks=expected_ranks.tolist(),
            pack_win_rates=pack_win_rates.tolist(),
            bottleneck_rivals=bottleneck_rival_names,
            rival_worst_threats=worst_threat_descs,
            per_rival_payoffs=per_rival_payoffs,
            user_plan_labels=u_labels,
            summary=summary,
        )


minimax_solver = AdversarialMinimaxSolver()
