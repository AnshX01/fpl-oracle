"""Bounded legal transfer/chip beam. No inferred rival captains or optimality claim."""

from typing import Any

from fpl_oracle.optimise.squad import squad_optimizer


_SOLVE_CACHES: dict[Any, Any] = {}
_RANKING_CACHE: dict[Any, Any] = {}
_LINEUP_CACHE: dict[Any, Any] = {}
_LINEUP_CACHE_LIMIT = 400_000


def evaluate_lineup(elements, player_map, risk="balanced", chip=None):
    """Best legal XI for a squad. Memoised: identical squad and projections give an identical answer."""
    key = (id(player_map), frozenset(elements), risk, chip)
    hit = _LINEUP_CACHE.get(key)
    if hit is not None and hit[0] is player_map:
        if hit[1] is None:
            raise ValueError("No legal XI in projected squad")
        return hit[1]
    if len(_LINEUP_CACHE) > _LINEUP_CACHE_LIMIT:
        _LINEUP_CACHE.clear()
    try:
        result = _evaluate_lineup_uncached(elements, player_map, risk, chip)
    except ValueError:
        _LINEUP_CACHE[key] = (player_map, None)
        raise
    _LINEUP_CACHE[key] = (player_map, result)
    return result


def _evaluate_lineup_uncached(elements, player_map, risk="balanced", chip=None):
    from fpl_oracle.optimise.transfers import VALID_FORMATIONS

    players = [player_map[e] for e in sorted(elements)]
    groups = {
        p: sorted(
            [r for r in players if r["position"] == p and not r.get("simulation_unavailable", False)],
            key=lambda r: (-float(r["expected_points"]), int(r["element"])),
        )
        for p in ("GKP", "DEF", "MID", "FWD")
    }
    best = None
    multiplier = 2 if chip == "3xc" else 1
    for formation in VALID_FORMATIONS:
        counts = dict(zip(("GKP", "DEF", "MID", "FWD"), (1, *formation), strict=True))
        for cap in players:
            if cap.get("simulation_unavailable", False):
                continue
            chosen = []
            for pos, count in counts.items():
                options = groups[pos]
                if cap["position"] == pos:
                    options = [cap] + [r for r in options if r["element"] != cap["element"]]
                chosen += options[:count]
            if len(chosen) != 11 or not any(r["element"] == cap["element"] for r in chosen):
                continue
            xp = float(cap["expected_points"])
            if risk == "points":
                score = xp
            elif risk == "conservative":
                score = 0.6 * xp + 0.4 * float(cap.get("p10", xp * 0.4))
            elif risk == "aggressive":
                score = 0.4 * xp + 0.6 * float(cap.get("p90", xp * 1.8))
            else:
                score = 0.7 * xp + 0.3 * float(cap.get("p90", xp * 1.8))
            starter_xp = sum(float(r["expected_points"]) for r in chosen)
            objective = (
                sum(float(r["expected_points"]) for r in players) if chip == "bboost" else starter_xp
            ) + multiplier * score
            if best is None or objective > best[0]:
                bench = sum(float(r["expected_points"]) for r in players) - starter_xp
                gross = starter_xp + multiplier * xp + (bench if chip == "bboost" else 0)
                best = (objective, gross, int(cap["element"]), formation)
    if best is None:
        raise ValueError("No legal XI in projected squad")
    return best[1:]


def search_sequences(
    optimizer,
    elements,
    bank,
    purchase,
    ft,
    gameweeks,
    pools,
    maps,
    available,
    locked_in,
    locked_out,
    excluded,
    risk="balanced",
    terminal_costs=None,
    beam_width=32,
    chips_by_set=None,
    previous_chip=None,
    candidate_limit=2,
    two_transfer_limit=1,
    per_position_limit=2,
    restructure_parent_limit=1,
    max_hits_per_move=1,
    hit_cooldown_until=0,
    enforce_hit_gain=True,
    forced_current_chip=None,
    forced_chip_schedule=None,
):

    costs = terminal_costs or {}
    states = [
        dict(
            elements=set(elements),
            bank=int(bank),
            purchase=dict(purchase),
            ft=ft,
            remaining=frozenset(available),
            history=[],
            score=0.0,
            first_chip=None,
        )
    ]
    # Squad solves depend only on the projections, locks and hit rules, so every
    # chip/date search over the same inputs can reuse them.
    solve_ctx = (
        id(pools),
        id(maps),
        tuple(gameweeks),
        optimizer.discount_factor,
        frozenset(locked_in),
        frozenset(locked_out),
        frozenset(excluded),
        risk,
        max_hits_per_move,
        hit_cooldown_until,
        optimizer.hit_penalty,
    )
    held = _SOLVE_CACHES.get(solve_ctx)
    if held is not None and held[0] is pools and held[1] is maps:
        solve_cache = held[2]
    else:
        solve_cache = {}
        if len(_SOLVE_CACHES) > 8:
            _SOLVE_CACHES.clear()
        _SOLVE_CACHES[solve_ctx] = (pools, maps, solve_cache)
    shared_key = (id(pools), id(maps), tuple(gameweeks), optimizer.discount_factor)
    shared = _RANKING_CACHE.get(shared_key)
    if shared is not None and shared[0] is pools and shared[1] is maps:
        ranking_pools, ranking_maps = shared[2], shared[3]
    else:
        ranking_pools = {}
        ranking_maps = {}
        for step, gw in enumerate(gameweeks):
            frame = pools[gw].copy()
            frame["expected_points"] = frame["element"].map(
                {
                    e: sum(
                        optimizer.discount_factor**j * float(maps[g].get(e, {}).get("expected_points", 0))
                        for j, g in enumerate(gameweeks[step:])
                    )
                    for e in maps[gw]
                }
            )
            ranking_pools[gw] = frame
            ranking_maps[gw] = {int(row["element"]): dict(row) for _, row in frame.iterrows()}
        if len(_RANKING_CACHE) > 8:
            _RANKING_CACHE.clear()
        _RANKING_CACHE[shared_key] = (pools, maps, ranking_pools, ranking_maps)
    for step, gw in enumerate(gameweeks):
        next_states = []
        restructure_parents = {}
        for parent in states:
            group = (parent["first_chip"], parent["remaining"])
            restructure_parents.setdefault(group, []).append(parent)
        restructure_ids = {
            id(parent) for parents in restructure_parents.values() for parent in parents[:restructure_parent_limit]
        }
        for state in states:
            remaining = state["remaining"]
            if step and gameweeks[step - 1] <= 19 < gw:
                # Do not infer the owner's second set from first-set availability.
                remaining = frozenset((chips_by_set or {}).get(2, []))
            moves = [dict(transfers_in=[], transfers_out=[], buy_costs={}, bank_delta=0)]
            immediate = optimizer._get_candidate_1_transfers(
                state["elements"],
                state["bank"],
                state["purchase"],
                pools[gw],
                maps[gw],
                locked_in,
                locked_out,
                excluded,
                max_per_pos=per_position_limit,
            )
            horizon = optimizer._get_candidate_1_transfers(
                state["elements"],
                state["bank"],
                state["purchase"],
                ranking_pools[gw],
                ranking_maps[gw],
                locked_in,
                locked_out,
                excluded,
                max_per_pos=per_position_limit,
            )
            # Keep immediate and future-value discovery paths. The beam still
            # scores actual weekly lineups, costs and FT resources, not rank scores.
            seen = set()
            for move in immediate[:candidate_limit] + horizon[:candidate_limit]:
                key = (tuple(move["transfers_in"]), tuple(move["transfers_out"]))
                if key not in seen:
                    seen.add(key)
                    moves.append(move)
            for frame, mapping, seeds in (
                (pools[gw], maps[gw], immediate),
                (ranking_pools[gw], ranking_maps[gw], horizon),
            ):
                pairs = optimizer._get_candidate_2_transfers(
                    elements=state["elements"],
                    bank=state["bank"],
                    purchase_prices=state["purchase"],
                    pool_df=frame,
                    player_map=mapping,
                    locked_in=locked_in,
                    locked_out=locked_out,
                    excluded_teams=excluded,
                    cand_1_moves=seeds,
                )
                for move in pairs[:two_transfer_limit]:
                    key = (tuple(sorted(move["transfers_in"])), tuple(sorted(move["transfers_out"])))
                    if key not in seen:
                        seen.add(key)
                        moves.append(move)
            try:
                evaluate_lineup(state["elements"], maps[gw], risk)
            except ValueError:
                # A points shortlist is not a feasibility test. Solve a full
                # legal-XI repair with all market candidates and priced hits.
                sells = {
                    e: optimizer.calculate_selling_price(state["purchase"][e], int(maps[gw][e]["value"]))
                    for e in state["elements"]
                }
                repair_key = (
                    "repair",
                    gw,
                    tuple(sorted(sells.items())),
                    state["bank"],
                    state["ft"],
                    max_hits_per_move,
                    hit_cooldown_until,
                )
                if repair_key not in solve_cache:
                    try:
                        solve_cache[repair_key] = squad_optimizer.solve_best_squad(
                            pools[gw],
                            state["bank"] + sum(sells.values()),
                            locked_in_ids=list(locked_in),
                            locked_out_ids=list(locked_out),
                            excluded_team_ids=list(excluded),
                            captain_mean_only=True,
                            require_talisman=False,
                            bench_weight=0,
                            repair_elements=state["elements"],
                            repair_sell_prices=sells,
                            repair_free_transfers=state["ft"],
                            repair_hit_penalty=optimizer.hit_penalty,
                            repair_max_transfers=min(
                                2, state["ft"] + (0 if gw < hit_cooldown_until else max_hits_per_move)
                            ),
                        )["squad"]
                    except (ValueError, RuntimeError):
                        solve_cache[repair_key] = None
                repaired = solve_cache[repair_key]
                if repaired is not None:
                    new = set(int(e) for e in repaired["element"])
                    ins, outs = new - state["elements"], state["elements"] - new
                    buys = {e: int(maps[gw][e]["value"]) for e in ins}
                    moves.append(
                        dict(
                            transfers_in=sorted(ins),
                            transfers_out=sorted(outs),
                            buy_costs=buys,
                            bank_delta=sum(sells[e] for e in outs) - sum(buys.values()),
                        )
                    )
            branches = [(None, m) for m in moves]
            for chip in sorted(remaining & {"3xc", "bboost"}):
                branches += [(chip, m) for m in moves]
            # Reserve restructuring for every resource state, including hold-now.
            # Global top-two pruning unfairly denied conserved chips future use.
            if id(state) in restructure_ids:
                for chip in sorted(remaining & {"freehit", "wildcard"}):
                    if chip == "freehit" and (
                        (state["history"] and state["history"][-1].get("chip") == "freehit")
                        or (not state["history"] and previous_chip == "freehit")
                    ):
                        continue
                    budget = state["bank"] + sum(
                        optimizer.calculate_selling_price(state["purchase"][e], int(maps[gw][e]["value"]))
                        for e in state["elements"]
                    )
                    key = (gw, budget, chip)
                    if key not in solve_cache:
                        frame = pools[gw].copy()
                        if chip == "wildcard":
                            # Score permanent restructure on the remaining discounted horizon.
                            frame["expected_points"] = frame["element"].map(
                                {
                                    e: sum(
                                        optimizer.discount_factor**j
                                        * float(maps[g].get(e, {}).get("expected_points", 0))
                                        for j, g in enumerate(gameweeks[step:])
                                    )
                                    for e in maps[gw]
                                }
                            )
                        try:
                            solve_cache[key] = squad_optimizer.solve_best_squad(
                                frame,
                                budget,
                                locked_in_ids=list(locked_in),
                                locked_out_ids=list(locked_out),
                                excluded_team_ids=list(excluded),
                                captain_mean_only=risk == "points",
                            )["squad"]
                        except (ValueError, RuntimeError):
                            solve_cache[key] = None
                    squad = solve_cache[key]
                    if squad is not None:
                        new = set(int(e) for e in squad["element"])
                        ins, outs = new - state["elements"], state["elements"] - new
                        sells = sum(
                            optimizer.calculate_selling_price(state["purchase"][e], int(maps[gw][e]["value"]))
                            for e in outs
                        )
                        buys = {e: int(maps[gw][e]["value"]) for e in ins}
                        branches.append(
                            (
                                chip,
                                dict(
                                    transfers_in=sorted(ins),
                                    transfers_out=sorted(outs),
                                    buy_costs=buys,
                                    bank_delta=sells - sum(buys.values()),
                                ),
                            )
                        )
            scheduled = (forced_chip_schedule or {}).get(gw)
            if scheduled:
                branches = [(chip, move) for chip, move in branches if chip == scheduled]
            else:
                reserved = set((forced_chip_schedule or {}).values())
                branches = [(chip, move) for chip, move in branches if chip not in reserved]
            if step == 0 and forced_current_chip:
                branches = [(chip, move) for chip, move in branches if chip == forced_current_chip]
            for chip, move in branches:
                new = (state["elements"] - set(move["transfers_out"])) | set(move["transfers_in"])
                new_bank = state["bank"] + move["bank_delta"]
                if new_bank < 0:
                    continue
                new_purchase = {e: p for e, p in state["purchase"].items() if e in new}
                new_purchase.update(move["buy_costs"])
                count = len(move["transfers_in"])
                special = chip in {"freehit", "wildcard"}
                hits = 0 if special else max(0, count - state["ft"])
                if not special and (count > 2 or hits > max_hits_per_move or (hits and gw < hit_cooldown_until)):
                    continue
                from fpl_oracle.optimise.hit_policy import expected_transfer_gain

                transfer_gain = expected_transfer_gain(
                    move["transfers_in"], move["transfers_out"], maps, gameweeks[step:]
                )
                if hits and enforce_hit_gain and transfer_gain + 1e-9 < hits * optimizer.hit_penalty:
                    continue
                next_ft = state["ft"] if special else min(5, max(0, state["ft"] - count) + 1)
                try:
                    gross, captain, formation = evaluate_lineup(new, maps[gw], risk, chip)
                except ValueError:
                    continue
                net = gross - hits * optimizer.hit_penalty
                record = dict(
                    gameweek=gw,
                    chip=chip,
                    elements=sorted(new),
                    captain=captain,
                    formation=formation,
                    transfers_in=move["transfers_in"],
                    transfers_out=move["transfers_out"],
                    transfers_count=count,
                    bank=new_bank,
                    banked_ft=next_ft,
                    hits=hits,
                    hit_cost=hits * optimizer.hit_penalty,
                    expected_player_gain=transfer_gain,
                    hit_requires_prior_success=bool(hits and any(r["hits"] for r in state["history"][-6:])),
                    gross_xp=gross,
                    net_xp=net,
                )
                first_chip = chip if step == 0 else state["first_chip"]
                next_states.append(
                    dict(
                        elements=set(state["elements"]) if chip == "freehit" else new,
                        bank=state["bank"] if chip == "freehit" else new_bank,
                        purchase=dict(state["purchase"]) if chip == "freehit" else new_purchase,
                        ft=next_ft,
                        remaining=remaining - {chip} if chip else remaining,
                        history=state["history"] + [record],
                        first_chip=first_chip,
                        score=state["score"] + optimizer.discount_factor**step * (net - costs.get(chip, 0)),
                    )
                )
        # A future-ranked player must survive the current-week beam too.
        # Use a no-more-transfers lookahead only for pruning, never for the
        # reported plan score or hit-gain gate.
        future_cache = {}

        def pruning_value(state, step=step, future_cache=future_cache):
            key = tuple(sorted(state["elements"]))
            if key in future_cache:
                return state["score"] + future_cache[key]
            future = 0.0
            for j, later in enumerate(gameweeks[step + 1 :], start=step + 1):
                try:
                    if risk == "points":
                        from fpl_oracle.optimise.transfers import VALID_FORMATIONS

                        groups = {
                            pos: sorted(
                                (
                                    float(maps[later][e]["expected_points"])
                                    for e in state["elements"]
                                    if maps[later][e]["position"] == pos
                                    and not maps[later][e].get("simulation_unavailable", False)
                                ),
                                reverse=True,
                            )
                            for pos in ("GKP", "DEF", "MID", "FWD")
                        }
                        scores = []
                        for formation in VALID_FORMATIONS:
                            counts = (1, *formation)
                            if all(len(groups[pos]) >= count for pos, count in zip(groups, counts, strict=True)):
                                chosen = [
                                    xp for pos, count in zip(groups, counts, strict=True) for xp in groups[pos][:count]
                                ]
                                scores.append(sum(chosen) + max(chosen))
                        if not scores:
                            raise ValueError("No legal future XI")
                        gross = max(scores)
                    else:
                        gross, _, _ = evaluate_lineup(state["elements"], maps[later], risk)
                except ValueError:
                    continue
                future += optimizer.discount_factor**j * gross
            future_cache[key] = future
            return state["score"] + future

        next_states.sort(key=pruning_value, reverse=True)
        groups: dict[Any, list] = {}
        for state in next_states:
            key = (state["first_chip"], state["remaining"], state["ft"])
            if len(groups.setdefault(key, [])) < 2:
                groups[key].append(state)
        # Bounded total; preserve one best branch per resource state before runners-up.
        selected = [v[0] for v in groups.values()]
        selected.sort(key=pruning_value, reverse=True)
        first_best = {}
        for candidate in selected:
            first_best.setdefault((candidate["first_chip"], candidate["ft"]), candidate)
        # Keep a frontier that retains each available chip so the endpoint can
        # report preservation break-even instead of silently spending everything.
        retain_best = []
        for resource in sorted(available):
            candidate = next((row for row in selected if resource in row["remaining"]), None)
            if candidate is not None and not any(candidate is row for row in first_best.values()):
                if not any(candidate is row for row in retain_best):
                    retain_best.append(candidate)
        states = list(first_best.values()) + retain_best
        states += [v for v in selected if not any(v is kept for kept in states)][: max(0, beam_width - len(states))]
    return sorted(states, key=lambda s: s["score"], reverse=True)
