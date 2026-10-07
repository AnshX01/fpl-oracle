"""Bounded legal transfer/chip beam. No inferred rival captains or optimality claim."""

from typing import Any

from fpl_oracle.optimise.squad import squad_optimizer


def evaluate_lineup(elements, player_map, risk="balanced", chip=None):
    from fpl_oracle.optimise.transfers import VALID_FORMATIONS

    players = [player_map[e] for e in sorted(elements)]
    groups = {
        p: sorted(
            [r for r in players if r["position"] == p], key=lambda r: (-float(r["expected_points"]), int(r["element"]))
        )
        for p in ("GKP", "DEF", "MID", "FWD")
    }
    best = None
    multiplier = 2 if chip == "3xc" else 1
    for formation in VALID_FORMATIONS:
        counts = dict(zip(("GKP", "DEF", "MID", "FWD"), (1, *formation), strict=True))
        for cap in players:
            chosen = []
            for pos, count in counts.items():
                options = groups[pos]
                if cap["position"] == pos:
                    options = [cap] + [r for r in options if r["element"] != cap["element"]]
                chosen += options[:count]
            if len(chosen) != 11 or not any(r["element"] == cap["element"] for r in chosen):
                continue
            xp = float(cap["expected_points"])
            if risk == "conservative":
                score = 0.6 * xp + 0.4 * float(cap.get("p10", xp * 0.4))
            elif risk == "aggressive":
                score = 0.4 * xp + 0.6 * float(cap.get("p90", xp * 1.8))
            else:
                score = 0.7 * xp + 0.3 * float(cap.get("p90", xp * 1.8))
            starter_xp = sum(float(r["expected_points"]) for r in chosen)
            objective = starter_xp + multiplier * score
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
    solve_cache: dict[Any, Any] = {}
    for step, gw in enumerate(gameweeks):
        next_states = []
        restructure_parents = {}
        for parent in states:
            group = (parent["first_chip"], parent["remaining"])
            restructure_parents.setdefault(group, []).append(parent)
        restructure_ids = {id(parent) for parents in restructure_parents.values() for parent in parents[:1]}
        for state in states:
            remaining = state["remaining"]
            if step and gameweeks[step - 1] <= 19 < gw:
                # Do not infer the owner's second set from first-set availability.
                remaining = frozenset((chips_by_set or {}).get(2, []))
            moves = [dict(transfers_in=[], transfers_out=[], buy_costs={}, bank_delta=0)]
            moves += optimizer._get_candidate_1_transfers(
                state["elements"],
                state["bank"],
                state["purchase"],
                pools[gw],
                maps[gw],
                locked_in,
                locked_out,
                excluded,
                max_per_pos=2,
            )[:2]
            moves += optimizer._get_candidate_2_transfers(
                elements=state["elements"],
                bank=state["bank"],
                purchase_prices=state["purchase"],
                pool_df=pools[gw],
                player_map=maps[gw],
                locked_in=locked_in,
                locked_out=locked_out,
                excluded_teams=excluded,
                cand_1_moves=moves[1:],
            )[:1]
            branches = [(None, m) for m in moves]
            for chip in sorted(remaining & {"3xc", "bboost"}):
                branches += [(chip, m) for m in moves]
            # Reserve restructuring for every resource state, including hold-now.
            # Global top-two pruning unfairly denied conserved chips future use.
            if id(state) in restructure_ids:
                for chip in sorted(remaining & {"freehit", "wildcard"}):
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
                next_ft = min(5, state["ft"] + 1) if special else min(5, max(0, state["ft"] - count) + 1)
                gross, captain, formation = evaluate_lineup(new, maps[gw], risk, chip)
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
        # Keep distinct resource states: chip conservation must survive early score pruning.
        next_states.sort(key=lambda s: s["score"], reverse=True)
        groups: dict[Any, list] = {}
        for state in next_states:
            key = (state["first_chip"], state["remaining"])
            if len(groups.setdefault(key, [])) < 2:
                groups[key].append(state)
        # Bounded total; preserve one best branch per resource state before runners-up.
        selected = [v[0] for v in groups.values()]
        selected.sort(key=lambda s: s["score"], reverse=True)
        first_best = {}
        for candidate in selected:
            first_best.setdefault(candidate["first_chip"], candidate)
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
