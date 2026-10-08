"""Local owner-confirmed deadline state. Never sends changes to FPL."""

import json
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import Column, String, Table, Text

from fpl_oracle.data.store import Base, data_store, engine


class TeamConfirmationRecord(Base):
    __tablename__ = "team_confirmations"
    key = Column(String(100), primary_key=True)
    payload = Column(Text, nullable=False)


cast(Table, TeamConfirmationRecord.__table__).create(bind=engine, checkfirst=True)


class TeamConfirmation:
    @staticmethod
    def identity(manager_id):
        return str(manager_id or "manual")

    def read(self, manager_id):
        with data_store.get_session() as session:
            row = session.get(TeamConfirmationRecord, self.identity(manager_id))
            return json.loads(row.payload) if row else None

    def write(self, manager_id, record):
        from fpl_oracle.server.safe_json import safe_json_serialize

        record = safe_json_serialize(record)
        with data_store.get_session() as session:
            row = session.get(TeamConfirmationRecord, self.identity(manager_id))
            if row is None:
                session.add(TeamConfirmationRecord(key=self.identity(manager_id), payload=json.dumps(record)))
            else:
                row.payload = json.dumps(record)
            session.commit()

    def remove(self, manager_id):
        with data_store.get_session() as session:
            row = session.get(TeamConfirmationRecord, self.identity(manager_id))
            if row:
                session.delete(row)
                session.commit()

    def locked(self, state):
        record = self.read(state.manager_id)
        return bool(
            record
            and record.get("locked")
            and record["gameweek"] == state.target_gw
            and record.get("deadline") == state.deadline_utc
            and (
                not state.deadline_utc
                or datetime.fromisoformat(state.deadline_utc.replace("Z", "+00:00")) > datetime.now(UTC)
            )
        )

    def recommendation(self, manager_id):
        with data_store.get_session() as session:
            row = session.get(TeamConfirmationRecord, "recommendation:" + self.identity(manager_id))
            return json.loads(row.payload) if row else None

    def save_recommendation(self, state, plan):
        record = dict(
            gameweek=state.target_gw,
            baseline=[p.element for p in state.squad],
            ins=[p["element"] for p in plan["transfers_in"]],
            outs=[p["element"] for p in plan["transfers_out"]],
            hit_cost=plan["hit_cost"],
            expected_gain=plan["trajectory"][0].get("expected_player_gain"),
        )
        key = "recommendation:" + self.identity(state.manager_id)
        with data_store.get_session() as session:
            row = session.get(TeamConfirmationRecord, key)
            if row is None:
                session.add(TeamConfirmationRecord(key=key, payload=json.dumps(record)))
            else:
                row.payload = json.dumps(record)
            session.commit()

    def followed_payload(self, state, bootstrap, recommendation):
        if not recommendation or recommendation.get("gameweek") != state.target_gw:
            raise ValueError("No current published advice to confirm.")
        rec = recommendation
        current = {p.element for p in state.squad}
        baseline = set(rec["baseline"])
        selected = (baseline - set(rec["outs"])) | set(rec["ins"])
        previous = self.read(state.manager_id)
        if (
            current == selected
            and previous
            and previous["gameweek"] == state.target_gw
            and previous.get("followed_recommendation") == rec
        ):
            return dict(previous)
        if (
            current != baseline
            or state.bank_tenths != rec["baseline_bank"]
            or state.free_transfers != rec["baseline_ft"]
        ):
            raise ValueError("Your team changed since that advice. Confirm what you did instead.")
        elements = {e.id: e for e in bootstrap.elements}
        purchases = {int(k): v for k, v in rec["baseline_purchase"].items() if int(k) in selected}
        from fpl_oracle.domain.manager_state import ManagerStateService

        for eid in rec["ins"]:
            paid = rec.get("buy_prices", {}).get(str(eid), rec.get("buy_prices", {}).get(eid))
            if paid != elements[eid].now_cost:
                raise ValueError("A buy price changed since the advice. Confirm the bank and moves instead.")
            purchases[eid] = paid
        selling = {
            eid: ManagerStateService.calculate_selling_price(purchases[eid], elements[eid].now_cost) for eid in selected
        }
        # A price move makes the stored predicted bank unsafe to use as fact.
        for eid in rec["outs"]:
            bought = int(rec["baseline_purchase"].get(str(eid), rec["baseline_purchase"].get(eid)))
            old_sell = int(rec["baseline_selling"].get(str(eid), rec["baseline_selling"].get(eid)))
            if ManagerStateService.calculate_selling_price(bought, elements[eid].now_cost) != old_sell:
                raise ValueError("A selling price changed since the advice. Confirm the bank and moves instead.")
        old_cost = previous["hit_cost"] if previous and previous["gameweek"] == state.target_gw else 0
        chip = rec.get("chip") or state.active_chip
        available = list(state.chips_remaining_set_1 if state.target_gw <= 19 else state.chips_remaining_set_2)
        return dict(
            gameweek=state.target_gw,
            player_ids=sorted(selected),
            bank_tenths=rec["remaining_bank"],
            free_transfers=state.free_transfers
            if chip in {"wildcard", "freehit"}
            else max(0, state.free_transfers - len(rec["ins"])),
            available_chips=[c for c in available if c != chip],
            active_chip=chip,
            hit_cost=old_cost + rec["hit_cost"],
            purchase_prices=purchases,
            selling_prices=selling,
            captain=rec["captain"],
            vice_captain=rec["vice_captain"],
            bench=rec["bench"],
            followed_recommendation=rec,
            locked=True,
            undo_state=previous,
        )

    def confirm(self, state, bootstrap, payload, recommendation=None):
        if state.deadline_utc and datetime.fromisoformat(state.deadline_utc.replace("Z", "+00:00")) <= datetime.now(
            UTC
        ):
            raise ValueError("Deadline passed. Reload the current gameweek before confirming.")
        ids = payload["player_ids"]
        if len(ids) != 15 or len(set(ids)) != 15:
            raise ValueError("Confirm 15 different players.")
        elements = {e.id: e for e in bootstrap.elements}
        if not set(ids).issubset(elements):
            raise ValueError("Some players are missing from current FPL data.")
        from collections import Counter

        positions = Counter(elements[e].element_type for e in ids)
        clubs = Counter(elements[e].team for e in ids)
        if positions != {1: 2, 2: 5, 3: 5, 4: 3} or max(clubs.values()) > 3:
            raise ValueError("Squad needs 2 keepers, 5 defenders, 5 midfielders, 3 forwards and at most 3 per club.")
        if payload["gameweek"] != state.target_gw:
            raise ValueError("Deadline changed. Reload and confirm the new gameweek.")
        if payload["bank_tenths"] < 0 or not 0 <= payload["free_transfers"] <= 5:
            raise ValueError("Check bank and remaining free transfers.")
        cost = payload["hit_cost"]
        if cost < 0 or cost % 4:
            raise ValueError("Taken hit cost must be zero or a multiple of 4.")
        chips = set(payload["available_chips"])
        if not chips.issubset({"wildcard", "freehit", "3xc", "bboost"}):
            raise ValueError("Unknown chip.")
        active = payload.get("active_chip")
        if active and active not in {"wildcard", "freehit", "3xc", "bboost"}:
            raise ValueError("Unknown active chip.")
        if active in chips:
            raise ValueError("An active chip cannot also be marked unused.")
        sold = {int(k): int(v) for k, v in payload["selling_prices"].items()}
        if set(sold) != set(ids) or any(v <= 0 or v > elements[e].now_cost for e, v in sold.items()):
            raise ValueError("Confirm each player's selling price from FPL.")
        from fpl_oracle.domain.manager_state import ManagerStateService

        purchases = {int(k): int(v) for k, v in payload["purchase_prices"].items()}
        if set(purchases) != set(ids) or any(
            v <= 0 or ManagerStateService.calculate_selling_price(v, elements[e].now_cost) != sold[e]
            for e, v in purchases.items()
        ):
            raise ValueError("Purchase and selling prices do not agree with FPL selling rules.")
        previous = self.read(state.manager_id)
        if recommendation and recommendation.get("gameweek") != state.target_gw:
            recommendation = None
        baseline = (
            set(previous["player_ids"])
            if previous and previous["gameweek"] == state.target_gw
            else {p.element for p in state.squad}
        )
        if recommendation and recommendation.get("baseline"):
            baseline = set(recommendation["baseline"])
        ins, outs = set(ids) - baseline, baseline - set(ids)
        expected = recommendation or {}
        expected_ins = set(expected.get("ins", []))
        expected_outs = set(expected.get("outs", []))
        followed = "no recommendation saved"
        if expected:
            followed = (
                "followed"
                if ins == expected_ins and outs == expected_outs
                else "partly followed"
                if ins & expected_ins or outs & expected_outs
                else "different"
            )
        origin = (
            set(previous.get("origin_ids", previous["player_ids"]))
            if previous and previous["gameweek"] == state.target_gw
            else {p.element for p in state.squad}
        )
        record = dict(
            payload,
            selling_prices=sold,
            purchase_prices=purchases,
            ins=sorted(ins),
            outs=sorted(outs),
            comparison=followed,
            confirmed_at=datetime.now(UTC).isoformat(),
            deadline=state.deadline_utc,
            source="user_confirmed",
            origin_ids=sorted(origin),
            actual_ins=sorted(set(ids) - origin),
            actual_outs=sorted(origin - set(ids)),
            manager_id=state.manager_id,
        )
        if previous and previous["gameweek"] == state.target_gw:
            if payload["hit_cost"] < previous["hit_cost"]:
                raise ValueError("Taken hit cost cannot decrease within a gameweek.")
            if not ins and not outs:
                record["ins"], record["outs"] = previous["ins"], previous["outs"]
                record["comparison"] = previous["comparison"]
        if payload.get("captain") and payload["captain"] not in ids:
            raise ValueError("Captain must be in your current team.")
        if payload.get("vice_captain") and (
            payload["vice_captain"] not in ids or payload["vice_captain"] == payload.get("captain")
        ):
            raise ValueError("Vice captain must be a different owned player.")
        bench = payload.get("bench", [])
        if bench and (
            len(bench) != 4
            or len(set(bench)) != 4
            or not set(bench).issubset(ids)
            or sum(elements[e].element_type == 1 for e in bench) != 1
        ):
            raise ValueError("Bench needs one keeper and three different owned outfield players.")
        self.write(state.manager_id, record)
        return record

    def apply(self, state, bootstrap):
        state.confirmation_required = True
        record = self.read(state.manager_id)
        if not record or record["gameweek"] != state.target_gw or record.get("deadline") != state.deadline_utc:
            state.team_confirmed = False
            if record and record["gameweek"] == state.current_gw and state.manager_id:
                expected = set(record["player_ids"])
                if record.get("active_chip") == "freehit":
                    expected = set(record.get("followed_recommendation", {}).get("baseline", []))
                official = {p.element for p in state.squad}
                if expected != official:
                    state.error_message = "Official published team differs from your last confirmed team. Check both teams before confirming the next deadline."
                elif record.get("locked") and not state.is_stale and not state.error_message:
                    # Carry only after the exact applied squad is seen in public picks.
                    # This does not claim to see later pending pre-deadline changes.
                    if record.get("active_chip") != "freehit":
                        from fpl_oracle.domain.manager_state import ManagerStateService, PriceProvenance

                        elems = {e.id: e for e in bootstrap.elements}
                        for player in state.squad:
                            paid = record["purchase_prices"].get(
                                str(player.element), record["purchase_prices"].get(player.element)
                            )
                            if paid is not None:
                                player.purchase_price = paid
                                player.selling_price = ManagerStateService.calculate_selling_price(
                                    paid, elems[player.element].now_cost
                                )
                                player.price_provenance = PriceProvenance.MANUAL_OVERRIDE
                    carried = dict(
                        gameweek=state.target_gw,
                        player_ids=sorted(official),
                        bank_tenths=state.bank_tenths,
                        free_transfers=state.free_transfers,
                        available_chips=state.chips_remaining_set_1
                        if state.target_gw <= 19
                        else state.chips_remaining_set_2,
                        active_chip=None,
                        hit_cost=0,
                        purchase_prices={p.element: p.purchase_price for p in state.squad},
                        selling_prices={p.element: p.selling_price for p in state.squad},
                        confirmed_at=datetime.now(UTC).isoformat(),
                        deadline=state.deadline_utc,
                        source="official_reconciled",
                        origin_ids=sorted(official),
                        actual_ins=[],
                        actual_outs=[],
                        ins=[],
                        outs=[],
                        comparison="official reconciled",
                        locked=False,
                        manager_id=state.manager_id,
                    )
                    self.write(state.manager_id, carried)
                    state.team_confirmed = True
                    state.confirmation_as_of = carried["confirmed_at"]
                    state.confirmation_comparison = "official reconciled"
            return state
        elements = {e.id: e for e in bootstrap.elements}
        if not set(record["player_ids"]).issubset(elements):
            state.team_confirmed = False
            return state
        from fpl_oracle.domain.manager_state import ManagerStateService, PlayerSquadState, PriceProvenance

        positions = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        state.squad = [
            PlayerSquadState(
                element=eid,
                web_name=elements[eid].web_name,
                team=elements[eid].team,
                position=positions[elements[eid].element_type],
                now_cost=elements[eid].now_cost,
                purchase_price=int(record["purchase_prices"][str(eid)]),
                selling_price=ManagerStateService.calculate_selling_price(
                    int(record["purchase_prices"][str(eid)]), elements[eid].now_cost
                ),
                price_provenance=PriceProvenance.MANUAL_OVERRIDE,
                status=elements[eid].status,
                news=elements[eid].news,
                chance_of_playing=elements[eid].chance_of_playing_next_round,
                is_captain=eid == record.get("captain"),
                is_vice_captain=eid == record.get("vice_captain"),
                is_starter=eid not in record.get("bench", []),
                bench_order=(record["bench"].index(eid) + 1) if eid in record.get("bench", []) else 0,
            )
            for eid in record["player_ids"]
        ]
        state.bank_tenths = record["bank_tenths"]
        state.bank_source = "confirmed_current_team"
        state.free_transfers = record["free_transfers"]
        state.ft_source = "confirmed_current_team"
        if state.target_gw <= 19:
            state.chips_remaining_set_1 = record["available_chips"]
        else:
            state.chips_remaining_set_2 = record["available_chips"]
        state.active_chip = record.get("active_chip")
        state.confidence = "user_confirmed"
        state.team_confirmed = not state.is_stale and not state.error_message
        if state.active_chip == "freehit":
            state.team_confirmed = False
            state.error_message = "Active Free Hit needs the permanent pre-chip team and bank for future weeks. Current advice is blocked rather than assuming the temporary squad is permanent."
        if state.deadline_utc and datetime.fromisoformat(state.deadline_utc.replace("Z", "+00:00")) <= datetime.now(
            UTC
        ):
            state.team_confirmed = False
            state.error_message = "Deadline passed; refresh and confirm the new gameweek."

        state.confirmation_as_of = record["confirmed_at"]
        state.confirmation_comparison = record["comparison"]
        return state


team_confirmation = TeamConfirmation()
