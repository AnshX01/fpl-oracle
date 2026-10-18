"""Decision dependencies, excluding live display counters and global ownership."""

import json

PLAYER_FIELDS = (
    "id",
    "code",
    "web_name",
    "first_name",
    "second_name",
    "team",
    "element_type",
    "now_cost",
    "cost_change_start",
    "status",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "news",
    "news_added",
    "scout_risks",
)
TEAM_FIELDS = (
    "id",
    "name",
    "short_name",
    "strength_attack_home",
    "strength_attack_away",
    "strength_defence_home",
    "strength_defence_away",
)
EVENT_FIELDS = ("id", "deadline_time", "is_current", "is_next", "finished", "data_checked")
FIXTURE_FIELDS = (
    "id",
    "event",
    "team_h",
    "team_a",
    "kickoff_time",
    "team_h_difficulty",
    "team_a_difficulty",
)


def picked(row, fields):
    return {field: row.get(field) for field in fields}


def rows(values, fields=None):
    result = [picked(row, fields) if fields else row for row in (values or [])]
    return sorted(result, key=lambda row: json.dumps(row, sort_keys=True, default=str))


def base_key(key):
    """Cache keys carry a season prefix ('2026-27:bootstrap-static'); compare on the source name."""
    head, _, rest = str(key).partition(":")
    return rest if rest and len(head) == 7 and head[4] == "-" else str(key)


def decision_source(key, value):
    key = base_key(key)
    parts = key.split(":")
    if parts[0] == "entry" and isinstance(value, dict):
        # Live points and ranks tick during matches; the decision uses money, transfers, chips and the squad.
        if len(parts) == 2:
            return picked(value, ("id", "last_deadline_bank", "last_deadline_value", "last_deadline_total_transfers"))
        if parts[-1] == "history":
            return {
                "gameweeks": rows(
                    value.get("current"), ("event", "event_transfers", "event_transfers_cost", "bank", "value")
                ),
                "chips": rows(value.get("chips")),
            }
        if parts[-1] == "picks":
            return {
                "chip": value.get("active_chip"),
                "picks": rows(value.get("picks"), ("element", "position", "is_captain", "is_vice_captain")),
                "money": picked(
                    value.get("entry_history") or {}, ("bank", "value", "event_transfers", "event_transfers_cost")
                ),
            }
    if key.startswith("league-classic:") and isinstance(value, dict):
        # Live point totals tick during matches; the advice depends on who is in the league, not the live score.
        standings = (value.get("standings") or {}).get("results") or []
        return sorted(row.get("entry") for row in standings)
    if key == "bootstrap-static":
        return {
            "players": rows(value.get("elements"), PLAYER_FIELDS),
            "teams": rows(value.get("teams"), TEAM_FIELDS),
            "events": rows(value.get("events"), EVENT_FIELDS),
            "chips": rows(value.get("chips")),
            "positions": rows(
                value.get("element_types"),
                ("id", "singular_name_short", "squad_select", "squad_min_play", "squad_max_play"),
            ),
            "rules": {
                "settings": picked(
                    value.get("game_settings") or {},
                    (
                        "squad_squadsize",
                        "squad_squadplay",
                        "squad_team_limit",
                        "squad_total_spend",
                        "transfers_sell_on_fee",
                        "max_extra_free_transfers",
                        "max_extra_transfers",
                        "element_sell_at_purchase_price",
                    ),
                ),
                "scoring": (value.get("game_config") or {}).get("scoring"),
            },
        }
    if key.startswith("fixtures:") and isinstance(value, list):
        return rows(value, FIXTURE_FIELDS)
    return value


def changes(key, before, after):
    old, new = decision_source(key, before), decision_source(key, after)
    if old == new:
        return []
    key = base_key(key)
    if key == "bootstrap-static":
        categories = [name for name in old if old[name] != new[name]]
        old_players = {r["id"]: r for r in old["players"]}
        new_players = {r["id"]: r for r in new["players"]}
        if "players" in categories and old_players.keys() == new_players.keys():
            fields = {
                field
                for eid in old_players
                for field in PLAYER_FIELDS
                if old_players[eid][field] != new_players[eid][field]
            }
            if fields <= {"now_cost", "cost_change_start"}:
                categories[categories.index("players")] = "prices"
            elif fields <= {
                "status",
                "chance_of_playing_next_round",
                "chance_of_playing_this_round",
                "news",
                "news_added",
                "scout_risks",
            }:
                categories[categories.index("players")] = "availability"
        return categories
    return ["fixtures" if key.startswith("fixtures:") else "owner_or_rival_data"]
