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
    "finished",
)


def picked(row, fields):
    return {field: row.get(field) for field in fields}


def rows(values, fields=None):
    result = [picked(row, fields) if fields else row for row in (values or [])]
    return sorted(result, key=lambda row: json.dumps(row, sort_keys=True, default=str))


def decision_source(key, value):
    if key == "bootstrap-static":
        return {
            "players": rows(value.get("elements"), PLAYER_FIELDS),
            "teams": rows(value.get("teams"), TEAM_FIELDS),
            "events": rows(value.get("events"), EVENT_FIELDS),
            "chips": rows(value.get("chips")),
            "positions": rows(value.get("element_types")),
            "rules": {k: value.get(k) for k in ("game_settings", "game_config")},
        }
    if key.startswith("fixtures:") and isinstance(value, list):
        return rows(value, FIXTURE_FIELDS)
    return value


def changes(key, before, after):
    old, new = decision_source(key, before), decision_source(key, after)
    if old == new:
        return []
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
