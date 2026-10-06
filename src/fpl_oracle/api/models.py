"""
Pydantic v2 models for FPL API data.
Defensive schemas with extra="allow" to ensure resilience against API changes.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FPLBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ElementType(FPLBaseModel):
    id: int
    plural_name: str
    plural_name_short: str
    singular_name: str
    singular_name_short: str
    squad_select: int = 0
    squad_min_play: int = 0
    squad_max_play: int = 0


class Team(FPLBaseModel):
    id: int
    name: str
    short_name: str
    strength: int | None = 3
    strength_overall_home: int | None = 1000
    strength_overall_away: int | None = 1000
    strength_attack_home: int | None = 1000
    strength_attack_away: int | None = 1000
    strength_defence_home: int | None = 1000
    strength_defence_away: int | None = 1000


class GameweekEvent(FPLBaseModel):
    id: int
    name: str
    deadline_time: str
    is_current: bool = False
    is_next: bool = False
    finished: bool = False
    data_checked: bool = False
    average_entry_score: int | None = None
    highest_score: int | None = None


class ChipDefinition(FPLBaseModel):
    name: str
    chip_type: str
    start_event: int
    stop_event: int


class Element(FPLBaseModel):
    id: int
    web_name: str
    first_name: str | None = ""
    second_name: str | None = ""
    team: int
    element_type: int
    now_cost: int  # In tenths, e.g. 100 = £10.0m
    selected_by_percent: str | float | None = "0.0"
    form: str | float | None = "0.0"
    points_per_game: str | float | None = "0.0"
    total_points: int = 0
    status: str = "a"  # a: available, d: doubtful, i: injured, s: suspended, u: unavailable
    news: str | None = ""
    news_added: str | None = None
    chance_of_playing_next_round: int | None = None
    chance_of_playing_this_round: int | None = None
    minutes: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    own_goals: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0
    influence: str | float | None = "0.0"
    creativity: str | float | None = "0.0"
    threat: str | float | None = "0.0"
    ict_index: str | float | None = "0.0"
    starts: int | None = 0
    expected_goals: str | float | None = "0.0"
    expected_assists: str | float | None = "0.0"
    expected_goal_involvements: str | float | None = "0.0"
    expected_goals_conceded: str | float | None = "0.0"
    defensive_contribution: int | None = 0
    transfers_in_event: int | None = 0
    transfers_out_event: int | None = 0
    price_change_percent: str | float | None = None
    price_change_hourly_rate: int | None = None
    price_change_projections: list[dict[str, Any]] | None = None


class BootstrapStatic(FPLBaseModel):
    chips: list[ChipDefinition] = []
    events: list[GameweekEvent] = []
    teams: list[Team] = []
    elements: list[Element] = []
    element_types: list[ElementType] = []
    game_settings: dict[str, Any] = {}
    game_config: dict[str, Any] | None = None


class FixtureStatItem(FPLBaseModel):
    value: int
    element: int


class FixtureStat(FPLBaseModel):
    identifier: str
    a: list[FixtureStatItem] = []
    h: list[FixtureStatItem] = []


class Fixture(FPLBaseModel):
    id: int
    code: int
    event: int | None = None
    finished: bool = False
    finished_provisional: bool = False
    kickoff_time: str | None = None
    minutes: int = 0
    started: bool = False
    team_a: int
    team_a_score: int | None = None
    team_h: int
    team_h_score: int | None = None
    team_h_difficulty: int | None = 3
    team_a_difficulty: int | None = 3
    stats: list[FixtureStat] = []


class PlayerMatchHistory(FPLBaseModel):
    element: int
    fixture: int
    opponent_team: int
    total_points: int = 0
    was_home: bool = True
    kickoff_time: str | None = None
    team_h_score: int | None = None
    team_a_score: int | None = None
    round: int
    minutes: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    own_goals: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0
    influence: str | float | None = "0.0"
    creativity: str | float | None = "0.0"
    threat: str | float | None = "0.0"
    ict_index: str | float | None = "0.0"
    starts: int | None = 0
    defensive_contribution: int | None = 0
    expected_goals: str | float | None = "0.0"
    expected_assists: str | float | None = "0.0"
    expected_goal_involvements: str | float | None = "0.0"
    expected_goals_conceded: str | float | None = "0.0"
    value: int = 50
    selected: int | None = 0
    transfers_balance: int | None = 0


class ElementSummary(FPLBaseModel):
    fixtures: list[dict[str, Any]] = []
    history: list[PlayerMatchHistory] = []
    history_past: list[dict[str, Any]] = []


class Pick(FPLBaseModel):
    element: int
    position: int  # 1-15: 1-11 starters, 12-15 bench
    multiplier: int = 1  # 1: normal, 2: captain, 3: triple captain, 0: benched
    is_captain: bool = False
    is_vice_captain: bool = False


class EntryHistory(FPLBaseModel):
    event: int
    points: int = 0
    total_points: int = 0
    rank: int | None = None
    rank_sort: int | None = None
    overall_rank: int | None = None
    bank: int = 0  # In tenths (£1.0m = 10)
    value: int = 1000  # In tenths
    event_transfers: int = 0
    event_transfers_cost: int = 0
    points_on_bench: int = 0


class SquadPicks(FPLBaseModel):
    active_chip: str | None = None
    entry_history: EntryHistory | None = None
    picks: list[Pick] = []


class ChipHistoryItem(FPLBaseModel):
    name: str  # e.g. "bboost", "3xc", "freehit", "wildcard"
    time: str | None = None
    event: int


class ManagerHistory(FPLBaseModel):
    current: list[EntryHistory] = []
    past: list[dict[str, Any]] = []
    chips: list[ChipHistoryItem] = []


class ManagerEntry(FPLBaseModel):
    id: int
    player_first_name: str = ""
    player_last_name: str = ""
    name: str = ""  # Team name
    summary_overall_points: int | None = 0
    summary_overall_rank: int | None = None
    current_event: int | None = None
    last_deadline_bank: int | None = 0
    last_deadline_value: int | None = 1000
    last_deadline_total_transfers: int | None = 0
    leagues: dict[str, Any] | None = None


class TransferHistoryItem(FPLBaseModel):
    element_in: int
    element_in_cost: int
    element_out: int
    element_out_cost: int
    entry: int
    event: int
    time: str


class ClassicStandingResult(FPLBaseModel):
    id: int | None = None
    club_badge_src: str | None = None
    event_total: int = 0
    player_name: str = ""
    rank: int = 0
    last_rank: int = 0
    rank_sort: int = 0
    total: int = 0
    entry: int
    entry_name: str = ""


class StandingsPage(FPLBaseModel):
    has_next: bool = False
    page: int = 1
    results: list[ClassicStandingResult] = []


class ClassicLeagueResponse(FPLBaseModel):
    league: dict[str, Any] = {}
    standings: StandingsPage = Field(default_factory=StandingsPage)
