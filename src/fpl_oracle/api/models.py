"""
Pydantic v2 models for FPL API data.
Defensive schemas with extra="allow" to ensure resilience against API changes.
"""

from typing import List, Optional, Dict, Any, Union
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
    strength: Optional[int] = 3
    strength_overall_home: Optional[int] = 1000
    strength_overall_away: Optional[int] = 1000
    strength_attack_home: Optional[int] = 1000
    strength_attack_away: Optional[int] = 1000
    strength_defence_home: Optional[int] = 1000
    strength_defence_away: Optional[int] = 1000

class GameweekEvent(FPLBaseModel):
    id: int
    name: str
    deadline_time: str
    is_current: bool = False
    is_next: bool = False
    finished: bool = False
    data_checked: bool = False
    average_entry_score: Optional[int] = None
    highest_score: Optional[int] = None

class ChipDefinition(FPLBaseModel):
    name: str
    chip_type: str
    start_event: int
    stop_event: int

class Element(FPLBaseModel):
    id: int
    web_name: str
    first_name: Optional[str] = ""
    second_name: Optional[str] = ""
    team: int
    element_type: int
    now_cost: int # In tenths, e.g. 100 = £10.0m
    selected_by_percent: Optional[Union[str, float]] = "0.0"
    form: Optional[Union[str, float]] = "0.0"
    points_per_game: Optional[Union[str, float]] = "0.0"
    total_points: int = 0
    status: str = "a" # a: available, d: doubtful, i: injured, s: suspended, u: unavailable
    news: Optional[str] = ""
    news_added: Optional[str] = None
    chance_of_playing_next_round: Optional[int] = None
    chance_of_playing_this_round: Optional[int] = None
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
    influence: Optional[Union[str, float]] = "0.0"
    creativity: Optional[Union[str, float]] = "0.0"
    threat: Optional[Union[str, float]] = "0.0"
    ict_index: Optional[Union[str, float]] = "0.0"
    starts: Optional[int] = 0
    expected_goals: Optional[Union[str, float]] = "0.0"
    expected_assists: Optional[Union[str, float]] = "0.0"
    expected_goal_involvements: Optional[Union[str, float]] = "0.0"
    expected_goals_conceded: Optional[Union[str, float]] = "0.0"
    defensive_contribution: Optional[int] = 0
    transfers_in_event: Optional[int] = 0
    transfers_out_event: Optional[int] = 0
    price_change_percent: Optional[Union[str, float]] = None
    price_change_hourly_rate: Optional[int] = None
    price_change_projections: Optional[List[Dict[str, Any]]] = None

class BootstrapStatic(FPLBaseModel):
    chips: List[ChipDefinition] = []
    events: List[GameweekEvent] = []
    teams: List[Team] = []
    elements: List[Element] = []
    element_types: List[ElementType] = []
    game_settings: Dict[str, Any] = {}
    game_config: Optional[Dict[str, Any]] = None

class FixtureStatItem(FPLBaseModel):
    value: int
    element: int

class FixtureStat(FPLBaseModel):
    identifier: str
    a: List[FixtureStatItem] = []
    h: List[FixtureStatItem] = []

class Fixture(FPLBaseModel):
    id: int
    code: int
    event: Optional[int] = None
    finished: bool = False
    finished_provisional: bool = False
    kickoff_time: Optional[str] = None
    minutes: int = 0
    started: bool = False
    team_a: int
    team_a_score: Optional[int] = None
    team_h: int
    team_h_score: Optional[int] = None
    team_h_difficulty: Optional[int] = 3
    team_a_difficulty: Optional[int] = 3
    stats: List[FixtureStat] = []

class PlayerMatchHistory(FPLBaseModel):
    element: int
    fixture: int
    opponent_team: int
    total_points: int = 0
    was_home: bool = True
    kickoff_time: Optional[str] = None
    team_h_score: Optional[int] = None
    team_a_score: Optional[int] = None
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
    influence: Optional[Union[str, float]] = "0.0"
    creativity: Optional[Union[str, float]] = "0.0"
    threat: Optional[Union[str, float]] = "0.0"
    ict_index: Optional[Union[str, float]] = "0.0"
    starts: Optional[int] = 0
    defensive_contribution: Optional[int] = 0
    expected_goals: Optional[Union[str, float]] = "0.0"
    expected_assists: Optional[Union[str, float]] = "0.0"
    expected_goal_involvements: Optional[Union[str, float]] = "0.0"
    expected_goals_conceded: Optional[Union[str, float]] = "0.0"
    value: int = 50
    selected: Optional[int] = 0
    transfers_balance: Optional[int] = 0

class ElementSummary(FPLBaseModel):
    fixtures: List[Dict[str, Any]] = []
    history: List[PlayerMatchHistory] = []
    history_past: List[Dict[str, Any]] = []

class Pick(FPLBaseModel):
    element: int
    position: int # 1-15: 1-11 starters, 12-15 bench
    multiplier: int = 1 # 1: normal, 2: captain, 3: triple captain, 0: benched
    is_captain: bool = False
    is_vice_captain: bool = False

class EntryHistory(FPLBaseModel):
    event: int
    points: int = 0
    total_points: int = 0
    rank: Optional[int] = None
    rank_sort: Optional[int] = None
    overall_rank: Optional[int] = None
    bank: int = 0 # In tenths (£1.0m = 10)
    value: int = 1000 # In tenths
    event_transfers: int = 0
    event_transfers_cost: int = 0
    points_on_bench: int = 0

class SquadPicks(FPLBaseModel):
    active_chip: Optional[str] = None
    entry_history: Optional[EntryHistory] = None
    picks: List[Pick] = []

class ChipHistoryItem(FPLBaseModel):
    name: str # e.g. "bboost", "3xc", "freehit", "wildcard"
    time: Optional[str] = None
    event: int

class ManagerHistory(FPLBaseModel):
    current: List[EntryHistory] = []
    past: List[Dict[str, Any]] = []
    chips: List[ChipHistoryItem] = []

class ManagerEntry(FPLBaseModel):
    id: int
    player_first_name: str = ""
    player_last_name: str = ""
    name: str = "" # Team name
    summary_overall_points: Optional[int] = 0
    summary_overall_rank: Optional[int] = None
    current_event: Optional[int] = None
    last_deadline_bank: Optional[int] = 0
    last_deadline_value: Optional[int] = 1000
    last_deadline_total_transfers: Optional[int] = 0
    leagues: Optional[Dict[str, Any]] = None

class TransferHistoryItem(FPLBaseModel):
    element_in: int
    element_in_cost: int
    element_out: int
    element_out_cost: int
    entry: int
    event: int
    time: str

class ClassicStandingResult(FPLBaseModel):
    id: Optional[int] = None
    club_badge_src: Optional[str] = None
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
    results: List[ClassicStandingResult] = []

class ClassicLeagueResponse(FPLBaseModel):
    league: Dict[str, Any] = {}
    standings: StandingsPage = Field(default_factory=StandingsPage)
