"""
Script to generate and synchronize the feature documentation table in
fpl_oracle_system_architecture.md directly from FEATURE_COLUMNS.
Supports --check flag for CI validation.
"""

import re
import sys
from pathlib import Path

from fpl_oracle.data.features import FEATURE_COLUMNS, FEATURE_SCHEMA_HASH

DOC_PATH = Path("fpl_oracle_system_architecture.md")

CLUSTERS = [
    (
        "Minutes & Starts",
        [c for c in FEATURE_COLUMNS if "minute" in c or "start" in c or "min60" in c],
        "Rolling average minutes, start rates, and appearance consistency across 3, 5, and 8 gameweeks.",
    ),
    (
        "Attacking Form",
        [
            c
            for c in FEATURE_COLUMNS
            if (
                c.startswith("roll_points_")
                or c.startswith("roll_xG_")
                or c.startswith("roll_xA_")
                or c == "roll_xGI_5"
                or c == "roll_goals_5"
                or c == "roll_assists_5"
            )
        ],
        "Underlying shot quality (xG), playmaking volume (xA/xGI), and attacking returns across 3, 5, and 8 matches.",
    ),
    (
        "Defensive & Goalkeeping Form",
        [
            c
            for c in FEATURE_COLUMNS
            if c in ("roll_xGC_5", "roll_clean_sheets_5", "roll_saves_5", "roll_goals_conceded_5", "roll_defcon_5")
        ],
        "Expected goals conceded, clean sheets, goalkeeper saves, and DefCon defensive contribution actions.",
    ),
    (
        "ICT & BPS",
        [c for c in FEATURE_COLUMNS if "ict" in c or "bps" in c],
        "FPL ICT index form and underlying Bonus Point System performance metrics.",
    ),
    (
        "Disciplinary & Rare Events",
        [c for c in FEATURE_COLUMNS if any(x in c for x in ["cards", "own_goals", "penalties"])],
        "Deduction risks including card frequency, own goals, penalty misses, and penalty save capability.",
    ),
    (
        "Match Context & Rest",
        [
            c
            for c in FEATURE_COLUMNS
            if c
            in (
                "was_home",
                "team_strength_attack",
                "opp_strength_defence",
                "net_strength_diff",
                "opponent_difficulty",
                "days_rest",
            )
        ],
        "Venue home/away split, official fixture difficulty, team strength differences, and match turnaround days rest.",
    ),
    (
        "Continuous Implied Match Dynamics",
        [c for c in FEATURE_COLUMNS if c.startswith("implied_")],
        "Continuous implied team goals, opponent goals, and clean sheet probabilities from Poisson attack-defence model.",
    ),
    (
        "Team Attack Form",
        [c for c in FEATURE_COLUMNS if c.startswith("team_roll_")],
        "Trailing club attacking volume: rolling actual goals and xG generated across 3, 5, and 8 matches.",
    ),
    (
        "Opponent Defensive Form",
        [c for c in FEATURE_COLUMNS if c.startswith("opp_roll_")],
        "Trailing opponent defensive vulnerability: conceded points, actual goals, and xGC across 3, 5, and 8 matches.",
    ),
    (
        "Positional One-Hot",
        [c for c in FEATURE_COLUMNS if c.startswith("pos_")],
        "One-hot position indicators (pos_GKP, pos_DEF, pos_MID, pos_FWD) for role-conditioned scoring rates.",
    ),
    (
        "Market & Status",
        [c for c in FEATURE_COLUMNS if c in ("value", "is_2026_27", "chance_of_playing")],
        "Current player price, active season indicator, and official availability probability percentage.",
    ),
]


def generate_markdown_table() -> str:
    total_features = len(FEATURE_COLUMNS)
    assert total_features == 62, f"Expected 62 features, got {total_features}"

    # Verify all columns accounted for
    clustered_cols = []
    for _, cols, _ in CLUSTERS:
        clustered_cols.extend(cols)
    assert set(clustered_cols) == set(FEATURE_COLUMNS), f"Missing: {set(FEATURE_COLUMNS) - set(clustered_cols)}"

    lines = [
        f"The {total_features} model features are grouped into {len(CLUSTERS)} functional clusters:\n",
        "| Feature Group | Features | Count | Description |",
        "|---|---|:---:|---|",
    ]
    for name, cols, desc in CLUSTERS:
        col_str = ", ".join(f"`{c}`" for c in cols)
        lines.append(f"| **{name}** | {col_str} | {len(cols)} | {desc} |")
    lines.append(
        f"| **Total** | *All {total_features} Canonical Feature Columns* | **{total_features}** | *Schema hash: `{FEATURE_SCHEMA_HASH[:16]}...`* |"
    )
    return "\n".join(lines)


def sync_architecture_doc(check_only: bool = False) -> bool:
    if not DOC_PATH.exists():
        print(f"Error: {DOC_PATH} not found.", file=sys.stderr)
        return False

    content = DOC_PATH.read_text(encoding="utf-8")

    if check_only:
        # Check requirements
        errors = []
        if "file:///C:" in content:
            errors.append("Doc contains file:///C: links")
        if "100% uptime" in content.lower():
            errors.append("Doc contains unprovable claim '100% uptime'")
        if "never fails" in content.lower():
            errors.append("Doc contains unprovable claim 'never fails'")
        if "57-column" in content:
            errors.append("Doc still references 57-column schema")
        if "57 model features" in content:
            errors.append("Doc still references 57 model features")
        if "62 model features" not in content or "62-column feature schema" not in content:
            errors.append("Doc missing 62 feature count references")
        for name, _, _ in CLUSTERS:
            if name not in content:
                errors.append(f"Cluster '{name}' missing from doc table")
        if errors:
            print("Architecture doc check failed:\n" + "\n".join(f" - {e}" for e in errors), file=sys.stderr)
            return False
        print("Architecture doc passes all verification checks!")
        return True

    # Mutate content
    # 1. Remove absolute file links
    new_text = re.sub(r"file:///C:[^)\s]+/src/", "src/", content)

    # 2. Fix unprovable claims
    new_text = new_text.replace("To guarantee 100% uptime:", "To provide high availability and graceful degradation:")
    new_text = new_text.replace(
        "- Never fails or crashes the user interface.",
        "- Falls back gracefully to offline rules-based heuristics if upstream LLM requests fail.",
    )

    # 3. Reword Dixon-Coles
    new_text = new_text.replace(
        "### Continuous Implied Match Strength & Dixon-Coles Poisson Model",
        "### Continuous Implied Match Strength & Independent Poisson Attack-Defence Model",
    )
    new_text = new_text.replace(
        "[Continuous Implied Match Strength & Dixon-Coles Poisson Model](#continuous-implied-match-strength--dixon-coles-poisson-model)",
        "[Continuous Implied Match Strength & Independent Poisson Attack-Defence Model](#continuous-implied-match-strength--independent-poisson-attack-defence-model)",
    )
    new_text = new_text.replace(
        "- **Poisson/Dixon-Coles match simulation**,",
        "- **Independent Poisson attack-defence match simulation (full bivariate Dixon-Coles with low-score rho scheduled for WI-3.1)**,",
    )
    new_text = new_text.replace(
        'DC["Poisson / Dixon-Coles\\nImplied Match Dynamics"]',
        'DC["Poisson Implied Match Dynamics\\n(Attack-Defence Rates)"]',
    )
    new_text = new_text.replace(
        "Rather than relying on static 1-5 Fixture Difficulty Ratings (FDR), FPL Oracle calculates continuous team attacking rates $\\lambda$ and defending rates $\\mu$ using a Dixon-Coles Poisson formulation:",
        "Rather than relying on static 1-5 Fixture Difficulty Ratings (FDR), FPL Oracle calculates continuous team attacking rates $\\lambda$ and defending rates $\\mu$ using an independent univariate Poisson attack-defence formulation (scheduled to be upgraded to a full bivariate Dixon-Coles model with low-score correlation parameter $\\rho$ in spec WI-3.1):",
    )

    # 4. Feature counts and table
    new_text = new_text.replace("57-column feature schema", "62-column feature schema")

    new_table = generate_markdown_table()
    table_pattern = re.compile(
        r"The \d+ model features are grouped into \d+ functional clusters:.*?(?=\n---|\n## )", re.DOTALL
    )
    new_text = table_pattern.sub(new_table + "\n", new_text)

    DOC_PATH.write_text(new_text, encoding="utf-8")
    print(f"Updated {DOC_PATH} with 62-column feature table, clean links, and verified wording.")
    return True


if __name__ == "__main__":
    check_mode = "--check" in sys.argv
    success = sync_architecture_doc(check_only=check_mode)
    sys.exit(0 if success else 1)
