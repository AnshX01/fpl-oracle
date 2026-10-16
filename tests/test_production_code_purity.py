"""
Test to ensure production code purity:
Asserts zero hardcoded player names exist in core production modules (advisor, optimise, rules, etc.).
"""

import re
from pathlib import Path


def test_zero_hardcoded_player_names_in_production_code():
    src_dir = Path("src/fpl_oracle")
    forbidden_names = [
        "haaland",
        "salah",
        "saka",
        "palmer",
        "calafiori",
        "tarkowski",
        "mbeumo",
        "szoboszlai",
        "belloumi",
        "verbruggen",
        "kinsky",
        "calvert-lewin",
        "gabriel",
    ]
    # Allowlist: NLP benchmark cases dataset which contains raw real press conference quotes
    excluded_files = {"held_out_benchmark_cases.py"}

    violations = []
    for py_file in src_dir.rglob("*.py"):
        if py_file.name in excluded_files:
            continue
        content = py_file.read_text(encoding="utf-8").lower()
        for name in forbidden_names:
            matches = re.findall(rf"\b{re.escape(name)}\b", content)
            if matches:
                violations.append(f"{py_file}: found '{name}' ({len(matches)} times)")

    assert not violations, "Hardcoded player names found in production code:\n" + "\n".join(violations)
