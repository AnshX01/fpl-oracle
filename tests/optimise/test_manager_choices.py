"""The manager's 'keep' and "don't buy" calls bind the plan."""

from tests.test_planner_audit_fixes import evaluate, scenario


def test_kept_player_stays_and_avoided_player_is_never_bought():
    squad, pool, proj = scenario(squad_xp=0.5, pool_xp=5.0)
    keep = int(squad.element.iloc[5])  # a weak owned player the plan would otherwise sell
    first = evaluate(squad, pool, proj, 2, chips_available=["wildcard"])
    bought = {t["element"] for t in first["recommended_plan"]["transfers_in"]}
    avoid = next(iter(bought))
    res = evaluate(squad, pool, proj, 2, chips_available=["wildcard"], locked_in_ids={keep}, locked_out_ids={avoid})
    plan = res["recommended_plan"]
    week1 = set(plan["trajectory"][0]["elements"])
    assert keep in week1
    assert avoid not in week1
