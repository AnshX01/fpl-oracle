from fpl_oracle.ml.holdout import compare_forecasts


def test_comparison_scores_each_source_on_the_same_players():
    freeze = {
        "predictions": [{"element": i, "expected_points": float(i)} for i in range(1, 31)],
        "snapshot_inputs": {"bootstrap": {"elements": [{"id": i, "ep_next": "2.0"} for i in range(1, 31)]}},
    }
    points = {i: float(i) for i in range(1, 31)}  # our forecast is exactly right
    minutes = {i: 90.0 for i in range(1, 31)} | {1: 0.0}
    res = compare_forecasts(freeze, points, minutes, {i: 3.0 for i in range(1, 31)})
    ours, fpl = res["sources"]["FPL Oracle"], res["sources"]["FPL ep_next"]
    assert res["players"] == 30
    assert ours["rmse"] == 0.0 and fpl["rmse"] > 0
    assert ours["top20_actual_points"] > fpl["top20_actual_points"] or fpl["top20_actual_points"] == ours["top20_actual_points"]
    assert set(ours["rmse_by_bucket"]) >= {"did not play", "5+ pts"}
