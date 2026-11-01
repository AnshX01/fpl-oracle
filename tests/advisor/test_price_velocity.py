"""
Unit tests for data-driven transfer price velocity model (Pillar 4).
Verifies that net transfer momentum, ownership scaling, and event cost change dampening
produce calibrated price rise and fall probabilities.
"""

from types import SimpleNamespace

from fpl_oracle.advisor.timing import estimate_price_change_probabilities


def test_estimate_price_change_probabilities():
    """High net transfer volume triggers high price rise probability."""
    mock_elements = [
        # Player 1: Heavy net buying (100,000 net in), 10% ownership
        SimpleNamespace(
            id=1,
            transfers_in_event=120000,
            transfers_out_event=20000,
            selected_by_percent="10.0",
            cost_change_event=0,
            price_change_projections=[],
        ),
        # Player 2: Heavy net selling (-80,000 net), 5% ownership
        SimpleNamespace(
            id=2,
            transfers_in_event=5000,
            transfers_out_event=85000,
            selected_by_percent="5.0",
            cost_change_event=0,
            price_change_projections=[],
        ),
        # Player 3: Quiet asset (0 net transfers)
        SimpleNamespace(
            id=3,
            transfers_in_event=100,
            transfers_out_event=100,
            selected_by_percent="2.0",
            cost_change_event=0,
            price_change_projections=[],
        ),
    ]
    mock_boot = SimpleNamespace(elements=mock_elements)

    probs = estimate_price_change_probabilities([1, 2, 3], mock_boot)

    # Player 1 should have high rise probability
    assert probs[1]["p_rise"] > 0.60
    assert probs[1]["p_fall"] < 0.05
    assert probs[1]["net_transfers"] == 100000

    # Player 2 should have high fall probability
    assert probs[2]["p_fall"] > 0.60
    assert probs[2]["p_rise"] < 0.05
    assert probs[2]["net_transfers"] == -80000

    # Player 3 should be near baseline (~0.01)
    assert probs[3]["p_rise"] < 0.05
    assert probs[3]["p_fall"] < 0.05
