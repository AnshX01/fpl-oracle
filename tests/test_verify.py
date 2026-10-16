"""
Tests for fpl_oracle.verify (Self-Check & Verification Suite).
"""

from unittest.mock import AsyncMock, MagicMock, patch

from fpl_oracle.verify import run_verification


def test_run_verification_pass():
    """Verify run_verification executes all checks and reports PASS."""
    mock_boot = MagicMock()
    mock_boot.elements = [MagicMock() for _ in range(600)]
    mock_fixtures = [MagicMock() for _ in range(350)]

    mock_profile = MagicMock()
    mock_profile.risk_preference = "balanced"

    with patch("fpl_oracle.api.fpl_client.fpl_client.get_current_and_next_gw", new_callable=AsyncMock) as m_gw, \
         patch("fpl_oracle.api.fpl_client.fpl_client.get_bootstrap_static", new_callable=AsyncMock) as m_boot, \
         patch("fpl_oracle.api.fpl_client.fpl_client.get_fixtures", new_callable=AsyncMock) as m_fix, \
         patch("fpl_oracle.data.store.data_store.get_profile", return_value=mock_profile), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("fpl_oracle.ml.predict.projection_engine.load_or_train"), \
         patch("pytest.main", return_value=0), \
         patch("fpl_oracle.backtest.backtest_harness.run_backtest", return_value={"oracle_total": 180}):

        m_gw.return_value = (5, 6)
        m_boot.return_value = (mock_boot, False)
        m_fix.return_value = (mock_fixtures, False)

        result = run_verification()
        assert result is True


def test_run_verification_api_failure():
    """Verify run_verification gracefully handles API error."""
    with patch("fpl_oracle.api.fpl_client.fpl_client.get_current_and_next_gw", side_effect=RuntimeError("API Connection Timeout")), \
         patch("fpl_oracle.data.store.data_store.get_profile", side_effect=RuntimeError("DB error")), \
         patch("pathlib.Path.exists", return_value=False), \
         patch("pytest.main", return_value=1), \
         patch("fpl_oracle.backtest.backtest_harness.run_backtest", side_effect=RuntimeError("Backtest failed")):

        result = run_verification()
        assert result is False
