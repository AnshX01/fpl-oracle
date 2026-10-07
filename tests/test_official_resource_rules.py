from types import SimpleNamespace

from fpl_oracle.optimise.transfers import transfer_optimizer


def test_special_chip_does_not_add_a_free_transfer():
    from fpl_oracle.domain.manager_state import ManagerStateService

    history = [dict(event=1, event_transfers=0), dict(event=2, event_transfers=0), dict(event=3, event_transfers=8)]
    for chip in ("wildcard", "freehit"):
        assert ManagerStateService.calculate_banked_free_transfers(history, [dict(event=3, name=chip)]) == 2
        model = SimpleNamespace(
            current=[SimpleNamespace(**h) for h in history], chips=[SimpleNamespace(event=3, name=chip)]
        )
        assert transfer_optimizer.calculate_banked_free_transfers(model) == 2


def test_initial_unlimited_deadline_does_not_bank_an_extra_ft():
    from fpl_oracle.domain.manager_state import ManagerStateService

    assert ManagerStateService.calculate_banked_free_transfers([dict(event=1, event_transfers=0)], []) == 1
    assert ManagerStateService.calculate_banked_free_transfers([dict(event=4, event_transfers=0)], []) == 1
    assert (
        transfer_optimizer.calculate_banked_free_transfers(
            SimpleNamespace(current=[SimpleNamespace(event=1, event_transfers=0)], chips=[])
        )
        == 1
    )
