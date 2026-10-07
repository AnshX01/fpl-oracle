import hashlib
import json
import threading
import time

from fpl_oracle.ml.bundle_lock import bundle_locked
from fpl_oracle.ml.model_registry import ModelRegistry


def test_interrupted_swap_restores_old_bundle(tmp_path):
    registry = ModelRegistry()
    registry.models_dir = tmp_path
    registry.manifest_path = tmp_path / "manifest.json"
    registry.versions_dir = tmp_path / "versions"
    old = registry.versions_dir / "old"
    old.mkdir(parents=True)
    (old / "weights.pkl").write_bytes(b"old")
    (tmp_path / "weights.pkl").write_bytes(b"mixed")
    digest = hashlib.sha256(b"old").hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(active_version="old", versions=[dict(version="old", file_hashes={"weights.pkl": digest})]))
    )
    (tmp_path / "promotion_journal.json").write_text(
        json.dumps(dict(new_version="new", previous_version="old", files=["weights.pkl"]))
    )
    registry.recover_interrupted_swap()
    assert (tmp_path / "weights.pkl").read_bytes() == b"old"
    assert not (tmp_path / "promotion_journal.json").exists()


def test_bundle_exclusion_blocks_promotion_until_inference_finishes():
    started = threading.Event()
    release = threading.Event()
    changed = []

    @bundle_locked
    def inference():
        started.set()
        assert release.wait(2)

    @bundle_locked
    def promotion():
        changed.append(True)

    t = threading.Thread(target=inference)
    t.start()
    assert started.wait(1)
    p = threading.Thread(target=promotion)
    p.start()
    time.sleep(0.03)
    assert not changed
    release.set()
    t.join()
    p.join()
    assert changed


def test_committed_manifest_finishes_journal_without_rollback(tmp_path):
    registry = ModelRegistry()
    registry.models_dir = tmp_path
    registry.versions_dir = tmp_path / "versions"
    registry.manifest_path = tmp_path / "manifest.json"
    (tmp_path / "weights.pkl").write_bytes(b"new")
    digest = hashlib.sha256(b"new").hexdigest()
    registry.manifest_path.write_text(
        json.dumps(dict(active_version="new", versions=[dict(version="new", file_hashes={"weights.pkl": digest})]))
    )
    (tmp_path / "promotion_journal.json").write_text(
        json.dumps(dict(new_version="new", previous_version="old", files=["weights.pkl"]))
    )
    assert registry.verify_weight_integrity()[0]
    assert (tmp_path / "weights.pkl").read_bytes() == b"new"
    assert not (tmp_path / "promotion_journal.json").exists()


def test_promotion_bundle_readback_and_calibration(tmp_path, monkeypatch):
    import pandas as pd

    from fpl_oracle.data.store import data_store
    from fpl_oracle.ml.model_registry import COMPONENT_WEIGHTS

    registry = ModelRegistry()
    registry.models_dir = tmp_path
    registry.versions_dir = tmp_path / "versions"
    registry.rejected_dir = tmp_path / "rejected"
    registry.manifest_path = tmp_path / "manifest.json"
    registry.manifest_path.write_text(
        json.dumps(dict(active_version="old", versions=[dict(version="old", status="production")]))
    )

    class FakeModel:
        def save(self, path):
            path.write_bytes(b"candidate-weight")

    models = {name: FakeModel() for name, _, _ in COMPONENT_WEIGHTS}
    monkeypatch.setattr(data_store, "save_model_version", lambda **kw: None)
    result = registry.promote_candidate_bundle(
        models,
        dict(z10=-1, z90=1),
        dict(ml_mae=1, base_mae=2, ml_spearman=0.7),
        new_version_tag="new",
        training_data_df=pd.DataFrame({"x": [1]}),
        training_provenance=dict(operation_id="op"),
    )
    assert result["promoted"]
    assert registry.verify_weight_integrity()[0]
    assert registry.get_active_version()["training_provenance"]["operation_id"] == "op"
    assert json.loads((tmp_path / "calibration.json").read_text())["z90"] == 1


def test_projection_reloads_bundle_and_calibration_on_version_change(monkeypatch):
    import pandas as pd

    from fpl_oracle.data.features import feature_engineering
    from fpl_oracle.ml.model_registry import model_registry
    from fpl_oracle.ml.predict import ProjectionEngine

    engine = ProjectionEngine()
    engine.is_loaded = True
    engine.loaded_version = "old"
    called = []

    def reload():
        called.append(True)
        engine.loaded_version = "new"

    monkeypatch.setattr(engine, "load_or_train", reload)
    monkeypatch.setattr(model_registry, "get_active_version", lambda: dict(version="new"))
    monkeypatch.setattr(feature_engineering, "extract_live_features_for_upcoming", lambda **kw: pd.DataFrame())
    engine.predict_gameweek(6, None, [])
    assert called and engine.loaded_version == "new"
