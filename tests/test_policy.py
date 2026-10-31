import os
import pytest
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from policy.calibration import TemperatureCalibrator
from policy.predict import SAGEPolicyPredictor
from retrieval.bm25 import ArtifactNotFoundError


def test_temperature_calibration():
    calibrator = TemperatureCalibrator(temperature=2.0)
    probs = np.array([0.1, 0.7, 0.2])
    calibrated = calibrator.calibrate_probabilities(probs)
    # Higher temperature softens distribution: max prob should decrease
    assert calibrated[1] < 0.7
    assert np.isclose(np.sum(calibrated), 1.0)


def test_conservative_tie_breaking():
    classes = [2, 3, 5, 10]
    # Exact tie between budget 3 and budget 5
    probs = np.array([0.1, 0.4, 0.4, 0.1])
    selected_k, conf = TemperatureCalibrator.select_budget_conservative(classes, probs)
    # Conservative tie breaking prefers smaller budget
    assert selected_k == 3


def test_missing_model_raises_error(tmp_path):
    missing_path = str(tmp_path / "non_existent.joblib")
    predictor = SAGEPolicyPredictor(model_path=missing_path)
    with pytest.raises(ArtifactNotFoundError):
        predictor.load_model()


def test_policy_predict_with_fixture_model(tmp_path):
    model_file = str(tmp_path / "test_rf.joblib")

    # Fit a tiny dummy model for fixture testing (n_features=10)
    n_features = 10
    classes = [2, 5, 10, 20]
    X_dummy = np.random.randn(20, n_features)
    y_dummy = np.random.choice(classes, size=20)
    rf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
    rf.fit(X_dummy, y_dummy)

    joblib.dump({"model": rf, "classes": classes, "temperature": 1.0}, model_file)

    predictor = SAGEPolicyPredictor(model_path=model_file)
    predictor.load_model()
    assert predictor.is_loaded

    test_feat = np.random.randn(1, n_features)
    decision = predictor.predict_budget(test_feat)
    assert decision["selected_k"] in classes
    assert 0.0 <= decision["confidence"] <= 1.0

