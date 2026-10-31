from __future__ import annotations
import os
import json
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from retrieval.bm25 import ArtifactNotFoundError
from policy.calibration import TemperatureCalibrator


class SAGEPolicyPredictor:
    """Inference-only predictor for the SAGE learned adaptive retrieval policy.
    Loads the trained Random Forest model exported from Google Colab and predicts
    the optimal discrete retrieval budget k for incoming queries.
    """

    CANDIDATE_BUDGETS = [2, 3, 5, 7, 10, 15, 20, 25, 30]

    def __init__(
        self,
        model_path: str = "artifacts/models/random_forest/sage_rf.joblib",
        metadata_path: str = "artifacts/metadata/model_metadata.json",
        temperature: float = 1.0,
        fallback_budget: int = 5
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.temperature = temperature
        self.fallback_budget = fallback_budget
        self.calibrator = TemperatureCalibrator(temperature=temperature)

        self.model = None
        self.preprocessor = None
        self.classes: List[int] = list(self.CANDIDATE_BUDGETS)
        self.expected_n_features: Optional[int] = None
        self.is_loaded = False

    def load_model(self) -> None:
        """Loads trained Random Forest model and metadata exported from Colab."""
        if not os.path.exists(self.model_path):
            raise ArtifactNotFoundError(
                f"Trained SAGE Random Forest model not found at: '{self.model_path}'.\n"
                f"Environment A (Google Colab) must be used to train and export the SAGE model.\n"
                f"Please run the Colab training notebook and download 'sage_rf.joblib' into '{os.path.dirname(self.model_path)}'."
            )

        try:
            artifact = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAGE model from {self.model_path}: {e}")

        # Support both raw model and dict-wrapped artifact from Colab
        if isinstance(artifact, dict):
            self.model = artifact.get("model")
            self.preprocessor = artifact.get("preprocessor")
            self.classes = [int(c) for c in artifact.get("classes", self.CANDIDATE_BUDGETS)]
            self.temperature = float(artifact.get("temperature", self.temperature))
            self.calibrator = TemperatureCalibrator(temperature=self.temperature)
        else:
            self.model = artifact
            if hasattr(self.model, "classes_"):
                self.classes = [int(c) for c in self.model.classes_]

        if hasattr(self.model, "n_features_in_"):
            self.expected_n_features = int(self.model.n_features_in_)

        self.is_loaded = True

    def validate_features(self, feature_vector: np.ndarray) -> None:
        """Validates that input features match the model's expected shape."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("SAGE model is not loaded. Call load_model() first.")

        if self.expected_n_features is not None:
            actual_n = feature_vector.shape[1] if feature_vector.ndim > 1 else len(feature_vector)
            if actual_n != self.expected_n_features:
                raise ValueError(
                    f"Feature dimension mismatch: SAGE model expected {self.expected_n_features} features, "
                    f"but received {actual_n}. Verify artifacts/metadata/feature_schema.json."
                )

    def predict_budget(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Predicts retrieval budget k in K for the given feature vector."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("SAGE model is not loaded. Call load_model() first.")

        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        self.validate_features(feature_vector)

        # Apply optional preprocessor (e.g. StandardScaler) if exported with model
        processed_features = feature_vector
        if self.preprocessor is not None:
            processed_features = self.preprocessor.transform(feature_vector)

        # Get raw class probabilities from Random Forest
        raw_probs = self.model.predict_proba(processed_features)[0]

        # Apply temperature calibration
        calibrated_probs = self.calibrator.calibrate_probabilities(raw_probs)

        # Conservative tie-breaking selection
        selected_k, confidence = self.calibrator.select_budget_conservative(
            classes=self.classes,
            probs=calibrated_probs
        )

        prob_distribution = {int(cls): float(p) for cls, p in zip(self.classes, calibrated_probs)}

        return {
            "selected_k": selected_k,
            "confidence": confidence,
            "temperature": self.temperature,
            "candidate_budgets": self.classes,
            "probability_distribution": prob_distribution
        }

