from __future__ import annotations
import numpy as np
from typing import List, Union, Tuple


class TemperatureCalibrator:
    """Applies temperature scaling and conservative tie-breaking to policy probability distributions.
    As specified in Section IV.D of the SAGE paper.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-4, float(temperature))

    def calibrate_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Scales predicted probability distribution using temperature:
        p_calibrated = softmax(log(probs) / T).
        """
        if self.temperature == 1.0:
            return probs

        # Convert probs to log space with numerical stability
        eps = 1e-12
        log_probs = np.log(np.clip(probs, eps, 1.0))
        scaled_logits = log_probs / self.temperature

        # Softmax over last dimension
        shifted = scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        calibrated = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return calibrated

    @staticmethod
    def select_budget_conservative(
        classes: List[int],
        probs: np.ndarray,
        tolerance: float = 1e-5
    ) -> Tuple[int, float]:
        """Selects argmax budget with tie-breaking in favor of smaller budgets.

        Args:
            classes: list of candidate budgets in ascending order, e.g. [2, 3, 5, 7, 10, 15, 20, 25, 30]
            probs: 1D probability array corresponding to classes
            tolerance: probability equality threshold for tie breaking

        Returns:
            (selected_budget, confidence_probability)
        """
        probs_1d = probs.squeeze()
        max_prob = np.max(probs_1d)

        # Find all classes whose probability is within tolerance of max_prob
        candidate_indices = [i for i, p in enumerate(probs_1d) if abs(p - max_prob) <= tolerance]

        # Break ties in favor of smaller budget (conservative choice per Section IV.B)
        candidate_budgets = [classes[i] for i in candidate_indices]
        chosen_budget = min(candidate_budgets)
        chosen_idx = classes.index(chosen_budget)

        return int(chosen_budget), float(probs_1d[chosen_idx])

