from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import StrategyParameters


@dataclass(frozen=True)
class ModelPrediction:
    probability_down: float
    probability_flat: float
    probability_up: float
    expected_return: float

    @property
    def confidence(self) -> float:
        return abs(self.probability_up - self.probability_down)


class IncrementalDirectionalModel:
    classes = np.array([0, 1, 2], dtype=np.int64)

    def __init__(
        self,
        strategy_parameters: StrategyParameters,
        class_return_means: dict[int, float] | None = None,
    ) -> None:
        self.strategy_parameters = strategy_parameters
        self.class_return_means = class_return_means or {0: -0.003, 1: 0.0, 2: 0.003}
        self.class_counts = {0: 1, 1: 1, 2: 1}
        self.is_fitted = False
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.training_steps = 0

    def fit_initial(self, features: np.ndarray, labels: np.ndarray, future_returns: np.ndarray) -> None:
        if len(features) == 0:
            raise ValueError("Initial training requires at least one row.")
        self._ensure_parameters(features.shape[1])
        random_generator = np.random.default_rng(self.strategy_parameters.random_state)
        for _ in range(self.strategy_parameters.initial_epochs):
            indices = random_generator.permutation(len(features))
            shuffled_features = features[indices].astype(float, copy=False)
            shuffled_labels = labels[indices].astype(np.int64, copy=False)
            for batch_start in range(0, len(shuffled_features), 512):
                batch_end = batch_start + 512
                self._gradient_step(shuffled_features[batch_start:batch_end], shuffled_labels[batch_start:batch_end])
        self.is_fitted = True
        self._bootstrap_return_stats(labels, future_returns)

    def update(self, features: np.ndarray, label: int, future_return: float) -> None:
        row = np.asarray(features, dtype=float).reshape(1, -1)
        labels = np.asarray([label], dtype=np.int64)
        self._ensure_parameters(row.shape[1])
        self._gradient_step(row, labels)
        self.is_fitted = True
        self._update_return_stat(label, future_return)

    def predict(self, features: np.ndarray) -> ModelPrediction:
        if not self.is_fitted or self.weights is None or self.bias is None:
            return ModelPrediction(0.0, 1.0, 0.0, 0.0)
        row = np.asarray(features, dtype=float).reshape(1, -1)
        probabilities = self._softmax(row @ self.weights.T + self.bias)[0]
        expected_return = sum(
            probabilities[index] * self.class_return_means[index] for index in range(len(self.classes))
        )
        return ModelPrediction(
            probability_down=float(probabilities[0]),
            probability_flat=float(probabilities[1]),
            probability_up=float(probabilities[2]),
            expected_return=float(expected_return),
        )

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> None:
        payload = {
            "strategy_parameters": self.strategy_parameters.to_dict(),
            "class_return_means": self.class_return_means,
            "class_counts": self.class_counts,
            "is_fitted": self.is_fitted,
            "weights": self.weights,
            "bias": self.bias,
            "training_steps": self.training_steps,
            "extra_state": extra_state or {},
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["IncrementalDirectionalModel", dict[str, Any]]:
        payload = joblib.load(path)
        strategy_parameters = StrategyParameters(**payload["strategy_parameters"])
        model = cls(strategy_parameters=strategy_parameters, class_return_means=payload["class_return_means"])
        model.class_counts = payload["class_counts"]
        model.is_fitted = payload["is_fitted"]
        model.weights = payload["weights"]
        model.bias = payload["bias"]
        model.training_steps = payload.get("training_steps", 0)
        extra_state = payload.get("extra_state", {})
        return model, extra_state

    def _bootstrap_return_stats(self, labels: np.ndarray, future_returns: np.ndarray) -> None:
        for class_id in range(len(self.classes)):
            class_mask = labels == class_id
            if not np.any(class_mask):
                continue
            class_returns = future_returns[class_mask]
            self.class_return_means[class_id] = float(np.median(class_returns))
            self.class_counts[class_id] = int(class_mask.sum())

    def _update_return_stat(self, label: int, future_return: float) -> None:
        self.class_counts[label] += 1
        prior_mean = self.class_return_means[label]
        adaptive_rate = max(0.02, min(0.20, 1.0 / self.class_counts[label]))
        self.class_return_means[label] = float(
            (1.0 - adaptive_rate) * prior_mean + adaptive_rate * future_return
        )

    def _ensure_parameters(self, feature_count: int) -> None:
        if self.weights is not None and self.bias is not None:
            return
        random_generator = np.random.default_rng(self.strategy_parameters.random_state)
        self.weights = random_generator.normal(0.0, 0.01, size=(len(self.classes), feature_count))
        self.bias = np.zeros(len(self.classes), dtype=float)

    def _gradient_step(self, features: np.ndarray, labels: np.ndarray) -> None:
        if self.weights is None or self.bias is None:
            self._ensure_parameters(features.shape[1])
        assert self.weights is not None
        assert self.bias is not None

        logits = features @ self.weights.T + self.bias
        probabilities = self._softmax(logits)
        targets = np.zeros_like(probabilities)
        targets[np.arange(len(labels)), labels] = 1.0
        errors = probabilities - targets

        batch_size = max(len(features), 1)
        gradient_weights = (errors.T @ features) / batch_size
        gradient_bias = errors.mean(axis=0)
        if self.strategy_parameters.alpha > 0.0:
            l2_component = (1.0 - self.strategy_parameters.l1_ratio) * self.weights
            l1_component = self.strategy_parameters.l1_ratio * np.sign(self.weights)
            gradient_weights += self.strategy_parameters.alpha * (l2_component + l1_component)

        learning_rate = self._learning_rate()
        self.weights -= learning_rate * gradient_weights
        self.bias -= learning_rate * gradient_bias
        self.training_steps += 1

    def _learning_rate(self) -> float:
        base_rate = 0.04
        decay = 1.0 + self.training_steps * 0.01
        return base_rate / np.sqrt(decay)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)
