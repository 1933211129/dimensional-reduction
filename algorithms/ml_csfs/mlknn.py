"""MLkNN 多标签分类器实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

__all__ = ["MLkNN", "MLkNNResult"]


@dataclass(slots=True)
class MLkNNResult:
    probabilities: np.ndarray
    predictions: np.ndarray


class MLkNN:
    """简化版 MLkNN 分类器。"""

    def __init__(
        self,
        k: int = 10,
        smoothing: float = 1.0,
        n_jobs: int = -1,
    ) -> None:
        if k < 1:
            raise ValueError("k 必须为正整数。")
        if smoothing < 0:
            raise ValueError("smoothing 需非负。")
        self.k = k
        self.smoothing = smoothing
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLkNN":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)

        if X.shape[0] != y.shape[0]:
            raise ValueError("训练样本数量不一致。")

        self._train_X = X
        self._train_y = y
        self._num_samples, self._num_labels = y.shape

        self._effective_k = min(self.k, self._num_samples - 1)
        if self._effective_k < 1:
            raise ValueError("训练集样本量不足以支撑 MLkNN (需要至少2条样本)。")

        k_neighbors = self._effective_k + 1

        self._nbrs = NearestNeighbors(
            n_neighbors=k_neighbors,
            metric="euclidean",
            algorithm="auto",
            n_jobs=self.n_jobs,
        )
        self._nbrs.fit(self._train_X)

        distances, indices = self._nbrs.kneighbors(self._train_X)
        neighbor_indices = indices[:, 1 : self._effective_k + 1]

        neighbor_labels = self._train_y[neighbor_indices]
        neighbor_counts = neighbor_labels.sum(axis=1)  # 形状：(n_samples, n_labels)

        self._prior_positive = (
            self.smoothing + self._train_y.sum(axis=0)
        ) / (self._num_samples + 2 * self.smoothing)
        self._prior_negative = 1.0 - self._prior_positive

        counts_positive = np.zeros(
            (self._num_labels, self.k + 1), dtype=np.float64
        )
        counts_negative = np.zeros_like(counts_positive)

        positive_mask = self._train_y == 1
        negative_mask = ~positive_mask

        for c in range(self.k + 1):
            mask_c = neighbor_counts == c
            counts_positive[:, c] = np.sum(mask_c & positive_mask, axis=0)
            counts_negative[:, c] = np.sum(mask_c & negative_mask, axis=0)

        denom_pos = counts_positive.sum(axis=1, keepdims=True)
        denom_neg = counts_negative.sum(axis=1, keepdims=True)

        self._cond_prob_positive = (
            counts_positive + self.smoothing
        ) / (denom_pos + (self.k + 1) * self.smoothing)

        self._cond_prob_negative = (
            counts_negative + self.smoothing
        ) / (denom_neg + (self.k + 1) * self.smoothing)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_nbrs") or self._effective_k is None:
            raise RuntimeError("请先调用 fit 进行训练。")

        X = np.asarray(X, dtype=np.float32)
        distances, indices = self._nbrs.kneighbors(X, n_neighbors=self._effective_k)
        neighbor_labels = self._train_y[indices]
        neighbor_counts = neighbor_labels.sum(axis=1)

        probabilities = np.empty((X.shape[0], self._num_labels), dtype=np.float64)
        for label_idx in range(self._num_labels):
            c_values = neighbor_counts[:, label_idx].astype(int)
            p_pos = self._prior_positive[label_idx] * self._cond_prob_positive[
                label_idx, c_values
            ]
            p_neg = self._prior_negative[label_idx] * self._cond_prob_negative[
                label_idx, c_values
            ]
            denom = p_pos + p_neg
            with np.errstate(divide="ignore", invalid="ignore"):
                proba = np.divide(p_pos, denom, out=np.zeros_like(p_pos), where=denom > 0)
            probabilities[:, label_idx] = proba

        return probabilities

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int8)

    def predict_with_prob(self, X: np.ndarray, threshold: float = 0.5) -> MLkNNResult:
        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(np.int8)
        return MLkNNResult(probabilities=probs, predictions=preds)

