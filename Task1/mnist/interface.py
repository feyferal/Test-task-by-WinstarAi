from __future__ import annotations

import abc
from typing import Optional

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float32]
ArrayI = NDArray[np.int64]


class MnistClassifierInterface(abc.ABC):
    """Unified API for all MNIST classifiers."""

    @abc.abstractmethod
    def train(
        self,
        X_train: ArrayF,
        y_train: ArrayI,
        X_val: Optional[ArrayF] = None,
        y_val: Optional[ArrayI] = None,
    ) -> None:
        """Fit on train; optionally use provided validation set."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X: ArrayF) -> ArrayI:
        """Return class predictions for X."""
        raise NotImplementedError
