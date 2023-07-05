from __future__ import annotations

import abc

import numpy as np


class OutOfDistributionDetector(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def low_score_is_low_confidence() -> bool:
        pass

    @abc.abstractmethod
    def fit(
        self, X_train: np.ndarray, Y_train: np.ndarray
    ) -> OutOfDistributionDetector:
        pass

    @abc.abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pass


class OutOfDistributionScorer(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self, detector: OutOfDistributionDetector
    ) -> OutOfDistributionScorer:
        self.__detector = detector

    @property
    def detector(self) -> OutOfDistributionDetector:
        return self.__detector

    @abc.abstractmethod
    def calibrate(
        self, X_valid: np.ndarray, Y_valid: np.ndarray
    ) -> OutOfDistributionScorer:
        pass

    @abc.abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the confidence scores for several test data inputs.

        Args:
          X_test:
            numpy array of shape `N_SAMPLES x N_FEATURES`

        Returns:
          The array confidence scores of shape `N_SAMPLES`
        """
        pass


class PercentileScorer(OutOfDistributionScorer):
    def __init__(
        self, detector: OutOfDistributionDetector
    ) -> PercentileScorer:
        super().__init__(detector)

    def calibrate(
        self, X_valid: np.ndarray, Y_valid: np.ndarray
    ) -> PercentileScorer:
        self.__S_valid = np.sort(self.detector.predict(X_valid))

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        S_test = np.searchsorted(
            self.__S_valid,
            self.detector.predict(X_test),
        ) / len(self.__S_valid)

        return (
            S_test
            if type(self.detector).low_score_is_low_confidence()
            else 1.0 - S_test
        )


class IdentityScorer(OutOfDistributionScorer):
    def __init__(self, detector: OutOfDistributionDetector) -> IdentityScorer:
        super().__init__(detector)

    def calibrate(
        self, X_valid: np.ndarray, Y_valid: np.ndarray
    ) -> IdentityScorer:
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        S_test = self.detector.predict(X_test)

        return (
            S_test
            if type(self.detector).low_score_is_low_confidence()
            else 1.0 - S_test
        )
