import abc
from typing import Union

import matplotlib as mpl
import numpy as np


class ToyExample(abc.ABC):
    @property
    @abc.abstractmethod
    def X_train(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def Y_train(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def X_valid(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def Y_valid(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def X_test(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def reconstruct(X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def is_in_distribution(X: np.ndarray) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def aspect_ratio(self) -> float:
        pass

    @abc.abstractmethod
    def plot(
        self,
        conf: np.ndarray,
        ax: mpl.axes.Axes,
        cmap: Union[str, mpl.colors.Colormap],
        with_scatter: bool = True,
    ):
        pass
