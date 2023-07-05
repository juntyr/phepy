from __future__ import annotations

from typing import Union

import matplotlib as mpl
import numpy as np

from . import ToyExample


class CircleToyExample(ToyExample):
    def __init__(
        self, rng: np.random.Generator, N: int = 10_000
    ) -> CircleToyExample:
        Y = np.sin(np.linspace(0, np.pi * 8.0, N))

        X1 = np.sin(np.linspace(0, np.pi * 2.0, N)) * (5 + Y)
        X2 = np.cos(np.linspace(0, np.pi * 2.0, N)) * (5 + Y)

        X1 += rng.normal(loc=0.0, scale=0.1, size=N)
        X2 += rng.normal(loc=0.0, scale=0.1, size=N)

        self.__X_train = np.stack([X1, X2], axis=1)
        self.__Y_train = Y

        X1 = np.sin(np.linspace(0, np.pi * 2.0, N)) * (5 + Y)
        X2 = np.cos(np.linspace(0, np.pi * 2.0, N)) * (5 + Y)

        X1 += rng.normal(loc=0.0, scale=0.1, size=N)
        X2 += rng.normal(loc=0.0, scale=0.1, size=N)

        self.__X_valid = np.stack([X1, X2], axis=1)
        self.__Y_valid = Y

        X1T, X2T = np.mgrid[-7.5:7.5:0.01, -7.5:7.5:0.01]
        X1T = X1T.flatten()
        X2T = X2T.flatten()

        self.__X_test = np.stack([X1T, X2T], axis=1)

    @property
    def X_train(self) -> np.ndarray:
        return np.copy(self.__X_train)

    @property
    def Y_train(self) -> np.ndarray:
        return np.copy(self.__Y_train)

    @property
    def X_valid(self) -> np.ndarray:
        return np.copy(self.__X_valid)

    @property
    def Y_valid(self) -> np.ndarray:
        return np.copy(self.__Y_valid)

    @property
    def X_test(self) -> np.ndarray:
        return np.copy(self.__X_test)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        # Project X onto the sin-perturbed circle
        angle = _my_arctan2(X[:, 0], X[:, 1])

        Y = np.sin(angle * 4.0)

        X1 = np.sin(angle) * (5.0 + Y)
        X2 = np.cos(angle) * (5.0 + Y)

        # return np.stack([X1, X2], axis=1)
        return X1.reshape((len(X), 1)) * np.array([[1, 0]]) + X2.reshape(
            (len(X), 1)
        ) * np.array([[0, 1]])

    def is_in_distribution(self, X: np.ndarray) -> np.ndarray:
        # Two standard deviations off the mean -> 95.45% interval
        return np.linalg.norm(X - self.reconstruct(X), axis=1) <= 0.2

    @property
    def aspect_ratio(self) -> float:
        return 1.0

    def plot(
        self,
        conf: np.ndarray,
        ax: mpl.axes.Axes,
        cmap: Union[str, mpl.colors.Colormap],
        with_scatter: bool = True,
    ):
        N_skip = len(self.X_train) // 40

        ax.imshow(
            conf.reshape(np.mgrid[-7.5:7.5:0.01, -7.5:7.5:0.01][0].shape).T,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            extent=[-7.5, 7.5, -7.5, 7.5],
            origin="lower",
            interpolation="bicubic",
            rasterized=True,
        )

        ax.set_xlim(-7.5, 7.5)
        ax.set_ylim(-7.5, 7.5)

        if with_scatter:
            ax.scatter(
                self.X_train[::N_skip, 0],
                self.X_train[::N_skip, 1],
                c="white",
                marker="x",
                lw=3,
                s=48,
            )
            ax.scatter(
                self.X_train[::N_skip, 0],
                self.X_train[::N_skip, 1],
                c="black",
                marker="x",
            )

        ax.axis("off")


def _my_arctan2(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    a = np.arctan(x1 / x2)

    a += np.pi * ((x2 < 0.0) & (x1 >= 0.0))
    a -= np.pi * ((x2 < 0.0) & (x1 < 0.0))

    return a
