from __future__ import annotations

from typing import Union

import matplotlib as mpl
import numpy as np

from . import ToyExample


class LineToyExample(ToyExample):
    def __init__(
        self, rng: np.random.Generator, N: int = 10_000
    ) -> LineToyExample:
        X1 = np.concatenate(
            [np.linspace(0, 4, N // 2), np.linspace(6, 10, N // 2)]
        )
        X2 = 1 + X1 * 0.5 + rng.normal(loc=0.0, scale=0.1, size=N)
        X1 += rng.normal(loc=0.0, scale=0.1, size=N)

        self.__X_train = np.stack([X1, X2], axis=1)
        self.__Y_train = X1 * np.sqrt(1.25)

        X1 = np.concatenate(
            [np.linspace(0, 4, N // 2), np.linspace(6, 10, N // 2)]
        )
        X2 = 1 + X1 * 0.5 + rng.normal(loc=0.0, scale=0.1, size=N)
        X1 += rng.normal(loc=0.0, scale=0.1, size=N)

        self.__X_valid = np.stack([X1, X2], axis=1)
        self.__Y_valid = X1 * np.sqrt(1.25)

        X1T, X2T = np.mgrid[-2:12:0.01, -1:8:0.01]
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
        # Project X onto the line
        return np.array([[0.0, 1.0]]) + np.array([[1.0, 0.5]]) * (
            np.matmul((X - np.array([0.0, 1.0])), np.array([[1.0], [0.5]]))
            / np.matmul(np.array([[1.0, 0.5]]), np.array([[1.0], [0.5]]))
        )

    def is_in_distribution(self, X: np.ndarray) -> np.ndarray:
        # Two standard deviations off the mean -> 95.45% interval
        return np.linalg.norm(X - self.reconstruct(X), axis=1) <= 0.2

    @property
    def aspect_ratio(self) -> float:
        return 14 / 9

    def plot(
        self,
        conf: np.ndarray,
        ax: mpl.axes.Axes,
        cmap: Union[str, mpl.colors.Colormap],
        with_scatter: bool = True,
    ):
        N_skip = len(self.X_train) // 40

        ax.imshow(
            conf.reshape(np.mgrid[-2:12:0.01, -1:8:0.01][0].shape).T,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            extent=[-2, 12, -1, 8],
            origin="lower",
            interpolation="bicubic",
            rasterized=True,
        )

        ax.set_xlim(-2, 12)
        ax.set_ylim(-1, 8)

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
