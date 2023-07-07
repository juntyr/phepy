from __future__ import annotations

from typing import Union

import matplotlib as mpl
import numpy as np

from . import ToyExample


class HaystackToyExample(ToyExample):
    def __init__(
        self, rng: np.random.Generator, N: int = 10_000
    ) -> HaystackToyExample:
        N *= 2

        n = 10
        a = 2

        # https://stats.stackexchange.com/a/124554
        A = np.matrix(
            [rng.normal(size=n) + rng.normal(size=1) * a for i in range(n)]
        )
        A = A * np.transpose(A)
        D_half = np.diag(np.diag(A) ** (-0.5))
        covs = D_half * A * D_half

        X = rng.multivariate_normal(np.zeros(shape=n), covs, size=N)
        X[:, 3] = -0.42

        # Since X[:,3] is const, it does not matter what its multiplier is
        C = rng.choice([-2, -1, 0, 0, 1, 2], size=n)

        Y = np.dot(X, C)

        I_train = rng.choice(N, size=N // 2, replace=False)
        I_valid = np.ones(N)
        I_valid[I_train] = 0
        (I_valid,) = np.nonzero(I_valid)

        self.__X_train = X[I_train]
        self.__Y_train = Y[I_train]
        self.__X_valid = X[I_valid]
        self.__Y_valid = Y[I_valid]

        XT = rng.multivariate_normal(np.zeros(shape=n), covs, size=N // 2)
        XT[:, 3] = (rng.random(size=N // 2) - 0.5) - 0.42
        # Provide some definitely ID points
        XT[np.random.choice(N // 2, size=N // 200), 3] = -0.42

        self.__X_test = XT

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
        return X * np.array(
            [[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        ) - np.array([[0.0, 0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    def is_in_distribution(self, X: np.ndarray) -> np.ndarray:
        # Can only be ID if the constant feature value matches
        return X[:, 3] == -0.42

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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([-0.42 - 0.4, -0.42, -0.42 + 0.4])
        ax.set_xticklabels([r"$-0.4 \sigma$", "const", r"$+0.4 \sigma$"])
        ax.get_yaxis().set_visible(False)

        ax.axvline(-0.42, c="white", lw=7, zorder=-1)

        # with_scatter is ignored
        ax.scatter(self.X_test[:, 3], conf, c="white", s=6, rasterized=True)
        ax.scatter(self.X_test[:, 3], conf, c="black", s=1, rasterized=True)

        ax.imshow(
            [[x / 100, x / 100] for x in range(100, -1, -1)],
            cmap=cmap,
            interpolation="bicubic",
            vmin=0.0,
            vmax=1.0,
            extent=[-0.93, 0.09, -0.01, 1.01],
            zorder=-2,
        )
