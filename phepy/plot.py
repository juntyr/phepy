from typing import Dict, List, Union

import matplotlib as mpl
import numpy as np

from .detector import OutOfDistributionScorer
from .toys import ToyExample


def plot_all_toy_examples(
    scorers: Dict[str, OutOfDistributionScorer],
    toys: List[ToyExample],
    cmap: Union[str, mpl.colors.Colormap],
    with_cbar: bool = True,
    with_titles: bool = True,
) -> mpl.figure.Figure:
    width_ratios = [toy.aspect_ratio for toy in toys]

    if with_cbar:
        width_ratios += [0.1]

    fig, axs = mpl.pyplot.subplots(
        len(scorers),
        len(toys) + with_cbar,
        figsize=(4 * np.sum(width_ratios), 4 * len(scorers)),
        gridspec_kw={"width_ratios": width_ratios},
    )
    axs = np.array(axs).reshape((len(scorers), len(toys) + with_cbar))

    for axr, (title, scorer) in zip(axs, scorers.items()):
        for ax, toy in zip(axr, toys):
            scorer.detector.fit(toy.X_train, toy.Y_train)
            scorer.calibrate(toy.X_valid, toy.Y_valid)

            conf = scorer.predict(toy.X_test)

            toy.plot(conf, ax, cmap)

        if with_titles and len(axr) > with_cbar:
            axr[0].text(
                0.5,
                0.95,
                title,
                ha="center",
                va="top",
                size=20,
                c="white",
                bbox=dict(facecolor="black", alpha=0.25, edgecolor="white"),
                transform=axr[0].transAxes,
            )

        if with_cbar and len(axr) > 1:
            axr[-1].spines["top"].set_visible(False)
            axr[-1].spines["right"].set_visible(False)
            axr[-1].spines["bottom"].set_visible(False)
            axr[-1].spines["left"].set_visible(False)

            axr[-1].xaxis.set_visible(False)
            axr[-1].set_yticks([0.1, 0.9])
            axr[-1].set_ylabel("confidence level $c$", labelpad=-13)
            axr[-1].yaxis.set_label_position("right")
            axr[-1].yaxis.tick_right()

            axr[-1].imshow(
                [[x / 100, x / 100] for x in range(100, -1, -1)],
                cmap=cmap,
                interpolation="bicubic",
                vmin=0.0,
                vmax=1.0,
                extent=[-0.05, 0.05, 0.0, 1.0],
                zorder=-2,
            )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    return fig
