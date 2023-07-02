import sys
import time
from dataclasses import dataclass
from gc import get_referents
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib as mpl
import numpy as np

from . import OutOfDistributionScorer, ToyExample


@dataclass
class Evaluation:
    fit_time: float
    calibrate_time: float
    predict_time: float

    detector_size: int
    scorer_size: int

    expected: np.ndarray
    confidence: np.ndarray


def plot_all_toy_examples(
    scorers: Dict[str, OutOfDistributionScorer],
    toys: List[ToyExample],
    cmap: Union[str, mpl.colors.Colormap],
    with_cbar: bool = True,
    with_titles: bool = True,
    with_scorer: Optional[
        Callable[
            [
                OutOfDistributionScorer,
                ToyExample,
                Evaluation,
                mpl.axes.Axes,
            ],
            None,
        ]
    ] = None,
) -> mpl.figure.Figure:
    """Plot the out-of-distribution (OOD) detection performance
    of all given scorers across all given toy examples.

    Note that the provided scorers and their detectors will be
    refitted and recalibrated for each toy example.

    Args:
      scorers:
        mapping from OOD scoring method name to an instance
        of the method
      toys:
        list of toy examples
      cmap:
        name or instance of a matplotlib Colormap that
        is used to encode the confidence score
      with_cbar:
        whether a colorbar should be produced for each
        row of subplots
      with_titles:
        whether each method's name should be added as a
        title to each row of subplots
      with_scorer:
        optional scoring function which can perform
        additional evaluation and plot its results

    Returns:
      The created matplotlib figure.
    """
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
            pre_fit = time.perf_counter()
            scorer.detector.fit(toy.X_train, toy.Y_train)
            post_fit = time.perf_counter()

            scorer.calibrate(toy.X_valid, toy.Y_valid)
            post_calibrate = time.perf_counter()

            conf = scorer.predict(toy.X_test)
            post_predict = time.perf_counter()

            detector_size = _getsize(scorer.detector)
            scorer_size = _getsize(scorer)

            toy.plot(conf, ax, cmap)

            if with_scorer is not None:
                with_scorer(
                    scorer,
                    toy,
                    Evaluation(
                        fit_time=post_fit - pre_fit,
                        calibrate_time=post_calibrate - post_fit,
                        predict_time=post_predict - post_calibrate,
                        detector_size=detector_size,
                        scorer_size=scorer_size - detector_size,
                        expected=toy.is_in_distribution(toy.X_test),
                        confidence=conf,
                    ),
                    ax,
                )

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


def _getsize(obj: Any) -> int:
    # https://stackoverflow.com/a/30316760
    #
    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    BLACKLIST = type, ModuleType, FunctionType

    if isinstance(obj, BLACKLIST):
        raise TypeError(
            f"getsize() does not take argument of type: {type(obj)}"
        )

    seen_ids = set()
    size = 0
    objects = [obj]

    while len(objects) > 0:
        need_referents = []

        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)

        objects = get_referents(*need_referents)

    return size
