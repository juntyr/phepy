# phepy &emsp; [![PyPi]][pypi-url]  [![License Apache-2.0]][apache-2.0] [![License MIT]][mit] [![Docs]][docs-stable] [![CI Status]][ci-status]

[PyPI]: https://img.shields.io/pypi/v/phepy
[pypi-url]: https://pypi.org/project/phepy

[Docs]: https://img.shields.io/pypi/v/phepy?color=blue&label=Docs
[docs-stable]: https://juntyr.github.io/phepy/

[License Apache-2.0]: https://img.shields.io/badge/License-Apache_2.0-yellowgreen.svg
[apache-2.0]: https://opensource.org/licenses/Apache-2.0

[License MIT]: https://img.shields.io/badge/License-MIT-yellow.svg
[mit]: https://opensource.org/licenses/MIT

[CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/phepy/ci.yml?branch=main&label=CI
[ci-status]: https://github.com/juntyr/phepy/actions/workflows/ci.yml?query=branch%3Amain

`phepy` is a Python package to visually evaluate out-of-distribution detectors using simple toy examples.

## Installation

### pip

The `phepy` package is available on the Python Package Index (PyPI) and can be installed using
```bash
pip install phepy
```
This command can also be run inside a conda environment to install `phepy` with conda.

### From Source

First, clone the git repository using
```bash
git clone https://github.com/juntyr/phepy.git
```
or
```bash
git clone git@github.com:juntyr/phepy.git
```

Next, enter the repository folder and use `pip` to install the program:
```bash
cd phepy && pip install .
```

## Usage Example

The following code snippet only provides a minimal example to get started, please refer to the [`examples`](https://github.com/juntyr/phepy/tree/main/examples) folder to find more extensive examples.

```python
# Import numpy, matplotlib, and sklearn
import numpy as np
import sklearn

from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Import phepy, three toy examples, the out-of-distribution (OOD)
#  detector and scorer, and the plotting helper function
import phepy

from phepy.detector import OutOfDistributionDetector, PercentileScorer
from phepy.plot import plot_all_toy_examples
from phepy.toys import ToyExample
from phepy.toys.line import LineToyExample
from phepy.toys.circle import CircleToyExample
from phepy.toys.haystack import HaystackToyExample


# Generate three toy test cases
line = LineToyExample(np.random.default_rng(42))
circle = CircleToyExample(np.random.default_rng(42))
haystack = HaystackToyExample(np.random.default_rng(42))


# Use the Local Outlier Factor (LOF) [^1] as an OOD detector
class LocalOutlierFactorDetector(OutOfDistributionDetector):
    @staticmethod
    def low_score_is_low_confidence() -> bool:
        return True

    def fit(
        self, X_train: np.ndarray, Y_train: np.ndarray
    ):
        self.__X_lof = LocalOutlierFactor(novelty=True).fit(X_train)
        
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.__X_lof.score_samples(X_test)


# Generate the plot for the LOF detector and the three test cases
fig = plot_all_toy_examples(
    scorers = {
        "Local Outlier Factor": PercentileScorer(LocalOutlierFactorDetector()),
    },
    toys = [line, circle, haystack],
    cmap = "coolwarm", # use e.g. "viridis" to be colour-blind safe
)

plt.show()
```

![By-example evaluation of Local Outlier Factor as a distance-based OOD detection method](https://raw.githubusercontent.com/juntyr/phepy/main/examples/minimal.png)

In the above figure, the single row showcases the Local Outlier Factor (LOF, [^1]) method, while the three columns contain the following three test cases:

* Two groups of training points are scattered along a line in the 2D feature space. The target variable only depends on the location along the line. In this example, points off the line are OOD.
* The training points are scattered around the sine-displaced boundary of a circle, but none are inside it. The target variable only depends on the location along the boundary. Again, points off the line are OOD.
* The training points are sampled from a 10-dimensional multivariate normal distribution, where one of the features is set to a constant. This example tests whether an OOD detection method can find a needle in a high-dimensional haystack, i.e. identify that points which do not share the constant are OOD.

The first two panels depict a subset of the training points using black `x` markers. Note that the first two plots do not have axis labels since the two axes map directly to the 2D feature space axes. The third plot differs and shows the *distribution* of confidence values on the y-axis for different x-axis values for the constant-in-training feature. The constant value is highlighted as a white line.

The Local Outlier Factor (LOF, [^1]) estimates the training data density around unseen data. Thus, it performs quite well on the line and circle examples where the data points are closely scattered. The LOF classifies the gap between the two groups of training inputs on the line as out-of-distribution (OOD), which may be too conservative if we assume that a machine-learning model can interpolate between the two groups. While LOF produces slightly lower confidence for the OOD inputs in the haystack, it does not clearly identify test data that do not have the constant feature seen in training as out-of-distribution.

## License

Licensed under either of

* Apache License, Version 2.0 ([`LICENSE-APACHE`](https://github.com/juntyr/phepy/blob/main/LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([`LICENSE-MIT`](https://github.com/juntyr/phepy/blob/main/LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Citation

Please refer to the [CITATION.cff](https://github.com/juntyr/phepy/blob/main/CITATION.cff) file and refer to https://citation-file-format.github.io to extract the citation in a format of your choice.

[^1]: M. M. Breunig *et al*. LOF: Identifying Density-Based Local Outliers. *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*. SIGMOD '00. Dallas, Texas, USA: Associ- ation for Computing Machinery, 2000, 93–104. Available from: [doi:10.1145/342009.335388](https://doi.org/10.1145/342009.335388).
