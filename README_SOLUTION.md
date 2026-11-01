# Predicting a post's success relative to its author

The task is to classify a post as **below / typical / above** its author's
usual reaction level. The final solution uses text, post structure and history;
video and images are not processed.

On the official test set, it achieves **macro-F1 0.6390**, compared with **0.5403**
for the original baseline. Aggregate results are in `reports/summary.json`,
and a step-by-step explanation is in `solution.ipynb`.

## Data and evaluation

| Split | Authors | Labelled posts |
|---|---:|---:|
| Train | 3,295 | 71,395 |
| Test | 824 | 17,623 |

The official author split is preserved: each author belongs entirely to train
or test. The label compares the current post's `log(1 + reactions)` with the
median over the author's own earlier posts, requiring at least five past posts.
Those first five posts can supply history even though they have no target label.

The primary metric is macro-F1, with QWK for the ordered classes and accuracy
as additional metrics. Models are selected on three author folds within train.
TF-IDF, SVD and classifiers are fitted only on the fit portion of each split.

Features exclude the current post's reactions, residual and label. History
contains only earlier posts. However, past reaction counts come from the snapshot
at collection time, so this reproduces the benchmark rather than a strict
prediction at publication time.

## Experiments

| Stage | Change | Finding |
|---|---|---|
| V1: features | Style, attachments, author dynamics and deviations from usual behaviour | Stronger than the original baseline |
| V1: text | TF-IDF/SVD and frozen multilingual E5-small | Content complements tabular features |
| V2: context | Organisation features and similar earlier posts by other authors | Selected 180-day KNN; organisation features were excluded |
| V2: model | 12 HGB settings for three families, 15 blends | Selected a blend of E5 and TF-IDF/SVD + KNN |
| Decision rule | Class multipliers before argmax | Further improvement in macro-F1 |

E5 is not fine-tuned: each post becomes a normalised 384-dimensional vector.
Text is limited to 512 tokens, prioritising the author's own text, shared text,
titles and available document text.

KNN searches earlier similar posts by **other fit authors**, with at most one
post per author. Training rows use inner author splits; validation/test posts
never enter the neighbour bank. Global organisation features also use only
earlier posts by fit authors.

## Final model

The probabilities from two HistGradientBoostingClassifier models are blended:

- **50%**: tabular features + E5, up to 63 leaves;
- **50%**: tabular features + TF-IDF/SVD + 180-day KNN,
  up to 63 leaves, at least 100 rows per leaf, L2 = 1.

Both models use a learning rate of 0.08 and at most 300 iterations with early stopping.
The following decision rule is applied after blending:

```text
label = argmax(0.8 × p_below, 1.0 × p_typical, 0.8 × p_above)
```

This changes the class decision; the weighted scores are not presented as new
normalised probabilities.

To evaluate the rule, each outer validation fold is split into two groups of authors:
multipliers are selected on one half and evaluated on the other, then vice versa.
The base model was trained without any authors from that outer fold. We tested
81 pairs of below/above weights from 0.8 to 1.2 in steps of 0.05; typical stays at 1.

Cross-evaluation of the **procedure using six rules** achieved macro-F1 0.6480,
compared with 0.6382 for V2. The final weights `[0.8, 1.0, 0.8]` were then fitted
on all OOF predictions; scoring them on those same rows is not validation.
Models and the rule were selected using train, but test results were already known
from earlier stages: this is a sequential experiment, not a fresh independent evaluation.

## Official test results

| Variant | Macro-F1 | QWK | Accuracy |
|---|---:|---:|---:|
| Original baseline | 0.5403 | 0.4603 | 0.5445 |
| V1 | 0.6142 | 0.5583 | 0.6251 |
| V2 | 0.6247 | 0.5658 | 0.6402 |
| **Final: V2 + decision rule** | **0.6390** | **0.5741** | **0.6692** |

The final gain is **9.870 percentage points over the baseline** and **1.433 over V2**.
For the latter difference, the 95% paired author bootstrap interval is
**[1.035; 1.862] percentage points**, based on 1,000 resamples. It describes
variation in the author population with predictions held fixed, excluding
training and selection uncertainty. Gains differ by class: relative to V2,
F1 for `below` fell slightly, while `typical` and `above` improved.

## Running the solution

Data is excluded from the repository. Place the extracted core dataset in `data/`.
Install the dependencies from `requirements-lock.txt`. With a verified E5 cache
in `data/cache/e5/`, the solution runs on CPU without Torch or a GPU.
Without the cache, the encoder uses CUDA by default, requiring a GPU, PyTorch
and the dependencies in `requirements-gpu.txt`.

From the project root:

```bash
python -m kairos.label
python calibrate_decision.py --config configs/solution_v2_decision.json
python evaluate.py --predictions results/v2_decision/final/predictions.csv
```

The wrapper trains the fixed V2 recipe on all train rows and applies the selected rule.
It does not need the validation fold list. Repeating the historical search exactly
with `configs/stage2_search.json` requires the local `configs/stage2_folds.json`;
this author list is not published with the code.
To repeat only the final step using saved V2 probabilities:

```bash
python calibrate_decision.py --action apply --config configs/solution_v2_decision.json
```

The reported metrics were obtained on Linux CPU with Python 3.12.3; a fresh V2 run
and repeated postprocessing produced identical results there. The full command
also runs on Mac with Python 3.10.20, but 998 of 17,623 predicted labels differ;
the cause has not been established.
The notebook reads only the aggregate report and runs without the dataset.
