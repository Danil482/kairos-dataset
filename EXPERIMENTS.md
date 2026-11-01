# Experiments: predicting relative post success

The final solution blends two HGB models with text features and applies a fixed
class decision rule. Official test macro-F1 improved from **0.540334 to 0.639033**.
Aggregate results are in [reports/summary.json](reports/summary.json);
run instructions are in [README_SOLUTION.md](README_SOLUTION.md) and [solution.ipynb](solution.ipynb).

## Task and protocol

The classes `below`, `typical` and `above` describe reactions relative to each
author's usual level. Official labels and the split were unchanged: 71,395 train
posts from 3,295 authors and 17,623 test posts from 824 other authors. Macro-F1 is
the primary metric; QWK accounts for class ordering, with accuracy as an additional metric.

Initial selection used an author validation split within train; finalists were
checked on three author folds. Stage two used the same fixed groups.
Every OOF prediction is for an author excluded from the corresponding model's training.
TF-IDF and SVD were fitted only on the fit portion; E5 was used without fine-tuning.

## Stage one: author history and text

The original baseline uses 16 numeric features and the post type. We tested
recent reaction history, posting rhythm, writing style, mentions and text vectors.
Enriched text included available content from shared posts and documents.
The results below use one common validation split, not the official test set.

| Variant | Validation macro-F1 |
|---|---:|
| Original baseline | 0.556797 |
| Baseline + last 5 posts | 0.568405 |
| Baseline + recent5 + recent10 + posting rhythm | 0.570814 |
| Baseline + writing style | 0.587662 |
| Baseline + mention features | 0.572509 |
| All extended tabular features, E2 | 0.607318 |
| Main text only: TF-IDF + logistic regression | 0.447454 |
| Enriched text: TF-IDF + logistic regression | 0.463845 |
| Tabular + TF-IDF/SVD + HGB | 0.624712 |
| Tabular + frozen E5 + HGB | 0.624970 |

Text was more useful alongside tabular features than on its own. The author's
text profile and similar past posts did not outperform adding vectors directly:
E5-history and E5-neighbors achieved 0.605271 and 0.616860, respectively.

Across three folds, baseline OOF macro-F1 was **0.553632**. A blend of 25% E2 and
75% of the tabular model with SVD achieved **0.621574**; the equivalent E5 blend
achieved **0.624701** and became V1. E5's advantage over the strong SVD variant was
only **0.313 percentage points**; no language-model fine-tuning was involved.

## Stage two: other authors' neighbours and organisations

We compared three families: **A**, extended tabular features; **B**, tabular + E5;
**C**, tabular + TF-IDF/SVD. Each was tested with organisation features, KNN over
all earlier history and 180-day KNN, followed by both organisation/KNN combinations.

KNN finds up to 20 posts closest by cosine similarity from other permitted fit
authors, strictly earlier than the query and with at most one post per author.
Features describe similarity, neighbour age, weighted class proportions and historical residuals.
Training rows receive these features through inner author folds: two parts during
outer CV and three for final training. Organisation features use earlier mentions
by fit authors, without labels or reactions.

The acceptance rule requires a pooled macro-F1 gain of at least 0.002, improvement
in at least two folds and no fold losing more than 0.005. A combination must pass
against both the selected individual block and its family without the new blocks.

| Block | OOF macro-F1 | Δ vs own family, pp | 95% author CI for Δ, pp | Accepted |
|---|---:|---:|---:|---|
| A / organisations | 0.601614 | −0.029 | [−0.227; +0.182] | no |
| A / KNN, all history | 0.610369 | +0.846 | [+0.586; +1.128] | yes |
| A / KNN, 180 days | 0.611227 | +0.932 | [+0.665; +1.193] | yes |
| B / organisations | 0.623577 | +0.073 | [−0.157; +0.306] | no |
| B / KNN, all history | 0.620770 | −0.207 | [−0.446; +0.052] | no |
| B / KNN, 180 days | 0.623557 | +0.071 | [−0.159; +0.332] | no |
| C / organisations | 0.619193 | −0.021 | [−0.250; +0.198] | no |
| C / KNN, all history | 0.619575 | +0.018 | [−0.236; +0.282] | no |
| C / KNN, 180 days | 0.621762 | +0.236 | [−0.003; +0.499] | yes |

We selected A/KNN180, B without additions and C/KNN180. Organisation features
and all six combinations failed the acceptance rule. For C/KNN180, the interval
includes zero: passing the selection rule is not the same as statistically convincing improvement.

## HGB tuning and blending

We tested 12 configurations per family, varying leaf count, minimum leaf size,
L2 regularisation, class balancing and learning rate paired with iteration count.
All 36 candidates and 15 A/B/C blends completed. The best individual models were
A/KNN180/leaves63 at **0.615631**, B/leaves63 at **0.629499**, and
C/KNN180/combo63 at **0.629927** OOF macro-F1.

V2 is **50% B/leaves63 + 50% C/KNN180/combo63**. It achieved **0.638197** versus
**0.623742** for the recomputed V1-recipe control on the same folds.
The gain is **1.445 percentage points**, with fold gains of +1.540 / +1.132 / +1.666;
the paired author 95% CI is **[1.206; 1.702] percentage points**. This is the joint
effect of features, tuning and blending, not an isolated estimate of KNN's contribution.
Historical V1 OOF (0.624701) differs slightly from the current control;
the cause of that difference has not been established.

## Final decision rule

After selecting V2, we tested 81 pairs of multipliers for `below` and `above`:
0.8–1.2 in steps of 0.05, with `typical` fixed at 1. Authors within each outer fold
were split into two halves; weights were fitted on one half and evaluated on the
other, then vice versa. Both halves were excluded from that fold's base-model training.

Evaluation of the six rules yielded macro-F1 **0.648045**, compared with **0.638197**
for V2: a gain of **0.985 percentage points**, with a 95% author CI of **[0.786; 1.193]**.
After this check, the single weight vector **[0.8, 1, 0.8]** was fitted on all OOF rows.
Its score of 0.647751 on those same rows is not validation. The final class is
`argmax([0.8, 1, 0.8] * probabilities)`; the original probabilities are unchanged.

## Official test

| Model | Macro-F1 | QWK | Accuracy |
|---|---:|---:|---:|
| Original baseline | 0.540334 | 0.460276 | 0.544459 |
| V1 | 0.614185 | 0.558349 | 0.625149 |
| V2 | 0.624701 | 0.565809 | 0.640243 |
| V2 + decision rule | **0.639033** | **0.574074** | **0.669239** |

V2 improved over V1 by 1.052 macro-F1 percentage points, with a 95% author CI
of [0.564; 1.521]. The final rule added **1.433 percentage points** over V2,
with a CI of **[1.035; 1.862]**. The total gain over baseline is **9.870 percentage points**.
Compared with V2, F1 for `below` fell from 0.539270 to 0.536159, `typical` rose
from 0.678778 to 0.722307, and `above` from 0.656053 to 0.658633:
most of the final step's gain comes from `typical`.

## Interpretation limits

All parameters were selected using train, but V1 and V2 test results were already
known during later stages. The final test therefore is not a completely fresh,
independent confirmation of the whole research process. The rule was not changed
after its final evaluation. Reusing OOF predictions for selection also introduces optimism.

Intervals use a paired bootstrap over authors, with 1000 resamples and seed 42.
They describe uncertainty for fixed predictions and exclude training, configuration
search and the full adaptive experimental history. Historical reaction counts
come from the supplied snapshot; their exact values at a new post's publication time are unavailable.
The baseline's original snapshot features are retained, so these results apply
to this dataset's protocol and do not establish performance in real time.
