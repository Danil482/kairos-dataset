# Lab 1 submission

| File | Contents |
|---|---|
| `predictions.csv` | One class per test post: `author`, `post`, `label` (0 below, 1 typical, 2 above). 17,623 rows. |
| `metrics.json` | Official test result: macro-F1 0.6683, QWK 0.6134, accuracy 0.7070, per-class scores, confusion matrix. |
| `analysis.json` | Train-only EDA, dev results for seeds 42 and 137, author CV, grouped ablation, failed variants, M4 timings. The notebook reads it. |
| `selection.json` | Selected model: decision weights, SHA-256 of the model, config, source files and predictions, upstream commit, limitations. |

The test was scored once with this model. It had been seen earlier with an older model, so the score is not a fresh independent test.
To check the file: `python evaluate.py --predictions lab1_submission/predictions.csv`.
