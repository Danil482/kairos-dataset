# Lab 1 submission — Andrei Pribavkin (wilsonSev)

Predict whether a LinkedIn post performs **below**, **typically**, or **above** its author's usual engagement level.

**Final method:** an equal-weight probability ensemble of two history-aware gradient boosters: one uses task-adapted multilingual sentence embeddings, the other uses TF-IDF compressed with SVD. Both also use post metadata, BM25 retrieval of earlier author posts, and recent author-history features.

**Submission files:** [English report](lab1_final_report.ipynb) · [Final predictions](results/predictions/lab1_final_v4.csv)

## Results

| Method | Mean validation macro-F1 | Follow-up test macro-F1 |
|---|---:|---:|
| Provided baseline | — | 0.5410 |
| Frozen embedding + SVD ensemble | 0.6098 | 0.5995 |
| **Task-adapted embedding + SVD ensemble** | **0.6129** | **0.6008** |

Final test QWK: **0.5433**; accuracy: **0.6084**. Test set: 17,623 posts; training set: 71,395 posts.

Validation uses three author-disjoint 80/20 splits of the official training set (seeds 42, 7, 2026). These splits overlap and were reused during experimentation. The official test was inspected in earlier rounds; the final configuration was selected using validation before its follow-up test assessment. **This is not an untouched blind-test result.** The improvement from adaptation is small and does not establish statistical significance.

Read the [final English report](lab1_final_report.ipynb) for the hypotheses, experiment tables, confusion matrix, limitations, and complete reproduction sequence. Saved aggregate results are included, so its display cells work without access to the dataset.

## Method in brief

1. Compute features using only each author's strictly earlier posts.
2. Retrieve up to three similar earlier posts with BM25 and summarize their relative performance.
3. Add recent engagement statistics and basic text-style features.
4. Encode text with the pinned multilingual MiniLM model (384 dimensions, maximum 128 tokens).
5. Adapt the encoder on 384 sampled fit-only posts using 768 same/different-class pairs and cosine-similarity MSE. This is **SetFit-inspired**, implemented directly with Sentence Transformers and PyTorch; it is not the SetFit package's default pipeline.
6. Train a balanced histogram gradient booster on these embeddings plus tabular features. Train a companion using 128 SVD dimensions of TF-IDF plus the same tabular features.
7. Average their three-class probabilities with fixed 50/50 weights and take the largest.

## Reproduce

Request the dataset from [Kairos](https://kairos-labs.ai/dataset/). Place the unpacked core dataset in `data/`, or set `KAIROS_DATA_DIR` to its directory. The required prediction CSV contains only pseudonymous author/post keys and predicted classes. Dataset text, media, true labels, passwords, training pairs, caches and model checkpoints are excluded.

Run the commands below from this submission directory (`cd submissions/wilsonSev` from the repository root). Use Python 3.12 for the complete encoder pipeline:

```bash
python3.12 -m venv .venv-embeddings
source .venv-embeddings/bin/activate
python -m pip install -r requirements-round2.txt
python -m pip install jupyterlab
python -m unittest discover -s tests -v
jupyter lab lab1_final_report.ipynb
```

Open the notebook from this repository directory. Its default mode displays saved aggregate results. To regenerate all validation experiments and final predictions, set `REBUILD_FINAL = True` in its reproduction cell and execute it. This downloads the pretrained encoder on the first run and requires the dataset, disk space for caches, and compute time. Subsequent adaptation uses that cached pretrained model. The exact tested dependency versions are recorded in `requirements-round2.txt`; baseline-only dependencies are in `requirements.txt`.

The regenerated submission is `results/predictions/lab1_final_v4.csv`. Evaluate it locally with:

```bash
python evaluate.py --data "$KAIROS_DATA_DIR" --predictions results/predictions/lab1_final_v4.csv
```

If you use the default `data/` folder, omit `--data`. Apart from the required prediction CSV, do not commit generated dataset-derived artifacts.

The source experiment ran on Apple Silicon with MPS. CPU execution is supported but slower, and small numeric differences across environments are possible. The repository export was checked with the existing feature tests and report display cells; a complete retraining from a newly installed environment was not performed for publication.

## Files

| File | Role |
|---|---|
| `lab1_final_report.ipynb` | Final English report and reproduction sequence |
| `prepare_fusion_data.py` | Training tables, style and BM25 features |
| `encode_posts.py` | Frozen pretrained text embeddings |
| `lab_round3.py` | Frozen embedding/SVD controls and ensemble |
| `adaptation_pilot.py` | Fit-only pair construction and validation |
| `adapt_text_encoder.py` | Encoder adaptation and text encoding |
| `final_adaptation.py` | Validation selection and final fitting |
| `kairos/` | Dataset utilities and feature implementations |
| `tests/` | Synthetic-data checks for history and retrieval features |
| `results/` | Aggregate experiment metrics only |

## Code formatting

Python source files and notebook code cells use Black with the settings in
`pyproject.toml`. Install the development dependencies to format or check them:

```bash
python -m pip install -r requirements-dev.txt
python -m black .
python -m black --check .
```

## Attribution and data use

Based on [Danil482/kairos-dataset](https://github.com/Danil482/kairos-dataset), starting from commit `623cbc87bc6c3ee261ad096ccea60fa3d608fa5a`. Original code copyright © 2026 Kairos; its [MIT license](LICENSE) is preserved. See the [upstream documentation](../../README.md) for dataset details and citation.

The code license does **not** license the dataset. The upstream rules prohibit re-hosting the data, sharing the password, or attempting re-identification. This submission contains code, aggregate evaluation results and the required prediction CSV. Course lecture PDFs and personal learning notebooks are not redistributed.
