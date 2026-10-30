# Kairos: Post Engagement Classification

Only posts were used in this work, without images or videos.

## Documentation

- [Approaches and experiment results](docs/SOLUTIONS.md)
- [Feature architecture and customization](docs/ARCHITECTURE-SPINE.md)

## First run

Requires Python 3.10+ and 7-Zip (`7zz` or `7z`). Request the dataset password
and download all five core archive volumes, `lab1_core.7z.001` through `.005`,
from the [dataset page](https://kairos-labs.ai/dataset/). Keep the volumes together.

```bash
./first_run.sh /path/to/lab1_core.7z.001
```

The script extracts `authors/`, `posts/`, `labels.csv`, and `split.json` into
`data/`, creates `.venv/`, installs `requirements.txt`, and starts the experiments.
7-Zip prompts for the archive password. Images and videos are not extracted.
If the data is already unpacked, extraction is skipped. With no arguments, the
script looks for `lab1_core.7z.001` in the repository root.

Feature preprocessing is performed by `baseline.py` before training. The E5 model
is downloaded on its first use; embeddings and the model are cached in `results/`.

## Subsequent runs

With the data and dependencies already available:

```bash
./run_short.sh
```

This runs the configurations and models enabled in `baseline.py`, including
feature preprocessing, training, and scoring. It does not unpack data or install
dependencies. Metrics are saved to `results/metrics/` and predictions to
`results/predictions/`.

Both scripts accept an alternative data directory:

```bash
./first_run.sh /path/to/lab1_core.7z.001 /path/to/data
./run_short.sh /path/to/data
```
