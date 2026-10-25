# Kairos Lab 1 — Artemiy Maslov

Test macro-F1 **0.6703**, QWK **0.6105**; author-OOF macro-F1 **0.6732**. All 17,623 predictions, metrics and experiment summaries are in [`solution/outputs/`](solution/outputs/).
The fixed ensemble combines tabular models, author history, a fine-tuned text encoder and CLIP features. Selection used three author-disjoint train folds (seed 42); the official test was scored once after freezing the selection.

Requires Linux x86-64, Docker with NVIDIA GPU support, 8 CPU cores, 32 GiB RAM and one BF16-capable GPU (tested on L40S). Python 3.12, CUDA 12.4 and Python dependencies are pinned. Obtain and unpack the [official release](https://kairos-labs.ai/dataset/) as described in the [original baseline README](docs/BASELINE.md#get-the-data). Data and weights stay outside Git.

```bash
docker build -t kairos-lab1 .
export DATA="/path/to/data" WEIGHTS="/path/to/weights" GPU_UUID="GPU-..."
./solution/run.sh verify-weights
./solution/run.sh predict --check-reference
./solution/run.sh evaluate
```

Trained weights are kept separately. Without them, train the same fixed configuration in fresh `WEIGHTS` and `WORK` directories (`DATA` and `GPU_UUID` remain set):

```bash
export WEIGHTS="/path/to/new-weights" WORK="$PWD/retrain"
./solution/run.sh fetch
./solution/run.sh train
./solution/run.sh predict
```

[`solution/config.json`](solution/config.json) fixes the model settings and blend weights. [Reproduction evidence](solution/outputs/reproduction.json) records exact matches for the prediction CSV and six intermediate feature files rebuilt from raw data with saved weights. Full real-data retraining was not repeated; training and two independent inference runs were tested on synthetic data. Other hardware may differ numerically. The baseline code is unchanged.

Run the 64-test suite without a GPU or dataset: `docker run --rm --network none --entrypoint python kairos-lab1 -m unittest discover -s tests -t .`.
