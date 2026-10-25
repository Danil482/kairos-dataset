# Kairos Post-Performance: Feature Report

## Headline

Adding text-embedding (semantic) features to the classical baseline moves all three metrics up, but by roughly the width of a single-run fluctuation. The honest reading is "no harm, plausibly a small gain, not yet proven."

| Configuration | Macro-F1 | QWK | Accuracy |
| --- | ---: | ---: | ---: |
| Non-deep features only | 0.5811 | 0.5174 | 0.5878 |
| + embedding/semantic features | 0.5880 | 0.5262 | 0.5946 |

## What each layer does

**Non-deep baseline (0.5811 / 0.5174 / 0.5878).** Per-post text and formatting statistics, calendar encodings, content-type and media flags, and leak-free history: rolling/expanding reaction and residual statistics, cadence, author-relative style deviation, level and spread — all computed strictly from earlier posts. This block is doing the heavy lifting. It captures how a post is written and where it sits in its author's trajectory, and it already clears a "typical"-heavy baseline on macro-F1, which `metrics.py` is explicitly built to punish.

**+ semantic layer (0.5880 / 0.5262 / 0.5946).** Cosine similarity to the author's prior 1/5/20 posts and centroids, BM25 and embedding relevance to listed skills, and kNN over past posts using both embedding and lexical similarity, aggregating neighbours' historical reactions. Label-free and strictly backward-looking, so the gain is not leakage.

## An honest read of the delta

Three things worth saying plainly:

1. **The gain is small.** ~0.007 absolute is comfortably inside the range you'd expect from a different random seed, a different CV fold split, or a slightly different train/test boundary. These are single runs. Until you have a paired bootstrap or repeated stratified CV over test posts, treat the direction as suggestive and the magnitude as unmeasured.

2. **All three metrics move together, which is mildly reassuring.** A class-collapse artefact would typically show accuracy rising while macro-F1 or QWK fell, or vice versa. That didn't happen. But coherence across metrics is not the same as statistical significance — it just rules out one specific failure mode.

3. **The QWK gain (+0.0088) is the largest in relative terms and the most plausible.** QWK weights distant ordinal errors more heavily than near misses. If the semantic features are doing anything real, it should be sharpening the tails — separating posts whose content genuinely departs from the author's norm — rather than reordering the crowded "typical" middle. That's consistent with what we see, but it's also consistent with noise. Don't over-read it.

## Why the ceiling is where it is

The baseline already had a strong reaction-history block. Semantic features are a genuinely orthogonal signal — meaning that surface statistics can't see — but they are second-order on top of a first-order signal that's already well captured. A lift in the low single-digit thousandths is what you should expect, not a step change. If you wanted a bigger jump, the lever is probably not more semantic features; it's better modelling of the history block or a stronger classifier head.

## Costs

The semantic pass is not free. One embedding run over all posts (fast with `model2vec`, seconds to a couple of minutes on CPU; slower with `sbert`), plus per-author kNN loops in `_embedding_knn_features` and `_bm25_knn_features` that are O(m²) in posts per author. `cache_dir` removes the embedding cost on reruns but not the kNN cost. For a ~0.007 gain, that's a real trade — worth it if you're already paying for the embedding cache, less obviously worth it if you're running this cold on every iteration.

## Bottom line

Keep the semantic features, but ship them with the caveat attached: the reported lift is real in direction and small in size, measured once. The responsible next step is cheap — repeated CV with paired significance testing — before anyone quotes these numbers as an improvement rather than a hint.
