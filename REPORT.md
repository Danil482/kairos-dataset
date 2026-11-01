# Lab 1 — Kairos: did the post beat its author's usual level?

**Author:** Pavel Rodygin · ITMO, Social Media Analytics, Agents & LLMs

## Task
Three-class ordinal classification of LinkedIn posts: *below* (< ×0.577 of the
author's usual reactions), *typical*, *above* (> ×3.182). The author's usual level is
the median of `log(1 + reactions)` over their earlier posts. Split is 80/20 by author;
headline metric is macro-F1, with QWK alongside.

## Method
`solution.py` extends the official baseline (gradient boosting over non-text metadata)
in two directions.

1. **Richer author history (no leakage).** All features use only posts published
   before the current one:
   - `last1_rel`, `last3_rel`, `last5_rel`: mean log-reactions of the last 1/3/5 posts
     minus the current level (captures momentum — an author whose recent posts are
     trending up relative to their long-run median is more likely to land *above*);
   - `hist_std`, `hist_max_rel`, `hist_min_rel`: spread and extremes of the history;
   - `trend5`: last five posts vs the five before them;
   - `posts_30d`: posting frequency in the previous 30 days;
   - `text_len_vs_author`: post length relative to the author's usual length.
2. **Text signal.** Word (1–2-gram) and character (3–5-gram, `char_wb`) TF-IDF with a
   class-balanced logistic regression. Its class probabilities are fed to the booster
   as three features. On train they are **out-of-fold** (5-fold `GroupKFold` by author),
   so the booster never sees probabilities produced from its own training rows.
   Character n-grams are robust to the pseudonymisation placeholders (`⟨URL⟩`,
   `⟨PERSON_n⟩`), emoji and informal spelling, which links to the normalisation
   discussion in Tutorial 1-2 (Q1): we keep raw forms because emoji, hashtags and
   mentions are themselves signals.

Final model: `HistGradientBoostingClassifier` (500 iterations, lr 0.05,
`class_weight="balanced"` so that macro-F1 is not dominated by *typical*).

## Results (test authors, 17,623 posts)

| model | macro-F1 | QWK | accuracy |
|---|---:|---:|---:|
| always *typical* | 0.2488 | 0.0000 | 0.5953 |
| official baseline | 0.5410 | 0.4576 | 0.5465 |
| baseline features, our GBM settings | 0.5393 | 0.4598 | 0.5437 |
| ours: baseline + extended author history (no text) | 0.5720 | 0.5101 | 0.5760 |
| text only (TF-IDF + LR) | 0.4518 | – | – |
| **ours: history + text (final)** | **0.5935** | **0.5401** | **0.5976** |

## Discussion
- The label is relative to the author, so raw popularity (followers, level) mostly
  identifies *who* wrote the post; the useful signals are deviations: recent momentum,
  content type, length and wording relative to the author's own habits.
- Reactions are a single snapshot taken a median of ~710 days after posting, so
  `age_days` is a collection artefact rather than a property of the post.
- Not used (time limit): images, PDFs and video. Natural next steps are multimodal
  late fusion (image embeddings, e.g. CLIP, plus text embeddings) and an LLM
  zero-shot judge given the author's last five posts, which the dataset authors
  report at ~0.63 macro-F1 on a 2,000-post sample.

## Reproduce
```
pip install -r requirements.txt
python solution.py            # or --no-text for the metadata-only variant
python evaluate.py --predictions results/predictions/solution.csv
```
