# Lab 1 submission — engagement prediction

**Author:** Kravchenko Timur, group M3336

Approach layered on top of the baseline (which stays untouched in `baseline.py`).

## Results on the official test split (17,623 posts, 824 authors)

| model | macro-F1 | QWK | accuracy |
|---|---|---|---|
| baseline (gradient boosting, meta only) | 0.5410 | 0.4576 | 0.5465 |
| Tier 1 — author-relative tabular (LightGBM) | 0.6103 | 0.5294 | 0.6484 |
| Tier 2 — + text embeddings (LightGBM) ← **submission** | **0.6361** | **0.5630** | **0.6801** |
| Tier 3 — + CLIP image embeddings (LightGBM) | 0.6357 | 0.5618 | 0.6791 |
| Tier 4 stacker (LR on OOF probas) | 0.6093 | 0.5571 | 0.6169 |

**Submission:** `results/predictions/supertimur.csv` ← Tier 2 model (**+0.095 macro-F1 over baseline**).

Model validation is 5-fold GroupKFold by author (so train / val authors never
overlap, matching the official 80/20 author split).

## Pipeline

### Tier 1 — author-relative tabular features (`my/features_tier1.py`)

Everything that references author history is computed with `expanding + shift(1)`
so a post never sees itself. On top of the baseline features we add:

- **Author-relative deltas**: `n_chars_delta`, `n_words_delta`, `n_hashtags_delta`,
  `n_mentions_delta`, `n_urls_delta`, `is_textless_vs_author`,
  `has_question_vs_author` — all `post_value − author_median`.
- **Cyclic time**: `hour_sin/cos`, `weekday_sin/cos`, `day_of_month_sin/cos`,
  and `hour_delta_circular` against the author's typical hour direction.
- **Author style markers**: `weekday_match` (post matches author's most common
  weekday), `content_type_rarity` (share of this `content_type` in author's history).
- **Rolling reactions**: `roll_mean_5`, `roll_std_5`, `roll_mean_20`,
  `roll_mean_all`, `roll_std_all`, `roll_trend_5` (OLS slope over past 5 posts).
- **Activity**: `posts_per_week_90d`, `cadence_cv_30d`, `days_since_first_post`,
  `hist_n_log`.
- **Collection artefacts**: `is_recent_post`, `age_days_log`.

Model: LightGBM multiclass, `class_weight="balanced"`, `num_leaves=63`,
`learning_rate=0.04`, early stopping on val fold.

### Tier 2 — text embeddings (`my/features_tier2.py`, `my/text_embed.py`)

- **Embeddings**: `intfloat/multilingual-e5-small`, 384-dim, mean-pooled,
  L2-normalised. `max_seq_length=160` to cover ~p85 of posts without padding
  waste. Precomputed once in `cache/text_embeddings_e5.parquet` (~2000s CPU).
- **PCA-128** of the embeddings, fit on train authors only.
- **Author-relative text**: `cos_to_author`, `l2_to_author`,
  `author_posts_so_far` — cosine / L2 between post embedding and the mean of
  the author's prior post embeddings.
- **Style stats**: `avg_word_len`, `caps_ratio`, `exclaim_count`,
  `question_count`, `emoji_count`, `bullet_count`, `placeholder_density`.

Also tried: PCA-32 (nearly identical score) and raw 384-dim (slightly worse —
LightGBM column subsampling dilutes signal).

### Tier 3 — image embeddings (`my/features_tier3.py`)

- **CLIP ViT-B/32** (OpenCLIP, `laion2b_s34b_b79k`) on 16,349 labelled posts
  that have a post image (~14.5% of posts). GPU inference in a Colab T4
  notebook (`notebooks/colab_clip_embeddings.ipynb`), ~5 min for all images.
- **PCA-48** of 512-dim CLIP embeddings on train authors.
- **Author-relative image**: `has_image`, `img_cos_to_author`,
  `img_l2_to_author`, `img_author_count`.

Didn't improve test macro-F1 (0.6361 → 0.6357). Signal from 14.5% coverage
is not enough to compensate for the extra noise in the 85.5% of posts filled
with zeros. Kept the module for reproducibility and in case author-centroid
image features help when combined with video (not pursued here).

### Tier 4 — ordinal regression + stacking (`my/train_tier4.py`)

- LightGBM **regressor** on `resid` with the frozen cutoffs applied to the
  predicted residual.
- CatBoost multiclass for model-type diversity.
- Logistic Regression **stacker** on OOF probability vectors.

Stacker ended up below the Tier 2 LightGBM alone, because the base models are
unequal in strength (LightGBM 0.636, CatBoost 0.601, ordinal 0.604) and
`class_weight="balanced"` in the LR stacker double-balances on top of the
already-balanced base models. Kept here as negative-result evidence rather
than the submission path.

### Tier 4b — Optuna search (`my/tune_lgbm.py`)

TPE sampler + median pruner, 5-fold GroupKFold, 25-trial budget. Best OOF
macro-F1 ≈ 0.6424 (seed 42, baseline Tier 2 features). This is essentially
the same as the untuned Tier 2 model — the configuration is already in a
good basin.

## Running

```bash
# features (cached in cache/*.parquet)
python -m my.features_tier1
python -m my.text_embed              # encode posts (~33 min CPU)
python -m my.features_tier2

# train Tier 2 (the submission)
python -m my.train_tier2

# official evaluation
python evaluate.py --predictions results/predictions/tier2_lightgbm.csv
```

## Notebook

`notebooks/lab1.ipynb` — EDA, feature description, ablation table, confusion
matrix, class-wise F1.

`notebooks/colab_clip_embeddings.ipynb` — Colab GPU notebook that produced the
CLIP image embeddings used by Tier 3.
