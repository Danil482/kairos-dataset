# Parsing

The data comes in different formats. It is important to support both so that
no data is skipped because of a different format.

# Estimates

Estimates of mean value and sample variance helpful to estimate data variance relatively to mean.

# TF-IDF

An alternative approach to BM25.

The following are computed:

- The TF-IDF vector itself: direct frequency analysis.
- The TF-IDF vector compressed using SVD: compressed data (removing linear dependence).
- Cosine distance multiplied by `resid` for "relevant" posts (recent posts and those with the closest cosine distance).

- Similarity to all posts.
- Similarity to the last 5 posts.
- Similarity to the 5 nearest recent posts.

# M5

A neural network that converts text into embeddings. The intent is to capture
"meaning". A Hugging Face implementation.

I chose this model because it is convenient: it accepts 512 input tokens, enough
to cover >95% of all post texts. It is designed to find semantic similarity, for
example between "How can you attract buyers for a new product?" and "Five ways
to promote a new product…".

The following are computed:

- The M5 vector: directly represents the "meaning" of the text.
- Similarity to all posts.
- Similarity to the last 5 posts.
- Similarity to the 5 nearest recent posts.

# BM25

New scores:

```python
neighbors_c = [p for p in top_n if p.label == c]

number_close_bm25[c] = len(neighbors_c)

sum_bm25[c] = sum(p.bm25_score for p in neighbors_c)

resid_sum_bm25[c] = sum(p.bm25_score * p.resid for p in neighbors_c)
```

Scores for the classes:

For each class (`below`, `typical`, `above`), 3 scores are added. The goal is
to reflect how posts with similar words perform:

- number_close_bm25_classname
- sum_bm25_classname
- resid_sum_bm25_classname

An example showing why it is useful to differentiate scores by class:

| History | Neighbors' resid | Mean |
|---|---|---|
| A | 0, 0, 0, 0, 0 | 0 |
| B | -1, -1, -1, +1.5, +1.5 | 0 |

Using only the mean or normalized mean of all nearest neighbors' residuals gives
the same number. In the first case, all posts are `typical`, and a person would
say the current post is also `typical`. In the second, most posts are `below`,
so a person would say `below` instead of `typical` (0).

A score across classes:

the standard deviation of `resid` in the nearest-neighbor sample.

# Formatting

Extend the baseline features with text formatting features.

| Group | Features | Calculation | Purpose |
|---|---|---|---|
| Emoji | n_emoji, n_unique_emoji, emoji_at_start | Number of emoji, number of distinct emoji, whether the text starts with an emoji | Describe emotional tone and presentation |
| Punctuation | n_exclamations, n_questions, n_repeated_punct | Number of !, number of ?, number of sequences of two or more !? marks | Distinguish restrained and expressive presentation |
| Letter case | uppercase_ratio, n_uppercase_words | Uppercase fraction among cased letters; number of all-uppercase words with at least two letters | Identify emphasis, announcements, and promotional presentation |
| Paragraphs and lists | n_paragraphs, first_paragraph_length, max_paragraph_length, n_list_items | Paragraphs are separated by blank lines; lengths are in characters; items are detected by bullets and numbering | Describe the structure and presentation of the text |
| Deviation from history | emoji_vs_history, uppercase_vs_history, paragraphs_vs_history | Current value minus the mean over the author's last 20 strictly earlier posts | Distinguish the author's usual style from a change in style |

Intuition: the lecture mentioned that this is useful information.

# Metric evaluation

| name | macro_f1 | f1_below | f1_typical | f1_above | qwk | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.5455 | 0.4891 | 0.5730 | 0.5745 | 0.4674 | 0.5509 |
| estimates | 0.5462 | 0.4943 | 0.5664 | 0.5780 | 0.4724 | 0.5496 |
| formatting | 0.5589 | 0.5014 | 0.5864 | 0.5890 | 0.4869 | 0.5645 |
| text_bm25 | 0.5636 | 0.5066 | 0.5870 | 0.5972 | 0.4969 | 0.5678 |
| text_tfidfsvd_e5 | 0.5832 | 0.5060 | 0.6259 | 0.6176 | 0.5141 | 0.5935 |
| comparison_tfidf | 0.5639 | 0.5080 | 0.5878 | 0.5957 | 0.4979 | 0.5685 |
| comparison_e5 | 0.5607 | 0.5130 | 0.5778 | 0.5915 | 0.4960 | 0.5635 |
| comparison_bm25 | 0.5636 | 0.5066 | 0.5870 | 0.5972 | 0.4969 | 0.5678 |
| comparison_tfidf_e5 | 0.5625 | 0.5146 | 0.5797 | 0.5930 | 0.5002 | 0.5653 |
| text_comparison | 0.5962 | 0.5212 | 0.6332 | 0.6341 | 0.5353 | 0.6047 |
| text_comparison_estimates | 0.5997 | 0.5239 | 0.6402 | 0.6350 | 0.5399 | 0.6094 |
| text_comparison_estimates_formatting | 0.6013 | 0.5297 | 0.6354 | 0.6387 | 0.5445 | 0.6090 |
| always_typical | 0.2488 | 0.0000 | 0.7463 | 0.0000 | 0.0000 | 0.5953 |
| random_in_train_mix | 0.3319 | 0.2030 | 0.6001 | 0.1926 | -0.0088 | 0.4386 |

These metrics were obtained from the experiments. The best score is achieved
with all available features.
