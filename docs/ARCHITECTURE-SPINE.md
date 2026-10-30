# How features are structured and added

All feature definitions, calculations, and model preparation are in
`kairos/features.py`. `baseline.py` selects column sets, trains models, and
saves predictions and metrics.

## CUSTOMIZATION POINTS

### 1. Feature groups and parameters in `kairos/features.py`

The main settings block:

```python
ARTEFACT_NUM = ["age_days", "hist_n"]

# history
ESTIMATES_NUM = ["hist_mean", "hist_sample_var"]

FORMATTING_HISTORY = 20
FORMATTING_NUM = [
    "n_emoji", "n_unique_emoji", "emoji_at_start",
    "n_exclamations", "n_questions", "n_repeated_punct",
    "uppercase_ratio", "n_uppercase_words",
    "n_paragraphs", "first_paragraph_length", "max_paragraph_length", "n_list_items",
    "emoji_vs_history", "uppercase_vs_history", "paragraphs_vs_history",
]

N_LAST = 20
N_NEAREST = 5
TEXT_COMPARISON_E5_NUM = [
    "similarity_all_e5",
    "similarity_last_5_e5",
    "last_nearest_resid_e5",
]
TEXT_COMPARISON_TFIDF_NUM = [
    "similarity_all_tfidf",
    "similarity_last_5_tfidf",
    "last_nearest_resid_tfidf",
]

BM25_TOP_N = 12
BM25_K1 = 1.5
BM25_B = 0.75
TEXT_BM25_NUM = [
    f"{stat}_bm25_{class_name}"
    for class_name in CLASS_NAMES.values()
    for stat in ("number_close", "norm_sum", "resid_sum")
]

TFIDF_MAX_FEATURES = 5000
SVD_COMPONENTS = 128
E5_DIMENSIONS = 384
E5_MODEL = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

# TF-IDF and SVD use only the leading names matching their actual dimensions.
TEXT_TFIDF_NUM = [f"text_tfidf_{i:04d}" for i in range(TFIDF_MAX_FEATURES)]
TEXT_SVD_NUM = [f"text_svd_{i:03d}" for i in range(SVD_COMPONENTS)]
TEXT_E5_NUM = [f"text_e5_{i:03d}" for i in range(E5_DIMENSIONS)]
TEXT_NUM = TEXT_TFIDF_NUM + TEXT_SVD_NUM + TEXT_E5_NUM
```

| Setting | What can be changed |
|---|---|
| `ARTEFACT_NUM` | Composition of the data collection feature group; values are computed in `build()` |
| `ESTIMATES_NUM` | Composition of the history statistics group; values are computed in `build()` |
| `FORMATTING_NUM` | 15 text formatting features and deviations from the author's style; computed in `build()` |
| `FORMATTING_HISTORY` | Number of strictly earlier posts for the mean style; defaults to 20, independently of `N_LAST` |
| `N_LAST` | Number of recent earlier posts used to find similar posts with cosine similarity and BM25 |
| `N_NEAREST` | Number of most similar candidates for the `cosine * resid` sum |
| `TEXT_COMPARISON_E5_NUM`, `TEXT_COMPARISON_TFIDF_NUM` | Names of the three output comparisons for each representation |
| `BM25_TOP_N` | Maximum number of neighbors with positive BM25, shared across the three classes |
| `BM25_K1`, `BM25_B` | Term frequency saturation and document length compensation in BM25 |
| `TEXT_BM25_NUM` | Nine neighbor scores: count, BM25 sum, and `BM25 * resid` sum for each class |
| `TFIDF_MAX_FEATURES` | Maximum TF-IDF vocabulary size |
| `SVD_COMPONENTS` | Maximum number of SVD components |
| `E5_MODEL`, `E5_REVISION` | Embedding model and its pinned revision |
| `E5_DIMENSIONS` | Expected dimensionality of the chosen model; must match its actual output |
| `TEXT_TFIDF_NUM`, `TEXT_SVD_NUM`, `TEXT_E5_NUM` | Coordinate names generated automatically from the specified dimensions |
| `TEXT_NUM` | Combined text vector coordinates used to determine processing in `preprocess()` |

The windows require `1 <= N_NEAREST <= N_LAST`, with both values being integers.
For BM25, `BM25_TOP_N` is a positive integer; if fewer neighbors are available,
all suitable ones are used. `BM25_K1 > 0`, `0 <= BM25_B <= 1`.
Changing `TFIDF_MAX_FEATURES` and `SVD_COMPONENTS` automatically changes the lists
of possible coordinates. Tables contain only the coordinates actually produced,
without zero padding. Parameters such as `min_df`, `ngram_range`, and other
`TfidfVectorizer` settings are changed in `prepare()`.

When adding or renaming a feature, keep the name list synchronized with the
column calculation in `build()` or `prepare()`. The order of the three names in
each `TEXT_COMPARISON_*_NUM` matches the output order of `_text_comparison()`:
all history, last five, weighted residual sum. To experiment with a subset of
these features, select the desired names in `CONFIGURATIONS`.

When replacing E5, keep the model, revision, dimensionality, input text format,
and cache key in `_e5_embeddings()` consistent. `E5_DIMENSIONS` describes the
model output and does not reduce the vector by itself. The scalar comparison
groups `TEXT_COMPARISON_*_NUM` remain outside `TEXT_NUM` so that they undergo
numeric scaling.

Feature sets for individual experiments are selected in
`baseline.py:CONFIGURATIONS` by combining lists from `features.py`.

### 2. Models to train in `baseline.py`

`MODELS` determines which classifiers are trained on each configuration:

```python
MODELS = {
    "hist_gradient_boosting": gradient_boosting,
    # "logistic_regression": logistic_regression,
}
```

To enable a model, add or uncomment its entry; to exclude it from the run,
comment out or remove the entry. In the version shown, only gradient boosting
is trained. The key is used in metrics and prediction filenames.

The value is a function that creates a new classifier. Training parameters
are changed in `gradient_boosting()` and `logistic_regression()`: for example,
the number of iterations, learning rate, class weights, or regularization.
For a new model, add such a function and an entry in `MODELS`. Consider the
supported input format: currently, sparse-to-dense conversion in the training
loop is enabled for the `hist_gradient_boosting` key.

### 3. Scaling and processing in `features.preprocess()`

`preprocess(feature_names)` defines processing rules for the selected columns.
This is where scalers, category encoding, and vector handling are changed:

| Columns | Current processing | Where to change it |
|---|---|---|
| Ordinary numbers, `FORMATTING_NUM`, `TEXT_COMPARISON_*_NUM`, and `TEXT_BM25_NUM` | `StandardScaler` | The `scaler` variable and the `("num", scaler, numeric)` rule |
| Categories from `CONTENT_CAT` | `OneHotEncoder` | The `"cat"` rule: encoding method, rarity threshold, and unknown categories |
| TF-IDF coordinates | Conversion of sparse pandas columns to scipy | The `"tfidf"` rule and `_tfidf_matrix()` function |
| SVD and E5 coordinates | Values passed through unchanged | The `"vectors"` rule with `"passthrough"` |

For example, to change numeric scaling, replace `StandardScaler` with the
chosen transformer and add its import. To change category encoding, edit the
`OneHotEncoder` block. To process particular numeric columns separately, give
them a separate rule in `ColumnTransformer` and exclude them from the general
`numeric` list.

The rules are currently shared across all configurations and models; changing
`preprocess()` affects every selected set containing the corresponding columns.
The `preprocess()` call itself creates a transformer. Its parameters are fitted
through `fit_transform(train)`; only `transform(test)` is used for test.

## Data flow

1. `kairos/data.py` reads posts, authors, and existing labels with the train/test split.
2. `kairos/label.py:author_level()` computes the author's level from earlier posts.
3. `features.build(levelled, followers)` creates a table of ordinary features for
   all posts. The row identifiers are `author` and `post`.
4. `baseline.py` joins the features with labels on these identifiers and splits
   the table using the existing `split` column.
5. `features.prepare(train, test, feature_names, results_dir, history=levelled)`
   adds text vectors and comparisons with earlier posts. Only representations
   needed by the selected configurations are computed. `history` contains the
   full history, including the initial unlabelled posts. Train/test indices and
   row order are preserved; history is matched by the `author`, `post` key.
6. `features.preprocess(feature_names)` creates a transformer for the selected
   columns. In `baseline.py`, it is fitted through `fit_transform(train)` and
   applied to test through `transform(test)`.
7. Classifiers are trained on the resulting matrices. The output format is
   defined by `baseline.py`, `kairos/metrics.py`, and `evaluate.py`.

## Groups are lists of column names

Groups are defined consistently in `features.py`:

```python
ESTIMATES_NUM = ["hist_mean", "hist_sample_var"]
TEXT_SVD_NUM = [f"text_svd_{i:03d}" for i in range(SVD_COMPONENTS)]
```

Each name denotes an individual numeric or categorical column in the table.
A vector of 128 numbers occupies 128 columns. Table cells do not contain lists
or arrays of vectors.

| Constant | Contents | Model preparation |
|---|---|---|
| `CONTENT_NUM` | Length, counts, posting time | `StandardScaler` |
| `CONTENT_CAT` | Post type | `OneHotEncoder` |
| `HISTORY_NUM` | Author's level, spread, rhythm, followers | `StandardScaler` |
| `ARTEFACT_NUM` | Post age at collection, history size | `StandardScaler` |
| `ESTIMATES_NUM` | History mean and sample variance | `StandardScaler` |
| `FORMATTING_NUM` | Emoji, punctuation, letter case, paragraphs, lists, and deviations from history | `StandardScaler` |
| `TEXT_TFIDF_NUM` | Up to 5000 TF-IDF coordinates | No additional scaling |
| `TEXT_SVD_NUM` | Up to 128 normalized SVD coordinates | No additional scaling |
| `TEXT_E5_NUM` | 384 normalized E5 coordinates | No additional scaling |
| `TEXT_COMPARISON_E5_NUM` | Three history comparisons based on E5 | `StandardScaler` |
| `TEXT_COMPARISON_TFIDF_NUM` | Three history comparisons based on TF-IDF | `StandardScaler` |
| `TEXT_BM25_NUM` | Nine BM25 neighbor scores across three classes | `StandardScaler` |

`TEXT_NUM` combines the text vector groups and is used to select processing.
The `_NUM` suffix means numeric columns; scaling rules are defined in
`preprocess()`. The scalar groups `TEXT_COMPARISON_*_NUM` are not part of
`TEXT_NUM` and are processed as ordinary numeric features. The same rule applies
to `TEXT_BM25_NUM`.

The `TEXT_TFIDF_NUM` and `TEXT_SVD_NUM` lists contain the maximum possible set of
coordinate names. `prepare()` creates only the columns actually produced,
without zero padding. The number of SVD components is limited by
`SVD_COMPONENTS`, the number of TF-IDF features, and the number of training posts.
For TF-IDF, the mapping from a coordinate number to a specific term is determined
by the vocabulary fitted on train in that run.

In `preprocess()`, the TF-IDF and SVD column selectors select existing coordinates
when fitted on train. The subsequent `transform(test)` uses the same columns.
Missing ordinary features or E5 coordinates remain an error.

## Experiment configurations

A value in `CONFIGURATIONS` is an ordinary list of columns:

```python
BASELINE = (
    features.CONTENT_NUM
    + features.HISTORY_NUM
    + features.ARTEFACT_NUM
    + features.CONTENT_CAT
)
CONFIGURATIONS = {
    "baseline": BASELINE,
    "estimates": BASELINE + features.ESTIMATES_NUM,
    "formatting": BASELINE + features.FORMATTING_NUM,
    "text_bm25": BASELINE + features.TEXT_BM25_NUM,
    "text_tf_idf_truncatedsvd": BASELINE + features.TEXT_SVD_NUM,
    "text_only_tf_idf_truncatedsvd": features.TEXT_SVD_NUM,
    "text_comparison_e5": BASELINE + features.TEXT_COMPARISON_E5_NUM,
    "text_comparison_tfidf": BASELINE + features.TEXT_COMPARISON_TFIDF_NUM,
}
```

The key is used in the metrics table and the prediction filename. The list
contains all required columns, including categories. A configuration with only
`TEXT_SVD_NUM` therefore receives only the actual text coordinates, at most 128.

`baseline.py` passes the union of the lists to `features.prepare()`, then applies
the same training loop to each configuration. Adding a feature set does not
require a separate training branch or matrix concatenation.

## How to add a feature

### A numeric feature of an individual post or its history

1. In `features.py`, add the name to a suitable group or create a new one:
   `MY_NUM = ["my_feature"]`.
2. In `features.build()`, compute `df["my_feature"]` for each row.
   For history, use only posts that precede the current one.
3. In `baseline.py`, add the configuration
   `"my_experiment": BASELINE + features.MY_NUM`.

New ordinary numeric columns automatically pass through `StandardScaler`.

### A categorical feature

Compute the column in `features.build()` and include its name in `CONTENT_CAT`.
`preprocess()` applies `OneHotEncoder` only to categories explicitly selected by
the configuration. Unknown test categories are handled with
`handle_unknown="ignore"`.

### A feature that needs a training corpus

1. In `features.py`, declare the list of output coordinate names.
2. In `features.prepare()`, determine whether this group was requested. Fit the
   transformation only on train, then transform test with the same instance.
3. Add the results as separate columns with the original train/test indices.
   For sparse vectors, use `DataFrame.sparse.from_spmatrix()`.
4. If the output is a ready-to-use text vector, add the group to `TEXT_NUM` and
   the corresponding list to `preprocess()` to pass coordinates through without
   additional scaling. Define other processing rules in `features.py` as well.
5. Add a configuration by combining column lists in `baseline.py`.

TF-IDF and SVD share the same TF-IDF fitted once on train. E5 uses a pinned
pretrained model and is not trained on the corpus. Empty texts receive a zero
E5 vector. Model loading and the embedding cache are in `features.py`; the cache
format includes the texts in order, the model, and its revision.

## Text formatting

The `FORMATTING_NUM` group contains 15 numeric features. They are computed in
`features.build()` through `_formatting()` for all posts, including the initial
unlabelled ones. They require no fitted vocabulary, embeddings, or labels.
The `formatting` configuration uses `BASELINE + features.FORMATTING_NUM`.
Like other ordinary numeric columns, they pass through `StandardScaler`.

| Group | Columns | Rule |
|---|---|---|
| Emoji | `n_emoji`, `n_unique_emoji`, `emoji_at_start` | Number of emoji, number of distinct sequences, and presence of an emoji at the start after skipping leading whitespace |
| Punctuation | `n_exclamations`, `n_questions`, `n_repeated_punct` | Number of `!`, number of `?`, number of continuous `[!?]{2,}` sequences; `!!!??` is one sequence |
| Letter case | `uppercase_ratio`, `n_uppercase_words` | Uppercase fraction among cased letters; number of words with at least two letters whose cased letters are all uppercase |
| Paragraphs | `n_paragraphs`, `first_paragraph_length`, `max_paragraph_length` | Nonempty paragraphs are separated by blank lines; lengths are computed with `len()` after stripping whitespace from paragraph boundaries |
| Lists | `n_list_items` | Number of lines that begin, after indentation, with a bullet or number, whitespace, and nonempty content |
| History | `emoji_vs_history`, `uppercase_vs_history`, `paragraphs_vs_history` | Current `n_emoji`, `uppercase_ratio`, `n_paragraphs` minus the corresponding mean over the author's last 20 strictly earlier posts |

Compound emoji are counted using Unicode graphemes: an emoji with a skin tone,
a ZWJ sequence, a flag, and a keycap emoji such as `1️⃣` each count as one item.
Uniqueness is determined by the full character sequence. Ordinary digits and
`#`, `*` characters without keycap formatting are not counted as emoji.
Unicode properties and graphemes use the `regex` dependency, explicitly listed
in `requirements.txt`.

In `uppercase_ratio`, digits, punctuation, and uncased letters do not enter the
denominator. If there are no cased letters, the result is zero. Uppercase word
length is checked by letter count, not code point count: a single letter with a
combining accent does not become a two-letter word.

Before analysis, CRLF and CR are converted to LF. A single line break remains
inside the paragraph and counts towards its length. A blank line may contain
spaces or tabs. For empty text, the paragraph count and both lengths are zero.
Supported bullets include `-`, `*`, `+`, common Unicode bullets, dashes, and check
marks, as well as numbers such as `1.`, `1)`, `(1)`, and compound numbering such
as `1.2.`. A marker must be followed by whitespace and content; horizontal
separators such as `---` and `***` are not counted as items.

The history window is controlled by a separate constant,
`FORMATTING_HISTORY = 20`, a positive integer. If history is insufficient,
available posts are used. With no history, all three deviations are zero.
Posts with the same `posted_at` do not enter each other's history. A post without
a date retains its current formatting features, but its deviations are zero
and it does not enter other posts' history. Original table indices and row order
are preserved.

## Comparison with the author's history

The groups `TEXT_COMPARISON_E5_NUM` and `TEXT_COMPARISON_TFIDF_NUM` each contain
three columns with the suffix of the corresponding representation:

- `similarity_all_e5` / `similarity_all_tfidf`: cosine similarity of the current
  vector to the mean vector of all the author's earlier posts.
- `similarity_last_5_e5` / `similarity_last_5_tfidf`: similarity to the mean vector
  of the last five earlier posts.
- `last_nearest_resid_e5` / `last_nearest_resid_tfidf`: sum of
  `cosine(current, previous) * previous.resid` over the most similar candidates.

Cosine is computed as `(a · b) / (||a|| * ||b||)`. For the first two features,
history vectors are averaged first, then cosine similarity to the mean is
computed; the mean of individual cosine similarities is not used here.

Parameters in `features.py`: `N_LAST = 20`, `N_NEAREST = 5`. These are the initial
experiment settings. Positive integers satisfying `1 <= N_NEAREST <= N_LAST`
are allowed. For the third feature, the last `N_LAST` posts are taken first,
posts with missing or infinite `resid` are excluded, and then up to `N_NEAREST`
candidates are selected in descending cosine order. Ties prefer the more recent
candidate; equal dates prefer the later row in the original history. Older
posts outside the `N_LAST` window do not replace excluded candidates.

History is restricted to the same author and strictly earlier `posted_at`.
Posts with the same date are processed as one group and do not enter each
other's history. Posts without dates are not used in history, and their
comparisons are zero. Initial unlabelled posts contribute to the mean vectors,
even though their `resid` is still missing. If history is short, available posts
are used. Cosine is zero for a zero vector; the sum is zero if there are no
candidates. Negative cosines are preserved. The sum is not normalized by the
sum of weights.

`history=levelled` is required only for comparison groups; calling `prepare()`
without this argument remains valid for the previous groups. History must
contain all requested train/test posts, and the `author`, `post` keys must be
unique. Only authors from train/test enter the calculation.

The vocabulary and IDF are still fitted only on labelled train texts. The same
transformer encodes the participating authors' history, including test authors.
E5 encodes this history in one call; the resulting vectors are reused for both
coordinates and comparisons. Without comparison groups, the previous encoding
order, `train_text + test_text`, and compatibility with its cache are preserved.
Computing comparisons does not require SVD or adding the original coordinates
to the table.

`_text_comparison()` uses a cumulative vector sum for the entire history and
small windows for recent posts. A full pairwise similarity matrix of all posts
is not built. TF-IDF retains sparse storage.

## BM25: neighbor scores by class

The `TEXT_BM25_NUM` group adds nine numeric columns. For each of `below`,
`typical`, and `above`, the following are computed:

```python
number_close_bm25_classname = len(neighbors_c)
norm_sum_bm25_classname = sum(p.bm25_score for p in neighbors_c)
resid_sum_bm25_classname = sum(p.bm25_score * p.resid for p in neighbors_c)
```

`norm_sum` is the feature name; by the specified formula, it is an ordinary sum
of BM25 scores, without division by the number of neighbors or the sum of all
scores. Column order: three scores for `below`, then three for `typical`, then
three for `above`. The class number mapping comes from `kairos.label.CLASS_NAMES`.

The `text_bm25` configuration contains `BASELINE + features.TEXT_BM25_NUM`.
In `prepare()`, it calls `_text_bm25(history, train_rows)`. The nine results are
matched to train/test using the existing `author`, `post` keys. They pass through
the ordinary `StandardScaler` in `preprocess()` and are not part of `TEXT_NUM`.
No new training branches, external libraries, or models are required.

The BM25 query is the current post's text. Candidates are the last `N_LAST = 20`
strictly earlier posts by the same author. Within this window, posts without a
finite `resid` are excluded, then up to `BM25_TOP_N = 12` neighbors with the
highest positive BM25 are selected. This is a shared top 12, which is then split
by class. Zero matches do not increase `number_close`. Ties prefer the more recent
post; equal dates prefer the later row in the original history. The current
post, future posts, and posts with the same date do not enter its candidates.
Posts without dates produce zero features.

A neighbor's class is determined from its `resid` through the existing
`fit_cutoffs()` and `classify()`. Cutoffs are computed only from labelled train
post residuals. Labels and residuals of current or future test posts do not
enter the calculation of its features. If a class has no suitable neighbors,
all three of its scores are zero.

BM25 uses a separate `CountVectorizer` with ordinary lowercase words and the
standard token pattern of at least two characters, without stemming or stop-word
removal. Vocabulary, document frequencies, and average document length are
determined from train texts. The BM25 vocabulary is not limited by
`TFIDF_MAX_FEATURES`; TF-IDF, SVD, and E5 are not computed for these nine features.

For query term `t` and historical document `d`, the formulas are:

```text
idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

BM25(q, d) = sum over distinct terms t in q:
    idf(t) * tf(t,d) * (k1 + 1)
    / (tf(t,d) + k1 * (1 - b + b * dl(d) / avgdl))
```

Here, `N` is the number of train documents, `df` is the number of train documents
containing the term, `tf` is its occurrence count in the document, `dl` is the
document length, and `avgdl` is the mean train document length. Length includes
all tokens, including those absent from the train vocabulary. Repeated words
in the query do not add a separate multiplier. Defaults: `BM25_K1 = 1.5`,
`BM25_B = 0.75`. An empty query, no matches, or an empty train vocabulary yields
zero scores.

BM25 frequency and weight matrices are stored sparsely. Only scores of current
posts against a small candidate window are computed densely; there is no full
pairwise similarity matrix of all posts.

## Validation rules

- Do not use test labels to fit transformations. Historical comparisons may
  use `resid` from earlier posts by the same author, including test authors;
  `resid` of the current or future posts does not enter the features.
- Fit the TF-IDF vocabulary, IDF, SVD, scaling, and category encoding only on
  train. Compute all historical features before adding the current post to
  the author's history.
- Preserve feature correspondence to the original `author`/`post` and table
  indices. Indices may be nonconsecutive after the train/test split.
- Keep TF-IDF sparse during data preparation and logistic regression training.
  In `baseline.py`, the sparse matrix is converted to dense for
  `HistGradientBoostingClassifier`, which requires this input.
- When changing the embedding model, its revision, or encoding method, update
  the cache key; when changing dimensionality, update the column name list.
- Check dimensions, row order, handling of unknown words and categories,
  absence of unrequested features, and the ability to train both models.
- After adding a new experiment, update its description in README.

Only at the user's explicit request, feature preparation contract checks can
be run from the repository root:

```bash
python -m unittest discover -s tests -v
```

They check that TF-IDF is fitted only on train, row correspondence, selection
of text columns only, sparsity, and compatibility with both classifiers.
The small corpus is checked without artificial zero padding of coordinates.
The E5 coordinate order test uses an encoder stub; downloading the model is
not required for these checks.
Comparison scenarios additionally check formulas, history windows, equal and
missing dates, negative cosines, residuals, and row correspondence.
BM25 scenarios check separate class scores with the same overall residual sum,
the BM25 formula, top 12, exclusion of zero matches, temporal boundaries, and
running only BM25 features without computing embeddings.
Formatting scenarios check compound emoji, Unicode letter case, punctuation,
paragraphs and lists, empty texts, preservation of indices, and a window of 20
strictly earlier posts without influence from future posts or other authors.
