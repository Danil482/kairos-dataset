import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from sklearn.model_selection import train_test_split
from kairos.data import load_labels, load_posts, load_followers
from kairos.label import author_level
from kairos import features
from kairos.bm25_features import author_bm25_features
from baseline import NUMERIC

root = Path("results/fusion")
root.mkdir(parents=True, exist_ok=True)
data = Path(os.environ.get("KAIROS_DATA_DIR", "data"))
labels = load_labels(data)
labels = labels[labels["split"] == "train"].copy()
# No official test authors/texts/labels enter these new experiments.
posts = load_posts(data)
posts = posts[posts.author.isin(labels.author)].copy()
posts.to_pickle(root / "train_posts.pkl")
levelled = author_level(posts)
table = features.build(levelled, load_followers(data))
text = table.text.fillna("")
letters = text.map(lambda s: sum(c.isalpha() for c in s))
upper = text.map(lambda s: sum(c.isupper() for c in s))
words = text.str.findall(r"[^\W\d_]+")
table["n_exclamations"] = text.str.count("!")
table["uppercase_ratio"] = upper / letters.clip(lower=1)
table["avg_word_length"] = words.map(lambda v: sum(map(len, v)) / max(1, len(v)))
styles = ["n_exclamations", "uppercase_ratio", "avg_word_length"]
retrieval = author_bm25_features(posts)
extra = ["bm25_history_delta", "bm25_match_count"]
columns = NUMERIC + styles + extra + features.CONTENT_CAT
combined = table[
    ["author", "post", "text"] + NUMERIC + styles + features.CONTENT_CAT
].merge(
    retrieval[["author", "post"] + extra], on=["author", "post"], validate="one_to_one"
)
combined = labels.merge(combined, on=["author", "post"], validate="one_to_one")
fit_authors, valid_authors = train_test_split(
    sorted(labels.author.unique()), test_size=0.2, random_state=42
)
combined["validation"] = combined.author.isin(valid_authors)
raw_age = posts[["author", "post"]].assign(
    raw_age=(posts.crawled_at - posts.posted_at).dt.total_seconds() / 86400
)
combined = combined.merge(raw_age, on=["author", "post"], validate="one_to_one")
age_fill = float(combined.loc[~combined.validation, "raw_age"].median())
combined["age_days"] = combined["raw_age"].fillna(age_fill)
combined = combined.drop(columns="raw_age")
assert combined.validation.sum() == 14548
assert set(combined.loc[combined.validation, "author"]).isdisjoint(
    combined.loc[~combined.validation, "author"]
)
combined.to_pickle(root / "train_table.pkl")
combined[["author", "post", "text"]].to_json(
    root / "encoder_input.jsonl", orient="records", lines=True, force_ascii=False
)
(root / "feature_config.json").write_text(
    json.dumps(
        {
            "numeric": NUMERIC + styles + extra,
            "categorical": features.CONTENT_CAT,
            "columns": columns,
            "seed": 42,
            "age_fill_scope": "labelled fit posts only",
            "age_fill_value": age_fill,
        },
        indent=2,
    )
)
print(
    "Cached only official-train authors:",
    len(posts),
    "history posts;",
    len(combined),
    "labelled posts",
    flush=True,
)
