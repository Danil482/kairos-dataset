"""Lab 1: structured baseline plus word/character TF-IDF."""
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from baseline import model as baseline_model
from kairos import features
from kairos.data import DATA_DIR, load_followers, load_labels, load_posts
from kairos.label import author_level

p = argparse.ArgumentParser()
p.add_argument("--data", default=DATA_DIR)
args = p.parse_args()
labels = load_labels(args.data)
posts = author_level(load_posts(args.data))
data = labels.merge(features.build(posts, load_followers(args.data)),
                    on=["author", "post"], validate="one_to_one")
train = data[data.split == "train"]
test = data[data.split == "test"]
vectors = FeatureUnion([
("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=160000)),
("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=4,
                         max_features=160000)),
])
text_model = Pipeline([
("vectors", vectors),
("classifier", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=60,
                             class_weight="balanced", random_state=42)),
])
structured = baseline_model().fit(train, train.label).predict_proba(test)
text_model.fit(train.text.fillna(""), train.label)
probabilities = .65*structured + .35*text_model.predict_proba(test.text.fillna(""))
output = Path("results/predictions/denis_minakov.csv")
output.parent.mkdir(parents=True, exist_ok=True)
test[["author", "post"]].assign(label=probabilities.argmax(1)).to_csv(
output, index=False)
print(f"wrote {output} ({len(test)} predictions)")
