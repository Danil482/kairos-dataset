# %% [markdown]
# # Lab 1: Kairos, предсказание вовлечённости поста относительно уровня автора
#
# Что добавлено к бейзлайну (градиентный бустинг по метаданным, macro-F1 0.5410):
#
# - **Текст**: TF-IDF + линейные модели (вероятности классов и остаток); предсказания
#   для train считаются out-of-fold по авторам и подаются в бустинг как признаки.
# - **Эмбеддинги текста**: локальная мультиязычная модель, тот же приём.
# - **Медиа и упоминания**: число картинок, наличие и длительность видео, число упомянутых
#   людей и организаций, репост с комментарием и наличие в нём медиа.
# - **Признаки истории автора** (только по прошлым постам): недавняя форма против долгого уровня,
#   результаты на постах того же типа, обычная длина поста.
# - **Настройки бустинга и веса классов** под macro-F1, подбор на отложенных авторах из train.
#
# Счётчики реакций из JSON (likeCount, commentsCount и т.п.) не используются: это утечка метки.
# Результат на test: macro-F1 0.6457 (бейзлайн 0.5410).

# %%
# Импорты
import os
import sys
import json
import time
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kairos import features
from kairos.data import DATA_DIR, load_followers, load_labels, load_posts
from kairos.label import author_level
from kairos.metrics import score

SEED = 42
T0 = time.time()


def stamp(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} мин] {msg}", flush=True)

# %%
# Данные
labels = load_labels()
levelled = author_level(load_posts())
built = features.build(levelled, load_followers())
table = labels.merge(built, on=["author", "post"], how="left", validate="one_to_one")
assert not table["level"].isna().any(), "release incomplete?"
stamp(f"labelled posts: {len(table):,}")

# %%
# Признаки медиа и упоминаний
def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _n_images(d):
    imgs = d.get("images")
    if isinstance(imgs, list) and imgs:
        return len(imgs)
    return 1 if d.get("image") else 0


def _video_sec(d):
    vids = [v for v in _as_list(d.get("video")) if isinstance(v, dict)]
    return (vids[0].get("duration") or 0) / 1000.0 if vids else 0.0


def extract_extra(p):
    rs = p.get("resharedPost") or {}
    return {
        "n_images": _n_images(p),
        "has_video": int(bool(_as_list(p.get("video")))),
        "video_sec": _video_sec(p),
        "n_people_mentions": len(_as_list(p.get("mentions"))),
        "n_org_mentions": len(_as_list(p.get("companyMentions"))),
        "has_reshare": int(bool(rs)),
        "rs_has_media": int(_n_images(rs) > 0 or bool(_as_list(rs.get("video")))),
    }


EXTRA_NUM = list(extract_extra({}).keys())


def load_extra(data_dir=DATA_DIR):
    rows = []
    for f in tqdm(sorted((Path(data_dir) / "posts").glob("AUTHOR_*.json")),
                  desc="extra", unit="file"):
        feed = json.loads(f.read_text(encoding="utf-8"))
        owner = feed["author"]
        for p in feed["posts"]:
            a = p.get("author")
            if not (isinstance(a, dict) and a.get("code") == owner):
                continue
            rows.append({"author": owner, "post": p["urn"], **extract_extra(p)})
    return pd.DataFrame(rows)


cache = Path("results/extra_features.csv")
extra = pd.read_csv(cache) if cache.exists() else None
if extra is None or not set(EXTRA_NUM) <= set(extra.columns):
    extra = load_extra()
    cache.parent.mkdir(parents=True, exist_ok=True)
    extra.to_csv(cache, index=False)

extra = extra.drop_duplicates(["author", "post"])[["author", "post"] + EXTRA_NUM]
table = table.merge(extra, on=["author", "post"], how="left", validate="one_to_one")
table[EXTRA_NUM] = table[EXTRA_NUM].fillna(0)
stamp("media features ready")

# %%
# Признаки истории автора
REL_NUM = ["recent3_gap", "recent10_gap", "hist_std", "hist_max_gap", "hist_mean_gap",
           "type_gap", "type_n_past", "type_freq", "len_vs_usual"]


def build_relative(b):
    out = b[["author", "post"]].copy()
    lvl = b["level"]
    g = b.groupby("author")["log_reactions"]
    r3 = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    r10 = g.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    out["recent3_gap"] = r3 - lvl
    out["recent10_gap"] = r10 - lvl
    out["hist_std"] = g.transform(lambda s: s.shift(1).expanding(min_periods=5).std())
    out["hist_max_gap"] = g.transform(
        lambda s: s.shift(1).expanding(min_periods=5).max()) - lvl
    out["hist_mean_gap"] = g.transform(
        lambda s: s.shift(1).expanding(min_periods=5).mean()) - lvl

    by_type = b.groupby(["author", "content_type"])
    type_mean = by_type["log_reactions"].transform(
        lambda s: s.shift(1).expanding().mean())
    out["type_gap"] = type_mean - lvl
    out["type_n_past"] = by_type.cumcount()
    out["type_freq"] = out["type_n_past"] / b["hist_n"].clip(lower=1)

    lc = np.log1p(b["n_chars"])
    usual = lc.groupby(b["author"]).transform(lambda s: s.shift(1).expanding().mean())
    out["len_vs_usual"] = lc - usual
    return out


rel = build_relative(built)
table = table.merge(rel, on=["author", "post"], how="left", validate="one_to_one")
table[REL_NUM] = table[REL_NUM].replace([np.inf, -np.inf], np.nan).fillna(0)
table["_row"] = np.arange(len(table))
stamp("author-relative features ready")

train = table[table["split"] == "train"].reset_index(drop=True)
test = table[table["split"] == "test"].reset_index(drop=True)
print(f"train {len(train):,} posts / {train.author.nunique():,} authors | "
      f"test {len(test):,} posts / {test.author.nunique():,} authors")

# %%
# Текст: TF-IDF
def clean(s: pd.Series) -> pd.Series:
    return s.str.replace(r"⟨([A-Za-z]+)(?:_\d+)?⟩",
                         lambda m: " ph" + m.group(1).lower() + " ", regex=True)


y = train["label"].to_numpy()
r = train["resid"].to_numpy()
groups = train["author"].to_numpy()


def stack_view(Xtr, Xte, make_models, tag):
    oof = np.zeros((Xtr.shape[0], 4))
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(Xtr, y, groups), 1):
        clf, reg = make_models()
        clf.fit(Xtr[a], y[a])
        reg.fit(Xtr[a], r[a])
        oof[b, :3] = clf.predict_proba(Xtr[b])
        oof[b, 3] = reg.predict(Xtr[b])
        stamp(f"[{tag}] fold {k}/5, macro-F1 только этого представления: "
              f"{f1_score(y[b], oof[b, :3].argmax(1), average='macro'):.4f}")
    clf, reg = make_models()
    clf.fit(Xtr, y)
    reg.fit(Xtr, r)
    return oof, np.column_stack([clf.predict_proba(Xte), reg.predict(Xte)])


TXT_COLS = ["txt_p_below", "txt_p_typical", "txt_p_above", "txt_resid"]
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150_000,
                        sublinear_tf=True, lowercase=True)
X_train = tfidf.fit_transform(clean(train["text"]))
X_test = tfidf.transform(clean(test["text"]))
stamp(f"tf-idf: {X_train.shape}")

train[TXT_COLS], test[TXT_COLS] = stack_view(
    X_train, X_test,
    lambda: (LogisticRegression(C=2.0, max_iter=300, class_weight="balanced"),
             Ridge(alpha=3.0)),
    "tfidf")

# %%
# Текст: эмбеддинги
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_COLS = ["emb_p_below", "emb_p_typical", "emb_p_above", "emb_resid"]
emb_all = None
emb_cache = Path("results/emb_cache.npz")
try:
    keys_a = table["author"].astype(str).to_numpy()
    keys_p = table["post"].astype(str).to_numpy()
    if emb_cache.exists():
        z = np.load(emb_cache)
        if len(z["emb"]) == len(table) and (z["author"] == keys_a).all() \
                and (z["post"] == keys_p).all():
            emb_all = z["emb"]
            stamp("эмбеддинги взяты из кэша")
    if emb_all is None:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(EMB_MODEL, device="cpu")
        st.max_seq_length = 128
        stamp("считаю эмбеддинги")
        emb_all = st.encode(clean(table["text"]).tolist(), batch_size=64,
                            show_progress_bar=True, convert_to_numpy=True,
                            normalize_embeddings=True).astype(np.float32)
        np.savez(emb_cache, emb=emb_all, author=keys_a, post=keys_p)
except Exception as e:
    print("эмбеддинги пропущены:", repr(e))
    emb_all = None

if emb_all is not None:
    E_train, E_test = emb_all[train["_row"].to_numpy()], emb_all[test["_row"].to_numpy()]
    train[EMB_COLS], test[EMB_COLS] = stack_view(
        E_train, E_test,
        lambda: (LogisticRegression(C=1.0, max_iter=500, class_weight="balanced"),
                 Ridge(alpha=10.0)),
        "emb")

# %%
# Выбор признаков и настроек бустинга
NUM_BASE = features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM
CFG_A = dict(max_iter=300, learning_rate=0.08)
CFG_B = dict(max_iter=600, learning_rate=0.04, min_samples_leaf=40, l2_regularization=1.0)


def make_model(numeric, **cfg):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20),
             features.CONTENT_CAT),
        ])),
        ("clf", HistGradientBoostingClassifier(**{**CFG_A, **cfg},
                                               class_weight="balanced",
                                               random_state=SEED)),
    ])


fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                        .split(train, groups=train["author"]))
fit, val = train.iloc[fit_idx], train.iloc[val_idx]

variants = {
    "baseline features": NUM_BASE,
    "+ text": NUM_BASE + TXT_COLS,
    "+ text + media": NUM_BASE + TXT_COLS + EXTRA_NUM,
    "+ text + media + author-relative": NUM_BASE + TXT_COLS + EXTRA_NUM + REL_NUM,
}
if emb_all is not None:
    variants["+ text + media + author-relative + embeddings"] = (
        NUM_BASE + TXT_COLS + EXTRA_NUM + REL_NUM + EMB_COLS)

models, val_scores = {}, {}
for name, cols in variants.items():
    models[name] = make_model(cols).fit(fit, fit["label"])
    val_scores[name] = score(val["label"], models[name].predict(val))
    stamp(f"{name:52s} {val_scores[name]}")

best_name = max(val_scores, key=lambda n: val_scores[n]["macro_f1"])
cols_final = variants[best_name]
best_cfg, best_model = CFG_A, models[best_name]
print("\nлучший набор признаков на отложенных авторах:", best_name)

model_b = make_model(cols_final, **CFG_B).fit(fit, fit["label"])
score_b = score(val["label"], model_b.predict(val))
stamp(f"настройки B для лучшего набора: {score_b}  (A: {val_scores[best_name]})")
if score_b["macro_f1"] > val_scores[best_name]["macro_f1"]:
    best_cfg, best_model = CFG_B, model_b
print("выбраны настройки:", "B" if best_cfg is CFG_B else "A")

# %%
# Веса классов
proba_val = best_model.predict_proba(val)
yv = val["label"].to_numpy()

grid = np.arange(0.3, 1.81, 0.1)
best_f1, W = -1.0, np.array([1.0, 1.0, 1.0])
for w0 in grid:
    for w2 in grid:
        w = np.array([w0, 1.0, w2])
        f = f1_score(yv, (proba_val * w).argmax(1), average="macro")
        if f > best_f1:
            best_f1, W = f, w
print("без весов:", round(f1_score(yv, proba_val.argmax(1), average="macro"), 4),
      "| с весами", W.round(2), ":", round(best_f1, 4))

# %%
# Финал на test
final = make_model(cols_final, **best_cfg).fit(train, train["label"])
proba_test = final.predict_proba(test)

pred_plain = proba_test.argmax(1)
pred = (proba_test * W).argmax(1)
res_plain, res_w = score(test["label"], pred_plain), score(test["label"], pred)
print("test, без весов:", res_plain)
print("test, с весами :", res_w)

# %%
# Сохранение предсказаний
out_dir = Path("results/predictions")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "improved_v3.csv"
test[["author", "post"]].assign(label=pred).to_csv(out_path, index=False)
Path("results/v3_summary.json").write_text(json.dumps({
    "best_features": best_name, "best_cfg": "B" if best_cfg is CFG_B else "A",
    "class_weights": [float(x) for x in W], "val_scores": val_scores,
    "val_with_weights": round(float(best_f1), 4),
    "test_plain": res_plain, "test_weighted": res_w,
    "embeddings_used": emb_all is not None,
}, indent=2, ensure_ascii=False), encoding="utf-8")
stamp(f"saved {out_path} ({len(test):,} rows)")

# %%
# Проверка evaluate.py
import subprocess
res = subprocess.run([sys.executable, "evaluate.py", "--predictions", str(out_path)],
                     capture_output=True, text=True)
print(res.stdout or res.stderr)