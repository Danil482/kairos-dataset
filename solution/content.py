import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from kairos import features
from kairos.label import author_level
BASE_NUMERIC=features.CONTENT_NUM+features.HISTORY_NUM+features.ARTEFACT_NUM
CATEGORICAL=['content_type','previous_content_type']
_IDENTITY=re.compile(r'⟨(PERSON|ORG|AUTHOR)_\d+⟩')

def text_value(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(text_value(v) for v in value)
    if isinstance(value, dict):
        return text_value(value.get("text", ""))
    return ""

def extra_content(data: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((data / "posts").glob("AUTHOR_*.json")):
        feed = json.loads(path.read_text(encoding="utf-8"))
        for post in feed["posts"]:
            if not isinstance(post.get("author"), dict) or post["author"].get("code") != feed["author"]:
                continue
            article = post.get("article") or {}
            document = post.get("document") or {}
            poll = post.get("poll") or {}
            attached = "\n".join(filter(None, [
                text_value(article.get("title")), text_value(article.get("subtitle")),
                text_value(document.get("title")), text_value(document.get("transcript")),
                text_value(poll.get("title")),
            ]))
            rows.append({"author": feed["author"], "post": post["urn"],
                         "attached_text": attached, "has_document": int(bool(document)),
                         "has_poll": int(bool(poll)), "has_article_title": int(bool(article.get("title")))})
    return pd.DataFrame(rows)

def enrich(posts: pd.DataFrame, followers: pd.Series) -> pd.DataFrame:
    df = features.build(author_level(posts).reset_index(drop=True), followers)
    groups = df["author"]
    # Keep missing values; each estimator must handle them using its fit partition.
    df["age_days"] = (df["crawled_at"] - df["posted_at"]).dt.total_seconds() / 86400
    df["posted_day"] = df["posted_at"].astype("int64") / 86400e9
    df["posted_year"] = df["posted_at"].dt.year
    df["posted_month"] = df["posted_at"].dt.month
    df["crawled_day"] = (df["crawled_at"].astype("int64") / 86400e9).where(df["crawled_at"].notna())

    # Every reaction-derived input is shifted within its author BEFORE aggregation.
    past = df.groupby("author", sort=False)["log_reactions"].shift(1)
    df["past_last_relative"] = past - df["level"]
    df["past_delta"] = past - past.groupby(groups, sort=False).shift(1)
    for window in (3, 5, 10, 20):
        rolled = past.groupby(groups, sort=False).rolling(window, min_periods=1)
        statistics = rolled.agg(["mean", "median", "std", "min", "max"]).droplevel(0).reindex(df.index)
        for name in statistics:
            value = statistics[name]
            df[f"past_{window}_{name}"] = value - df["level"] if name != "std" else value
    for span in (3, 10, 30):
        smooth = past.groupby(groups, sort=False).transform(lambda s: s.ewm(span=span, adjust=False).mean())
        df[f"past_ewm_{span}_relative"] = smooth - df["level"]
    df["past_recent_trend"] = df["past_3_mean"] - df["past_20_mean"]
    past_resid = df.groupby("author", sort=False)["resid"].shift(1)
    for window in (3, 10):
        df[f"past_resid_{window}"] = (
            past_resid.groupby(groups, sort=False).rolling(window, min_periods=1)
            .mean().droplevel(0).reindex(df.index))

    previous_type = df.groupby("author", sort=False)["content_type"].shift(1)
    df["previous_content_type"] = previous_type.fillna("unknown")
    df["same_type_as_previous"] = df["content_type"].eq(previous_type).astype(int)
    df["past_type_frequency"] = df.groupby(["author", "content_type"]).cumcount() / df["hist_n"].clip(lower=1)
    past_length = df.groupby("author", sort=False)["n_chars"].shift(1)
    for window in (5, 20):
        typical_length = (past_length.groupby(groups, sort=False).rolling(window, min_periods=1)
                          .mean().droplevel(0).reindex(df.index))
        df[f"length_vs_past_{window}"] = np.log1p(df["n_chars"]) - np.log1p(typical_length)
    for days in (7, 30, 90):
        counts = np.zeros(len(df), dtype=np.int32)
        for ids in df.groupby("author", sort=False).indices.values():
            times = df.iloc[ids]["posted_at"].astype("int64").to_numpy()
            counts[ids] = np.arange(len(ids)) - np.searchsorted(times, times - int(days * 86400e9), side="left")
        df[f"past_posts_{days}d"] = counts
    text = df["text"]
    length = text.str.len().clip(lower=1)
    df["n_exclamations"] = text.str.count("!")
    df["n_digits"] = text.str.count(r"\d")
    df["latin_fraction"] = text.str.count(r"[A-Za-z]") / length
    df["cyrillic_fraction"] = text.str.count(r"[А-Яа-яЁё]") / length
    df["uppercase_fraction"] = text.str.count(r"[A-ZА-ЯЁ]") / length
    df["n_emoji"] = text.str.count(r"[\U0001F300-\U0001FAFF]")
    return df

def content(data: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((data / 'posts').glob('AUTHOR_*.json')):
        feed = json.loads(path.read_text())
        for post in feed['posts']:
            if not isinstance(post.get('author'), dict) or post['author'].get('code') != feed['author']:
                continue
            shared = post.get('resharedPost') or {}
            entity, celebration = post.get('entity') or {}, post.get('celebration') or {}
            document, poll = post.get('document') or {}, post.get('poll') or {}
            article = shared.get('article') or {}
            texts = [text_value(shared.get('text')), text_value(article.get('title')),
                     text_value(article.get('subtitle')), text_value(entity.get('title')),
                     text_value(entity.get('subtitle')), text_value(celebration.get('text'))]
            options = poll.get('options') or []
            if isinstance(options, list):
                texts.extend(text_value(option.get('text') or option.get('title'))
                             for option in options if isinstance(option, dict))
            delay = np.nan
            if shared.get('postedDateTimestamp') and post.get('postedDateTimestamp'):
                delay = (post['postedDateTimestamp'] - shared['postedDateTimestamp']) / 86400000
            rows.append({'author':feed['author'], 'post':post['urn'],
                'extra_public_text':'\n'.join(filter(None,texts)), 'has_reshared_content':int(bool(shared)),
                'reshared_content_type':shared.get('contentType') or 'none', 'reshared_delay_days':delay,
                'is_brand_partnership':int(bool(post.get('isBrandPartnership'))),
                'document_pages':document.get('totalPageCount') or 0,
                'content_image_count':len(post.get('images') or []),
                'structured_mentions':len(post.get('mentions') or []),
                'company_mentions':len(post.get('companyMentions') or []),
                'poll_option_count':len(options), 'has_entity':int(bool(entity)),
                'shared_text_length':len(text_value(shared.get('text')))})
    return pd.DataFrame(rows)

def normalized_times(posts: pd.DataFrame) -> pd.DataFrame:
    result = posts.copy()
    for name in ('posted_at', 'crawled_at'):
        if name in result:
            result[name] = result[name].dt.as_unit('ns')
    return result

def aligned(additions, table):
    keys = ['author', 'post']
    if additions.duplicated(keys).any():
        raise ValueError('Duplicate feature keys')
    index = pd.MultiIndex.from_frame(table[keys])
    source = additions.set_index(keys)
    if not index.isin(source.index).all():
        raise ValueError('Missing post in revised feature population')
    result = source.reindex(index)
    result.index = table.index
    return result

def cdf_features(posts, cutoffs):
    table = author_level(normalized_times(posts)).reset_index(drop=True)
    names = [f'current_cdf_{window}_{kind}' for window in (5, 20, 0)
             for kind in ('below', 'above', 'zero')]
    values = np.full((len(table), len(names)), np.nan, dtype=np.float32)
    low, high = cutoffs
    for ids in table.groupby('author', sort=False).indices.values():
        rows = table.iloc[ids]
        times = rows.posted_at.astype('int64').to_numpy()
        reactions, levels = rows.log_reactions.to_numpy(), rows.level.to_numpy()
        for current in range(5, len(ids)):
            earlier = np.flatnonzero(times[:current] < times[current])
            for column, window in enumerate((5, 20, 0)):
                previous = earlier[-window:] if window else earlier
                if not len(previous):
                    continue
                history = reactions[previous]
                values[ids[current], column * 3:column * 3 + 3] = [
                    np.mean(history < levels[current] + low),
                    np.mean(history > levels[current] + high), np.mean(history == 0)]
    return pd.concat([table[['author', 'post']], pd.DataFrame(values, columns=names)], axis=1)

def history_indices(posts, length=20):
    indices = np.full((len(posts), length), -1, dtype=np.int32)
    ages = np.zeros((len(posts), length), dtype=np.float32)
    for ids in posts.groupby('author', sort=False).indices.values():
        times = posts.iloc[ids].posted_at.dt.as_unit('ns').astype('int64').to_numpy()
        for i in range(len(ids)):
            end = np.searchsorted(times, times[i], side='left')
            previous = np.arange(max(0, end - length), end)
            if len(previous):
                indices[ids[i], -len(previous):] = ids[previous]
                ages[ids[i], -len(previous):] = (times[i] - times[previous]) / 86400e9
    return indices, ages

def past_semantics(posts: pd.DataFrame, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.full((len(posts), 3), np.nan, dtype=np.float32)
    delta = np.full((len(posts), 64), np.nan, dtype=np.float32)
    projection = np.random.default_rng(42).normal(0, 1 / 8, (384, 64)).astype("float32")
    for ids in posts.groupby("author", sort=False).indices.values():
        current = np.asarray(vectors[ids])
        prefix = np.vstack([np.zeros((1, 384)), np.cumsum(current, axis=0, dtype=np.float64)])
        index = np.arange(len(ids))
        for column, window in enumerate((1, 5, 20)):
            count = np.minimum(index, window)
            mean = (prefix[index] - prefix[np.maximum(0, index - window)]) / np.maximum(count[:, None], 1)
            denominator = np.linalg.norm(mean, axis=1)
            valid = (count > 0) & (denominator > 0)
            value = np.full(len(ids), np.nan, dtype="float32")
            value[valid] = np.einsum("ij,ij->i", current[valid], mean[valid]) / denominator[valid]
            cosine[ids, column] = value
            if window == 5:
                delta[ids] = ((current - mean) @ projection).astype("float32")
                delta[ids[count == 0]] = np.nan
    return cosine, delta

def past_vectors(posts, vectors):
    dimension = vectors.shape[1]
    cosine = np.full((len(posts), 3), np.nan, dtype='float32')
    delta = np.full((len(posts), 64), np.nan, dtype='float32')
    projection = np.random.default_rng(42).normal(0, 1 / 8, (dimension, 64)).astype('float32')
    for ids in posts.groupby('author', sort=False).indices.values():
        current = np.asarray(vectors[ids])
        prefix = np.vstack([np.zeros((1, dimension)), np.cumsum(current, axis=0, dtype='float64')])
        index = np.arange(len(ids))
        for column, window in enumerate([1, 5, 20]):
            count = np.minimum(index, window)
            mean = (prefix[index] - prefix[np.maximum(index - window, 0)]) / np.maximum(count[:, None], 1)
            norm = np.linalg.norm(mean, axis=1)
            valid = (count > 0) & (norm > 0)
            cosine[ids[valid], column] = np.einsum('ij,ij->i', current[valid], mean[valid]) / norm[valid]
            if window == 5:
                delta[ids] = (current - mean) @ projection
                delta[ids[count == 0]] = np.nan
    return cosine, delta

def visual_history(posts: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    df = author_level(posts).reset_index(drop=True)
    names = ["visual_previous_cosine", "visual_past_mean_cosine", "visual_nearest_cosine",
             "visual_nearest_reaction", "visual_weighted_reaction", "visual_previous_count"]
    values = np.full((len(df), len(names)), np.nan, dtype="float32")
    for ids in df.groupby("author", sort=False).indices.values():
        images = np.asarray(vectors[ids])
        present = np.linalg.norm(images, axis=1) > .5
        times = df.iloc[ids].posted_at.astype("int64").to_numpy()
        reactions = df.iloc[ids].log_reactions.to_numpy()
        level = df.iloc[ids].level.to_numpy()
        for i in np.flatnonzero(present):
            earlier = np.flatnonzero(present[:i] & (times[:i] < times[i]))
            if not len(earlier):
                continue
            similarity = images[earlier] @ images[i]
            mean = images[earlier].mean(axis=0)
            weights = np.exp((similarity - similarity.max()) / .1)
            weights /= weights.sum()
            values[ids[i]] = [similarity[-1], images[i] @ mean / max(np.linalg.norm(mean), 1e-8),
                              similarity.max(), reactions[earlier[np.argmax(similarity)]] - level[i],
                              np.average(reactions[earlier], weights=weights) - level[i], len(earlier)]
    return pd.concat([df[["author", "post"]], pd.DataFrame(values, columns=names)], axis=1)


GENRES={
 'portrait':'a professional portrait photograph of a person',
 'team':'a group photograph of a team of people',
 'conference':'a speaker giving a presentation on a conference stage',
 'office':'an office workspace with desks and computers',
 'code':'computer programming source code on a screen',
 'certificate':'an award or a professional achievement certificate',
 'slides':'a business presentation slide with text',
 'chart':'a chart or a statistical data visualization',
 'infographic':'an educational infographic with text and diagrams',
 'quote':'a motivational quotation printed on a background',
 'meme':'a funny meme or a cartoon',
 'interface':'a screenshot of a software application interface',
 'diagram':'a software architecture or technical engineering diagram',
 'book':'a book cover or a stack of books',
 'vacancy':'a job advertisement or a hiring announcement',
 'city':'a city street or a building',
 'nature':'an outdoor landscape with mountains or a beach',
 'food':'food or a restaurant meal',
 'family':'a family photograph',
 'sports':'a sports or fitness activity',
 'graduation':'a university graduation ceremony',
 'celebration':'a party or a celebration',
 'networking':'people networking at a professional event',
 'product':'a product or a piece of electronic hardware',
 'research':'a scientific laboratory or a research experiment',
 'art':'a painting or a work of art',
 'logo':'a company logo or a corporate branding graphic',
 'social_screenshot':'a screenshot of a social media conversation',
 'meeting':'a work meeting or a video conference',
 'workshop':'a classroom or an educational workshop',
}
