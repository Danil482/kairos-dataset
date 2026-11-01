from __future__ import annotations
import hashlib
import re
from collections import Counter, deque
import numpy as np
import pandas as pd
from kairos.text_features import _WORD_RE, _nfkc, _safe_surface
HISTORY_FEATURE_NAMES = ['time_missing', 'time_hour_sin', 'time_hour_cos', 'time_weekday_sin', 'time_weekday_cos', 'time_month_sin', 'time_month_cos', 'time_is_weekend', 'history_has_prior', 'history_n_prior', 'history_n_7d', 'history_n_30d', 'history_days_since_previous', 'history_days_since_first', 'history_template_prior_count', 'history_previous_word_jaccard', 'history_recent_word_jaccard_mean', 'history_recent_word_jaccard_max']
for _window in (5, 20):
    HISTORY_FEATURE_NAMES += [f'history_{_window}_chars_mean', f'history_{_window}_chars_std', f'history_{_window}_words_mean', f'history_{_window}_hashtags_mean', f'history_{_window}_current_length_ratio']

def _describe(text):
    safe, tags, _ = _safe_surface(_nfkc(text))
    tokens = _WORD_RE.findall(re.sub('[⟨<][^⟩>]+[⟩>]', ' ', safe).casefold())
    token_set = set(tokens[:2048])
    canonical = ' '.join(safe.casefold().split())
    digest = hashlib.sha256(canonical.encode('utf-8')).digest() if tokens else None
    return (len(safe), len(tokens), len(tags), token_set, digest)

def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0

def build_text_history(posts: pd.DataFrame) -> pd.DataFrame:
    if not posts.index.is_unique:
        raise ValueError('history input index must be unique')
    columns = ['author', 'post', 'text', 'posted_at']
    data = posts[columns].copy()
    data['posted_at'] = pd.to_datetime(data['posted_at'], utc=True, errors='coerce')
    values_array = np.zeros((len(data), len(HISTORY_FEATURE_NAMES)), dtype=np.float32)
    positions = {index: position for position, index in enumerate(data.index)}
    columns_map = {name: i for i, name in enumerate(HISTORY_FEATURE_NAMES)}
    values_array[data['posted_at'].isna().to_numpy(), columns_map['time_missing']] = 1.0
    for _, author in data.groupby('author', sort=False):
        recent = deque(maxlen=20)
        month_times = deque()
        week_times = deque()
        counts = Counter()
        n_prior, first_time, previous_time = (0, None, None)
        ordered = author.dropna(subset=['posted_at']).sort_values(['posted_at', 'post'], kind='stable')
        for timestamp, simultaneous in ordered.groupby('posted_at', sort=False):
            while month_times and (timestamp - month_times[0]).total_seconds() > 30 * 86400:
                month_times.popleft()
            while week_times and (timestamp - week_times[0]).total_seconds() > 7 * 86400:
                week_times.popleft()
            pending = []
            for row in simultaneous.itertuples():
                chars, words, tags, tokens, digest = _describe(row.text)
                hour = timestamp.hour + timestamp.minute / 60
                values = {'time_hour_sin': np.sin(2 * np.pi * hour / 24), 'time_hour_cos': np.cos(2 * np.pi * hour / 24), 'time_weekday_sin': np.sin(2 * np.pi * timestamp.weekday() / 7), 'time_weekday_cos': np.cos(2 * np.pi * timestamp.weekday() / 7), 'time_month_sin': np.sin(2 * np.pi * (timestamp.month - 1) / 12), 'time_month_cos': np.cos(2 * np.pi * (timestamp.month - 1) / 12), 'time_is_weekend': float(timestamp.weekday() >= 5), 'history_has_prior': float(n_prior > 0), 'history_n_prior': n_prior, 'history_n_7d': len(week_times), 'history_n_30d': len(month_times), 'history_days_since_previous': (timestamp - previous_time).total_seconds() / 86400 if previous_time is not None else 0, 'history_days_since_first': (timestamp - first_time).total_seconds() / 86400 if first_time is not None else 0, 'history_template_prior_count': counts.get(digest, 0) if digest is not None else 0}
                similarities = [_jaccard(tokens, item[3]) for item in list(recent)[-5:]]
                values.update({'history_previous_word_jaccard': similarities[-1] if similarities else 0, 'history_recent_word_jaccard_mean': np.mean(similarities) if similarities else 0, 'history_recent_word_jaccard_max': max(similarities, default=0)})
                for window in (5, 20):
                    context = list(recent)[-window:]
                    mean_chars = np.mean([item[0] for item in context]) if context else 0
                    values.update({f'history_{window}_chars_mean': mean_chars, f'history_{window}_chars_std': np.std([item[0] for item in context]) if context else 0, f'history_{window}_words_mean': np.mean([item[1] for item in context]) if context else 0, f'history_{window}_hashtags_mean': np.mean([item[2] for item in context]) if context else 0, f'history_{window}_current_length_ratio': chars / max(1, mean_chars) if context else 0})
                for name, value in values.items():
                    values_array[positions[row.Index], columns_map[name]] = value
                pending.append((chars, words, tags, tokens, digest))
            for item in pending:
                recent.append(item[:4])
                if item[4] is not None:
                    counts[item[4]] += 1
                month_times.append(timestamp)
                week_times.append(timestamp)
                n_prior += 1
            first_time = first_time if first_time is not None else timestamp
            previous_time = timestamp
    return pd.DataFrame(values_array, index=data.index, columns=HISTORY_FEATURE_NAMES)
