import numpy as np
import pandas as pd

def engagement_history(posts):
    required = {'author', 'post', 'posted_at', 'reactions', 'content_type', 'text'}
    if not required.issubset(posts.columns):
        raise ValueError('Missing source columns')
    if posts[['author', 'post']].duplicated().any():
        raise ValueError('Duplicate post keys')
    rows = []
    for author, group in posts.groupby('author', sort=False):
        prior = []
        kinds = []
        times = []
        lengths = []
        for stamp, tied in group.dropna(subset=['posted_at']).sort_values('posted_at').groupby('posted_at', sort=True):
            logs = np.asarray(prior, dtype=float)
            level = float(np.median(logs)) if len(logs) else 0.0
            for row in tied.itertuples():
                values = {'author': author, 'post': row.post, 'snapshot_prior_count': len(logs)}
                for window in (3, 5, 10, 20):
                    recent = logs[-window:]
                    prefix = f'snapshot_prior{window}'
                    if len(recent):
                        q10, q25, q50, q75, q90 = np.quantile(recent, [0.1, 0.25, 0.5, 0.75, 0.9])
                        stats = [recent.mean(), recent.std(), q10, q25, q50, q75, q90, recent[-1], q50 - level, recent.mean() - level, float((recent < level).mean()), float((recent > level).mean())]
                    else:
                        stats = [0.0] * 12
                    names = ['mean', 'std', 'q10', 'q25', 'median', 'q75', 'q90', 'last', 'median_delta_level', 'mean_delta_level', 'fraction_under_level', 'fraction_over_level']
                    values.update({prefix + '_' + name: float(value) for name, value in zip(names, stats)})
                if len(logs):
                    n = min(20, len(logs))
                    latest = logs[-n:]
                    x = np.arange(n, dtype=float)
                    slope = float(np.polyfit(x, latest, 1)[0]) if n >= 2 else 0.0
                    ema = float(logs[0])
                    for v in logs[1:]:
                        ema = 0.2 * float(v) + 0.8 * ema
                    same = np.asarray([v for v, k in zip(prior, kinds) if k == row.content_type], dtype=float)
                    shrink = (same.sum() + 5 * logs.mean()) / (len(same) + 5)
                    length = np.log1p(len(row.text or ''))
                    near = np.abs(np.asarray(lengths) - length) <= 0.5
                    nearvals = logs[near]
                    length_mean = (nearvals.sum() + 5 * logs.mean()) / (len(nearvals) + 5)
                    values.update(snapshot_prior_slope20=slope, snapshot_prior_ema_delta=ema - level, snapshot_prior_type_count=len(same), snapshot_prior_type_mean_delta=shrink - level, snapshot_prior_type_std=float(same.std()) if len(same) > 1 else 0.0, snapshot_prior_similar_length_count=len(nearvals), snapshot_prior_similar_length_delta=length_mean - level, snapshot_prior_zero_fraction=float((logs == 0).mean()), snapshot_prior_recent_span_days=float((stamp - times[-n]).total_seconds() / 86400), snapshot_prior_last_gap_days=float((stamp - times[-1]).total_seconds() / 86400))
                else:
                    for name in ('slope20', 'ema_delta', 'type_count', 'type_mean_delta', 'type_std', 'similar_length_count', 'similar_length_delta', 'zero_fraction', 'recent_span_days', 'last_gap_days'):
                        values['snapshot_prior_' + name] = 0.0
                rows.append(values)
            for row in tied.itertuples():
                prior.append(float(np.log1p(max(0, row.reactions))))
                kinds.append(row.content_type)
                times.append(stamp)
                lengths.append(np.log1p(len(row.text or '')))
    return pd.DataFrame(rows)
