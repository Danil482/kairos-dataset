import numpy as np
import pandas as pd

def build_causal_history(posts, cutoffs):
    result = []
    for author, group in posts.groupby('author', sort=False):
        prior, residuals, times, kinds, previous_labels = ([], [], [], [], [])
        for stamp, tied in group.dropna(subset=['posted_at']).sort_values('posted_at', kind='stable').groupby('posted_at', sort=True):
            a = np.asarray(prior, float)
            level = float(np.median(a)) if len(a) else 0.0
            r = np.asarray(residuals, float)
            labels = np.asarray(previous_labels, int)
            days = np.asarray([(stamp - v).total_seconds() / 86400 for v in times])
            for row in tied.itertuples():
                f = {'author': author, 'post': row.post}
                for window in (3, 5, 10, 20, 50):
                    tail = r[-window:]
                    usable = tail[np.isfinite(tail)]
                    past_labels = labels[-window:]
                    past_labels = past_labels[past_labels >= 0]
                    probs = (np.bincount(past_labels, minlength=3) + 2) / (len(past_labels) + 6)
                    f.update({f'past_class_{window}_{i}': float(probs[i]) for i in range(3)})
                    f[f'past_residual_mean_{window}'] = float(usable.mean()) if len(usable) else 0.0
                    f[f'past_residual_median_{window}'] = float(np.median(usable)) if len(usable) else 0.0
                    f[f'past_residual_std_{window}'] = float(usable.std()) if len(usable) else 0.0
                for horizon in (7, 30, 90, 365):
                    mask = days <= horizon
                    recent = a[mask]
                    lab = labels[mask]
                    lab = lab[lab >= 0]
                    probs = (np.bincount(lab, minlength=3) + 2) / (len(lab) + 6)
                    f[f'calendar_count_{horizon}'] = float(np.log1p(mask.sum()))
                    f[f'calendar_mean_delta_{horizon}'] = float((recent.sum() + 5 * level) / (len(recent) + 5) - level)
                    f.update({f'calendar_class_{horizon}_{i}': float(probs[i]) for i in range(3)})
                same = np.asarray([kind == row.content_type for kind in kinds])
                same_labels = labels[same] if len(same) else np.asarray([], int)
                same_labels = same_labels[same_labels >= 0]
                probs = (np.bincount(same_labels, minlength=3) + 3) / (len(same_labels) + 9)
                f.update({f'past_same_type_class_{i}': float(probs[i]) for i in range(3)})
                for i in range(3):
                    run = 0
                    for value in reversed(previous_labels):
                        if value != i:
                            break
                        run += 1
                    f[f'past_class_run_{i}'] = float(run)
                f['below_feasible'] = float(level + cutoffs['low'] > 0)
                f['level_to_below_threshold'] = level + cutoffs['low']
                f['minimum_above_reactions_log'] = level + cutoffs['high']
                for n in (5, 20):
                    count = min(n, len(a))
                    if count >= 3 and np.ptp(days[-count:]) > 0.01:
                        x = -days[-count:]
                        x = x - x.mean()
                        slope = float(np.dot(x, a[-count:] - a[-count:].mean()) / (np.dot(x, x) + 1))
                        f[f'calendar_slope_{n}'] = slope
                        f[f'calendar_extrapolated_delta_{n}'] = float(a[-count:].mean() + slope * days[-count:].mean() - level)
                    else:
                        f[f'calendar_slope_{n}'] = 0.0
                        f[f'calendar_extrapolated_delta_{n}'] = 0.0
                result.append(f)
            for row in tied.itertuples():
                value = float(np.log1p(max(0, row.reactions)))
                residual = value - level if len(prior) >= 5 else np.nan
                label = -1 if not np.isfinite(residual) else 0 if residual < cutoffs['low'] else 2 if residual > cutoffs['high'] else 1
                prior.append(value)
                residuals.append(residual)
                previous_labels.append(label)
                times.append(stamp)
                kinds.append(row.content_type)
    table = pd.DataFrame(result).set_index(['author', 'post'])
    if table.index.has_duplicates or not np.isfinite(table.to_numpy()).all():
        raise ValueError('Invalid causal history')
    return table
