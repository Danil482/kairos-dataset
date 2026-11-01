from __future__ import annotations
import numpy as np
import pandas as pd
MIN_HISTORY = 5
QUANTILES = (0.2, 0.8)
BELOW, TYPICAL, ABOVE = (0, 1, 2)
CLASS_NAMES = {BELOW: 'below', TYPICAL: 'typical', ABOVE: 'above'}

def author_level(posts: pd.DataFrame, min_history: int=MIN_HISTORY) -> pd.DataFrame:
    out = posts.sort_values(['author', 'posted_at'], kind='mergesort').copy()
    out['log_reactions'] = np.log1p(out['reactions'].clip(lower=0).astype(float))
    by_author = out.groupby('author')['log_reactions']
    out['level'] = by_author.transform(lambda s: s.shift(1).expanding(min_periods=min_history).median())
    out['hist_n'] = by_author.cumcount()
    out['resid'] = out['log_reactions'] - out['level']
    return out

def fit_cutoffs(train_resid: pd.Series) -> tuple[float, float]:
    low, high = np.quantile(train_resid.dropna(), QUANTILES)
    return (float(low), float(high))

def classify(resid: pd.Series, low: float, high: float) -> pd.Series:
    return pd.Series(np.where(resid < low, BELOW, np.where(resid > high, ABOVE, TYPICAL)), index=resid.index, dtype='int8')

def relabel(posts: pd.DataFrame, train_authors) -> pd.DataFrame:
    levelled = author_level(posts)
    labelled = levelled[levelled['resid'].notna()].copy()
    is_train = labelled['author'].isin(set(train_authors))
    low, high = fit_cutoffs(labelled.loc[is_train, 'resid'])
    labelled['label'] = classify(labelled['resid'], low, high)
    labelled.attrs['cutoffs'] = (low, high)
    return labelled

def main() -> None:
    import argparse
    from kairos.data import DATA_DIR, load_labels, load_posts, load_split
    ap = argparse.ArgumentParser(description='Check labels.csv against the posts.')
    ap.add_argument('--data', default=DATA_DIR, help='unpacked release (default: data/)')
    args = ap.parse_args()
    split, released = (load_split(args.data), load_labels(args.data))
    ours = relabel(load_posts(args.data), split['train'])
    low, high = ours.attrs['cutoffs']
    shipped = split['cutoffs_on_log_residual']
    print(f"cutoffs  low {low:+.6f} (split.json {shipped['low']:+.6f})  high {high:+.6f} (split.json {shipped['high']:+.6f})")
    both = released.merge(ours[['author', 'post', 'label']], on=['author', 'post'], how='outer', suffixes=('_released', '_recomputed'), indicator=True)
    agree = (both['label_released'] == both['label_recomputed']).sum()
    print(f"labels.csv {len(released):,}  recomputed {len(ours):,}  in both {(both['_merge'] == 'both').sum():,}  agree {agree:,}")
    if agree != len(released) or len(ours) != len(released):
        raise SystemExit('recomputed labels differ from labels.csv')
    print('every label reproduces')
if __name__ == '__main__':
    main()
