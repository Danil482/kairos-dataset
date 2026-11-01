from __future__ import annotations
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from kairos.data import DATA_DIR, load_labels
from kairos.label import CLASS_NAMES
from kairos.metrics import score

def main() -> None:
    ap = argparse.ArgumentParser(description='Score predictions against the labels.')
    ap.add_argument('--predictions', required=True, help='CSV: author, post, label')
    ap.add_argument('--split', default='test', choices=('test', 'train'))
    ap.add_argument('--data', default=DATA_DIR, help='unpacked release (default: data/)')
    args = ap.parse_args()
    truth = load_labels(args.data)
    truth = truth[truth['split'] == args.split]
    pred = pd.read_csv(args.predictions)
    if 'label' not in pred and 'class' in pred:
        pred['label'] = pred['class'].map({name: k for k, name in CLASS_NAMES.items()})
    if not {'author', 'post', 'label'} <= set(pred.columns):
        raise SystemExit('the CSV needs `author`, `post` and `label` (or `class`) columns')
    if pred.duplicated(['author', 'post']).any():
        raise SystemExit('some posts are predicted twice')
    if not pred['label'].isin(CLASS_NAMES).all():
        raise SystemExit('labels must be 0, 1, 2 or below, typical, above')
    both = truth.merge(pred[['author', 'post', 'label']], on=['author', 'post'], how='left', suffixes=('', '_pred'))
    missing = int(both['label_pred'].isna().sum())
    if missing:
        raise SystemExit(f'{missing:,} of {len(truth):,} {args.split} posts have no prediction; every one needs a label')
    extra = len(pred) - len(truth)
    if extra:
        print(f'note: {extra:,} predicted posts are not in the {args.split} split and are ignored')
    y_true, y_pred = (both['label'], both['label_pred'].astype(int))
    m = score(y_true, y_pred)
    print(f'{args.split}: {len(both):,} posts')
    print(f"  macro-F1  {m['macro_f1']:.4f}   (primary)")
    print(f"  QWK       {m['qwk']:.4f}")
    print(f"  accuracy  {m['accuracy']:.4f}")
    per_class = f1_score(y_true, y_pred, average=None, labels=list(CLASS_NAMES))
    print('  F1 by class  ' + '  '.join((f'{CLASS_NAMES[k]} {v:.4f}' for k, v in zip(CLASS_NAMES, per_class))))
    print('  confusion, rows = truth below / typical / above:')
    for row in confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES)):
        print('    ' + ' '.join((f'{v:7,d}' for v in row)))
if __name__ == '__main__':
    main()
