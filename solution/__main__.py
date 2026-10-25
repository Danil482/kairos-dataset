"""Portable CLI; no SSH, server paths, model search, or automatic test scoring."""
import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from solution import pipeline as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fetch', 'verify-weights', 'prepare', 'encode', 'features', 'predict', 'train'])
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--work', type=Path, default=Path('.cache/solution'))
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--check-reference', action='store_true')
    parser.add_argument('--deadline', help='Optional ISO timestamp for an externally bounded reproduction job')
    args = parser.parse_args()
    cfg = p.settings()
    deadline = datetime.fromisoformat(args.deadline).timestamp() if args.deadline else float('inf')
    if args.command == 'fetch':
        from kairos_lab.acquire import fetch_models
        args.work.mkdir(parents=True, exist_ok=True)
        fetch_models(args.weights / 'pretrained', args.work, cfg['config'], None)
    elif args.command == 'verify-weights':
        p.verify_weights(args.weights, cfg)
        print('All fixed model and pretrained weight hashes passed')
    elif args.command == 'train':
        p.train(args.data, args.weights, args.work, cfg, args.device, deadline)
    elif args.command == 'predict':
        p.predict(args.data, args.weights, args.work, args.output, cfg, args.device, args.check_reference, deadline)
    else:
        p.configure(args.device, cfg['config']['threads'])
        p.prepare(args.data, args.work, cfg)
        if args.command in ('encode', 'features'):
            p.raw_embeddings(args.data, args.weights, args.work, cfg, args.device)
        if args.command == 'encode':
            p.verify_weights(args.weights, cfg)
            p.adapted_vectors(args.weights, args.work, cfg, args.device, deadline)
        if args.command == 'features':
            p.past_tables(args.data, args.work, cfg)


if __name__ == '__main__':
    main()
