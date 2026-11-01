from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

def _release(data_dir: Path | str) -> Path:
    data_dir = Path(data_dir)
    if not (data_dir / 'posts').is_dir() or not (data_dir / 'labels.csv').is_file():
        raise SystemExit(f'No release in {data_dir}. Request the password and download the core archive at https://kairos-labs.ai/dataset/, then unpack it there:\n    7z x lab1_core.7z.001 -o{data_dir}')
    return data_dir

def load_posts(data_dir: Path | str=DATA_DIR, own_only: bool=True) -> pd.DataFrame:
    data_dir = _release(data_dir)
    rows = []
    for f in tqdm(sorted((data_dir / 'posts').glob('AUTHOR_*.json')), desc='posts', unit='file'):
        feed = json.loads(f.read_text(encoding='utf-8'))
        owner = feed['author']
        for p in feed['posts']:
            a = p.get('author')
            own = isinstance(a, dict) and a.get('code') == owner
            if own_only and (not own):
                continue
            rows.append({'author': owner, 'post': p['urn'], 'own': own, 'posted_at': p.get('postedDateTimestamp'), 'crawled_at': feed.get('crawledAt'), 'reactions': int(p.get('totalReactionCount') or 0), 'text': p.get('text') or '', 'content_type': p.get('contentType') or 'unknown'})
    if not rows:
        raise ValueError('release has no own posts in posts/AUTHOR_*.json')
    df = pd.DataFrame(rows)
    df['posted_at'] = pd.to_datetime(df['posted_at'], unit='ms', utc=True, errors='coerce')
    df['crawled_at'] = pd.to_datetime(df['crawled_at'], utc=True, errors='coerce')
    return df

def load_followers(data_dir: Path | str=DATA_DIR) -> pd.Series:
    data_dir = _release(data_dir)
    counts = {}
    for f in sorted((data_dir / 'authors').glob('AUTHOR_*.json')):
        profile = json.loads(f.read_text(encoding='utf-8'))
        counts[profile.get('author', f.stem)] = profile.get('followers')
    return pd.to_numeric(pd.Series(counts, name='followers'), errors='coerce')

def load_labels(data_dir: Path | str=DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(_release(data_dir) / 'labels.csv')

def load_split(data_dir: Path | str=DATA_DIR) -> dict:
    return json.loads((_release(data_dir) / 'split.json').read_text(encoding='utf-8'))
