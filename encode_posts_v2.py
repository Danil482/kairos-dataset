"""Optional local frozen text encoder. Requires sentence-transformers.
First: python improved_v2.py --stage prepare
Then: python encode_posts_v2.py
Finally: python improved_v2.py --stage validate --embeddings results/v2_cache/embeddings.npy
No post text is sent to a remote inference service; only public model files download.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, default=Path('results/v2_cache'))
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    source = args.cache/'texts.jsonl'
    # Iterate physical lines: str.splitlines also splits Unicode characters inside JSON strings.
    with source.open() as f:
        rows = [json.loads(line) for line in f]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator='\n')
    writer.writerow(['author', 'post'])
    writer.writerows((r['author'], r['post']) for r in rows)
    digest = hashlib.sha256(buf.getvalue().encode()).hexdigest()
    text_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    chunks = args.cache/f'embedding_chunks_{text_digest[:16]}'
    chunks.mkdir(exist_ok=True)
    model = SentenceTransformer(MODEL, revision=REVISION,
        cache_folder=str(args.cache/'models'), device=args.device)
    model.max_seq_length = 128
    texts = [re.sub(r'⟨[^⟩]+⟩', ' person ', r['text']) for r in rows]
    print('Encoder device:', model.device, flush=True)
    files = []
    start = time.time()
    for i in range(0, len(texts), 2048):
        file = chunks/f'{i:06d}.npy'
        files.append(file)
        if not file.exists():
            e = model.encode(texts[i:i+2048], batch_size=args.batch_size,
                normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            np.save(file, e.astype('float32'))
        print(min(i+2048, len(texts)), '/', len(texts), round(time.time()-start), 's', flush=True)
    emb = np.concatenate([np.load(f) for f in files])
    assert emb.shape == (len(texts), 384) and np.isfinite(emb).all()
    np.save(args.cache/'embeddings.npy', emb)
    config = dict(model=MODEL, revision=REVISION, max_seq_length=128,
        normalize_embeddings=True, rows=len(texts), key_sha256=digest, text_sha256=text_digest)
    (args.cache/'embedding_config.json').write_text(json.dumps(config, indent=2))

if __name__ == '__main__':
    main()
