import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from kairos.text_features import _nfkc, _safe_surface


def encode(frame, config, directory, device):
    keys = pd.MultiIndex.from_frame(frame[['author', 'post']])
    result = np.zeros((len(frame), config['dimension']), np.float32)
    present = np.zeros(len(frame), bool)
    if (directory/'ids.csv').exists() and (directory/'embeddings.npy').exists():
        cached_ids = pd.read_csv(directory/'ids.csv')
        cached_index = pd.MultiIndex.from_frame(cached_ids[['author', 'post']])
        if cached_index.has_duplicates:
            raise ValueError('Duplicate cached embedding keys')
        rows = cached_index.get_indexer(keys)
        cached = np.load(directory/'embeddings.npy', mmap_mode='r')
        for i in np.flatnonzero(rows >= 0):
            if hashlib.sha256(frame.text.iloc[i].encode()).hexdigest() == cached_ids.text_sha256.iloc[rows[i]]:
                result[i] = cached[rows[i]]
                present[i] = True
    present |= frame.text.str.strip().eq('').to_numpy()
    missing = np.flatnonzero(~present)
    if not len(missing):
        return result
    import torch
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'])
    dtype = torch.float16 if device == 'mps' else torch.float32
    model = AutoModel.from_pretrained(config['model'], revision=config['revision'], dtype=dtype, attn_implementation='sdpa').to(device).eval()
    prefix = tokenizer.encode('query: ', add_special_tokens=False)
    capacity = 512-len(prefix)-tokenizer.num_special_tokens_to_add(pair=False)
    for start in range(0, len(missing), 1024):
        locations = missing[start:start+1024]
        texts = [_safe_surface(_nfkc(text))[0] for text in frame.text.iloc[locations]]
        token_lists = tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)['input_ids']
        chunks, owners, weights, buckets = [], [], [], {}
        for owner, tokens in enumerate(token_lists):
            covered = 0
            for offset in range(0, len(tokens), capacity-32):
                end = min(len(tokens), offset+capacity)
                values = [tokenizer.cls_token_id, *prefix, *tokens[offset:end], tokenizer.sep_token_id]
                index = len(chunks)
                chunks.append(values)
                owners.append(owner)
                weights.append(end-covered)
                buckets.setdefault((len(values)+31)//32, []).append(index)
                covered = end
                if end >= len(tokens):
                    break
        vectors = np.zeros((len(chunks), config['dimension']), np.float32)
        with torch.inference_mode():
            for bucket, indices in sorted(buckets.items()):
                batch_size = min(128, max(1, 16384//(bucket*32)))
                for left in range(0, len(indices), batch_size):
                    selected = indices[left:left+batch_size]
                    width = max(len(chunks[i]) for i in selected)
                    ids = torch.full((len(selected), width), tokenizer.pad_token_id, dtype=torch.long)
                    mask = torch.zeros_like(ids)
                    for row, index in enumerate(selected):
                        ids[row, :len(chunks[index])] = torch.as_tensor(chunks[index])
                        mask[row, :len(chunks[index])] = 1
                    mask = mask.to(device)
                    hidden = model(input_ids=ids.to(device), attention_mask=mask).last_hidden_state.float()
                    attention = mask.unsqueeze(-1).float()
                    pooled = (hidden*attention).sum(1)/attention.sum(1).clamp_min(1)
                    vectors[selected] = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
        combined = np.zeros((len(locations), config['dimension']), np.float32)
        totals = np.zeros(len(locations), np.float32)
        for index, owner in enumerate(owners):
            combined[owner] += vectors[index]*weights[index]
            totals[owner] += weights[index]
        combined /= np.maximum(totals[:, None], 1)
        combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12)
        result[locations] = combined
    if not np.isfinite(result).all():
        raise ValueError('Invalid embeddings')
    return result
