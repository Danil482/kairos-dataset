import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key, value in {'TORCH_HOME': '.cache/torch', 'XDG_CACHE_HOME': '.cache', 'TMPDIR': '.tmp'}.items():
    os.environ.setdefault(key, str(ROOT / value))
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.4')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.2')

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kairos.provenance import sha256


class PoolNetwork(nn.Module):
    def __init__(self, metadata_size, hidden, projection, dropout, pooling):
        super().__init__()
        self.pooling = pooling
        self.match = nn.Linear(384, projection, bias=False)
        self.value = nn.Sequential(nn.Linear(384, projection), nn.GELU())
        self.bias = nn.Linear(4, 1, bias=False)
        nn.init.zeros_(self.bias.weight)
        self.head = nn.Sequential(nn.Linear(metadata_size + projection + 4, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout / 2), nn.Linear(hidden // 2, 3))

    def forward(self, metadata, current, past, attributes, mask):
        if self.pooling == 'attention':
            query = nn.functional.normalize(self.match(current), dim=-1)
            keys = nn.functional.normalize(self.match(past), dim=-1)
            scores = (query[:, None] * keys).sum(-1) / .2 + self.bias(attributes).squeeze(-1)
        else:
            scores = torch.zeros_like(mask, dtype=current.dtype)
        weights = torch.softmax(scores.masked_fill(~mask, -1e4), dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        values = torch.cat([self.value(past), attributes], dim=-1)
        context = (weights[:, :, None] * values).sum(dim=1)
        return self.head(torch.cat([metadata, context], dim=1))


class HistoryClassifier:
    def __init__(self, schema, config, seed):
        self.schema, self.config, self.seed = schema, config, seed
        self.signatures = {name: sha256(ROOT / 'work' / name) for name in
                           ['round3-history-posts.pkl', 'round3-history.npz', 'embeddings.npy']}
        self.resources_ = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['resources_'] = None
        return state

    def resources(self, device):
        if self.resources_ is not None and self.resources_['device'] == device:
            return self.resources_
        for name, digest in self.signatures.items():
            if sha256(ROOT / 'work' / name) != digest:
                raise ValueError('History inference resource changed')
        posts = pd.read_pickle(ROOT / 'work/round3-history-posts.pkl')
        history = dict(np.load(ROOT / 'work/round3-history.npz'))
        vectors = np.load(ROOT / 'work/embeddings.npy', mmap_mode='r')
        data = {'device': device, 'lookup': pd.MultiIndex.from_frame(posts[['author', 'post']]),
                'embeddings': torch.from_numpy(np.vstack([np.zeros((1, 384), dtype='float32'), vectors])).to(device),
                'indices': torch.from_numpy(history['indices'].astype('int64') + 1).to(device),
                'ages': torch.from_numpy(history['ages']).to(device),
                'log_reactions': torch.from_numpy(np.r_[np.float32(0), history['log_reactions']]).to(device),
                'level': torch.from_numpy(history['level']).to(device),
                'type': torch.from_numpy(np.r_[np.int32(-1), history['content_type']].astype('int64')).to(device)}
        self.resources_ = data
        return data

    def rows(self, frame, data):
        values = data['lookup'].get_indexer(pd.MultiIndex.from_frame(frame[['author', 'post']]))
        if (values < 0).any():
            raise ValueError('Post missing from fixed history resource')
        return torch.tensor(values, dtype=torch.long, device=data['device'])

    @staticmethod
    def batch(data, rows):
        ids = data['indices'][rows]
        current = data['embeddings'][rows + 1]
        past = data['embeddings'][ids]
        attributes = torch.stack([
            ((data['log_reactions'][ids] - data['level'][rows, None]) / 3).clamp(-4, 4),
            torch.log1p(data['ages'][rows]) / 6,
            (data['type'][ids] == data['type'][rows + 1, None]).to(torch.float32),
            (past * current[:, None]).sum(-1)], dim=-1)
        return current, past, attributes, ids > 0

    def metadata(self, frame):
        from solution.models import model_frame
        return model_frame(frame, self.schema, self.config)

    def fit(self, frame, labels):
        torch.set_num_threads(4)
        torch.manual_seed(self.seed)
        device = self.config.get('device', 'mps' if torch.backends.mps.is_available() else 'cpu')
        features = self.metadata(frame)
        cats = [name for name in features if not pd.api.types.is_numeric_dtype(features[name])]
        numeric = [name for name in features if name not in cats]
        self.processing_ = ColumnTransformer([
            ('num', Pipeline([('fill', SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True)),
                               ('scale', StandardScaler())]), numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), cats)], sparse_threshold=0)
        metadata = torch.from_numpy(np.asarray(self.processing_.fit_transform(features), dtype='float32')).to(device)
        y = np.asarray(labels, dtype='int64')
        self.classes_ = np.unique(y)
        if self.classes_.tolist() != [0, 1, 2] or not torch.isfinite(metadata).all():
            raise ValueError('History model requires finite features and all three classes')
        labels = torch.from_numpy(y.copy()).to(device)
        self.network_ = PoolNetwork(metadata.shape[1], self.config['hidden'], self.config['projection'],
                                    self.config['dropout'], self.config['pooling']).to(device)
        data = self.resources(device)
        positions = self.rows(frame, data)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.config['rate'],
                                       weight_decay=self.config['decay'], foreach=False)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'])
        self.network_.train()
        for epoch in range(self.config['epochs']):
            total = 0.
            order = torch.randperm(len(frame))
            for cpu_ids in order.split(self.config['batch_size']):
                ids = cpu_ids.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.network_(metadata[ids], *self.batch(data, positions[ids]))
                loss = nn.functional.cross_entropy(logits, labels[ids])
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite history training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network_.parameters(), 5, foreach=False)
                optimizer.step()
                total += float(loss.detach()) * len(ids)
            schedule.step()
            print('HISTORY_EPOCH', epoch + 1, round(total / len(frame), 5), flush=True)
        self.loss_ = total / len(frame)
        self.network_.eval().cpu()
        self.resources_ = None
        if device == 'mps':
            torch.mps.empty_cache()
        return self

    def predict_proba(self, frame):
        metadata = torch.from_numpy(np.asarray(self.processing_.transform(self.metadata(frame)), dtype='float32'))
        self.network_.eval()
        data = self.resources('cpu')
        rows = self.rows(frame, data)
        output = []
        with torch.inference_mode():
            for start in range(0, len(frame), 512):
                stop = start + 512
                logits = self.network_(metadata[start:stop], *self.batch(data, rows[start:stop]))
                output.append(torch.softmax(logits, dim=1).numpy())
        return np.concatenate(output)
