from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DenseClassifier(ClassifierMixin, BaseEstimator):
    normalization = "layer"

    def __init__(self, hidden=256, epochs=40, rate=.001, dropout=.3, decay=.01,
                 balanced=False, seed=42, batch_size=512, normalization="layer"):
        self.hidden = hidden
        self.epochs = epochs
        self.rate = rate
        self.dropout = dropout
        self.decay = decay
        self.balanced = balanced
        self.seed = seed
        self.batch_size = batch_size
        self.normalization = normalization

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.set_num_threads(4)
        torch.manual_seed(self.seed)
        X = np.array(X, dtype="float32", copy=True)
        y = np.array(y, dtype="int64", copy=True)
        self.classes_ = np.unique(y)
        if self.classes_.tolist() != [0, 1, 2] or not np.isfinite(X).all():
            raise ValueError("Dense classifier needs finite features and classes 0, 1, 2")
        self.n_features_in_ = X.shape[1]
        if self.normalization not in {"layer", "batch"}:
            raise ValueError("normalization must be layer or batch")
        if self.normalization == "batch" and self.batch_size < 2:
            raise ValueError("Batch normalization requires batches with at least two rows")
        self.network_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden),
            nn.BatchNorm1d(self.hidden) if self.normalization == "batch" else nn.LayerNorm(self.hidden),
            nn.GELU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden // 2), nn.GELU(), nn.Dropout(self.dropout / 2),
            nn.Linear(self.hidden // 2, 3))
        inputs = torch.from_numpy(X)
        labels = torch.from_numpy(y)
        weights = torch.tensor(len(y) / (3 * np.bincount(y, minlength=3)), dtype=torch.float32) if self.balanced else None
        loss_function = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.rate, weight_decay=self.decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        self.network_.train()
        for epoch in range(self.epochs):
            order = torch.randperm(len(y))
            total = 0.0
            batches = list(order.split(self.batch_size))
            if self.normalization == "batch" and len(batches[-1]) == 1:
                batches[-2] = torch.cat([batches[-2], batches[-1]])
                batches.pop()
            for ids in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(self.network_(inputs[ids]), labels[ids])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(ids)
            schedule.step()
            if epoch % 10 == 9 or epoch == self.epochs - 1:
                print("MLP epoch", epoch + 1, "loss", round(total / len(y), 5), flush=True)
        self.network_.eval().cpu()
        self.loss_ = total / len(y)
        return self

    def predict_proba(self, X):
        import torch
        self.network_.eval()
        batches = []
        with torch.inference_mode():
            for batch in np.array_split(np.asarray(X, dtype="float32"), max(1, (len(X) + 2047) // 2048)):
                batches.append(torch.softmax(self.network_(torch.from_numpy(batch)), dim=1).numpy())
        return np.concatenate(batches)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
