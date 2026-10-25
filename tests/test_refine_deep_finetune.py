"""Reuse the original suite and exercise a 12-layer, three-epoch exact resume."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
from transformers import XLMRobertaConfig, XLMRobertaModel

base_path = Path(__file__).with_name('test_finetune_base.py')
if not base_path.exists():
    base_path = Path(__file__).with_name('test_refine_finetune.py')
spec = importlib.util.spec_from_file_location('deep_finetune_base', base_path)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)


class DeepFineTuneTests(base.FineTuneTests):
    def test_all_twelve_layers_three_epochs_resume_and_frozen_embeddings(self):
        snapshot = self.root / 'twelve-layer-snapshot'
        config = XLMRobertaConfig(vocab_size=32, hidden_size=12, num_hidden_layers=12,
                   num_attention_heads=3, intermediate_size=24, max_position_embeddings=64)
        torch.manual_seed(17)
        original = XLMRobertaModel(config, add_pooling_layer=False)
        original.save_pretrained(snapshot)
        embedding = {k: v.clone() for k, v in original.embeddings.state_dict().items()}
        initial_layers = [{k: v.clone() for k, v in layer.state_dict().items()} for layer in original.encoder.layer]
        rng = np.random.default_rng(21)
        ids = rng.integers(5, 30, (24, 12), dtype=np.int32); masks = np.ones_like(ids, dtype=np.int8)
        numeric = rng.normal(size=(24, 5)).astype('float32'); labels = np.tile(np.arange(3), 8)
        trial = base.small_spec(); trial.update(last_layers=12, epochs=3)
        args = (ids, masks, numeric, labels, snapshot, trial)
        a, pa = base.ft.fit_encoder(*args, self.root/'deep-a', self.root, 'deep12', 'cpu', 42, float('inf'), base.utils)
        base.ft.fit_encoder(*args, self.root/'deep-b', self.root, 'deep12', 'cpu', 42, float('inf'), base.utils, until_step=7)
        b, pb = base.ft.fit_encoder(*args, self.root/'deep-b', self.root, 'deep12', 'cpu', 42, float('inf'), base.utils)
        np.testing.assert_array_equal(base.ft.predict_encoder(a, pa, ids, masks, numeric, 'cpu', trial, base.utils),
                                      base.ft.predict_encoder(b, pb, ids, masks, numeric, 'cpu', trial, base.utils))
        for key, value in embedding.items():
            torch.testing.assert_close(value, b.backbone.embeddings.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in b.backbone.embeddings.parameters()))
        self.assertEqual(len(b.backbone.encoder.layer), 12)
        for old, layer in zip(initial_layers, b.backbone.encoder.layer):
            self.assertTrue(all(p.requires_grad for p in layer.parameters()))
            self.assertTrue(any(not torch.equal(value, layer.state_dict()[key]) for key, value in old.items()))
        self.assertLessEqual(len(list((self.root/'deep-b').glob('optimizer-step-*.pt'))), 2)
        self.assertEqual(len(list((self.root/'deep-b').glob('epoch-end-*.pt'))), 3)


if __name__ == '__main__':
    unittest.main()
