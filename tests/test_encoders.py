import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kairos_lab.common import identity, mark
from kairos_lab.dataset import prepare
from kairos_lab.encoders import encode_images, encode_text
from kairos_lab.train import train_models, score_final
from kairos_lab.verify import verify
from tests.fixtures import make_release


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "GPU dependencies are tested in the GPU image and local optional environment")
class EncoderTests(unittest.TestCase):
    def test_tiny_local_models_text_chunks_media_and_resume(self):
        import torch
        from PIL import Image
        from transformers import (BertConfig, BertModel, BertTokenizerFast, CLIPConfig,
                                  CLIPImageProcessor, CLIPModel)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, work, models = root / "data", root / "work", root / "models"
            config = make_release(data)
            config["enable_images"] = True
            text = config["text_encoder"]
            text.update(repo="synthetic-bert", revision="fixture", max_length=16, max_chunks=3, batch_size=2)
            image = config["image_encoder"]
            image.update(repo="synthetic-clip", revision="fixture", batch_size=2)
            text_dir = models / "text_encoder/fixture"
            text_dir.mkdir(parents=True)
            vocab = text_dir / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nlearn\ncode\ntoday\njoin\nour\nlaunch\n")
            tokenizer = BertTokenizerFast(vocab_file=str(vocab))
            tokenizer.save_pretrained(text_dir)
            model = BertModel(BertConfig(vocab_size=11, hidden_size=8, num_hidden_layers=1,
                                         num_attention_heads=2, intermediate_size=16))
            model.save_pretrained(text_dir)
            mark(text_dir / "verified.json", identity(text), [p for p in text_dir.iterdir() if p.is_file()])
            image_dir = models / "image_encoder/fixture"
            image_dir.mkdir(parents=True)
            settings = CLIPConfig(text_config={"vocab_size": 11, "hidden_size": 8, "intermediate_size": 16,
                 "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 16},
                 vision_config={"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                 "num_attention_heads": 2, "image_size": 16, "patch_size": 8}, projection_dim=8)
            CLIPModel(settings).save_pretrained(image_dir)
            CLIPImageProcessor(size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}).save_pretrained(image_dir)
            mark(image_dir / "verified.json", identity(image), [p for p in image_dir.iterdir() if p.is_file()])
            path = data / "media/test.png"
            Image.new("RGB", (20, 30), color="red").save(path)
            (data / "media/manifest.jsonl").write_text(json.dumps({"author": "AUTHOR_0000",
                            "post": "POST_0000_0005", "path": "media/test.png"}) + "\n")
            prepare(data, work, config)
            encode_text(models, work, config, "cpu")
            encode_images(data, models, work, config, "cpu")
            train_models(work, config)
            score_final(data, work, config)
            verify(work, config)
            self.assertTrue((work / "models/semantic-0/complete.json").exists())
            self.assertTrue((work / "models/multimodal/complete.json").exists())
            text_values = np.load(work / "cache/text_embeddings.npy")
            image_values = np.load(work / "cache/image_embeddings.npy")
            self.assertEqual(text_values.shape, (288, 8))
            self.assertEqual(image_values.shape, (288, 9))
            self.assertEqual(int((image_values[:, -1] > 0).sum()), 1)
            self.assertTrue(np.isfinite(text_values).all())
            encode_text(models, work, config, "cpu")
            encode_images(data, models, work, config, "cpu")
