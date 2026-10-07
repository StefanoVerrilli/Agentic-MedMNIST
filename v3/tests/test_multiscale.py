from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

import torch

from contracts import CandidateProposal
from lightning_components import PathMNISTLitModule, build_network
from search import candidate_hash, default_candidates, lightning_config_payload, proposal_to_config, proposal_to_representation


class MultiScaleTests(unittest.TestCase):
    def test_all_scale_pairs_pooling_and_positions_have_finite_gradients(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for scales in itertools.combinations((2, 4, 7, 14), 2):
                for pooling in ("mean", "cls", "attention"):
                    for position in ("learned", "sinusoidal"):
                        with self.subTest(scales=scales, pooling=pooling, position=position):
                            model = build_network("multi_scale_transformer", in_channels=3, n_classes=9,
                                hidden=16, depth=1, dropout=0.0, scales=scales, num_heads=4,
                                pooling=pooling, positional_encoding=position)
                            outputs = model(torch.randn(2, 3, 28, 28))
                            self.assertEqual(outputs.shape, (2, 9))
                            outputs.square().mean().backward()
                            for tokenizer in model.tokenizers:
                                self.assertTrue(torch.isfinite(tokenizer.weight.grad).all())
                            self.assertTrue(torch.isfinite(model.scale_embedding.grad).all())
        finally:
            torch.set_num_threads(previous)

    def test_contract_rejects_duplicate_invalid_scales_and_bad_heads(self):
        payload = default_candidates(1)[0].model_dump()
        for update in ({"scales": [2]}, {"scales": [2, 2]}, {"scales": [3, 7]},
                       {"scales": [2, "7"]}, {"scales": None},
                       {"num_heads": 3}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                CandidateProposal.model_validate({**payload, **update})

    def test_canonical_hash_ignores_scale_order_and_inactive_knobs(self):
        payload = default_candidates(1)[0].model_dump()
        first = CandidateProposal.model_validate({**payload, "scales": [7, 2, 4], "patch_size": 14})
        second = CandidateProposal.model_validate({**payload, "scales": [2, 4, 7], "tokenizer_layers": 5})
        self.assertEqual(candidate_hash(first), candidate_hash(second))
        cnn = CandidateProposal.model_validate({**payload, "model_family": "tiny_cnn", "scales": [2, 14]})
        self.assertEqual(cnn.scales, (2, 4, 7))

    def test_export_and_checkpoint_reload_preserve_every_scale(self):
        candidate = default_candidates(1)[0].model_copy(update={"scales": (4, 7, 14), "hidden": 16, "depth": 1})
        config = proposal_to_config(candidate, seed=42, device="cpu", epochs=1, source="test")
        payload = lightning_config_payload(config, proposal_to_representation(candidate),
            data_root=None, train_limit=27, val_limit=18, test_limit=18, download=False)
        self.assertEqual(payload["model"]["scales"], (4, 7, 14))
        model = PathMNISTLitModule(**payload["model"]).eval()
        inputs = torch.randn(2, 3, 28, 28)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            import lightning.pytorch as pl
            torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                        "pytorch-lightning_version": pl.__version__}, path)
            restored = PathMNISTLitModule.load_from_checkpoint(path, map_location="cpu").eval()
            with torch.no_grad():
                torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)

    def test_wrong_image_shape_is_rejected(self):
        model = build_network("multi_scale_transformer", in_channels=3, n_classes=9,
                              hidden=16, depth=1, dropout=0, scales=(4, 7))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 32, 32))
