import tempfile
import unittest
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from contracts import CandidateProposal
from feature_pyramid import GroundingTransformer, RenderingTransformer, SelfTransformer
from lightning_components import PathMNISTLitModule, build_network
from search import candidate_hash, default_candidates, lightning_config_payload, proposal_to_config, proposal_to_representation


class FeaturePyramidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_mixture_against_independent_scalar_reference(self):
        torch.manual_seed(17)
        for operator in (SelfTransformer(8), GroundingTransformer(8)):
            operator.double()
            query = torch.randn(1, 8, 2, 2, dtype=torch.double)
            key = torch.randn(1, 8, 1, 3, dtype=torch.double)
            q, k = query.flatten(2)[0].T, key.flatten(2)[0].T
            gates = torch.softmax(operator.gate.weight @ k.mean(0), 0)
            expected = torch.zeros(4, 3, dtype=torch.double)
            for component in range(operator.components):
                section = slice(component * 8 // operator.components, (component + 1) * 8 // operator.components)
                for i in range(4):
                    scores = torch.stack([-(q[i, section] - k[j, section]).square().sum().sqrt()
                                          if operator.euclidean else (q[i, section] * k[j, section]).sum()
                                          for j in range(3)])
                    expected[i] += gates[component] * scores.softmax(0)
            actual = operator.attention_weights(query, key)[0]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual.sum(-1), torch.ones(4, dtype=torch.double))

    def test_projected_values_and_attention_direction(self):
        for operator in (SelfTransformer(8), GroundingTransformer(8)):
            fine = torch.randn(2, 8, 7, 7)
            source = fine if isinstance(operator, SelfTransformer) else torch.randn(2, 8, 4, 4)
            weights = operator.attention_weights(operator.query(fine), operator.key(source))
            expected = (weights @ operator.value(source).flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(fine)
            actual = operator(fine) if isinstance(operator, SelfTransformer) else operator(fine, source)
            torch.testing.assert_close(actual, expected)

    def test_rendering_matches_convolution_formula_at_all_odd_resolutions(self):
        sizes = (28, 14, 7, 4)
        for src in range(4):
            for dst in range(src + 1, 4):
                operator = RenderingTransformer(8, 2 ** (dst - src))
                fine = torch.randn(1, 8, sizes[src], sizes[src])
                coarse = torch.randn(1, 8, sizes[dst], sizes[dst])
                def conv(layer, x):
                    return F.conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding)
                weighted = conv(operator.query, coarse) * conv(operator.key, fine).mean((-2, -1), keepdim=True)
                expected = conv(operator.refine, conv(operator.query_refine, weighted) +
                                conv(operator.downsample, conv(operator.value, fine)))
                torch.testing.assert_close(operator(coarse, fine), expected)

    def test_four_levels_original_branch_inputs_and_finite_gradients(self):
        model = build_network("feature_pyramid_transformer", in_channels=3, n_classes=9, hidden=8, depth=1, dropout=0)
        pyramid = model.pyramid(torch.randn(2, 3, 28, 28))
        self.assertEqual([x.shape[-1] for x in pyramid], [28, 14, 7, 4])
        observations = []
        handles = [module.register_forward_pre_hook(lambda module, args: observations.append(args))
                   for module in [*model.transformer.grounding.values(), *model.transformer.rendering.values()]]
        outputs = model.transformer(pyramid)
        for handle in handles:
            handle.remove()
        self.assertEqual(len(observations), 12)
        for args in observations:
            self.assertTrue(all(any(tensor is original for original in pyramid) for tensor in args))
        model.head(torch.cat([x.mean((-2, -1)) for x in outputs], 1)).square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_candidate_canonicalization_and_export(self):
        candidate = next(x for x in default_candidates(1) if x.model_family == "feature_pyramid_transformer")
        self.assertEqual((candidate.hidden, candidate.depth, candidate.batch_size, candidate.optimizer), (256, 1, 32, "adamw"))
        changed = CandidateProposal.model_validate({**candidate.model_dump(), "depth": 48, "num_heads": 3,
            "pooling": "cls", "scales": [14, 2], "patch_size": 14, "tokenizer_layers": 5})
        self.assertEqual(candidate_hash(candidate), candidate_hash(changed))
        self.assertEqual(candidate_hash(candidate), candidate_hash(candidate.model_copy(update={"depth": 3})))
        config = proposal_to_config(changed, seed=42, device="cpu", epochs=1, source="test")
        payload = lightning_config_payload(config, proposal_to_representation(changed), data_root=None,
            train_limit=27, val_limit=18, test_limit=18, download=False)
        self.assertEqual(payload["model"]["model_family"], "feature_pyramid_transformer")
        self.assertEqual(payload["model"]["depth"], 1)

    def test_short_lightning_training_checkpoint_and_replay(self):
        torch.manual_seed(42)
        model = PathMNISTLitModule(model_family="feature_pyramid_transformer", hidden=8, depth=4, dropout=0, scheduler="none")
        self.assertEqual(model.hparams.depth, 1)
        inputs = torch.randn(4, 3, 28, 28)
        loader = DataLoader(TensorDataset(inputs, torch.tensor([0, 1, 2, 3])), batch_size=2)
        before = model.network.head[1].weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False,
                enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                num_sanity_val_steps=0, default_root_dir=directory)
            trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
            self.assertFalse(torch.equal(before, model.network.head[1].weight))
            path = Path(directory) / "fpt.ckpt"
            trainer.save_checkpoint(path)
            restored = PathMNISTLitModule.load_from_checkpoint(path, map_location="cpu").eval()
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
                torch.testing.assert_close(restored(inputs), restored(inputs), rtol=0, atol=0)

    def test_invalid_width_and_image_size(self):
        with self.assertRaises(ValueError):
            build_network("feature_pyramid_transformer", in_channels=3, n_classes=9, hidden=10, depth=1, dropout=0)
        model = build_network("feature_pyramid_transformer", in_channels=3, n_classes=9, hidden=8, depth=1, dropout=0)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 32, 32))
