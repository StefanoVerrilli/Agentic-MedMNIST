"""Opt-in real SSH/container checks; no local generated-code fallback."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from contracts import Blackboard, GeneratedCodeDecision, GeneratedSource, WorkerConfiguration
from generated_agents import generated_config
from ml import prepare_data, train_model
from contracts import RepresentationDecision
from remote import SSHWorker, archive_bundle
from tests.helpers import make_bundle

PROBE = '''
import os
import socket
from pathlib import Path
torch.set_num_threads(1)

def check_isolation():
    assert os.getuid() == 65534
    assert not Path('/var/run/docker.sock').exists()
    assert not Path('/root/.ssh').exists()
    assert not Path('/input/data.npz').is_symlink()
    assert next(line.split(':')[1].strip() for line in Path('/proc/self/status').read_text().splitlines()
                if line.startswith('CapEff:')) == '0000000000000000'
    for filename in ['/original.py', '/input/framework/should_not_exist.py', '/input/bundle/should_not_exist.py']:
        try:
            Path(filename).write_text('forbidden')
        except OSError:
            pass
        else:
            raise AssertionError('original filesystem is writable')
    try:
        socket.create_connection(('1.1.1.1', 443), timeout=1).close()
    except OSError:
        pass
    else:
        raise AssertionError('outbound network is available')

original_train = train
original_predict = predict

def train(context):
    check_isolation()
    assert set(context['data']) == {'train_images', 'train_targets', 'val_images', 'val_targets'}
    return original_train(context)

def predict(context):
    check_isolation()
    assert set(context['data']) == {'images'}
    return original_predict(context)
'''


@unittest.skipUnless(all(os.environ.get(key) for key in (
    "AGENTIC_TEST_WORKER_HOST", "AGENTIC_TEST_WORKER_ROOT", "AGENTIC_TEST_WORKER_IMAGE")),
    "Linux SSH worker not configured; real isolation test not executed")
class WorkerIntegrationTests(unittest.TestCase):
    def test_real_adaptive_segments_match_continuous_training(self):
        """Opt-in only: opaque state is deserialized exclusively inside containers."""
        import numpy as np
        from autonomous import experiment_config
        from contracts import AutonomousExperimentDecision
        worker = SSHWorker(WorkerConfiguration(host=os.environ["AGENTIC_TEST_WORKER_HOST"],
            root=os.environ["AGENTIC_TEST_WORKER_ROOT"], image=os.environ["AGENTIC_TEST_WORKER_IMAGE"]))
        device = os.environ.get("AGENTIC_TEST_WORKER_DEVICE", "cpu")
        worker.preflight(device)
        source = Path(__file__).resolve().parents[1].joinpath("default_experiment.py").read_text(encoding="utf-8")
        decision = AutonomousExperimentDecision(bundle_id="adaptive_probe", hypothesis="Verify exact remote continuation",
            files=[GeneratedSource(path="experiment.py", code=source + PROBE)], epochs=4, batch_size=9,
            parameters={"hidden": 16, "depth": 1, "scales": [4, 7], "dropout": .1},
            training=dict(lr=.001, weight_decay=.0001, class_weighting=False, optimizer="adamw", scheduler="none",
                label_smoothing=0., early_stopping_patience=0, early_stopping_monitor="val_accuracy",
                early_stopping_min_delta=0., gradient_clip_val=1., adam_beta1=.9, adam_beta2=.999, optimizer_eps=1e-8))
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(Path(directory) / "run")
            reference = archive_bundle(bb, decision, "integration_test")
            config = experiment_config(decision, reference, worker.configuration, 42, device, "integration_test")
            worker.execute(bb.root, config, "verify", {}, epochs=1)
            prepared = prepare_data(make_bundle(), RepresentationDecision(normalization="standardize",
                augmentations=[], rationale="Synthetic data for isolated continuation validation"))
            continuous = train_model(prepared, config, bb.blob_dir / "continuous.ckpt")
            segmented = train_model(prepared, config.model_copy(update={"segment_targets": [2, 4]}), bb.blob_dir / "segmented.ckpt")
            self.assertEqual(continuous.result.history, segmented.result.history)
            self.assertTrue(Path(segmented.result.resume_path).is_file())
            np.testing.assert_array_equal(continuous.model.predict(prepared, "val", 9)[1],
                                          segmented.model.predict(prepared, "val", 9)[1])

    def test_real_read_only_files_network_sealing_checkpoint_and_inference(self):
        worker = SSHWorker(WorkerConfiguration(host=os.environ["AGENTIC_TEST_WORKER_HOST"],
            root=os.environ["AGENTIC_TEST_WORKER_ROOT"], image=os.environ["AGENTIC_TEST_WORKER_IMAGE"]))
        device = os.environ.get("AGENTIC_TEST_WORKER_DEVICE", "cpu")
        preflight = worker.preflight(device)
        self.assertEqual(preflight["isolation"]["network"], "none")
        source = Path(__file__).resolve().parents[1].joinpath("default_experiment.py").read_text(encoding="utf-8")
        decision = GeneratedCodeDecision(bundle_id="isolation_probe", hypothesis="Verify real worker isolation",
            files=[GeneratedSource(path="experiment.py", code=source + PROBE)],
            parameters={"hidden": 16, "depth": 1, "scales": [4, 7]}, epochs=1, batch_size=9)
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(Path(directory) / "run")
            reference = archive_bundle(bb, decision, "integration_test")
            config = generated_config(decision, reference, worker.configuration, 42, device, 1, "integration_test")
            checked = worker.execute(bb.root, config, "verify", {}, epochs=1)
            self.assertTrue((checked / "result.json").exists())
            prepared = prepare_data(make_bundle(), RepresentationDecision(normalization="standardize",
                augmentations=[], rationale="Integration test on synthetic official-shaped splits"))
            result = train_model(prepared, config, bb.blob_dir / "probe.ckpt")
            probabilities, logits, targets = result.model.predict(prepared, "test", 9)
            self.assertEqual(logits.shape, (18, 9))
            self.assertEqual(probabilities.shape, (18, 9))
            self.assertEqual(len(targets), 18)
