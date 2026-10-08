"""Regression coverage for progressive repair and operational continuation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contracts import AutonomousSearchDecision, AutonomousRecoveryDecision, Blackboard
from llm import OllamaReasoner, OllamaDecisionError, ReasonedDecision
from tests.test_llm import autonomous_decision
from tests.test_autonomous import experiment, AdaptiveWorker, ScriptedReasoner
from tests import test_autonomous as support
from tests.test_generated import SimulatedWorker, WORKER

class StatefulRepairTests(unittest.TestCase):
    def test_length_epsilon_then_missing_cosine_is_monotonic(self):
        original = autonomous_decision(1.0)
        original['experiment']['files'][0]['code'] = original['experiment']['files'][0]['code'].strip()
        original['experiment']['training']['scheduler'] = 'cosine'
        repaired = {**original['experiment']['training'], 'cosine_eta_min': 0.0}
        outputs = iter([(original, 'length'), (original, 'stop'), ({'optimizer_eps': '1e-8'}, 'stop'), (repaired, 'stop')])
        scopes, events, prompts = [], [], []
        def transport(url, payload, timeout):
            scopes.append(payload['format']['title'])
            prompts.append(json.loads(json.dumps(payload['messages'])))
            content, reason = next(outputs)
            return {'message': {'content': json.dumps(content)}, 'done_reason': reason}
        result = OllamaReasoner('http://llm', retries=2, required=True, transport=transport).decide(
            stage='autonomous_search.action_7', system='test', user='test', response_model=AutonomousSearchDecision,
            fallback=dict(action='finish_search', rationale='No fallback allowed'),
            audit=lambda e, **kw: events.append(dict(event=e, **kw)))
        self.assertEqual(result.value.experiment.training.optimizer_eps, 1e-8)
        self.assertEqual(result.value.experiment.training.cosine_eta_min, 0.)
        self.assertEqual(scopes, ['AutonomousSearchDecision', 'AutonomousSearchDecision', '_OptimizerEpsilonRepair', 'AutonomousTrainingOptions'])
        self.assertIn('concise', prompts[1][-1]['content'])
        self.assertEqual(json.loads(prompts[-1][-1]['content'])['current_document']['optimizer_eps'], 1e-8)
        final = result.value.model_dump(mode='json')
        for key in ('action', 'rationale'):
            self.assertEqual(final[key], original[key])
        for key in ('files', 'epochs', 'batch_size', 'bundle_id', 'parameters', 'hypothesis'):
            self.assertEqual(final['experiment'][key], original['experiment'][key])
        self.assertTrue(any(e['event'] == 'llm_output_truncated' for e in events))
        finished = [e for e in events if e['event'] == 'llm_repair_finished']
        self.assertEqual(len(finished), 2)
        self.assertEqual(finished[-1]['changed_paths'], ['experiment.training.cosine_eta_min'])
        AutonomousSearchDecision.model_validate(final)

    def test_training_repair_cannot_inject_source_or_action(self):
        invalid = autonomous_decision()
        invalid['experiment']['training']['lr'] = -1
        injected = {**autonomous_decision()['experiment']['training'], 'action': 'finish_search', 'files': []}
        outputs = iter((invalid, injected))
        reasoner = OllamaReasoner('http://llm', retries=1, required=True,
            transport=lambda *a: {'message': {'content': json.dumps(next(outputs))}})
        with self.assertRaises(OllamaDecisionError):
            reasoner.decide(stage='test', system='test', user='test', response_model=AutonomousSearchDecision,
                            fallback=dict(action='finish_search', rationale='No fallback allowed'))

    def test_recovery_rejects_new_experiments(self):
        with self.assertRaises(ValueError):
            AutonomousRecoveryDecision.model_validate(autonomous_decision())

class PipelineReasoner(ScriptedReasoner):
    def decide(self, **kwargs):
        if kwargs['response_model'] in {AutonomousSearchDecision, AutonomousRecoveryDecision}:
            if not self.actions:
                raise OllamaDecisionError('synthetic exhausted repair')
            return super().decide(**kwargs)
        value = kwargs['response_model'].model_validate(kwargs['fallback'])
        return ReasonedDecision(value, self.model, False, 1, None)

class ResumeTests(unittest.TestCase):
    def setUp(self):
        SimulatedWorker.calls = []

    def run_parent(self, directory):
        from run import build_parser, validate_args, resolve_limits, run_once
        from agents import IngestionAgent
        from tests.helpers import make_bundle
        actions = [dict(action='new_trial', rationale='Test unique configuration',
                        experiment=experiment().model_copy(update={'parameters': {'revision': n}}).model_dump()) for n in range(6)]
        backend = PipelineReasoner(actions)
        def ingestion(**kwargs):
            return IngestionAgent(loader=lambda **kw: make_bundle(), **kwargs)
        args = build_parser().parse_args(['--execution-mode', 'agent_autonomous', '--device', 'cpu', '--ollama-base', 'http://simulated'])
        validate_args(args)
        root = Path(directory) / 'parent' / 'seed_42'
        with contextlib.redirect_stdout(io.StringIO()), patch('run.OllamaReasoner', return_value=backend), \
                patch('run.ollama_model_digest', return_value='a'*64), patch('run.IngestionAgent', side_effect=ingestion), \
                patch('autonomous.LocalWorker', AdaptiveWorker), patch('remote.LocalWorker', AdaptiveWorker):
            with self.assertRaises(OllamaDecisionError):
                run_once(args, 42, root, resolve_limits(args))
        return root, args

    def test_resume_six_trials_action_seven_and_pipeline_completion(self):
        from autonomous import AutonomousSearchState, restore_training_result
        from resume import inspect_parent
        from run import run_once, resolve_limits
        from governance import check_integrity
        with tempfile.TemporaryDirectory() as directory:
            root, args = self.run_parent(directory)
            before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            parent, _ = inspect_parent(root)
            state = AutonomousSearchState.restore(parent)
            self.assertEqual((state.sequence, state.candidate_sequence), (6, 6))
            self.assertEqual((len(state.trials), len(state.latest)), (6, 6))
            old = state.trials[0].model_copy(update={'training_result': None})
            self.assertEqual(restore_training_result(old), state.results[old.candidate_id])
            args.resume_run = str(root)
            child = Path(directory) / 'child' / 'seed_42'
            backend = PipelineReasoner([dict(action='new_trial', rationale='New configuration after pause',
                experiment=experiment().model_copy(update={'parameters': {'revision': 7}}).model_dump()),
                dict(action='finish_search', rationale='Enough evidence')])
            initial_trains = sum(c[0] == 'train' for c in SimulatedWorker.calls)
            with contextlib.redirect_stdout(io.StringIO()), patch('run.OllamaReasoner', return_value=backend), \
                    patch('run.ollama_model_digest', return_value='a'*64), \
                    patch('agents.IngestionAgent.run', side_effect=AssertionError('ingestion cannot rerun')), \
                    patch('agents.PreprocessingAgent.run', side_effect=AssertionError('preprocessing cannot rerun')), \
                    patch('autonomous.LocalWorker', AdaptiveWorker), patch('remote.LocalWorker', AdaptiveWorker):
                summary = run_once(args, 42, child, resolve_limits(args))
            self.assertTrue(summary['status'].startswith('completed'), summary)
            self.assertEqual(backend.calls[0]['stage'], 'autonomous_search.action_7')
            self.assertEqual(sum(c[0] == 'train' for c in SimulatedWorker.calls) - initial_trains, 1)
            resumed = Blackboard.open(child)
            self.assertEqual(resumed.get('trial_0007').candidate_id, 'autonomous_t0007')
            self.assertEqual(resumed.get('search_report').completed_trials, 7)
            self.assertEqual(resumed.get('resume_manifest').first_new_decision, 'autonomous_search.action_7')
            self.assertEqual(check_integrity(child), [])
            after = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            for trial in state.trials:
                for relative in (trial.checkpoint_path, trial.resume_path):
                    self.assertEqual((child / relative).read_bytes(), (root / relative).read_bytes())
            historical = resumed.get('resume_manifest').inherited_transcripts
            new_names = sorted(p.name for p in (child / 'blobs/reasoning').glob('*.json') if p.name not in historical)
            self.assertGreater(int(new_names[0][:-5]), max(int(n[:-5]) for n in historical))

    def test_integrity_failure_rejects_checkpoint_and_latest_state(self):
        from resume import create_child
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.run_parent(directory)
            trial = Blackboard.open(root).get('trial_0001')
            for relative in (trial.checkpoint_path, trial.resume_path):
                path = root / relative
                original = path.read_bytes()
                path.write_bytes(b'corrupted')
                child = Path(directory) / 'rejected' / 'seed_42'
                with self.assertRaisesRegex(ValueError, 'integrity|checksum'):
                    create_child(root, child, code_sha256='a'*64)
                self.assertFalse(child.exists())
                path.write_bytes(original)

    def test_valid_trials_allow_live_recovery_finish_search(self):
        from autonomous import AutonomousSearchAgent
        class RecoveryReasoner(PipelineReasoner):
            def decide(self, **kwargs):
                if kwargs['stage'] == 'autonomous_search.action_2':
                    raise OllamaDecisionError('unrepairable proposal')
                return super().decide(**kwargs)
        with tempfile.TemporaryDirectory() as directory, patch('autonomous.LocalWorker', AdaptiveWorker), patch('remote.LocalWorker', AdaptiveWorker):
            bb = support.AutonomousTests().populate(directory)
            backend = RecoveryReasoner([dict(action='new_trial', rationale='Test valid trial', experiment=experiment().model_dump()),
                                       dict(action='finish_search', rationale='Stop after evidence')])
            AutonomousSearchAgent(backend, worker=WORKER, seed=42, device='cpu').run(bb)
            self.assertEqual(backend.calls[-1]['stage'], 'autonomous_search.recovery_2')
            self.assertIs(backend.calls[-1]['response_model'], AutonomousRecoveryDecision)
            self.assertEqual(bb.get('search_report').completed_trials, 1)

    def test_review_resume_does_not_execute_completed_search(self):
        from autonomous import AutonomousSearchAgent, AutonomousSearchState
        from contracts import ExecutionState
        from search import rank_trials
        from run import run_once, resolve_limits
        with tempfile.TemporaryDirectory() as directory:
            root, args = self.run_parent(directory)
            bb = Blackboard.open(root)
            state = AutonomousSearchState.restore(bb)
            selected = rank_trials(state.latest.values(), accuracy_tolerance=.005, equal_epoch_budgets=False)
            agent = AutonomousSearchAgent(PipelineReasoner([]), worker=WORKER, seed=42, device='cpu')
            agent._freeze(bb, selected, state.results[selected.candidate_id], state.trials, state.trial_names)
            bb.record_event('review_exception', stage='model_search', error_type='OllamaDecisionError', error='timeout')
            bb.put('execution_status', ExecutionState(status='paused:review:model_search', stage='model_search'), producer='test')
            args.resume_run = str(root)
            child = Path(directory) / 'review_child' / 'seed_42'
            with contextlib.redirect_stdout(io.StringIO()), patch('run.OllamaReasoner', return_value=PipelineReasoner([])), \
                    patch('run.ollama_model_digest', return_value='a'*64), \
                    patch('autonomous.AutonomousSearchAgent.run', side_effect=AssertionError('search cannot rerun')), \
                    patch('remote.LocalWorker', AdaptiveWorker):
                summary = run_once(args, 42, child, resolve_limits(args))
            self.assertTrue(summary['status'].startswith('completed'), summary)
            self.assertTrue(Blackboard.open(child).get('resume_manifest').review_only)

    def test_historical_failed_llm_migration_and_terminal_refusal(self):
        from contracts import ExecutionState
        from resume import inspect_parent
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.run_parent(directory)
            bb = Blackboard.open(root)
            bb.put('execution_status', ExecutionState(status='failed:model_search', stage='model_search'), producer='test')
            restored, _ = inspect_parent(root)
            self.assertEqual(restored.get('execution_status').status, 'failed:model_search')
            bb.record_event('stage_exception', stage='model_search', error_type='ValueError', error='checksum mismatch')
            with self.assertRaisesRegex(ValueError, 'terminal failure'):
                inspect_parent(root)

    def test_pause_classification_and_cli_exclusivity(self):
        from orchestrator import exception_status
        from run import main
        for exc in (OllamaDecisionError('schema'), TimeoutError(), KeyboardInterrupt()):
            self.assertEqual(exception_status(exc), 'paused')
        for exc in (ValueError('checksum mismatch'), OSError('corrupt disk'), RuntimeError('invariant')):
            self.assertEqual(exception_status(exc), 'failed')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--resume-run', 'parent', '--replay-run', 'parent'])
