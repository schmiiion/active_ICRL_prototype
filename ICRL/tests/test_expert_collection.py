"""Trajectory semantics, filtering, and reproducibility of snapshot collection."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box

from ICRL.action_wrapper import ActionClip
from ICRL.collect_experts import collect, load_snapshot, rollout_episode


class RecordingActor:
    def __init__(self):
        self.inputs = []

    def predict(self, obs, deterministic=False):
        self.inputs.append(obs.clone())
        return torch.full((1, 1), 2.0)


class SharedBufferEnv:
    """A terminal violation must reject the whole episode, not its safe prefix."""

    need_evaluation = False
    need_time_limit_wrapper = False
    clip_actions = False
    action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    observation_space = Box(-100.0, 100.0, shape=(1,), dtype=np.float32)

    def __init__(self, unsafe_seeds=()):
        self.observation = torch.tensor([10.0])
        self.unsafe_seeds = unsafe_seeds
        self.actions = []
        self.closed = False

    def reset(self, seed=None, options=None):
        self.seed, self.steps = seed, 0
        self.observation.fill_(10.0)
        return self.observation, {}

    def step(self, action):
        self.actions.append(action.clone())
        self.steps += 1
        self.observation.add_(2.0)
        truncated = self.steps == 2
        cost = float(truncated and self.seed in self.unsafe_seeds)
        return (self.observation, torch.tensor(3.0), torch.tensor(cost),
                torch.tensor(False), torch.tensor(truncated), {})

    def close(self):
        self.closed = True


class ExpertCollectionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='icrl-expert-test-')
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.run = root / 'run'
        (self.run / 'torch_save').mkdir(parents=True)
        (self.run / 'torch_save/epoch-200.pt').write_bytes(b'test checkpoint')
        (self.run / 'config.json').write_text(json.dumps({'env_id': 'test'}))
        self.config = dict(run_dir=str(self.run), output=str(root / 'data'),
                           checkpoint='epoch-200.pt', episodes=1, max_attempts=3,
                           seed=100, deterministic=False, max_episode_cost=0.0,
                           min_episode_reward=None, torch_threads=1)
        self.previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self.previous_threads)

    def run_collection(self, env):
        training = dict(env_id='HCWithPos-v0', env_cfgs={'backend': 'modern'})
        wrapped = ActionClip(env, clip_actions=self.config.get('clip_actions', False))
        with patch('ICRL.collect_experts.load_snapshot',
                   return_value=(wrapped, RecordingActor(), None, training)):
            with contextlib.redirect_stdout(io.StringIO()):
                return collect(self.config)

    def test_snapshot_inherits_clipping_and_allows_explicit_override(self):
        for saved, override, expected in [(None, None, False), (False, None, False),
                                           (True, None, True), (True, False, False),
                                           (False, True, True)]:
            with self.subTest(saved=saved, override=override):
                training = dict(env_id='HCWithPos-v0', algo_cfgs={'obs_normalize': False},
                                model_cfgs=dict(actor={'hidden_sizes': [4], 'activation': 'tanh'},
                                                weight_initialization_mode='orthogonal',
                                                actor_type='gaussian_learning'))
                if saved is not None:
                    training['icrl_cfgs'] = {'clip_actions': saved}
                (self.run / 'config.json').write_text(json.dumps(training))
                raw_env = SharedBufferEnv()
                with (patch('ICRL.collect_experts.make', return_value=raw_env),
                      patch('ICRL.collect_experts.torch.load', return_value={'pi': {}}),
                      patch('ICRL.collect_experts.ActorBuilder') as builder):
                    actor = builder.return_value.build_actor.return_value
                    actor.predict.return_value = torch.tensor([[2.0]])
                    env, actor, normalizer, _ = load_snapshot(
                        self.run, 'epoch-200.pt', clip_actions=override)
                try:
                    episode = rollout_episode(env, actor, normalizer, seed=100)
                    value = 1.0 if expected else 2.0
                    np.testing.assert_array_equal(episode['actions'], [[value], [value]])
                    torch.testing.assert_close(torch.stack(raw_env.actions), torch.full((2, 1), value))
                finally:
                    env.close()

    def test_clipped_collection_manifest_and_numpy_actions_describe_executed_commands(self):
        self.config['clip_actions'] = True
        output = self.run_collection(SharedBufferEnv())
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertIs(manifest['clip_actions'], True)
        self.assertIn('clipped', manifest['actions'])
        with np.load(output / 'trajectory-000000.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['actions'], [[1.0], [1.0]])

    def test_raw_transition_alignment_frozen_normalization_and_unclipped_actions(self):
        env, actor = SharedBufferEnv(), RecordingActor()
        normalizer = {'_mean': torch.tensor([10.0]), '_std': torch.tensor([2.0]),
                      '_clip': torch.tensor([0.5])}
        before = {key: value.clone() for key, value in normalizer.items()}
        episode = rollout_episode(env, actor, normalizer, seed=100)
        np.testing.assert_array_equal(episode['observations'], [[10.0], [12.0]])
        np.testing.assert_array_equal(episode['next_observations'], [[12.0], [14.0]])
        np.testing.assert_array_equal(episode['actions'], [[2.0], [2.0]])
        torch.testing.assert_close(torch.stack(env.actions), torch.tensor([[2.0], [2.0]]))
        torch.testing.assert_close(torch.cat(actor.inputs), torch.tensor([[0.0], [0.5]]))
        np.testing.assert_array_equal(episode['terminated'], [False, False])
        np.testing.assert_array_equal(episode['truncated'], [False, True])
        for key, value in normalizer.items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_terminal_violation_rejects_whole_episode_and_records_acceptance(self):
        env = SharedBufferEnv(unsafe_seeds={100})
        output = self.run_collection(env)
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertTrue(env.closed)
        self.assertTrue(manifest['complete'])
        self.assertEqual(manifest['accepted'], 1)
        self.assertEqual(manifest['attempted'], 2)
        self.assertEqual(manifest['acceptance_rate'], 0.5)
        self.assertEqual([row['seed'] for row in manifest['attempts']], [100, 101])
        self.assertEqual([row['accepted'] for row in manifest['attempts']], [False, True])
        self.assertEqual([row['cost'] for row in manifest['attempts']], [1.0, 0.0])
        self.assertEqual([row['length'] for row in manifest['attempts']], [2, 2])
        files = list(output.glob('*.npz'))
        self.assertEqual(len(files), 1)
        with np.load(files[0], allow_pickle=False) as saved:
            self.assertEqual(saved['costs'].sum(), 0)
            self.assertEqual(saved['observations'].shape, (2, 1))
            self.assertEqual(saved['next_observations'][-1, 0], 14.0)
            for key in saved.files:
                self.assertNotEqual(saved[key].dtype, np.dtype(object))

    def test_exhausted_filter_attempts_leave_truthful_manifest_and_fail(self):
        env = SharedBufferEnv(unsafe_seeds={100, 101, 102})
        self.config['max_attempts'] = 2
        with self.assertRaisesRegex(RuntimeError, 'Collected 0/1'):
            self.run_collection(env)
        output = Path(self.config['output'])
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertTrue(env.closed)
        self.assertFalse(manifest['complete'])
        self.assertEqual(manifest['attempted'], 2)
        self.assertEqual(manifest['accepted'], 0)
        self.assertEqual(manifest['acceptance_rate'], 0.0)
        self.assertEqual(list(output.glob('*.npz')), [])

    def test_same_seed_reproduces_stochastic_actions(self):
        class RandomActor:
            def predict(self, obs, deterministic=False):
                return torch.randn((1, 1))

        first = rollout_episode(SharedBufferEnv(), RandomActor(), None, seed=42)
        second = rollout_episode(SharedBufferEnv(), RandomActor(), None, seed=42)
        other = rollout_episode(SharedBufferEnv(), RandomActor(), None, seed=43)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(np.array_equal(first['actions'], other['actions']))


if __name__ == '__main__':
    unittest.main()
