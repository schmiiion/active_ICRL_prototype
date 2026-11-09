"""Native sampling preserves raw data and source-style feasible reward metrics."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box

from ICRL.native_sampling import NativeSampler
from omnisafe.utils.config import Config


class RecordingActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def predict(self, obs, deterministic=False):
        self.inputs.append((obs.clone(), deterministic))
        return torch.full((1, 1), 2.0)


class Stats:
    def __init__(self):
        self.values = {'_mean': torch.tensor([10.0]), '_std': torch.tensor([2.0]),
                       '_clip': torch.tensor([0.5])}

    def state_dict(self):
        return self.values


class EpisodeEnv:
    need_time_limit_wrapper = False
    need_auto_reset_wrapper = False
    need_evaluation = False
    action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    observation_space = Box(-100.0, 100.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.obs = torch.tensor([10.0])
        self.seeds, self.actions = [], []
        self.closed = False

    def reset(self, seed=None, options=None):
        self.seeds.append(seed)
        self.steps = 0
        self.obs.fill_(10.0)
        return self.obs, {}

    def step(self, action):
        self.actions.append(action.clone())
        self.steps += 1
        self.obs.add_(2.0)
        # The second transition violates; returning to safety must not resume reward_nc.
        return (self.obs, torch.tensor(float(self.steps)), torch.tensor(float(self.steps == 2)),
                torch.tensor(False), torch.tensor(self.steps == 3), {})

    def close(self):
        self.closed = True


class NativeSamplingTest(unittest.TestCase):
    def make_sampler(self, normalized=True, *, clip_actions=False):
        self.env, self.actor, self.stats = EpisodeEnv(), RecordingActor(), Stats()
        normalizers = {'obs_normalizer': self.stats} if normalized else {}
        native = SimpleNamespace(_device=torch.device('cpu'),
                                 _cfgs=Config(env_cfgs={'backend': 'modern'},
                                              icrl_cfgs={'clip_actions': clip_actions}), actor=self.actor,
                                 train_env=SimpleNamespace(save=lambda: normalizers))
        with patch('ICRL.native_sampling.make', return_value=self.env) as make:
            sampler = NativeSampler(native, {'eval_env_id': 'HCWithPos-v0'}, seed=123)
            make.assert_called_once_with('HCWithPos-v0', num_envs=1, device='cpu', backend='modern')
        self.addCleanup(sampler.close)
        return sampler

    def test_clipped_execution_is_recorded_consistently_in_samples_and_evaluation(self):
        sampler = self.make_sampler(clip_actions=True)
        obs, actions, _, lengths = sampler.sample(1)
        np.testing.assert_array_equal(obs[0], [[10.0], [12.0], [14.0]])
        np.testing.assert_array_equal(actions[0], [[1.0], [1.0], [1.0]])
        self.assertEqual(lengths, [3])
        sampler.evaluate(1)
        torch.testing.assert_close(torch.stack(self.env.actions), torch.ones((6, 1)))

    def test_raw_pre_states_unclipped_actions_and_frozen_normalization(self):
        sampler = self.make_sampler()
        before = {key: value.clone() for key, value in self.stats.values.items()}
        obs, actions, rewards, lengths = sampler.sample(2)
        for episode in obs:
            np.testing.assert_array_equal(episode, [[10.0], [12.0], [14.0]])
        for episode in actions:
            np.testing.assert_array_equal(episode, [[2.0], [2.0], [2.0]])
        np.testing.assert_array_equal(rewards[0], [1.0, 2.0, 3.0])
        self.assertEqual(lengths, [3, 3])
        torch.testing.assert_close(torch.cat([value for value, _ in self.actor.inputs[:3]]),
                                   torch.tensor([[0.0], [0.5], [0.5]]))
        self.assertEqual(self.env.seeds, [123, None])
        self.assertTrue(self.actor.training)
        for key, value in before.items():
            torch.testing.assert_close(self.stats.values[key], value, rtol=0, atol=0)
            torch.testing.assert_close(sampler.normalizer[key], value, rtol=0, atol=0)
            self.assertNotEqual(sampler.normalizer[key].data_ptr(), value.data_ptr())

    def test_evaluation_excludes_first_violation_and_every_later_reward(self):
        sampler = self.make_sampler(normalized=False)
        metrics = sampler.evaluate(1, deterministic=True)
        self.assertEqual(metrics, {'reward': 6.0, 'feasible_reward': 1.0, 'violation_rate': 1.0})
        self.assertEqual(sampler.last_records[0]['cost'], 1.0)
        self.assertTrue(all(deterministic for _, deterministic in self.actor.inputs))
        torch.testing.assert_close(self.actor.inputs[0][0], torch.tensor([[10.0]]))

    def test_statistics_resync_before_each_sampling_phase(self):
        sampler = self.make_sampler()
        sampler.sample(1)
        self.stats.values['_mean'].fill_(8.0)
        self.stats.values['_clip'].fill_(5.0)
        sampler.sample(1)
        torch.testing.assert_close(self.actor.inputs[3][0], torch.tensor([[1.0]]))
        self.assertEqual(self.env.seeds, [123, None])


if __name__ == '__main__':
    unittest.main()
