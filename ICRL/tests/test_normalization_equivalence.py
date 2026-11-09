"""Execute Liu's original normalizers on the same transition stream."""

import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from ICRL.normalization import LiuNormalize


REFERENCE = Path(os.environ.get('LIU_ICRL_REFERENCE',
                 Path(__file__).resolve().parents[3] / 'ICRL-benchmarks-public'))


class Stream:
    num_envs = 2
    observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float64)
    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self):
        return np.full((2, 3), 0.25)

    def step_wait(self):
        return self.transition


class Delegate:
    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space, self.action_space = venv.observation_space, venv.action_space

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()


class NormalizationEquivalence(unittest.TestCase):
    def test_configured_clipping_changes_observation_reward_and_cost(self):
        cfg = dict(dont_normalize_obs=False, dont_normalize_reward=False,
                   dont_normalize_cost=False, reward_gamma=0.99, cost_gamma=0.99,
                   clip_obs=0.25, clip_reward=0.1, clip_cost=0.2,
                   normalization_epsilon=1e-6)
        normalizer = LiuNormalize((1,), 1, cfg)
        np.testing.assert_array_equal(normalizer.reset(np.array([[100.0]])), [[0.25]])
        _, reward, cost = normalizer.step(np.array([[100.0]]), np.array([100.0]),
                                          np.array([100.0]), np.array([False]))
        np.testing.assert_array_equal(reward, [0.1])
        np.testing.assert_array_equal(cost, [0.2])

    def test_original_transition_statistics_and_scalings(self):
        rms_path = REFERENCE / 'stable_baselines3/common/running_mean_std.py'
        if not rms_path.exists():
            self.skipTest('Liu reference checkout is not available')
        rms = types.ModuleType('rms')
        exec(compile(rms_path.read_text(), str(rms_path), 'exec'), rms.__dict__)
        base = types.ModuleType('base')
        base.VecEnv, base.VecEnvWrapper = object, Delegate
        namespace = {}
        path = REFERENCE / 'stable_baselines3/common/vec_env/vec_normalize.py'
        with patch.dict(sys.modules, {
            'stable_baselines3.common.running_mean_std': rms,
            'stable_baselines3.common.vec_env.base_vec_env': base,
        }):
            exec(compile(path.read_text(), str(path), 'exec'), namespace)
        cfg = dict(dont_normalize_obs=False, dont_normalize_reward=False,
                   dont_normalize_cost=False, reward_gamma=0.99, cost_gamma=0.97)
        for training in (True, False):
            stream = Stream()
            expected = namespace['VecNormalizeWithCost'](stream, training=training, cost_gamma=0.97)
            actual = LiuNormalize((3,), 2, cfg, training)
            # Evaluation normalization is disabled by make_eval_env.
            expected.norm_reward = expected.norm_cost = training
            np.testing.assert_array_equal(expected.reset(), actual.reset(stream.reset()))
            rng = np.random.RandomState(123)
            for _ in range(30):
                obs, reward, cost = rng.randn(2, 3), rng.randn(2), rng.rand(2)
                done = rng.rand(2) > 0.6
                stream.transition = (obs, reward, done, [{'cost': c} for c in cost])
                eo, er, _, infos = expected.step_wait()
                ao, ar, ac = actual.step(obs, reward, cost, done)
                for a, b in [(eo, ao), (er, ar), (np.array([i['cost'] for i in infos]), ac)]:
                    np.testing.assert_array_equal(a, b)
                for left, right in [(expected.obs_rms, actual.obs_rms),
                                    (expected.ret_rms, actual.ret_rms),
                                    (expected.cost_rms, actual.cost_rms)]:
                    np.testing.assert_array_equal(left.mean, right.mean)
                    np.testing.assert_array_equal(left.var, right.var)
                    self.assertEqual(left.count, right.count)


if __name__ == '__main__':
    unittest.main()
