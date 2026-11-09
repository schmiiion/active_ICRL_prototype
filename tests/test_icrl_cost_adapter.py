"""Check learned-cost semantics independently of a MuJoCo simulator."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box

from ICRL.learned_cost_adapter import LearnedCostAdapter, LearnedCostWrapper
from omnisafe.envs.core import CMDP
from omnisafe.utils.config import Config


class CountingEnv(CMDP):
    """Deterministic observations, two-step episodes, and non-unit action bounds."""

    need_time_limit_wrapper = False
    need_evaluation = False

    def __init__(self, num_envs: int = 1) -> None:
        self._num_envs = num_envs
        self.need_auto_reset_wrapper = num_envs == 1
        self._observation_space = Box(-1000.0, 1000.0, shape=(2,), dtype=np.float32)
        self._action_space = Box(-2.0, 4.0, shape=(1,), dtype=np.float32)
        self.episode = 0
        self.episode_step = 0
        self.step_calls = 0
        self.last_action = None
        self.closed = False

    def _observation(self):
        obs = torch.tensor([10.0 * self.episode, 20.0 * self.episode])
        obs += self.episode_step
        return obs if self.num_envs == 1 else obs.repeat(self.num_envs, 1)

    def _scalar(self, value):
        tensor = torch.as_tensor(value)
        return tensor if self.num_envs == 1 else tensor.repeat(self.num_envs)

    def reset(self, seed=None, options=None):
        self.episode += 1
        self.episode_step = 0
        return self._observation(), {}

    def step(self, action):
        self.step_calls += 1
        self.episode_step += 1
        self.last_action = action.clone()
        return (
            self._observation(),
            self._scalar(1.0),
            self._scalar(9.0),
            self._scalar(self.episode_step == 2),
            self._scalar(False),
            {},
        )

    def set_seed(self, seed):
        pass

    def render(self):
        return None

    def close(self):
        self.closed = True


class LearnedCostAdapterTests(unittest.TestCase):
    def make_adapter(self, callback=None, *, obs_normalize=False, cost_normalize=False,
                     clip_actions=False):
        env = CountingEnv()
        cfgs = Config(
            train_cfgs={'device': 'cpu'},
            icrl_cfgs={'clip_actions': clip_actions},
            algo_cfgs={
                'obs_normalize': obs_normalize,
                'reward_normalize': False,
                'cost_normalize': cost_normalize,
                'use_cost': True,
            },
        )
        with (
            patch('omnisafe.adapter.online_adapter.support_envs', return_value=['Counting-v0']),
            patch('omnisafe.adapter.online_adapter.make', return_value=env) as make,
        ):
            adapter = LearnedCostAdapter('Counting-v0', 1, 0, cfgs, callback)
            self.assertEqual(make.call_count, 1)
        self.addCleanup(adapter.close)
        return adapter, env

    def test_execution_clipping_is_outside_learned_cost_and_does_not_mutate_policy_action(self):
        for enabled, expected_callback, expected_physical in [(True, 1.0, 4.0), (False, 2.0, 7.0)]:
            with self.subTest(clip_actions=enabled):
                seen = []

                def cost_function(obs, actions):
                    seen.append(actions.copy())
                    return actions[:, 0] ** 2

                adapter, env = self.make_adapter(cost_function, clip_actions=enabled)
                adapter.reset()
                action = torch.tensor([[2.0]])
                _, _, cost, _, _, _ = adapter.step(action)
                torch.testing.assert_close(action, torch.tensor([[2.0]]))
                np.testing.assert_array_equal(seen[0], [[expected_callback]])
                torch.testing.assert_close(env.last_action, torch.tensor([expected_physical]))
                self.assertAlmostEqual(cost.item(), expected_callback ** 2)

    def test_only_one_simulator_is_created(self):
        with patch.object(CountingEnv, 'need_evaluation', True):
            adapter, env = self.make_adapter()
        self.assertIsNone(adapter._eval_env)
        adapter.close()
        self.assertTrue(env.closed)

    def test_raw_pre_action_observation_and_policy_action(self):
        calls = []

        def cost_function(obs, actions):
            calls.append((obs.copy(), actions.copy(), torch.is_grad_enabled()))
            return np.array([2.0])

        adapter, env = self.make_adapter(cost_function, obs_normalize=True)
        adapter.reset()
        action = torch.tensor([[0.5]], requires_grad=True)
        adapter.step(action)
        _, _, cost, _, _, info = adapter.step(action)

        np.testing.assert_array_equal(calls[0][0], [[10.0, 20.0]])
        np.testing.assert_array_equal(calls[1][0], [[11.0, 21.0]])
        np.testing.assert_array_equal(calls[0][1], [[0.5]])
        self.assertFalse(calls[0][2])
        torch.testing.assert_close(env.last_action, torch.tensor([2.5]))
        torch.testing.assert_close(cost, torch.tensor([2.0]))
        torch.testing.assert_close(info['true_cost'], torch.tensor([9.0]))
        torch.testing.assert_close(info['original_cost'], cost)
        self.assertFalse(cost.requires_grad)

    def test_autoreset_cost_uses_new_initial_observation(self):
        observations = []

        def cost_function(obs, actions):
            observations.append(obs.copy())
            return obs[:, 0]

        adapter, env = self.make_adapter(cost_function)
        adapter.reset()
        action = torch.zeros((1, 1))
        adapter.step(action)
        obs, _, cost, terminated, _, info = adapter.step(action)
        self.assertTrue(terminated.item())
        torch.testing.assert_close(cost, torch.tensor([11.0]))
        torch.testing.assert_close(obs, torch.tensor([[20.0, 40.0]]))
        torch.testing.assert_close(info['final_observation'], torch.tensor([[12.0, 22.0]]))
        adapter.step(action)
        np.testing.assert_array_equal(observations[2], [[20.0, 40.0]])
        self.assertEqual(env.step_calls, 3)

    def test_normalized_step_cost_but_raw_learned_lagrange_cost(self):
        adapter, _ = self.make_adapter(
            lambda obs, actions: obs[:, 0], cost_normalize=True,
        )
        adapter.reset()
        for _ in range(2):
            _, reward, cost, _, _, info = adapter.step(torch.zeros((1, 1)))
            adapter._log_value(reward=reward, cost=cost, info=info)

        self.assertNotAlmostEqual(cost.item(), 11.0)
        self.assertEqual(info['original_cost'].item(), 11.0)
        self.assertEqual(info['true_cost'].item(), 9.0)
        self.assertEqual(adapter._ep_cost.item(), 21.0)

    def test_scalar_and_vector_cost_shapes(self):
        for num_envs in (1, 2):
            with self.subTest(num_envs=num_envs):
                env = CountingEnv(num_envs)
                observed_shapes = []

                def cost_function(obs, actions):
                    observed_shapes.append((obs.shape, actions.shape))
                    return np.arange(num_envs, dtype=np.float32).reshape(num_envs, 1)

                wrapper = LearnedCostWrapper(env, torch.device('cpu'), cost_function)
                wrapper.reset()
                action_shape = (1,) if num_envs == 1 else (num_envs, 1)
                _, _, cost, _, _, info = wrapper.step(torch.zeros(action_shape))
                self.assertEqual(observed_shapes, [((num_envs, 2), (num_envs, 1))])
                self.assertEqual(cost.shape, torch.Size([] if num_envs == 1 else [num_envs]))
                self.assertEqual(info['true_cost'].shape, cost.shape)
                torch.testing.assert_close(cost.reshape(-1), torch.arange(num_envs).float())

    def test_unbound_or_invalid_callback_does_not_advance_environment(self):
        adapter, env = self.make_adapter()
        adapter.reset()
        action = torch.zeros((1, 1))
        self.assertFalse(adapter.has_cost_function)
        with self.assertRaises(RuntimeError):
            adapter.step(action)
        with self.assertRaises(TypeError):
            adapter.set_cost_function(None)
        for result in (np.array([1.0, 2.0]), np.array([np.nan]), np.array([np.inf])):
            with self.subTest(result=result):
                adapter.set_cost_function(lambda obs, actions: result)
                with self.assertRaises(ValueError):
                    adapter.step(action)
        self.assertEqual(env.step_calls, 0)

    def test_step_requires_reset(self):
        adapter, env = self.make_adapter(lambda obs, actions: np.zeros(1))
        with self.assertRaises(RuntimeError):
            adapter.step(torch.zeros((1, 1)))
        self.assertEqual(env.step_calls, 0)

    def test_live_callback_and_replacement_preserve_normalizers(self):
        class Constraint:
            value = 2.0

            def cost_function(self, obs, actions):
                return np.full(len(obs), self.value)

        constraint = Constraint()
        adapter, env = self.make_adapter(constraint.cost_function, obs_normalize=True)
        adapter.reset()
        normalizer = adapter.save()['obs_normalizer']
        count = normalizer.state_dict()['_count'].clone()
        constraint.value = 3.0
        _, _, cost, _, _, _ = adapter.step(torch.zeros((1, 1)))
        self.assertEqual(cost.item(), 3.0)
        adapter.set_cost_function(lambda obs, actions: np.full(len(obs), 4.0))
        self.assertIs(adapter.save()['obs_normalizer'], normalizer)
        self.assertGreater(normalizer.state_dict()['_count'].item(), count.item())
        self.assertEqual(env.episode, 1)
        _, _, cost, _, _, _ = adapter.step(torch.zeros((1, 1)))
        self.assertEqual(cost.item(), 4.0)


if __name__ == '__main__':
    unittest.main()
