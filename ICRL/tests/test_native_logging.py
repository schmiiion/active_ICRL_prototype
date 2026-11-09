"""Native rollouts retain the raw episode statistics needed for Liu's figures."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from ICRL.learned_cost_adapter import LearnedCostAdapter
from omnisafe.envs.core import CMDP
from omnisafe.utils.config import Config


class EpisodeSequence(CMDP):
    need_time_limit_wrapper = False
    need_auto_reset_wrapper = True
    need_evaluation = False

    def __init__(self):
        self._num_envs = 1
        self._observation_space = Box(-1000, 1000, (1,), dtype=np.float32)
        self._action_space = Box(-1, 1, (1,), dtype=np.float32)
        self.episode = -1

    def reset(self, seed=None, options=None):
        self.episode += 1
        self.length = 0
        return torch.tensor([10.0 * self.episode]), {}

    def step(self, action):
        reward = (2.0, 3.0, 5.0)[self.length]
        cost = float(self.episode == 0 and self.length == 1)
        self.length += 1
        return (torch.tensor([10.0 * self.episode + self.length]), torch.tensor(reward),
                torch.tensor(cost), torch.tensor(False), torch.tensor(self.length == 3), {})

    def set_seed(self, seed):
        pass

    def render(self):
        return None

    def close(self):
        pass


class NativeLoggingTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='native-episode-log-')
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)

    def adapter(self, writer=None, normalize=True):
        config = Config(train_cfgs={'device': 'cpu'}, algo_cfgs={
            'obs_normalize': normalize, 'reward_normalize': normalize,
            'cost_normalize': normalize, 'use_cost': True,
        })
        raw = EpisodeSequence()
        with (patch('omnisafe.adapter.online_adapter.support_envs', return_value=['Sequence-v0']),
              patch('omnisafe.adapter.online_adapter.make', return_value=raw)):
            adapter = LearnedCostAdapter('Sequence-v0', 1, 7, config,
                                         lambda obs, action: np.full(len(obs), 0.4, np.float32))
        self.addCleanup(adapter.close)
        adapter.configure_episode_logging(self.path / 'episodes.csv', 'VICRL', 7,
                                          writer=writer, count_steps=lambda: 0)
        return adapter

    def test_raw_metrics_are_separate_from_optimization_cost_and_written_to_tensorboard(self):
        writer = SummaryWriter(str(self.path / 'tb'))
        self.addCleanup(writer.close)
        adapter = self.adapter(writer)
        adapter.reset()
        for _ in range(3):
            _, reward, cost, _, _, info = adapter.step(torch.zeros((1, 1)))
            adapter._log_value(reward, cost, info)
        self.assertAlmostEqual(adapter._ep_cost.item(), 1.2, places=6)
        self.assertAlmostEqual(adapter._ep_ret.item(), 10.0, places=6)
        self.assertEqual(info['true_cost'].item(), 0.0)
        # The second episode never violates the true constraint.
        for _ in range(3):
            adapter.step(torch.zeros((1, 1)))
        writer.flush()
        with (self.path / 'episodes.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual([float(row['reward']) for row in rows], [10.0, 10.0])
        self.assertEqual([float(row['reward_nc']) for row in rows], [2.0, 10.0])
        self.assertEqual([float(row['constraint']) for row in rows], [1.0, 0.0])
        self.assertEqual([float(row['true_cost']) for row in rows], [1.0, 0.0])
        np.testing.assert_allclose([float(row['learned_cost']) for row in rows], [1.2, 1.2])
        self.assertEqual([int(row['length']) for row in rows], [3, 3])
        self.assertEqual([int(row['total_steps']) for row in rows], [3, 6])
        events = EventAccumulator(str(self.path / 'tb')).Reload()
        np.testing.assert_allclose([row.value for row in events.Scalars('Figure2/ConstraintViolationRaw')], [1, 0])
        np.testing.assert_allclose([row.value for row in events.Scalars('Figure2/FeasibleRewardRaw')], [2, 10])
        self.assertEqual([row.step for row in events.Scalars('Figure2/ConstraintViolationRaw')], [0, 1])
        self.assertNotIn('Figure2/ConstraintViolationRate', events.Tags()['scalars'])
        np.testing.assert_allclose([row.value for row in events.Scalars('Episodes/TrueCost')], [1, 0])
        np.testing.assert_allclose([row.value for row in events.Scalars('Episodes/LearnedCost')], [1.2, 1.2])

    def test_explicit_epoch_reset_discards_partial_episode_without_inventing_a_completion(self):
        adapter = self.adapter(normalize=False)
        adapter.reset()
        adapter.step(torch.zeros((1, 1)))
        adapter.reset()
        for _ in range(3):
            adapter.step(torch.zeros((1, 1)))
        with (self.path / 'episodes.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]['length']), 3)
        self.assertEqual(float(rows[0]['reward']), 10.0)
        self.assertAlmostEqual(float(rows[0]['learned_cost']), 1.2, places=6)
        self.assertEqual(int(rows[0]['total_steps']), 4)

    def test_figure2_forward_window_has_source_alignment_and_tail_length(self):
        writer = SummaryWriter(str(self.path / 'tb'))
        self.addCleanup(writer.close)
        adapter = self.adapter(writer, normalize=False)
        adapter.reset()
        for _ in range(100 * 3):
            adapter.step(torch.zeros((1, 1)))
        writer.flush()
        events = EventAccumulator(str(self.path / 'tb')).Reload()
        self.assertNotIn('Figure2/ConstraintViolationRate', events.Tags()['scalars'])
        for _ in range(3):
            adapter.step(torch.zeros((1, 1)))
        writer.flush()
        events.Reload()
        costs = events.Scalars('Figure2/ConstraintViolationRate')
        rewards = events.Scalars('Figure2/FeasibleReward')
        self.assertEqual(len(costs), 1)  # Liu's helper returns N - 100 points.
        self.assertEqual(costs[0].step, 0)
        self.assertAlmostEqual(costs[0].value, 0.01)
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0].step, 0)
        self.assertAlmostEqual(rewards[0].value, (2 + 99 * 10) / 100, places=5)


if __name__ == '__main__':
    unittest.main()
