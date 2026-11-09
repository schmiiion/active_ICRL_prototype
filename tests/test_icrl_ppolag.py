"""Exercise repeated inner phases using actual PPO-Lag updates on a tiny CMDP."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ICRL.ICRL_PPOLag import ICRL_PPOLag
from omnisafe.utils.config import get_default_kwargs_yaml
from test_icrl_cost_adapter import CountingEnv


class ICRLPPOLagTests(unittest.TestCase):
    def make_learner(self, cost_function=None, *, total_steps=16, clip_actions=False):
        output = tempfile.TemporaryDirectory(prefix='icrl-ppolag-test-')
        self.addCleanup(output.cleanup)
        env = CountingEnv()
        config = {
            'env_id': 'SafetyPointCircle1-v0',
            'seed': 7,
            'device': 'cpu',
            'torch_threads': 1,
            'parallel': 1,
            'vector_env_nums': 1,
            'icrl': {'total_steps': total_steps},
            'inner_cfgs': {
                'steps_per_epoch': 4,
                'batch_size': 4,
                'update_iters': 1,
                'obs_normalize': True,
            },
            'model_cfgs': {
                'linear_lr_decay': True,
                'actor': {'hidden_sizes': [8]},
                'critic': {'hidden_sizes': [8]},
            },
            'lagrange_cfgs': {'cost_limit': 0.1},
            'logger_cfgs': {
                'log_dir': output.name,
                'use_tensorboard': True,
                'use_wandb': False,
                'save_model_freq': 1,
            },
        }
        cfgs = get_default_kwargs_yaml('PPOLag', config['env_id'], 'on-policy')
        cfgs.recurisve_update({
            'env_id': config['env_id'], 'algo': 'PPOLag', 'exp_name': 'ICRL-PPOLag-test',
            'seed': config['seed'],
            'train_cfgs': {key: config[key] for key in (
                'device', 'torch_threads', 'parallel', 'vector_env_nums')},
            'algo_cfgs': config['inner_cfgs'], 'model_cfgs': config['model_cfgs'],
            'lagrange_cfgs': config['lagrange_cfgs'], 'logger_cfgs': config['logger_cfgs'],
        })
        cfgs.train_cfgs.total_steps = config['icrl']['total_steps']
        cfgs.recurisve_update({'icrl_cfgs': {'clip_actions': clip_actions}})
        with patch('omnisafe.adapter.online_adapter.make', return_value=env):
            learner = ICRL_PPOLag(config['env_id'], cfgs, cost_function)
        self.addCleanup(learner.close)
        # Suppress tables/progress-independent logger output during tiny updates.
        printer = patch.object(learner.logger._console, 'print')
        printer.start()
        self.addCleanup(printer.stop)
        return learner, env

    def test_clipped_execution_preserves_raw_ppo_actions_and_log_probabilities(self):
        callback_actions = []

        def cost_function(obs, actions):
            callback_actions.append(actions.copy())
            return np.full(len(obs), 0.2, dtype=np.float32)

        learner, env = self.make_learner(cost_function, total_steps=4, clip_actions=True)
        with torch.no_grad():
            layers = [module for module in learner.actor.mean.modules()
                      if isinstance(module, torch.nn.Linear)]
            for layer in layers:
                layer.weight.zero_()
                layer.bias.zero_()
            layers[-1].bias.fill_(2.0)
            learner.actor.log_std.fill_(-2.0)
        recorded = []
        store = learner._buf.store

        def record(**data):
            recorded.append((data['act'].clone(), data['logp'].clone()))
            return store(**data)

        with patch.object(learner._buf, 'store', side_effect=record):
            learner.train_for(4)
        raw_actions = torch.cat([action for action, _ in recorded])
        log_probs = torch.cat([logp for _, logp in recorded])
        self.assertTrue(bool((raw_actions > 1).all()))
        expected = torch.distributions.Normal(torch.full_like(raw_actions, 2.0),
                                               torch.full_like(raw_actions, np.exp(-2.0)))
        torch.testing.assert_close(log_probs, expected.log_prob(raw_actions).sum(-1))
        np.testing.assert_array_equal(np.concatenate(callback_actions), np.ones((4, 1)))
        torch.testing.assert_close(env.last_action, torch.tensor([4.0]))
        saved = json.loads((Path(learner.logger.log_dir) / 'config.json').read_text())
        self.assertIs(saved['icrl_cfgs']['clip_actions'], True)

    def test_two_phases_preserve_training_state_and_use_updated_costs(self):
        coefficient = [0.2]

        def cost_function(obs, actions):
            return np.full(len(obs), coefficient[0], dtype=np.float32)

        learner, env = self.make_learner(cost_function)
        ac = learner._actor_critic
        identities = (
            learner.actor, ac.reward_critic, ac.cost_critic,
            ac.actor_optimizer, ac.reward_critic_optimizer, ac.cost_critic_optimizer,
            learner._lagrange, learner._lagrange.lambda_optimizer,
            learner.train_env, learner.logger,
        )
        normalizer = learner.train_env.save()['obs_normalizer']
        first = learner.train_for(8)
        count = normalizer._count.clone()
        first_optimizer_steps = [state['step'].item() for state in ac.actor_optimizer.state.values()]
        self.assertEqual(learner.num_timesteps, 8)
        self.assertFalse(env.closed)
        self.assertFalse(learner.logger._output_file.closed)
        self.assertAlmostEqual(first[1], 0.4, places=5)
        self.assertAlmostEqual(ac.actor_scheduler.get_last_lr()[0], 0.00015)

        # No set_cost_function call: the same callback sees the changed constraint.
        coefficient[0] = 0.6
        second = learner.train_for(8)
        self.assertEqual(learner.num_timesteps, 16)
        self.assertAlmostEqual(second[1], 1.2, places=5)
        self.assertGreater(normalizer._count.item(), count.item())
        self.assertIs(normalizer, learner.train_env.save()['obs_normalizer'])
        continued = (
            learner.actor, ac.reward_critic, ac.cost_critic,
            ac.actor_optimizer, ac.reward_critic_optimizer, ac.cost_critic_optimizer,
            learner._lagrange, learner._lagrange.lambda_optimizer,
            learner.train_env, learner.logger,
        )
        for before, after in zip(identities, continued):
            self.assertIs(before, after)
        for before, state in zip(first_optimizer_steps, ac.actor_optimizer.state.values()):
            self.assertGreater(state['step'].item(), before)
        self.assertEqual(ac.actor_scheduler.last_epoch, 4)
        self.assertAlmostEqual(ac.actor_scheduler.get_last_lr()[0], 0.0)

        with open(Path(learner.logger.log_dir) / 'progress.csv', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(float(row['TotalEnvSteps'])) for row in rows], [4, 8, 12, 16])
        self.assertEqual([int(float(row['Train/Epoch'])) for row in rows], [0, 1, 2, 3])
        self.assertTrue(list((Path(learner.logger.log_dir) / 'tb').glob('events.*')))
        self.assertTrue((Path(learner.logger.log_dir) / 'torch_save' / 'epoch-4.pt').exists())
        learner.close()
        learner.close()
        self.assertTrue(env.closed)
        self.assertTrue(learner.logger._output_file.closed)
        self.assertIsNone(learner.logger._tensorboard_writer.all_writers)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            learner.train_for(4)

    def test_splitting_training_matches_uninterrupted_updates(self):
        def cost_function(obs, actions):
            return np.full(len(obs), 0.2, dtype=np.float32)

        whole, _ = self.make_learner(cost_function)
        whole.train_for(16)
        split, _ = self.make_learner(cost_function)
        split.train_for(8)
        split.train_for(8)
        for key, value in whole._actor_critic.state_dict().items():
            torch.testing.assert_close(value, split._actor_critic.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(
            whole._lagrange.lagrangian_multiplier,
            split._lagrange.lagrangian_multiplier,
            rtol=0, atol=0,
        )

    def test_rejects_invalid_or_unbound_training_before_rollout(self):
        learner, env = self.make_learner()
        with self.assertRaisesRegex(RuntimeError, 'cost function'):
            learner.train_for(4)
        learner.set_cost_function(lambda obs, actions: np.ones(len(obs), dtype=np.float32))
        for steps in (0, -4, True, 4.5, 3, 20):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                learner.train_for(steps)
        self.assertEqual(env.step_calls, 0)
        self.assertEqual(learner.num_timesteps, 0)

    def test_learn_uses_remaining_budget_and_closes(self):
        learner, env = self.make_learner(
            lambda obs, actions: np.zeros(len(obs), dtype=np.float32),
            total_steps=8,
        )
        learner.train_for(4)
        learner.learn()
        self.assertEqual(learner.num_timesteps, 8)
        self.assertTrue(env.closed)


if __name__ == '__main__':
    unittest.main()
