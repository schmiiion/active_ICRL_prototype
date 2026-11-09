"""The selected YAML owns experiment settings unless an override is explicit."""

from pathlib import Path
import tempfile
import unittest

import yaml

from ICRL.experiment_config import METHODS, ROOT, TASKS, load_experiment_config


class ExperimentConfigTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='icrl-config-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'experiment.yaml'
        self.config = yaml.safe_load((ROOT / 'config' / 'VICRL_HC.yaml').read_text())

    def write_config(self):
        self.path.write_text(yaml.safe_dump(self.config))
        return self.path

    def test_every_method_environment_has_a_complete_config(self):
        for method in METHODS:
            for task in TASKS:
                with self.subTest(method=method, task=task):
                    config, source = load_experiment_config(method=method, task=task)
                    self.assertEqual(config['method'], method)
                    self.assertEqual(config['task'], task)
                    self.assertEqual(source.name, f'{method}_{task}.yaml')
                    self.assertEqual(config['run']['solver'], 'omnisafe')
                    self.assertEqual(config['run']['backend'], 'modern')
                    self.assertEqual(config['run']['workers'], 1)
                    self.assertEqual(config['omnisafe']['train_cfgs'], dict(vector_env_nums=1, parallel=1))
                    self.assertEqual(set(config['PPO']), {'forward_timesteps'})
                    self.assertEqual(set(config['env']), {
                        'train_env_id', 'eval_env_id', 'cost_boundary', 'max_episode_steps',
                    })
                    for section, values in self.config['omnisafe'].items():
                        self.assertEqual(set(config['omnisafe'][section]), set(values))
                    self.assertIs(config['clip_actions'], True)
                    self.assertTrue(Path(config['running']['expert_path']).is_dir())
                    self.assertTrue(Path(config['reference_config']).is_file())
                    self.assertTrue(Path(config['run']['output']).is_absolute())

    def test_native_lagrange_initializers_use_floating_point_values(self):
        # Native Lagrange constructs a gradient-tracked tensor without a dtype override.
        for method in METHODS:
            for task in TASKS:
                with self.subTest(method=method, task=task):
                    config, _ = load_experiment_config(method=method, task=task)
                    self.assertIsInstance(
                        config['omnisafe']['lagrange_cfgs']['lagrangian_multiplier_init'], float,
                    )

    def test_non_hc_native_baselines_preserve_task_and_reference_hyperparameters(self):
        for method in METHODS:
            for task in ('Ant', 'Walker', 'Swimmer'):
                with self.subTest(method=method, task=task):
                    config, _ = load_experiment_config(method=method, task=task)
                    reference = yaml.safe_load(Path(config['reference_config']).read_text())
                    ppo = reference['PPO']
                    native = config['omnisafe']
                    self.assertEqual(native['algo_cfgs']['steps_per_epoch'], 10000)
                    self.assertEqual(config['PPO']['forward_timesteps'], ppo['forward_timesteps'])
                    for field, original in (
                        ('update_iters', 'n_epochs'), ('batch_size', 'batch_size'),
                        ('target_kl', 'target_kl'), ('entropy_coef', 'ent_coef'),
                        ('max_grad_norm', 'max_grad_norm'), ('gamma', 'reward_gamma'),
                        ('cost_gamma', 'cost_gamma'), ('lam', 'reward_gae_lambda'),
                        ('lam_c', 'cost_gae_lambda'), ('clip', 'clip_range'),
                    ):
                        self.assertEqual(native['algo_cfgs'][field], ppo[original])
                    self.assertEqual(native['model_cfgs']['actor']['lr'], ppo['learning_rate'])
                    self.assertEqual(native['model_cfgs']['critic']['lr'], ppo['learning_rate'])
                    self.assertEqual(native['model_cfgs']['actor']['hidden_sizes'], ppo['policy_layers'])
                    self.assertEqual(native['model_cfgs']['critic']['hidden_sizes'], ppo['reward_vf_layers'])
                    self.assertEqual(native['model_cfgs']['critic']['hidden_sizes'], ppo['cost_vf_layers'])
                    self.assertEqual(native['lagrange_cfgs']['cost_limit'], ppo['budget'])
                    self.assertEqual(native['lagrange_cfgs']['lagrangian_multiplier_init'], ppo['penalty_initial_value'])
                    self.assertEqual(native['lagrange_cfgs']['lambda_lr'], ppo['penalty_learning_rate'])
                    for field in ('n_iters', 'n_eval_episodes', 'save_every', 'expert_rollouts',
                                  'sample_rollouts', 'store_sample_rollouts', 'use_buffer'):
                        self.assertEqual(config['running'][field], reference['running'][field])
                    for field in ('train_env_id', 'eval_env_id'):
                        self.assertEqual(config['env'][field], reference['env'][field])
                    self.assertEqual(config['env']['max_episode_steps'], 500)
                    self.assertEqual(config['env']['cost_boundary'], 0.5 if task == 'Swimmer' else -3.0)
                    self.assertEqual(Path(config['running']['expert_path']).name, f'Blocked{task}')
                    for field in ('cn_learning_rate', 'cn_reg_coeff', 'cn_layers', 'cn_batch_size',
                                  'clip_obs', 'cn_normalize', 'cn_target_kl_old_new',
                                  'cn_target_kl_new_old', 'train_gail_lambda', 'cn_eps',
                                  'backward_iters', 'anneal_clr_by_factor'):
                        self.assertEqual(config['CN'][field], reference['CN'][field])
                    if method == 'VICRL':
                        self.assertEqual(config['CN']['di_prior'], reference['CN']['di_prior'])
                        self.assertEqual(config['CN']['mode'], reference['CN']['mode'])
                    else:
                        self.assertEqual(config['CN']['no_importance_sampling'], reference['CN']['no_importance_sampling'])
                        self.assertEqual(config['CN']['per_step_importance_sampling'], reference['CN']['per_step_importance_sampling'])

    def test_action_clipping_can_be_disabled_explicitly(self):
        self.config['clip_actions'] = False
        config, _ = load_experiment_config(self.write_config())
        self.assertIs(config['clip_actions'], False)

    def test_action_clipping_must_be_explicit(self):
        del self.config['clip_actions']
        with self.assertRaisesRegex(ValueError, 'clip_actions must be an explicit YAML boolean'):
            load_experiment_config(self.write_config())

    def test_action_clipping_requires_a_yaml_boolean(self):
        for value in ('false', 0, 1, None):
            with self.subTest(value=value):
                self.config['clip_actions'] = value
                with self.assertRaisesRegex(ValueError, 'clip_actions'):
                    load_experiment_config(self.write_config())

    def test_absent_console_options_preserve_yaml_values(self):
        self.config['device'] = 'cuda:1'
        self.config['run'].update(seed=456, workers=1, backend='modern', torch_threads=2)
        config, _ = load_experiment_config(
            self.write_config(), overrides=dict(device=None, seed=None, workers=None, backend=None,
                                                torch_threads=None, iterations=None))
        self.assertEqual(config['device'], 'cuda:1')
        self.assertEqual(config['run']['seed'], 456)
        self.assertEqual(config['run']['workers'], 1)
        self.assertEqual(config['run']['backend'], 'modern')
        self.assertEqual(config['run']['torch_threads'], 2)
        self.assertEqual(config['running']['n_iters'], self.config['running']['n_iters'])

    def test_paths_are_relative_to_the_selected_yaml(self):
        self.config['running']['expert_path'] = 'datasets/nested/experts'
        self.config['run']['output'] = 'results/run'
        config, _ = load_experiment_config(self.write_config())
        self.assertEqual(Path(config['running']['expert_path']), (self.path.parent / 'datasets/nested/experts').resolve())
        self.assertEqual(Path(config['run']['output']), (self.path.parent / 'results/run').resolve())

    def test_explicit_overrides_are_recorded_in_resolved_config(self):
        config, _ = load_experiment_config(self.write_config(), overrides={
            'device': 'cpu', 'backend': 'modern', 'seed': 321, 'workers': 1,
            'iterations': 2, 'forward_steps': 4096, 'rollout_steps': 1024,
            'expert_data': 'custom/data', 'output': 'custom/results',
        })
        self.assertEqual(config['device'], 'cpu')
        self.assertEqual(config['run']['backend'], 'modern')
        self.assertEqual(config['run']['seed'], 321)
        self.assertEqual(config['run']['workers'], 1)
        self.assertEqual(config['running']['n_iters'], 2)
        self.assertEqual(config['PPO']['forward_timesteps'], 4096)
        self.assertEqual(config['omnisafe']['algo_cfgs']['steps_per_epoch'], 1024)
        self.assertEqual(config['running']['expert_path'], str(Path('custom/data').resolve()))
        self.assertEqual(config['run']['output'], str(Path('custom/results').resolve()))

    def test_unsupported_solver_and_conflicting_selectors_fail_clearly(self):
        for solver in ('liu', 'unsupported', None):
            with self.subTest(solver=solver):
                self.config['run']['solver'] = solver
                with self.assertRaisesRegex(ValueError, 'run.solver must be omnisafe'):
                    load_experiment_config(self.write_config())
        self.config['run']['solver'] = 'omnisafe'
        with self.assertRaisesRegex(ValueError, 'run.solver must be omnisafe'):
            load_experiment_config(self.write_config(), overrides={'solver': 'liu'})
        with self.assertRaisesRegex(ValueError, '--method conflicts'):
            load_experiment_config(self.write_config(), method='MECL')
        with self.assertRaisesRegex(ValueError, '--task conflicts'):
            load_experiment_config(self.path, task='Ant')
        self.config['group'] = 'ICRL'
        with self.assertRaisesRegex(ValueError, 'VICRL requires group VICRL'):
            load_experiment_config(self.write_config())
        with self.assertRaisesRegex(ValueError, 'Provide --config'):
            load_experiment_config()

    def test_other_simulator_backends_are_rejected(self):
        self.config['run']['backend'] = 'unsupported'
        with self.assertRaisesRegex(ValueError, 'run.backend must be modern'):
            load_experiment_config(self.write_config())

    def test_forward_budget_must_contain_complete_native_epochs(self):
        self.config['PPO']['forward_timesteps'] = 20001
        with self.assertRaisesRegex(ValueError, 'divisible'):
            load_experiment_config(self.write_config())

    def test_only_one_environment_and_one_process_are_supported(self):
        for section, field in ((self.config['run'], 'workers'),
                               (self.config['omnisafe']['train_cfgs'], 'vector_env_nums'),
                               (self.config['omnisafe']['train_cfgs'], 'parallel')):
            with self.subTest(field=field):
                section[field] = 2
                with self.assertRaisesRegex(ValueError, 'one environment and one process'):
                    load_experiment_config(self.write_config())
                section[field] = 1


if __name__ == '__main__':
    unittest.main()
