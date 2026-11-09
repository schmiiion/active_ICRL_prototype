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
                    expected_solver = 'omnisafe' if task == 'HC' else 'liu'
                    self.assertEqual(config['run']['solver'], expected_solver)
                    self.assertEqual(config['run']['backend'], 'modern')
                    self.assertIs(config['clip_actions'], True)
                    self.assertTrue(Path(config['running']['expert_path']).is_dir())
                    self.assertTrue(Path(config['reference_config']).is_file())
                    self.assertTrue(Path(config['run']['output']).is_absolute())

    def test_action_clipping_can_be_disabled_and_historical_configs_keep_their_behavior(self):
        self.config['clip_actions'] = False
        config, _ = load_experiment_config(self.write_config())
        self.assertIs(config['clip_actions'], False)
        del self.config['clip_actions']
        for solver, expected in [('omnisafe', False), ('liu', True)]:
            with self.subTest(solver=solver):
                self.config['run']['solver'] = solver
                config, _ = load_experiment_config(self.write_config())
                self.assertIs(config['clip_actions'], expected)

    def test_action_clipping_requires_a_yaml_boolean(self):
        self.config['clip_actions'] = 'false'
        with self.assertRaisesRegex(ValueError, 'clip_actions'):
            load_experiment_config(self.write_config())

    def test_absent_console_options_preserve_yaml_values(self):
        self.config['run']['solver'] = 'liu'
        self.config['device'] = 'cuda:1'
        self.config['run'].update(seed=456, workers=3, backend='modern', torch_threads=2)
        config, _ = load_experiment_config(
            self.write_config(), overrides=dict(device=None, seed=None, workers=None, backend=None,
                                                torch_threads=None, iterations=None))
        self.assertEqual(config['device'], 'cuda:1')
        self.assertEqual(config['run']['seed'], 456)
        self.assertEqual(config['run']['workers'], 3)
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
        self.config['run']['solver'] = 'unsupported'
        with self.assertRaisesRegex(ValueError, 'run.solver must be liu or omnisafe'):
            load_experiment_config(self.write_config())
        self.config['run']['solver'] = 'liu'
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

    def test_native_solver_checks_forward_budget_and_single_environment(self):
        self.config['PPO']['forward_timesteps'] = 20001
        with self.assertRaisesRegex(ValueError, 'divisible'):
            load_experiment_config(self.write_config())
        self.config['PPO']['forward_timesteps'] = 20000
        self.config['run']['workers'] = 2
        with self.assertRaisesRegex(ValueError, 'one environment'):
            load_experiment_config(self.write_config())


if __name__ == '__main__':
    unittest.main()
