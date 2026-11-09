"""Load one complete ICRL experiment; command-line values are optional overrides."""

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
METHODS = ('MECL', 'VICRL')
TASKS = ('HC', 'Ant', 'Walker', 'Swimmer')


def _path(value, directory):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


def load_experiment_config(config_path=None, *, method=None, task=None, overrides=None):
    """Return the resolved YAML and its path, without creating files or simulators.

    Paths in YAML are relative to that YAML. Explicit command-line path overrides
    are relative to the working directory, matching ordinary terminal commands.
    """
    if config_path is None:
        if method is None or task is None:
            raise ValueError('Provide --config PATH, or both --method and --task.')
        config_path = ROOT / 'config' / f'{method}_{task}.yaml'
    source = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict):
        raise ValueError(f'Expected an experiment mapping in {source}.')
    config = deepcopy(config)
    if method is not None and config['method'] != method:
        raise ValueError('--method conflicts with the selected config.')
    if task is not None and config['task'] != task:
        raise ValueError('--task conflicts with the selected config.')
    if config['method'] not in METHODS or config['task'] not in TASKS:
        raise ValueError('Experiment method/task must select MECL/VICRL and HC/Ant/Walker/Swimmer.')
    expected_group = 'VICRL' if config['method'] == 'VICRL' else 'ICRL'
    if config['group'] != expected_group:
        raise ValueError(f"method {config['method']} requires group {expected_group}.")

    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    run = config['run']
    for key in ('seed', 'constraint_seed', 'sampling_seed', 'workers', 'backend',
                'solver', 'torch_threads', 'output'):
        if key in overrides:
            run[key] = overrides[key]
    if 'device' in overrides:
        config['device'] = overrides['device']
    for key, section, name in (('iterations', 'running', 'n_iters'),
                               ('forward_steps', 'PPO', 'forward_timesteps')):
        if key in overrides:
            config[section][name] = overrides[key]

    if run.get('solver') != 'omnisafe':
        raise ValueError('run.solver must be omnisafe; only native OmniSafe PPO-Lag is supported.')
    if not isinstance(config.get('clip_actions'), bool):
        raise ValueError('clip_actions must be an explicit YAML boolean (true or false).')
    if 'rollout_steps' in overrides:
        config['omnisafe']['algo_cfgs']['steps_per_epoch'] = overrides['rollout_steps']
    native = config['omnisafe']
    if run['workers'] != 1 or native['train_cfgs']['vector_env_nums'] != 1 or native['train_cfgs']['parallel'] != 1:
        raise ValueError('The ICRL runner currently uses one environment and one process.')
    if config['PPO']['forward_timesteps'] % native['algo_cfgs']['steps_per_epoch']:
        raise ValueError('PPO.forward_timesteps must be divisible by omnisafe.algo_cfgs.steps_per_epoch.')
    if run['backend'] != 'modern':
        raise ValueError('Only the installed Gymnasium/MuJoCo backend is supported; run.backend must be modern.')
    dataset = overrides.get('expert_data', config['running']['expert_path'])
    config['running']['expert_path'] = str(_path(dataset, Path.cwd() if 'expert_data' in overrides else source.parent))
    if run.get('output') is None:
        run['output'] = str(ROOT / 'runs' / f"{config['method']}_{config['task']}_seed{run['seed']}")
    else:
        run['output'] = str(_path(run['output'], Path.cwd() if 'output' in overrides else source.parent))
    if config.get('reference_config') is not None:
        config['reference_config'] = str(_path(config['reference_config'], source.parent))
    return config, source
