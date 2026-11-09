"""Read raw state/action demonstrations from NumPy episodes or Liu's pickles."""

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def validate_demonstration_environment(path, env_id, env_config, backend):
    """Reject collector datasets generated under different physics/task settings."""
    manifest = Path(path) / 'manifest.json'
    if not manifest.exists():
        return  # Liu's released pickles predate the collector manifest.
    dataset = json.loads(manifest.read_text())
    expected = {'backend': backend,
                **{key: env_config[key] for key in ('cost_boundary', 'max_episode_steps')}}
    recorded = dataset['env_cfgs']
    if dataset['env_id'] != env_id:
        raise ValueError(f"Expert dataset environment {dataset['env_id']} differs from {env_id}.")
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise ValueError(f'Expert dataset {key}={recorded.get(key)!r} differs from training {value!r}.')


def load_trajectory(path):
    """Return raw pre-action observations and actions from one episode."""
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as episode:
            return {key: episode[key] for key in ('observations', 'actions')}
    if path.suffix == '.pkl':
        episode = pickle.loads(path.read_bytes())
        return {'observations': np.asarray(episode['original_observations']),
                'actions': np.asarray(episode['actions'])}
    raise ValueError(f'Expected a .npz or .pkl trajectory: {path}')


def load_demonstrations(path, count):
    """Select the first N lexicographic episodes, preserving Liu's ordering."""
    path = Path(path)
    numpy_files, pickle_files = sorted(path.glob('*.npz')), sorted(path.glob('*.pkl'))
    if numpy_files and pickle_files:
        raise ValueError(f'Keep NumPy and Liu pickle demonstrations in separate directories: {path}')
    files = (numpy_files or pickle_files)[:count]
    if len(files) != count:
        raise FileNotFoundError(f'Expected {count} expert demonstrations in {path}')
    trajectories = [load_trajectory(file) for file in files]
    return (np.concatenate([item['observations'] for item in trajectories]),
            np.concatenate([item['actions'] for item in trajectories]),
            {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files})
