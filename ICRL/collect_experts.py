"""Export complete NumPy trajectories from a native OmniSafe actor snapshot.

Run: python -m ICRL.collect_experts --config ICRL/config/collect_experts_HC.yaml
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

import ICRL.blocked_envs  # Register the project's custom tasks.
from omnisafe.envs.core import make
from omnisafe.envs.wrapper import ActionScale, TimeLimit
from omnisafe.models.actor.actor_builder import ActorBuilder

from ICRL.action_wrapper import ActionClip


def load_snapshot(run_dir, checkpoint, *, clip_actions=None):
    """Restore a snapshot, inheriting its action setting unless explicitly overridden."""
    training = json.loads((run_dir / 'config.json').read_text())
    if clip_actions is None:
        clip_actions = training.get('icrl_cfgs', {}).get('clip_actions', False)
    if not isinstance(clip_actions, bool):
        raise ValueError('clip_actions must be a YAML boolean (true or false).')
    saved = torch.load(run_dir / 'torch_save' / checkpoint, map_location='cpu', weights_only=False)
    env = make(training['env_id'], num_envs=1, device='cpu', **training.get('env_cfgs', {}))
    try:
        device = torch.device('cpu')
        if env.need_time_limit_wrapper:
            env = TimeLimit(env, time_limit=env.max_episode_steps, device=device)
        env = ActionScale(env, low=-1.0, high=1.0, device=device)
        env = ActionClip(env, device=device, clip_actions=clip_actions)
        model = training['model_cfgs']
        actor = ActorBuilder(env.observation_space, env.action_space,
                             model['actor']['hidden_sizes'], model['actor']['activation'],
                             model['weight_initialization_mode']).build_actor(model['actor_type'])
        actor.load_state_dict(saved['pi'])
        actor.eval()
        normalizer = saved['obs_normalizer'] if training['algo_cfgs']['obs_normalize'] else None
        return env, actor, normalizer, training
    except BaseException:
        env.close()
        raise


@torch.no_grad()
def rollout_episode(env, actor, normalizer, seed, deterministic=False):
    """Record (s_t, a_t, r_t, c_t, s_{t+1}), retaining the true terminal state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    trajectory = {key: [] for key in ('observations', 'actions', 'next_observations',
                                     'rewards', 'costs', 'terminated', 'truncated')}
    while True:
        policy_obs = obs
        if normalizer is not None:
            # Read checkpoint statistics directly: normalization must not update them.
            policy_obs = torch.clamp((obs - normalizer['_mean']) / normalizer['_std'],
                                     min=-normalizer['_clip'], max=normalizer['_clip'])
        action = actor.predict(policy_obs.unsqueeze(0), deterministic=deterministic).squeeze(0)
        trajectory['observations'].append(obs.detach().cpu().numpy().copy())
        next_obs, reward, cost, terminated, truncated, _ = env.step(action)
        executed = env.last_action if isinstance(env, ActionClip) else action
        trajectory['actions'].append(executed.detach().cpu().numpy().copy())
        for key, value in zip(('next_observations', 'rewards', 'costs', 'terminated', 'truncated'),
                              (next_obs, reward, cost, terminated, truncated)):
            trajectory[key].append(value.detach().cpu().numpy().copy())
        obs = next_obs
        if bool(terminated or truncated):
            return {key: np.stack(values) for key, values in trajectory.items()}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(config):
    """Collect the requested accepted episodes, with a manifest of all attempts."""
    run_dir, output = Path(config['run_dir']), Path(config['output'])
    torch.set_num_threads(config['torch_threads'])
    if not 0 < config['episodes'] <= config['max_attempts']:
        raise ValueError('Require 0 < episodes <= max_attempts.')
    output.mkdir(parents=True, exist_ok=False)
    env, actor, normalizer, training = load_snapshot(
        run_dir, config['checkpoint'], clip_actions=config.get('clip_actions'))
    records, accepted = [], 0
    manifest = dict(
        checkpoint=str(run_dir / 'torch_save' / config['checkpoint']),
        checkpoint_sha256=sha256(run_dir / 'torch_save' / config['checkpoint']),
        training_config_sha256=sha256(run_dir / 'config.json'),
        collection=config, env_id=training['env_id'], env_cfgs=training.get('env_cfgs', {}),
        observation_normalization='frozen checkpoint statistics' if normalizer is not None else 'none',
        clip_actions=env.clip_actions,
        actions=('executed policy coordinates, clipped to action bounds before ActionScale'
                 if env.clip_actions else 'unclipped policy coordinates passed through ActionScale'),
        versions={name: version(name) for name in ('numpy', 'torch', 'gymnasium', 'mujoco')},
        attempts=records,
    )
    try:
        for attempt in range(config['max_attempts']):
            seed = config['seed'] + attempt
            episode = rollout_episode(env, actor, normalizer, seed, config['deterministic'])
            reward = float(episode['rewards'].sum(dtype=np.float64))
            cost = float(episode['costs'].sum(dtype=np.float64))
            keep = cost <= config['max_episode_cost'] and (
                config['min_episode_reward'] is None or reward >= config['min_episode_reward'])
            record = dict(seed=seed, reward=reward, cost=cost, length=len(episode['actions']), accepted=keep)
            if keep:
                file = output / f'trajectory-{accepted:06d}.npz'
                np.savez_compressed(file, **episode)
                record.update(file=file.name, sha256=sha256(file))
                accepted += 1
            records.append(record)
            print(f"attempt={attempt + 1} accepted={accepted}/{config['episodes']} "
                  f'reward={reward:.2f} cost={cost:g}', flush=True)
            if accepted == config['episodes']:
                break
    finally:
        env.close()
        manifest.update(accepted=accepted, attempted=len(records), complete=accepted == config['episodes'],
                        acceptance_rate=accepted / len(records) if records else 0.0)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if accepted < config['episodes']:
        raise RuntimeError(f"Collected {accepted}/{config['episodes']} acceptable episodes; see {output / 'manifest.json'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    source = args.config.expanduser().resolve()
    config = yaml.safe_load(source.read_text())
    for key in ('run_dir', 'output'):
        config[key] = str((source.parent / Path(config[key]).expanduser()).resolve())
    print(collect(config))


if __name__ == '__main__':
    main()
