"""Raw-state cost replacement, SB3 normalization, and Figure 2 episode metrics."""

import csv
from pathlib import Path

import numpy as np

from ICRL.blocked_envs import HORIZONS, make_raw_env, true_cost
from ICRL.normalization import LiuNormalize


class LiuVectorEnv:
    """Synchronous equivalent of independent SubprocVecEnv workers.

    The constraint sees raw pre-step states and executed actions.
    The true constraint sees post-step states. Auto-reset precedes normalization.
    """
    def __init__(self, env_id, num_envs, seed, env_config, *, training=True,
                 cost_function=None, episode_file=None, method='', count_steps=lambda: 0,
                 backend='modern', clip_actions=True):
        self.env_id, self.num_envs = env_id, num_envs
        self.clip_actions = clip_actions
        self.cost_boundary = env_config.get('cost_boundary')
        self.max_episode_steps = env_config.get('max_episode_steps', HORIZONS[env_id])
        if backend != 'modern':
            raise ValueError('Only the installed Gymnasium/MuJoCo backend is supported (modern).')
        self.envs = [make_raw_env(env_id) for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.normalizer = LiuNormalize(self.observation_space.shape, num_envs, env_config, training)
        self.cost_function = cost_function
        self.initial_seeds = [seed + i for i in range(num_envs)]
        self.lengths = np.zeros(num_envs, int)
        self.returns = np.zeros(num_envs)
        self.feasible_returns = np.zeros(num_envs)
        self.violated = np.zeros(num_envs, bool)
        self.episodes, self.last_episodes = [], []
        self.worker_episodes = np.zeros(num_envs, int)
        self.raw_obs = None
        self.method, self.seed, self.count_steps = method, seed, count_steps
        self.output = None
        if episode_file is not None:
            Path(episode_file).parent.mkdir(parents=True, exist_ok=True)
            self.output = open(episode_file, 'w', newline='')
            self.writer = csv.DictWriter(self.output, fieldnames=[
                'method', 'env', 'seed', 'episode', 'worker', 'worker_episode', 'reward', 'reward_nc',
                'constraint', 'length', 'total_steps'])
            self.writer.writeheader()

    def reset(self):
        observations = []
        for i, env in enumerate(self.envs):
            observations.append(env.reset(seed=self.initial_seeds[i])[0])
            self.initial_seeds[i] = None
        self.raw_obs = np.stack(observations)
        self.lengths[:] = 0
        self.returns[:] = 0
        self.feasible_returns[:] = 0
        self.violated[:] = False
        return self.normalizer.reset(self.raw_obs)

    def step(self, actions):
        obs, rewards, dones = [], [], []
        self.last_episodes = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            violation = bool(true_cost(self.env_id, next_obs, self.cost_boundary))
            self.lengths[i] += 1
            done = terminated or truncated or self.lengths[i] >= self.max_episode_steps
            self.returns[i] += reward
            self.violated[i] |= violation
            if not self.violated[i]:
                self.feasible_returns[i] += reward
            if done:
                self.worker_episodes[i] += 1
                row = dict(method=self.method, env=self.env_id, seed=self.seed,
                           episode=len(self.episodes) + 1, worker=i,
                           worker_episode=int(self.worker_episodes[i]), reward=self.returns[i],
                           reward_nc=self.feasible_returns[i], constraint=int(self.violated[i]),
                           length=int(self.lengths[i]), total_steps=self.count_steps() + self.num_envs)
                self.episodes.append(row)
                self.last_episodes.append((i, row))
                if self.output is not None:
                    self.writer.writerow(row)
                    self.output.flush()
                next_obs = env.reset()[0]
                self.lengths[i], self.returns[i], self.feasible_returns[i], self.violated[i] = 0, 0, 0, False
            obs.append(next_obs)
            rewards.append(reward)
            dones.append(done)
        if self.cost_function is None:
            raise RuntimeError('Attach a state-action cost callback before stepping.')
        cost = np.asarray(self.cost_function(self.raw_obs.copy(), np.array(actions, copy=True))).reshape(self.num_envs)
        self.last_raw_cost = cost.copy()
        self.raw_obs = np.stack(obs)
        return (*self.normalizer.step(self.raw_obs, np.asarray(rewards), cost, np.asarray(dones)), np.asarray(dones))

    def close(self):
        for env in self.envs:
            env.close()
        if self.output is not None:
            self.output.close()


def sample_trajectories(learner, env, count, deterministic=False):
    """Liu's single-thread sampler records raw pre-step state + clipped action."""
    obs = env.reset()
    obs_games, action_games, reward_games, lengths = [], [], [], []
    for _ in range(count):
        observations, actions, rewards = [], [], []
        done = False
        while not done:
            action = learner.predict(obs, deterministic=deterministic)
            observations.append(env.raw_obs[0].copy())
            actions.append(action[0].copy())
            obs, reward, _, dones = env.step(action)
            rewards.append(reward[0])
            done = bool(dones[0])
        obs_games.append(np.asarray(observations))
        action_games.append(np.asarray(actions))
        reward_games.append(np.asarray(rewards))
        lengths.append(len(observations))
    return obs_games, action_games, reward_games, lengths


def evaluate(learner, env, episodes, deterministic=False):
    start = len(env.episodes)
    obs = env.reset()
    while len(env.episodes) - start < episodes:
        obs, _, _, _ = env.step(learner.predict(obs, deterministic=deterministic))
    records = env.episodes[start:]
    return {key: float(np.mean([r[column] for r in records])) for key, column in
            [('reward', 'reward'), ('feasible_reward', 'reward_nc'), ('violation_rate', 'constraint')]}
