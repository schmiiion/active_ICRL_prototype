"""Liu's VecNormalizeWithCost semantics, isolated from OmniSafe's normalizers.

Observations are centered; rewards/costs are divided by the standard deviation
of discounted returns, without centering. All statistics use float64.
"""

from copy import deepcopy

import numpy as np


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, values):
        mean, var, count = values.mean(axis=0), values.var(axis=0), len(values)
        delta = mean - self.mean
        total = self.count + count
        self.mean = self.mean + delta * count / total
        self.var = (self.var * self.count + var * count
                    + delta**2 * self.count * count / total) / total
        self.count = total


class LiuNormalize:
    # Class defaults also keep the completed reproduction's pickled normalizer readable.
    clip_obs = clip_reward = clip_cost = 10.0
    epsilon = 1e-8

    def __init__(self, obs_shape, num_envs, config, training=True):
        self.obs_rms = RunningMeanStd(obs_shape)
        self.ret_rms, self.cost_rms = RunningMeanStd(), RunningMeanStd()
        self.ret, self.cost_ret = np.zeros(num_envs), np.zeros(num_envs)
        self.training = training
        self.norm_obs = not config['dont_normalize_obs']
        self.norm_reward = training and not config['dont_normalize_reward']
        self.norm_cost = training and not config['dont_normalize_cost']
        self.gamma, self.cost_gamma = config['reward_gamma'], config['cost_gamma']
        self.clip_obs = config.get('clip_obs', 10.0)
        self.clip_reward = config.get('clip_reward', 10.0)
        self.clip_cost = config.get('clip_cost', 10.0)
        self.epsilon = config.get('normalization_epsilon', 1e-8)

    def observation(self, obs):
        if not self.norm_obs:
            return obs.copy()
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                       -self.clip_obs, self.clip_obs)

    def reset(self, obs):
        self.ret[:] = 0
        self.cost_ret[:] = 0
        # The original updates return statistics, but NOT observation statistics,
        # on reset. Preserve this seemingly unusual initialization.
        if self.training:
            self.ret_rms.update(self.ret)
            self.cost_rms.update(self.cost_ret)
        return self.observation(obs)

    def step(self, obs, reward, cost, done):
        if self.training:
            self.obs_rms.update(obs)
            self.ret = self.ret * self.gamma + reward
            self.ret_rms.update(self.ret)
            self.cost_ret = self.cost_ret * self.cost_gamma + cost
            self.cost_rms.update(self.cost_ret)
        norm_reward = (np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon),
                               -self.clip_reward, self.clip_reward)
                       if self.norm_reward else reward.copy())
        norm_cost = (np.clip(cost / np.sqrt(self.cost_rms.var + self.epsilon),
                             -self.clip_cost, self.clip_cost)
                     if self.norm_cost else cost.copy())
        self.ret[done], self.cost_ret[done] = 0, 0
        return self.observation(obs), norm_reward, norm_cost

    def sync_from(self, source):
        # sync_envs_normalization copies these two, not cost_rms, in Liu's repo.
        self.obs_rms = deepcopy(source.obs_rms)
        self.ret_rms = deepcopy(source.ret_rms)
