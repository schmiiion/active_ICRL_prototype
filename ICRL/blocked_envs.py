"""Liu et al. (ICLR 2023) Blocked tasks on OmniSafe's installed MuJoCo runtime.

Task rewards, observations, reset sampling and positional constraints follow
the benchmark; simulation uses Gymnasium and the installed ``mujoco`` package.
Import this module before ``omnisafe.envs.make`` to register the four IDs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv

from omnisafe.envs.core import CMDP, env_register


ASSET_DIR = Path(__file__).with_name('env_assets')
HORIZONS = {'HCWithPos-v0': 1000, 'AntWall-v0': 500,
            'WalkerWithPos-v0': 500, 'SwimmerWithPos-v0': 500}


def benchmark_random_state(seed: int | None = None) -> np.random.RandomState:
    """Benchmark reset RNG, preserving the seeds used by the trained experts."""
    if seed is None:
        seed = int.from_bytes(os.urandom(8), 'little')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer or None')
    seed = int(seed) % (2**64)
    digest = hashlib.sha512(str(seed).encode('utf8')).digest()[:8]
    value = int.from_bytes(digest, 'little')
    words = []
    while value:
        value, word = divmod(value, 2**32)
        words.append(word)
    rng = np.random.RandomState()
    rng.seed(words or [0])
    return rng


def true_cost(env_id: str, observation: np.ndarray, boundary: float | None = None) -> np.ndarray:
    """Original strict positional predicate, evaluated on the POST-step state."""
    if env_id not in HORIZONS:
        raise ValueError(f'Unknown Liu environment: {env_id}')
    if boundary is None:
        boundary = 0.5 if env_id == 'SwimmerWithPos-v0' else -3.0
    x = np.asarray(observation)[..., 0]
    return (x > boundary if env_id == 'SwimmerWithPos-v0' else x < boundary).astype(np.float32)


class _BenchmarkReset:
    """Benchmark reset distributions and stepping on the installed MuJoCo engine."""

    _reset_scale = 0.1
    _normal_velocity_noise = False

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
        # Additional contact postprocessing changes Ant's task observations
        # and penalty. Preserve the benchmark's sequence of mj_step calls.

    def reset(self, *, seed=None, options=None):
        if seed is not None or not hasattr(self, '_benchmark_rng'):
            self._benchmark_rng = benchmark_random_state(seed)
        self._np_random = self._benchmark_rng
        return super().reset(seed=None, options=options)

    def reset_model(self):
        scale = self._reset_scale
        qpos = self.init_qpos + self.np_random.uniform(-scale, scale, self.model.nq)
        noise = (self.np_random.standard_normal(self.model.nv) * scale
                 if self._normal_velocity_noise
                 else self.np_random.uniform(-scale, scale, self.model.nv))
        self.set_state(qpos, self.init_qvel + noise)
        return self._get_obs()


class _PositionObservation(_BenchmarkReset):
    def _get_obs(self):
        # In particular Walker velocities are NOT clipped, unlike Walker2d-v4.
        return np.concatenate((self.data.qpos.ravel(), self.data.qvel.ravel())).copy()

    def _init_mujoco(self, xml_file, frame_skip, obs_dim, **kwargs):
        MujocoEnv.__init__(self, str(ASSET_DIR / xml_file), frame_skip,
                          observation_space=spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64),
                          **kwargs)


class BlockedHalfCheetah(_PositionObservation, HalfCheetahEnv):
    """HalfCheetahWithPos, REWARD_TYPE='old'; full 18-dimensional state."""

    _normal_velocity_noise = True

    def __init__(self, **kwargs):
        self._init_mujoco('half_cheetah.xml', 5, 18, **kwargs)

    def step(self, action):
        before = float(self.data.qpos[0])
        self.do_simulation(action, self.frame_skip)
        after = float(self.data.qpos[0])
        ctrl = -0.1 * np.square(action).sum()
        run = abs(after - before) / self.dt
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(), run + ctrl, False, False, {
            'reward_run': run, 'reward_ctrl': ctrl, 'xpos': after,
        }


class BlockedWalker(_PositionObservation, Walker2dEnv):
    """WalkerWithPos; absolute speed, alive bonus, original fall termination."""

    _reset_scale = 0.005

    def __init__(self, **kwargs):
        self._init_mujoco('walker2d.xml', 4, 18, **kwargs)

    def step(self, action):
        before = float(self.data.qpos[0])
        self.do_simulation(action, self.frame_skip)
        after, height, angle = self.data.qpos[:3]
        ctrl = -1e-3 * np.square(action).sum()
        run = abs(after - before) / self.dt
        terminated = not (0.8 < height < 2.0 and -1.0 < angle < 1.0)
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(), run + ctrl + 1.0, terminated, False, {
            'reward_run': run, 'reward_ctrl': ctrl, 'x_position': float(after),
        }


class BlockedSwimmer(_PositionObservation, SwimmerEnv):
    """SwimmerWithPos; full state and cost for x > +0.5, not x < -3."""

    def __init__(self, **kwargs):
        self._init_mujoco('swimmer.xml', 4, 10, **kwargs)

    def step(self, action):
        before = float(self.data.qpos[0])
        self.do_simulation(action, self.frame_skip)
        after = float(self.data.qpos[0])
        ctrl = -1e-4 * np.square(action).sum()
        run = abs(after - before) / self.dt
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(), run + ctrl, False, False, {
            'reward_run': run, 'reward_ctrl': ctrl, 'x_position': after,
        }


class BlockedAnt(_BenchmarkReset, AntEnv):
    """AntWall with its original XML, full positions, and contact observations."""

    _normal_velocity_noise = True

    def __init__(self, **kwargs):
        super().__init__(xml_file=str(ASSET_DIR / 'ant_circle.xml'),
                         healthy_reward=1.0, terminate_when_unhealthy=False,
                         reset_noise_scale=0.1,
                         exclude_current_positions_from_observation=False,
                         use_contact_forces=True, **kwargs)

    def step(self, action):
        before = self.get_body_com('torso')[:2].copy()
        self.do_simulation(action, self.frame_skip)
        after = self.get_body_com('torso')[:2].copy()
        vx, vy = abs(after - before) / self.dt
        ctrl, contact, healthy = self.control_cost(action), self.contact_cost, self.healthy_reward
        distance = np.linalg.norm(after, ord=2)
        reward = (distance + healthy) - (ctrl + contact)
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(), reward, False, False, {
            'reward_forward': vx, 'reward_ctrl': -ctrl, 'reward_contact': -contact,
            'reward_survive': healthy, 'x_position': after[0], 'y_position': after[1],
            'distance_from_origin': distance, 'x_velocity': vx, 'y_velocity': vy,
            'forward_reward': vx,
        }


RAW_CLASSES = {'HCWithPos-v0': BlockedHalfCheetah, 'AntWall-v0': BlockedAnt,
               'WalkerWithPos-v0': BlockedWalker, 'SwimmerWithPos-v0': BlockedSwimmer}


def make_raw_env(env_id: str, **kwargs) -> MujocoEnv:
    """Construct a task without normalization, time limits or automatic reset."""
    try:
        cls = RAW_CLASSES[env_id]
    except KeyError as exc:
        raise ValueError(f'Unknown Liu environment: {env_id}') from exc
    return cls(**kwargs)


@env_register
class LiuBlockedEnv(CMDP):
    """OmniSafe's six-tuple API; raw MuJoCo equations stay inside this package."""

    _support_envs = list(HORIZONS)
    need_time_limit_wrapper = True
    need_auto_reset_wrapper = True
    need_evaluation = False

    def __init__(self, env_id: str, num_envs: int = 1, device: str = 'cpu',
                 backend: str = 'modern',
                 cost_boundary: float | None = None, max_episode_steps: int | None = None,
                 **kwargs: Any):
        super().__init__(env_id)
        if num_envs != 1:
            raise ValueError('LiuBlockedEnv requires num_envs=1; use independent workers for seeds.')
        if backend != 'modern':
            raise ValueError('Only the installed MuJoCo backend (modern) is supported.')
        self._env_id = env_id
        self._cost_boundary = cost_boundary
        self._max_episode_steps = HORIZONS[env_id] if max_episode_steps is None else max_episode_steps
        self._num_envs = 1
        self._device = torch.device(device)
        self._env = make_raw_env(env_id, **kwargs)
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self._metadata = self._env.metadata

    @property
    def raw_env(self):
        return self._env

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action.detach().cpu().numpy())
        cost = true_cost(self._env_id, obs, self._cost_boundary)
        info['lag_cost'] = int(cost)
        values = (obs, reward, cost)
        obs_t, reward_t, cost_t = (
            torch.as_tensor(value, dtype=torch.float32, device=self._device) for value in values
        )
        return (obs_t, reward_t, cost_t,
                torch.as_tensor(terminated, dtype=torch.bool, device=self._device),
                torch.as_tensor(truncated, dtype=torch.bool, device=self._device), info)

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed):
        self.reset(seed=seed)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
