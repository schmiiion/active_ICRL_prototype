"""OmniSafe training environments whose costs come from an ICRL constraint network."""

from __future__ import annotations

from collections import deque
import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.envs.core import CMDP, Wrapper
from omnisafe.envs.wrapper import (
    ActionScale,
    AutoReset,
    CostNormalize,
    ObsNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)
from omnisafe.utils.config import Config

from ICRL.action_wrapper import ActionClip


CostFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


class LearnedCostWrapper(Wrapper):
    """Evaluate costs on raw pre-action observations and policy-coordinate actions.

    The callback receives NumPy batches with a leading environment dimension, including
    when there is only one environment. Native costs remain available as ``true_cost``.
    """

    def __init__(
        self,
        env: CMDP,
        device: torch.device,
        cost_function: CostFunction | None = None,
    ) -> None:
        super().__init__(env=env, device=device)
        self._cost_function: CostFunction | None = None
        self._raw_obs: torch.Tensor | None = None
        self.episode_callback = None
        self.completed_episodes = {}
        self.worker_episodes = np.zeros(self.num_envs, dtype=int)
        self.total_transitions = 0
        self._reset_episode_statistics()
        if cost_function is not None:
            self.set_cost_function(cost_function)

    @property
    def has_cost_function(self) -> bool:
        """Whether a learned cost is attached; there is no native-cost fallback."""
        return self._cost_function is not None

    def set_cost_function(self, cost_function: CostFunction) -> None:
        """Replace the callback without resetting the environment or normalizers."""
        if not callable(cost_function):
            raise TypeError('cost_function must be callable.')
        self._cost_function = cost_function

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Remember the raw observation used for the next state-action cost."""
        obs, info = super().reset(seed=seed, options=options)
        self._raw_obs = obs.detach().clone()
        self._reset_episode_statistics()
        return obs, info

    def _reset_episode_statistics(self):
        self.episode_reward = np.zeros(self.num_envs)
        self.episode_feasible_reward = np.zeros(self.num_envs)
        self.episode_true_cost = np.zeros(self.num_envs)
        self.episode_learned_cost = np.zeros(self.num_envs)
        self.episode_violation = np.zeros(self.num_envs, dtype=bool)
        self.episode_length = np.zeros(self.num_envs, dtype=int)
        self.rollout_transitions = 0
        self.completed_episodes = {}

    def _record_transition(self, reward, true_cost, learned_cost, terminated, truncated):
        raw_reward = reward.detach().cpu().numpy().reshape(self.num_envs)
        raw_true_cost = true_cost.detach().cpu().numpy().reshape(self.num_envs)
        raw_learned_cost = learned_cost.detach().cpu().numpy().reshape(self.num_envs)
        self.rollout_transitions += self.num_envs
        self.total_transitions += self.num_envs
        self.episode_length += 1
        self.episode_reward += raw_reward
        self.episode_true_cost += raw_true_cost
        self.episode_learned_cost += raw_learned_cost
        self.episode_violation |= raw_true_cost > 0
        # Liu's reward_nc stops before the first violating transition.
        self.episode_feasible_reward += np.where(self.episode_violation, 0, raw_reward)
        self.completed_episodes = {}
        done = (terminated | truncated).detach().cpu().numpy().reshape(self.num_envs)
        for idx in np.flatnonzero(done):
            self.worker_episodes[idx] += 1
            row = dict(worker=int(idx), worker_episode=int(self.worker_episodes[idx]),
                       reward=float(self.episode_reward[idx]),
                       reward_nc=float(self.episode_feasible_reward[idx]),
                       true_cost=float(self.episode_true_cost[idx]),
                       learned_cost=float(self.episode_learned_cost[idx]),
                       constraint=int(self.episode_violation[idx]), length=int(self.episode_length[idx]))
            self.completed_episodes[int(idx)] = row
            if self.episode_callback is not None:
                self.episode_callback(row, self.total_transitions)
            for values in (self.episode_reward, self.episode_feasible_reward, self.episode_true_cost,
                           self.episode_learned_cost, self.episode_violation, self.episode_length):
                values[idx] = 0

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Substitute the learned cost before outer normalization wrappers run."""
        if self._cost_function is None:
            raise RuntimeError('Call set_cost_function() before stepping the training environment.')
        if self._raw_obs is None:
            raise RuntimeError('Call reset() before stepping the training environment.')

        # Calculate c(s_t, a_t), before stepping or possibly auto-resetting the simulator.
        obs_batch = self._raw_obs.reshape(self.num_envs, *self.observation_space.shape)
        action_batch = action.reshape(self.num_envs, *self.action_space.shape)
        with torch.no_grad():
            costs = self._cost_function(
                obs_batch.cpu().numpy().copy(),
                action_batch.detach().cpu().numpy().copy(),
            )
            learned_cost = torch.as_tensor(costs, dtype=torch.float32, device=self._device)
        if learned_cost.numel() != self.num_envs:
            raise ValueError('cost_function must return one cost per environment.')
        if not torch.isfinite(learned_cost).all():
            raise ValueError('cost_function returned a non-finite cost.')

        obs, reward, true_cost, terminated, truncated, info = super().step(action)
        learned_cost = learned_cost.detach().reshape_as(true_cost)
        info['true_cost'] = true_cost
        # OmniSafe logs original_cost and uses its episode sum for the Lagrange update.
        info['unscaled_learned_cost'] = learned_cost
        info['original_cost'] = learned_cost
        self._record_transition(reward, true_cost, learned_cost, terminated, truncated)
        # AutoReset is inside this wrapper: obs is already the next episode's initial
        # observation when an episode ended, not info['final_observation'].
        self._raw_obs = obs.detach().clone()
        return obs, reward, learned_cost, terminated, truncated, info


class LearnedCostAdapter(OnPolicyAdapter):
    """Keep OmniSafe's rollout machinery while replacing only the training cost."""

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
        cost_function: CostFunction | None = None,
    ) -> None:
        if cost_function is not None and not callable(cost_function):
            raise TypeError('cost_function must be callable.')
        super().__init__(env_id=env_id, num_envs=num_envs, seed=seed, cfgs=cfgs)
        if cost_function is not None:
            self.set_cost_function(cost_function)

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        """Place learned costs outside action scaling and inside all normalizers."""
        if self._env.need_time_limit_wrapper:
            if not self._env.max_episode_steps:
                raise ValueError('The environment must define max_episode_steps.')
            self._env = TimeLimit(
                self._env,
                time_limit=self._env.max_episode_steps,
                device=self._device,
            )
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)

        # The callback and nominal sampler use the same policy-coordinate actions.
        self._env = ActionScale(self._env, low=-1.0, high=1.0, device=self._device)
        self._learned_cost_wrapper = LearnedCostWrapper(
            self._env,
            device=self._device,
        )
        self._env = ActionClip(
            self._learned_cost_wrapper, device=self._device,
            clip_actions=self._cfgs.get('icrl_cfgs', {}).get('clip_actions', False),
        )
        if obs_normalize:
            self._env = ObsNormalize(self._env, device=self._device)
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)
        # ICRL owns a separate, shared sampling/evaluation environment. Prevent
        # OnlineAdapter.__init__ from allocating an unused second simulator here.
        self._env.need_evaluation = False

    @property
    def has_cost_function(self) -> bool:
        """Whether training can use a learned cost."""
        return self._learned_cost_wrapper.has_cost_function

    def set_cost_function(self, cost_function: CostFunction) -> None:
        """Attach a cost callback; in-place network updates need no reattachment."""
        self._learned_cost_wrapper.set_cost_function(cost_function)

    def configure_episode_logging(self, path, method, seed, writer=None, count_steps=lambda: 0):
        """Write raw completed episodes; TensorBoard's Figure2 axis is episode count."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._episode_stream = path.open('w', newline='')
        fields = ['method', 'env', 'seed', 'episode', 'worker', 'worker_episode', 'reward',
                  'reward_nc', 'constraint', 'length', 'total_steps', 'true_cost', 'learned_cost']
        csv_writer = csv.DictWriter(self._episode_stream, fieldnames=fields)
        csv_writer.writeheader()
        recent = deque(maxlen=101)
        episode_count = 0
        start_step = count_steps()
        start_transitions = self._learned_cost_wrapper.total_transitions

        def record(values, transitions):
            nonlocal episode_count
            episode_count += 1
            step = start_step + transitions - start_transitions
            row = dict(method=method, env=self._env_id, seed=seed, episode=episode_count,
                       total_steps=step, **values)
            csv_writer.writerow(row)
            self._episode_stream.flush()
            recent.append(values)
            if writer is not None:
                for tag, column in [('Reward', 'reward'), ('FeasibleReward', 'reward_nc'),
                                    ('TrueCost', 'true_cost'), ('LearnedCost', 'learned_cost'),
                                    ('Violation', 'constraint'), ('Length', 'length')]:
                    writer.add_scalar(f'Episodes/{tag}', values[column], step)
                writer.add_scalar('Figure2/ConstraintViolationRaw', values['constraint'], episode_count - 1)
                writer.add_scalar('Figure2/FeasibleRewardRaw', values['reward_nc'], episode_count - 1)
                if len(recent) == 101:
                    # Match Liu's forward 100-episode mean and discarded last window.
                    window = list(recent)[:100]
                    writer.add_scalar('Figure2/ConstraintViolationRate',
                                      np.mean([r['constraint'] for r in window]), episode_count - 101)
                    writer.add_scalar('Figure2/FeasibleReward',
                                      np.mean([round(r['reward_nc'], 2) for r in window]), episode_count - 101)

        self._learned_cost_wrapper.episode_callback = record

    def _log_metrics(self, logger, idx):
        super()._log_metrics(logger, idx)
        row = self._learned_cost_wrapper.completed_episodes[idx]
        logger.store({'Metrics/TrueEpCost': row['true_cost'],
                      'Metrics/FeasibleEpRet': row['reward_nc'],
                      'Metrics/ConstraintViolationRate': row['constraint']})

    def close(self):
        try:
            super().close()
        finally:
            if hasattr(self, '_episode_stream'):
                self._episode_stream.close()
