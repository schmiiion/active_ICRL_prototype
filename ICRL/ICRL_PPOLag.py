"""A persistent PPO-Lag learner for the inner loop of ICRL."""

from __future__ import annotations

import math
import os
import time
from copy import deepcopy
from typing import Any

import torch

from ICRL.learned_cost_adapter import CostFunction, LearnedCostAdapter
from omnisafe.algorithms import PPOLag
from omnisafe.models.base import Actor
from omnisafe.utils.config import Config


class ICRL_PPOLag(PPOLag):
    """Continue PPO-Lag across constraint updates without rebuilding its state.

    ``cfgs`` is a complete OmniSafe PPOLag configuration. ``total_steps`` describes
    the entire outer-loop run, so learning-rate schedules keep progressing
    across calls to ``train_for``. This implementation runs in one process.
    """

    def __init__(
        self,
        env_id: str,
        cfgs: Config,
        cost_function: CostFunction | None = None,
    ) -> None:
        cfgs = deepcopy(cfgs)
        steps_per_epoch = cfgs.algo_cfgs.steps_per_epoch
        total_steps = cfgs.train_cfgs.total_steps
        cfgs.train_cfgs.epochs = total_steps // steps_per_epoch

        num_envs = cfgs.train_cfgs.vector_env_nums
        if not isinstance(num_envs, int) or num_envs <= 0 or steps_per_epoch % num_envs:
            raise ValueError('steps_per_epoch must be divisible by vector_env_nums.')

        # These runtime settings are normally applied by omnisafe.Agent.
        torch.set_num_threads(cfgs.train_cfgs.torch_threads)
        os.environ['OMNISAFE_DEVICE'] = str(cfgs.train_cfgs.device)
        self._initial_cost_function = cost_function
        self._epochs_completed = 0
        self._start_time: float | None = None
        self._closed = False
        self.last_metrics = {}
        super().__init__(env_id=env_id, cfgs=cfgs)

    def _init_log(self):
        super()._init_log()
        for key in ('Metrics/TrueEpCost', 'Metrics/FeasibleEpRet', 'Metrics/ConstraintViolationRate'):
            self._logger.register_key(key, window_length=self._cfgs.logger_cfgs.window_lens)
        for key in ('Grad/Actor', 'Grad/RewardCritic', 'Grad/CostCritic'):
            self._logger.register_key(key)

    def _log_gradient(self, key, model):
        norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
        self._logger.store({key: torch.stack(norms).norm().item()})

    def _update_actor(self, *args, **kwargs):
        super()._update_actor(*args, **kwargs)
        self._log_gradient('Grad/Actor', self._actor_critic.actor)

    def _update_reward_critic(self, *args, **kwargs):
        super()._update_reward_critic(*args, **kwargs)
        self._log_gradient('Grad/RewardCritic', self._actor_critic.reward_critic)

    def _update_cost_critic(self, *args, **kwargs):
        super()._update_cost_critic(*args, **kwargs)
        self._log_gradient('Grad/CostCritic', self._actor_critic.cost_critic)


    def _init_env(self) -> None:
        """Create the learner's own adapter with learned costs inside normalization."""
        self._env = LearnedCostAdapter(
            env_id=self._env_id,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            seed=self._seed,
            cfgs=self._cfgs,
            cost_function=self._initial_cost_function,
        )
        self._steps_per_epoch = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )

    @property
    def train_env(self) -> LearnedCostAdapter:
        """The training adapter owned by this learner."""
        return self._env

    @property
    def actor(self) -> Actor:
        """The live actor, reusable for sequential sampling and evaluation."""
        return self._actor_critic.actor

    @property
    def num_timesteps(self) -> int:
        """Total environment transitions trained on across all inner phases."""
        return self._epochs_completed * self._cfgs.algo_cfgs.steps_per_epoch

    def set_cost_function(self, cost_function: CostFunction) -> None:
        """Replace the cost callback without resetting the learner or normalizers.

        Updating an already attached network in place does not require this call.
        """
        if self._closed:
            raise RuntimeError('This learner is closed.')
        self.train_env.set_cost_function(cost_function)

    def train_for(self, n_steps: int) -> tuple[float, float, float]:
        """Train for additional steps; return final logged reward, learned cost, length.

        Steps are counted across all vector environments and must fill whole PPO
        epochs. Networks, optimizers, multiplier, schedules and normalizers survive
        each call. As in OmniSafe, simulation episodes reset at each epoch.
        Call ``close()`` once after the outer loop finishes.
        """
        if self._closed:
            raise RuntimeError('This learner is closed.')
        steps_per_epoch = self._cfgs.algo_cfgs.steps_per_epoch
        if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError('n_steps must be a positive integer.')
        if n_steps % steps_per_epoch:
            raise ValueError('n_steps must be a multiple of steps_per_epoch.')
        if self.num_timesteps + n_steps > self._cfgs.train_cfgs.total_steps:
            raise ValueError('Requested steps exceed the configured total_steps horizon.')
        if not self.train_env.has_cost_function:
            raise RuntimeError('Attach a learned cost function before training.')

        if self._start_time is None:
            self._start_time = time.time()
        # The attached network may have changed in place since the last phase.
        # Lambda must not use the logger's rolling costs from an older constraint.
        self._logger._data['Metrics/EpCost'].clear()
        self._actor_critic.train()
        self._logger.log(f'INFO: Continue ICRL PPO-Lag for {n_steps} environment steps')

        for _ in range(n_steps // steps_per_epoch):
            epoch = self._epochs_completed
            epoch_time = time.time()
            self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
            )
            self._logger.store({'Time/Rollout': time.time() - epoch_time})
            ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
            if not math.isfinite(ep_cost):
                raise RuntimeError(
                    'No finite completed-episode cost is available for PPO-Lag. '
                    'Increase steps_per_epoch so each rollout contains a full episode.',
                )

            update_time = time.time()
            self._update()  # Inherited PPO-Lag: update lambda, actor, and both critics.
            self._logger.store({'Time/Update': time.time() - update_time})
            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)
            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._epochs_completed += 1
            self._logger.store({
                'TotalEnvSteps': self.num_timesteps,
                'Train/Epoch': epoch,
                'Train/LR': (
                    self._actor_critic.actor_scheduler.get_last_lr()[0]
                    if self._cfgs.model_cfgs.actor.lr is not None else 0.0
                ),
                'Time/FPS': steps_per_epoch / (time.time() - epoch_time),
                'Time/Total': time.time() - self._start_time,
                'Time/Epoch': time.time() - epoch_time,
            })
            # Capture before dump_tabular clears non-windowed metrics.
            metrics = tuple(self._logger.get_stats(key)[0] for key in (
                'Metrics/EpRet', 'Metrics/EpCost', 'Metrics/EpLen',
            ))
            self.last_metrics = {key: self._logger.get_stats(key)[0] for key in self._logger._data}
            self._logger.dump_tabular()
            if (
                self._epochs_completed % self._cfgs.logger_cfgs.save_model_freq == 0
                or self._epochs_completed == self._cfgs.train_cfgs.epochs
            ):
                self._logger.torch_save()

        return metrics

    def learn(self) -> tuple[float, float, float]:
        """Run the remaining budget and close, like OmniSafe's one-shot entry point."""
        try:
            return self.train_for(self._cfgs.train_cfgs.total_steps - self.num_timesteps)
        finally:
            self.close()

    def close(self) -> None:
        """Close the environment and log outputs once, after all outer iterations."""
        if self._closed:
            return
        try:
            self._env.close()
        finally:
            self._logger.close()
            writer = getattr(self._logger, '_tensorboard_writer', None)
            if writer is not None:
                writer.close()
            self._closed = True
