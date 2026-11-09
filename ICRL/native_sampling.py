"""Raw nominal trajectories and Figure 2 metrics for native OmniSafe policies."""

import numpy as np
import torch

import ICRL.blocked_envs  # Register the custom tasks.
from omnisafe.envs.core import make
from omnisafe.envs.wrapper import ActionScale, TimeLimit

from ICRL.action_wrapper import ActionClip


class NativeSampler:
    """Sample the live policy with frozen training observation statistics.

    The simulator returns true costs. Policy inputs alone are normalized;
    constraint learning receives raw pre-action states and executed policy actions.
    """

    def __init__(self, learner, env_config, seed=0):
        self.learner = learner
        self.device = learner._device
        settings = learner._cfgs.env_cfgs.todict()
        self.env = make(env_config['eval_env_id'], num_envs=1,
                        device=str(self.device), **settings)
        if self.env.need_time_limit_wrapper:
            self.env = TimeLimit(self.env, time_limit=self.env.max_episode_steps,
                                 device=self.device)
        # No AutoReset: the final successor remains the true terminal state.
        self.env = ActionScale(self.env, low=-1.0, high=1.0, device=self.device)
        self.env = ActionClip(
            self.env, device=self.device,
            clip_actions=learner._cfgs.get('icrl_cfgs', {}).get('clip_actions', False),
        )
        self._seed = seed
        self.normalizer = None
        self.last_records = []

    def sync_normalizer(self):
        source = self.learner.train_env.save().get('obs_normalizer')
        self.normalizer = (None if source is None else
                           {key: value.detach().clone() for key, value in source.state_dict().items()})

    @torch.no_grad()
    def _episode(self, deterministic):
        obs, _ = self.env.reset(seed=self._seed)
        self._seed = None
        observations, actions, rewards = [], [], []
        reward_sum = feasible_reward = cost_sum = 0.0
        violated = False
        while True:
            policy_obs = obs
            if self.normalizer is not None:
                norm = self.normalizer
                policy_obs = torch.clamp((obs - norm['_mean']) / norm['_std'],
                                         min=-norm['_clip'], max=norm['_clip'])
            action = self.learner.actor.predict(policy_obs.unsqueeze(0),
                                                deterministic=deterministic).squeeze(0)
            observations.append(obs.detach().cpu().numpy().copy())
            obs, reward, cost, terminated, truncated, _ = self.env.step(action)
            actions.append(self.env.last_action.detach().cpu().numpy().copy())
            reward, cost = float(reward), float(cost)
            rewards.append(reward)
            reward_sum += reward
            cost_sum += cost
            violated |= cost > 0
            if not violated:
                feasible_reward += reward
            if bool(terminated or truncated):
                record = dict(reward=reward_sum, reward_nc=feasible_reward,
                              constraint=int(violated), cost=cost_sum, length=len(actions))
                return np.asarray(observations), np.asarray(actions), np.asarray(rewards), record

    def sample(self, count, deterministic=False):
        """Return raw observation/action/reward lists and complete episode lengths."""
        if count <= 0:
            raise ValueError('Sample at least one episode.')
        self.sync_normalizer()
        actor = self.learner.actor
        was_training = actor.training
        actor.eval()
        try:
            episodes = [self._episode(deterministic) for _ in range(count)]
        finally:
            actor.train(was_training)
        self.last_records = [episode[3] for episode in episodes]
        return ([episode[0] for episode in episodes], [episode[1] for episode in episodes],
                [episode[2] for episode in episodes], [record['length'] for record in self.last_records])

    def evaluate(self, count, deterministic=False):
        """Return raw reward, reward before the first violation, and episode CVR."""
        self.sample(count, deterministic)
        return {key: float(np.mean([row[column] for row in self.last_records]))
                for key, column in [('reward', 'reward'), ('feasible_reward', 'reward_nc'),
                                    ('violation_rate', 'constraint')]}

    def close(self):
        self.env.close()
