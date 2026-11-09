"""ICRL-only clipping of executed actions, leaving PPO samples untouched."""

import torch

from omnisafe.envs.core import Wrapper


class ActionClip(Wrapper):
    """Optionally bound an environment-bound copy and expose the executed action."""

    def __init__(self, env, device=torch.device('cpu'), *, clip_actions=False):
        super().__init__(env, device=device)
        self.clip_actions = clip_actions
        self._low = torch.as_tensor(self.action_space.low, device=device)
        self._high = torch.as_tensor(self.action_space.high, device=device)
        self.last_action = None

    def step(self, action):
        # Never clamp in place: the collector still needs the original sample
        # and its original log probability for the PPO likelihood ratio.
        self.last_action = (torch.clamp(action, min=self._low, max=self._high)
                            if self.clip_actions else action)
        return super().step(self.last_action)
