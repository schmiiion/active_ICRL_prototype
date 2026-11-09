"""Reference PPO-Lagrangian updates using OmniSafe's actor and value models.

This reproduction solver deliberately lives outside omnisafe/: its minibatch
advantage normalization, unclipped cost surrogate, shared Adam, and softplus
dual update differ from OmniSafe's default PPOLag. See reproduction_audit.md.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.critic.v_critic import VCritic


class LiuPolicy(nn.Module):
    def __init__(self, observation_space, action_space, config):
        super().__init__()
        activation = config.get('activation', 'tanh')
        activation_class = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'elu': nn.ELU}[activation]
        # Keep OmniSafe model construction from consuming the reference stream.
        with torch.random.fork_rng():
            self.actor = GaussianLearningActor(observation_space, action_space,
                                               config['policy_layers'], activation)
            self.reward_critic = VCritic(observation_space, action_space,
                                        config['reward_vf_layers'], activation)
            self.cost_critic = VCritic(observation_space, action_space,
                                      config['cost_vf_layers'], activation)
        # SB3 creates the three hidden branches interleaved, then the three heads.
        sizes = [config['policy_layers'], config['reward_vf_layers'], config['cost_vf_layers']]
        branches, dims = [[], [], []], [observation_space.shape[0]] * 3
        for i in range(max(map(len, sizes))):
            for branch, widths in enumerate(sizes):
                if i < len(widths):
                    branches[branch].extend([nn.Linear(dims[branch], widths[i]), activation_class()])
                    dims[branch] = widths[i]
        heads = [nn.Linear(dims[0], action_space.shape[0]), nn.Linear(dims[1], 1), nn.Linear(dims[2], 1)]
        for branch in branches:
            for layer in branch:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, config.get('hidden_gain', math.sqrt(2)))
                    nn.init.zeros_(layer.bias)
        gains = [config.get('policy_output_gain', 0.01)] + [config.get('value_output_gain', 1.0)] * 2
        for head, gain in zip(heads, gains):
            nn.init.orthogonal_(head.weight, gain)
            nn.init.zeros_(head.bias)
        self.actor.mean = nn.Sequential(*branches[0], heads[0])
        self.reward_critic.critic_0 = nn.Sequential(*branches[1], heads[1])
        self.reward_critic.net_lst = [self.reward_critic.critic_0]
        self.cost_critic.critic_0 = nn.Sequential(*branches[2], heads[2])
        self.cost_critic.net_lst = [self.cost_critic.critic_0]
        self.actor.log_std.data.fill_(config.get('log_std_init', 0.0))

    def evaluate(self, obs, actions):
        dist = self.actor(obs)
        return (self.reward_critic(obs)[0], self.cost_critic(obs)[0],
                dist.log_prob(actions).sum(-1), dist.entropy().sum(-1))

    @torch.no_grad()
    def step(self, obs):
        dist = self.actor(obs)
        # SB3's DiagGaussianDistribution.sample delegates to Normal.rsample.
        action = dist.rsample()
        return action, self.reward_critic(obs)[0], self.cost_critic(obs)[0], dist.log_prob(action).sum(-1)


class DualVariable:
    """Original softplus dual, updated using raw mean per-step rollout cost."""
    def __init__(self, config):
        initial = np.log(max(np.exp(config['penalty_initial_value']) - 1, 1e-8))
        self.log_nu = nn.Parameter(initial * torch.ones(1))
        self.minimum = initial
        self.budget = config['budget']
        self.optimizer = torch.optim.Adam([self.log_nu], lr=config['penalty_learning_rate'],
                                          eps=config.get('dual_optimizer_epsilon', 1e-8))

    def nu(self):
        return F.softplus(self.log_nu)

    def update(self, cost):
        self.optimizer.zero_grad()
        loss = -self.nu() * (cost - self.budget)
        loss.backward()
        self.optimizer.step()
        # Preserve Nu.clamp's original log/exp expression, including its floor.
        self.log_nu.data.clamp_(min=np.log(max(np.exp(self.minimum), 1e-8)))
        return loss.item()


def gae(rewards, values, dones, last_value, gamma, lam):
    """dones[t] describes the transition at t; time limits end GAE in Liu."""
    advantages = np.zeros_like(values, dtype=np.float32)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        # The original buffer stores float32 dones, but its collector passes
        # bool final dones separately: preserve that last-step float64 promotion.
        nonterminal = 1.0 - (dones[t].astype(bool) if t == len(rewards) - 1 else dones[t])
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def ppo_loss(policy, batch, config, penalty):
    vr, vc, logp, entropy = policy.evaluate(batch['obs'], batch['act'])
    ar = batch['adv_r'] - batch['adv_r'].mean()
    ar = ar / (batch['adv_r'].std() + config.get('advantage_epsilon', 1e-8))
    ac = batch['adv_c'] - batch['adv_c'].mean()
    ratio = torch.exp(logp - batch['logp'])
    clip = config['clip_range']
    policy_loss = (-torch.minimum(ar * ratio, ar * ratio.clamp(1 - clip, 1 + clip)).mean()
                   + penalty * (ac * ratio).mean()) / (1 + penalty)
    loss_r, loss_c = F.mse_loss(vr, batch['ret_r']), F.mse_loss(vc, batch['ret_c'])
    loss = (policy_loss - config['ent_coef'] * entropy.mean()
            + config['reward_vf_coef'] * loss_r + config['cost_vf_coef'] * loss_c)
    return loss, {'policy_loss': policy_loss.item(), 'reward_value_loss': loss_r.item(),
                  'cost_value_loss': loss_c.item(), 'approx_kl': (batch['logp'] - logp).mean().item()}


class LiuPPOLag:
    def __init__(self, env, config, device='cpu'):
        self.env, self.config, self.device = env, config, torch.device(device)
        self.policy = LiuPolicy(env.observation_space, env.action_space, config).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['learning_rate'],
                                          eps=config.get('optimizer_epsilon', 1e-5))
        self.dual = DualVariable(config)
        self.num_timesteps = 0
        self.last_rollout = None

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict(self, obs, deterministic=False):
        dist = self.policy.actor(self.tensor(obs))
        action = dist.mean if deterministic else dist.rsample()
        action = action.cpu().numpy()
        return (np.clip(action, self.env.action_space.low, self.env.action_space.high)
                if getattr(self.env, 'clip_actions', True) else action)

    def collect(self, obs):
        data = {key: [] for key in ['obs', 'act', 'logp', 'reward', 'cost', 'raw_cost', 'vr', 'vc', 'done']}
        for _ in range(self.config['n_steps']):
            action, vr, vc, logp = self.policy.step(self.tensor(obs))
            raw_action = action.cpu().numpy()
            executed = (np.clip(raw_action, self.env.action_space.low, self.env.action_space.high)
                        if getattr(self.env, 'clip_actions', True) else raw_action)
            next_obs, reward, cost, done = self.env.step(executed)
            values = [obs, raw_action, logp.cpu().numpy(), reward, cost,
                      self.env.last_raw_cost, vr.cpu().numpy(), vc.cpu().numpy(), done]
            for key, value in zip(data, values):
                data[key].append(np.array(value, copy=True))
            obs = next_obs
            self.num_timesteps += self.env.num_envs
        data = {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}
        # Liu's collector passes the values computed BEFORE its final step,
        # rather than evaluating the returned next observation. Preserve this
        # source quirk for reproducibility, including at nonterminal boundaries.
        last_r, last_c = data['vr'][-1], data['vc'][-1]
        c = self.config
        data['adv_r'], data['ret_r'] = gae(data['reward'], data['vr'], data['done'], last_r,
                                          c['reward_gamma'], c['reward_gae_lambda'])
        data['adv_c'], data['ret_c'] = gae(data['cost'], data['vc'], data['done'], last_c,
                                          c['cost_gamma'], c['cost_gae_lambda'])
        self.last_rollout = data
        return obs, data

    def update(self, data):
        flat = {key: self.tensor(value.swapaxes(0, 1).reshape(-1, *value.shape[2:]))
                for key, value in data.items()}
        count = len(flat['obs'])
        records = []
        for epoch in range(self.config['n_epochs']):
            kls = []
            permutation = np.random.permutation(count)
            for start in range(0, count, self.config['batch_size']):
                indices = permutation[start:start + self.config['batch_size']]
                batch = {key: value[indices] for key, value in flat.items()}
                loss, metrics = ppo_loss(self.policy, batch, self.config, self.dual.nu().item())
                self.optimizer.zero_grad()
                loss.backward()
                metrics['grad_norm'] = float(nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm']))
                self.optimizer.step()
                records.append(metrics)
                kls.append(metrics['approx_kl'])
            if (self.config['target_kl'] is not None
                    and np.mean(kls) > self.config.get('kl_stop_factor', 1.5) * self.config['target_kl']):
                break
        metrics = {key: float(np.mean([row[key] for row in records])) for key in records[0]}
        metrics['dual_loss'] = self.dual.update(float(data['raw_cost'].mean()))
        metrics['nu'] = self.dual.nu().item()
        metrics['ppo_epochs'] = epoch + 1
        return metrics

    def train_for(self, steps):
        # .learn(reset_num_timesteps=True) resets episodes each outer iteration,
        # but continues episodes across its successive PPO rollout batches.
        obs = self.env.reset()
        updates = []
        for _ in range(math.ceil(steps / (self.env.num_envs * self.config['n_steps']))):
            obs, data = self.collect(obs)
            if self.num_timesteps == self.env.num_envs * self.config['n_steps'] and hasattr(self, 'audit_dir'):
                np.savez_compressed(self.audit_dir / 'first_ppo_rollout.npz', **data)
                torch.save(dict(policy=self.policy.state_dict(), optimizer=self.optimizer.state_dict(),
                                numpy_rng=np.random.get_state()), self.audit_dir / 'policy_before_update.pt')
            updates.append(self.update(data))
        return {key: float(np.mean([row[key] for row in updates])) for key in updates[0]}
