"""Native OmniSafe PPO-Lag as a persistent forward solver for the ICRL loop."""

from copy import deepcopy

from ICRL.ICRL_PPOLag import ICRL_PPOLag
from omnisafe.utils.config import Config, check_all_configs, get_default_kwargs_yaml


class OmniSafePPOLag:
    def __init__(self, config, cost_function, output):
        run = config['run']
        env_id = config['env']['train_env_id']
        cfgs = get_default_kwargs_yaml('PPOLag', env_id, 'on-policy')
        cfgs.recurisve_update(deepcopy(config['omnisafe']))
        # Persist ICRL execution settings in the native snapshot's config.json.
        # Native PPO ignores this section; only our adapter/exporter consumes it.
        cfgs.icrl_cfgs = Config(clip_actions=config['clip_actions'])
        cfgs.seed = run['seed']
        cfgs.train_cfgs.device = config['device']
        cfgs.train_cfgs.torch_threads = run['torch_threads']
        cfgs.train_cfgs.total_steps = config['running']['n_iters'] * config['PPO']['forward_timesteps']
        cfgs.train_cfgs.epochs = cfgs.train_cfgs.total_steps // cfgs.algo_cfgs.steps_per_epoch
        cfgs.env_cfgs.recurisve_update({key: config['env'][key]
                                      for key in ('cost_boundary', 'max_episode_steps')})
        cfgs.env_cfgs.backend = run['backend']
        cfgs.logger_cfgs.log_dir = str(output / 'forward')
        cfgs.logger_cfgs.use_tensorboard = True
        cfgs.exp_name = f'PPOLag-{{{env_id}}}'
        cfgs.env_id, cfgs.algo = env_id, 'PPOLag'
        check_all_configs(cfgs, 'on-policy')
        self.native = ICRL_PPOLag(env_id, cfgs, cost_function)

    @property
    def num_timesteps(self):
        return self.native.num_timesteps

    @property
    def train_env(self):
        return self.native.train_env

    def train_for(self, steps):
        self.native.train_for(steps)
        stats = self.native.last_metrics
        names = {'Loss/Loss_pi': 'policy_loss', 'Loss/Loss_reward_critic': 'reward_value_loss',
                 'Loss/Loss_cost_critic': 'cost_value_loss', 'Train/KL': 'approx_kl',
                 'Metrics/LagrangeMultiplier': 'nu', 'Train/StopIter': 'ppo_epochs'}
        return {name: stats[key] for key, name in names.items() if key in stats}

    def state_dict(self):
        learner = self.native
        ac = learner._actor_critic
        return dict(actor_critic=ac.state_dict(), actor_optimizer=ac.actor_optimizer.state_dict(),
                    reward_critic_optimizer=ac.reward_critic_optimizer.state_dict(),
                    cost_critic_optimizer=ac.cost_critic_optimizer.state_dict(),
                    actor_scheduler=ac.actor_scheduler.state_dict(),
                    lagrange_multiplier=learner._lagrange.lagrangian_multiplier.detach().clone(),
                    lagrange_optimizer=learner._lagrange.lambda_optimizer.state_dict(),
                    normalizers={key: value.state_dict() for key, value in learner.train_env.save().items()},
                    epochs_completed=learner._epochs_completed, config=learner._cfgs.todict())

    def close(self):
        self.native.close()
