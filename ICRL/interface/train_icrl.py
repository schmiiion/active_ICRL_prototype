"""Train one ICRL experiment: python -m ICRL.interface.train_icrl --config ICRL/config/VICRL_HC.yaml."""

import argparse
import csv
import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from ICRL.constraint_models.constraint_net.constraint_net import ConstraintNet
from ICRL.constraint_models.constraint_net.variational_constraint_net import VariationalConstraintNet
from ICRL.demonstrations import load_demonstrations, validate_demonstration_environment
from ICRL.experiment_config import load_experiment_config
from ICRL.liu_ppo import LiuPPOLag
from ICRL.native_ppo import OmniSafePPOLag
from ICRL.native_sampling import NativeSampler
from ICRL.rollouts import LiuVectorEnv, evaluate, sample_trajectories
from ICRL.utils.memory_buffer import IRLDataQueue


def make_constraint(config, obs_dim, act_dim, expert_obs, expert_actions):
    cn, running = config['CN'], config['running']
    parameters = dict(
        obs_dim=obs_dim, acs_dim=act_dim, hidden_sizes=cn['cn_layers'], batch_size=cn['cn_batch_size'],
        lr_schedule=lambda p: cn['cn_learning_rate'] * cn['anneal_clr_by_factor'] ** (running['n_iters'] * (1 - p)),
        expert_obs=expert_obs, expert_acs=expert_actions, is_discrete=False,
        task=config['group'], regularizer_coeff=cn['cn_reg_coeff'],
        obs_select_dim=cn.get('obs_select_dim'), acs_select_dim=cn.get('acs_select_dim'),
        clip_obs=cn['clip_obs'], activation=cn.get('activation', 'relu'),
        initial_obs_mean=None, initial_obs_var=None,
        optimizer_kwargs={'eps': cn.get('optimizer_epsilon', 1e-5)},
        action_low=-np.ones(act_dim), action_high=np.ones(act_dim),
        target_kl_old_new=cn['cn_target_kl_old_new'], target_kl_new_old=cn['cn_target_kl_new_old'],
        train_gail_lambda=cn['train_gail_lambda'], eps=cn['cn_eps'], device=config['device'])
    if config['group'] == 'VICRL':
        return VariationalConstraintNet(
            **parameters, di_prior=cn['di_prior'], mode=cn['mode'],
            no_importance_sampling=cn.get('no_importance_sampling', True),
            per_step_importance_sampling=cn.get('per_step_importance_sampling', False))
    return ConstraintNet(**parameters, no_importance_sampling=cn['no_importance_sampling'],
                         per_step_importance_sampling=cn['per_step_importance_sampling'])


def train(args):
    config, source = load_experiment_config(
        getattr(args, 'config', None), method=getattr(args, 'method', None),
        task=getattr(args, 'task', None), overrides=vars(args))
    run = config['run']
    method, task = config['method'], config['task']
    torch.set_num_threads(run['torch_threads'])
    output = Path(run['output'])
    running, cn = config['running'], config['CN']
    dataset = Path(running['expert_path'])
    validate_demonstration_environment(dataset, config['env']['train_env_id'], config['env'], run['backend'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    expert_obs, expert_actions, hashes = load_demonstrations(dataset, running['expert_rollouts'])
    # Reference evaluation-env construction reseeds the parent to 0 before CN.
    random.seed(run['constraint_seed'])
    np.random.seed(run['constraint_seed'])
    torch.manual_seed(run['constraint_seed'])
    constraint = make_constraint(config, expert_obs.shape[-1], expert_actions.shape[-1], expert_obs, expert_actions)
    env_id = config['env']['train_env_id']
    native = run['solver'] == 'omnisafe'
    tensorboard = SummaryWriter(str(output / 'tb'))
    # PPOLagrangian._setup_model reseeds after constraint creation.
    random.seed(run['seed'])
    np.random.seed(run['seed'])
    torch.manual_seed(run['seed'])
    if native:
        learner = OmniSafePPOLag(config, constraint.cost_function, output)
        train_env = learner.train_env
        train_env.configure_episode_logging(output / 'episodes.csv', method, run['seed'],
                                            tensorboard, lambda: learner.num_timesteps)
        sampling_env = NativeSampler(learner.native, config['env'], run['sampling_seed'])
        config['omnisafe_resolved'] = learner.native._cfgs.todict()
        (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    else:
        train_env = LiuVectorEnv(env_id, run['workers'], run['seed'], config['env'],
                                 cost_function=constraint.cost_function, method=method,
                                 episode_file=output / 'episodes.csv', clip_actions=config['clip_actions'])
        sampling_env = LiuVectorEnv(config['env']['eval_env_id'], 1, run['sampling_seed'], config['env'],
                                    training=False, cost_function=constraint.cost_function,
                                    clip_actions=config['clip_actions'])
        random.seed(run['seed'])
        np.random.seed(run['seed'])
        torch.manual_seed(run['seed'])
        learner = LiuPPOLag(train_env, config['PPO'], config['device'])
        learner.audit_dir = output
        train_env.count_steps = lambda: learner.num_timesteps
    queue = IRLDataQueue(running['store_sample_rollouts'], run['seed']) if running['use_buffer'] else None
    backend_versions = {'gymnasium': version('gymnasium'), 'mujoco': version('mujoco')}
    metadata = dict(method=method, task=task, seed=run['seed'], workers=run['workers'],
                    clip_actions=config['clip_actions'],
                    expert_files=hashes, expert_transitions=len(expert_obs),
                    expert_trajectories=running['expert_rollouts'], torch=torch.__version__, numpy=np.__version__,
                    source_config=str(source), source_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    backend=run['backend'], solver=run['solver'], run=run, backend_versions=backend_versions,
                    simulator=f"MuJoCo {backend_versions['mujoco']} / Gymnasium {backend_versions['gymnasium']}")
    if native:
        metadata.update(forward_algorithm='omnisafe.algorithms.PPOLag',
                        nominal_sampling='raw true-cost environment; frozen training observation statistics',
                        figure2_smoothing='forward mean of 100 episodes; N-100 points; reward rounded to two decimals',
                        forward_log_dir=learner.native.logger.log_dir,
                        tensorboard_dirs=[str(output / 'tb'), str(Path(learner.native.logger.log_dir) / 'tb')])
    repo = Path(__file__).resolve().parents[2]
    sources = ['ICRL/interface/train_icrl.py', 'ICRL/action_wrapper.py', 'ICRL/native_ppo.py', 'ICRL/ICRL_PPOLag.py',
               'ICRL/native_sampling.py', 'ICRL/learned_cost_adapter.py', 'ICRL/blocked_envs.py',
               'ICRL/utils/memory_buffer.py',
               'ICRL/constraint_models/constraint_net/constraint_net.py',
               'ICRL/constraint_models/constraint_net/variational_constraint_net.py',
               'omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag.py',
               'omnisafe/algorithms/on_policy/base/policy_gradient.py']
    metadata['code_sha256'] = {file: hashlib.sha256((repo / file).read_bytes()).hexdigest() for file in sources}
    if config.get('reference_config') is not None:
        reference = Path(config['reference_config'])
        metadata.update(reference_config=str(reference),
                        reference_config_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    tensorboard.add_text('Experiment/Config', '```yaml\n' + yaml.safe_dump(config, sort_keys=False) + '\n```')
    tensorboard.add_text('Experiment/Provenance', json.dumps(metadata, indent=2))
    fields = ['iteration', 'total_steps', 'reward', 'feasible_reward', 'violation_rate',
              'policy_loss', 'reward_value_loss', 'cost_value_loss', 'approx_kl', 'grad_norm',
              'dual_loss', 'nu', 'ppo_epochs', 'constraint_parameter_change', 'cost_mean', 'cost_std',
              'expert_cost_mean', 'true_cost', 'expert_transitions', 'nominal_transitions',
              'nominal_trajectories', 'replay_queue_trajectories']
    try:
        with (output / 'iterations.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for iteration in range(running['n_iters']):
                metrics = learner.train_for(config['PPO']['forward_timesteps'])
                if native:
                    observations, actions, rewards, lengths = sampling_env.sample(
                        running['sample_rollouts'], deterministic=running.get('sample_deterministic', False))
                else:
                    sampling_env.normalizer.sync_from(train_env.normalizer)
                    observations, actions, rewards, lengths = sample_trajectories(
                        learner, sampling_env, running['sample_rollouts'],
                        deterministic=running.get('sample_deterministic', False))
                if queue is not None:
                    queue.put(observations, actions, rewards, lengths)
                    observations, actions, rewards, lengths = queue.get(running['sample_rollouts'])
                obs, act = np.concatenate(observations), np.concatenate(actions)
                previous = torch.cat([p.detach().flatten().cpu() for p in constraint.parameters()])
                obs_mean = obs_var = None
                if cn.get('cn_normalize', False):
                    if native:
                        norm = train_env.save()['obs_normalizer'].state_dict()
                        obs_mean, obs_var = (norm[key].cpu().numpy().copy() for key in ('_mean', '_var'))
                    else:
                        obs_mean, obs_var = train_env.normalizer.obs_rms.mean.copy(), train_env.normalizer.obs_rms.var.copy()
                if iteration == 0:
                    torch.save(dict(model=constraint.state_dict(), optimizer=constraint.optimizer.state_dict(),
                                    obs_mean=obs_mean, obs_var=obs_var,
                                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state()),
                               output / 'constraint_before_update.pt')
                    np.savez_compressed(output / 'first_nominal_samples.npz', observations=obs,
                                        actions=act, lengths=np.asarray(lengths))
                backward = constraint.train_nn(
                    iterations=cn['backward_iters'], nominal_obs=obs, nominal_acs=act, episode_lengths=np.asarray(lengths),
                    obs_mean=obs_mean, obs_var=obs_var,
                    current_progress_remaining=1 - iteration / running['n_iters'])
                current = torch.cat([p.detach().flatten().cpu() for p in constraint.parameters()])
                with torch.random.fork_rng():
                    costs = constraint.cost_function(obs, act)
                    expert_costs = constraint.cost_function(expert_obs, expert_actions)
                if native:
                    assessment = sampling_env.evaluate(running['n_eval_episodes'],
                                                       deterministic=running.get('eval_deterministic', False))
                    assessment['true_cost'] = float(np.mean([row['cost'] for row in sampling_env.last_records]))
                else:
                    sampling_env.normalizer.sync_from(train_env.normalizer)
                    assessment = evaluate(learner, sampling_env, running['n_eval_episodes'],
                                          deterministic=running.get('eval_deterministic', False))
                row = dict(iteration=iteration, total_steps=learner.num_timesteps, **metrics, **assessment,
                           constraint_parameter_change=float(torch.linalg.vector_norm(current - previous)),
                           cost_mean=float(np.mean(costs)), cost_std=float(np.std(costs)),
                           expert_cost_mean=float(np.mean(expert_costs)), expert_transitions=len(expert_obs),
                           nominal_transitions=len(obs), nominal_trajectories=len(lengths),
                           replay_queue_trajectories=0 if queue is None else len(queue.store_obs))
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row), flush=True)
                for prefix, values in [('Outer', row), ('Constraint', backward)]:
                    for key, value in values.items():
                        array = np.asarray(value)
                        if array.size == 1:
                            tensorboard.add_scalar(f'{prefix}/{key}', float(array.item()), learner.num_timesteps)
                if native:
                    for key, value in learner.native.last_metrics.items():
                        tensorboard.add_scalar(f'Forward/{key}', value, learner.num_timesteps)
                gradients = [p.grad.detach().flatten() for p in constraint.parameters() if p.grad is not None]
                tensorboard.add_scalar('Constraint/GradientNorm',
                                       float(torch.linalg.vector_norm(torch.cat(gradients))), learner.num_timesteps)
                tensorboard.add_scalar('Constraint/LearningRate', constraint.optimizer.param_groups[0]['lr'],
                                       learner.num_timesteps)
                if method == 'VICRL':
                    with torch.no_grad():
                        for name, states, controls in [('nominal', obs, act), ('expert', expert_obs, expert_actions)]:
                            alpha, beta = constraint.network(constraint.prepare_data(states, controls)).unbind(-1)
                            for key, value in [('alpha', alpha), ('beta', beta),
                                               ('expected_cost', 1 - alpha / (alpha + beta))]:
                                tensorboard.add_scalar(f'Posterior/{name}/{key}_mean', value.mean().item(),
                                                       learner.num_timesteps)
                                tensorboard.add_scalar(f'Posterior/{name}/{key}_std', value.std(unbiased=False).item(),
                                                       learner.num_timesteps)
                tensorboard.flush()
                (output / f'backward_{iteration:03d}.json').write_text(
                    json.dumps(backward, indent=2, default=lambda value: np.asarray(value).tolist()))
                checkpoint = (learner.state_dict() if native else
                              dict(policy=learner.policy.state_dict(), optimizer=learner.optimizer.state_dict(),
                                   dual=learner.dual.log_nu.detach(), normalizer=train_env.normalizer))
                checkpoint.update(constraint=constraint.state_dict(), constraint_optimizer=constraint.optimizer.state_dict(),
                                  iteration=iteration, total_steps=learner.num_timesteps,
                                  torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                                  python_rng=random.getstate(), replay_queue=None if queue is None else queue.__dict__)
                torch.save(checkpoint, output / 'checkpoint.pt')
                if (iteration + 1) % running['save_every'] == 0 or iteration + 1 == running['n_iters']:
                    torch.save(checkpoint, output / f'checkpoint_{iteration + 1:03d}.pt')
    finally:
        if native:
            learner.close()
        else:
            train_env.close()
        sampling_env.close()
        tensorboard.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Experiment YAML containing all settings')
    parser.add_argument('--method', choices=['MECL', 'VICRL'], help='Compatibility selector with --task')
    parser.add_argument('--task', choices=['HC', 'Ant', 'Walker', 'Swimmer'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device')
    parser.add_argument('--torch-threads', type=int)
    parser.add_argument('--output')
    parser.add_argument('--expert-data')
    parser.add_argument('--iterations', type=int, help='Override paper outer-loop count for diagnostics')
    parser.add_argument('--forward-steps', type=int, help='Override paper forward budget for diagnostics')
    parser.add_argument('--rollout-steps', type=int, help='Override paper steps per worker for diagnostics')
    args = parser.parse_args()
    try:
        train(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
