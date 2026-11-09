"""Replay a recorded simulator run against Liu's original numerical updates.

Usage: python -m ICRL.validate_run ICRL/runs/audit_VICRL_HC
This compares the real rollout inputs, losses, gradients, parameters and Adam
states. It does not train another policy or claim trajectory-level identity
between different MuJoCo/PyTorch releases.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import types

import gymnasium as gym
import numpy as np
import torch
import yaml

from ICRL.constraint_models.constraint_net.constraint_net import ConstraintNet
from ICRL.constraint_models.constraint_net.variational_constraint_net import VariationalConstraintNet
from ICRL.demonstrations import load_trajectory
from ICRL.liu_ppo import LiuPPOLag
from ICRL.tests.test_constraint_equivalence import _reference_classes
from ICRL.tests.test_ppo_equivalence import _reference_ppo


def _compare(left, right, *, atol=0.0, rtol=0.0, path="root"):
    """Assert matching trees and return their largest absolute discrepancy."""
    if isinstance(left, dict):
        assert left.keys() == right.keys(), f"Different keys at {path}"
        return max((_compare(left[key], right[key], atol=atol, rtol=rtol,
                             path=f"{path}.{key}") for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"Different lengths at {path}"
        return max((_compare(lhs, rhs, atol=atol, rtol=rtol, path=f"{path}[{i}]")
                    for i, (lhs, rhs) in enumerate(zip(left, right))), default=0.0)
    if isinstance(left, torch.Tensor):
        left, right = left.detach().cpu().numpy(), right.detach().cpu().numpy()
    if isinstance(left, (bool, np.bool_)):
        assert left == right, f"Different flags at {path}"
        return 0.0
    if isinstance(left, (np.ndarray, float, int, np.number)):
        np.testing.assert_allclose(left, right, rtol=rtol, atol=atol, err_msg=path)
        return float(np.max(np.abs(np.asarray(left) - np.asarray(right)))) if np.size(left) else 0.0
    assert left == right, f"Different values at {path}: {left!r}, {right!r}"
    return 0.0


def _record_gradients(optimizer, named_parameters):
    history = []
    params = list(named_parameters)
    def hook(_optimizer, _args, _kwargs):
        history.append({name: param.grad.detach().clone() for name, param in params})
    optimizer.register_step_pre_hook(hook)
    return history


def validate_constraint(run, config, expert_directory=None):
    metadata = json.loads((run / "metadata.json").read_text())
    configured = Path(config["running"]["expert_path"])
    directory = Path(expert_directory) if expert_directory else (
        configured if configured.is_absolute() else
        Path(__file__).resolve().parent / "data/expert_data" / configured.name)
    records = []
    for name, expected_hash in sorted(metadata["expert_files"].items()):
        content = (directory / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_hash, f"Expert data changed: {name}"
        records.append(load_trajectory(directory / name))
    expert_obs = np.concatenate([row["observations"] for row in records])
    expert_actions = np.concatenate([row["actions"] for row in records])
    snapshot = torch.load(run / "constraint_before_update.pt", map_location="cpu", weights_only=False)
    with np.load(run / "first_nominal_samples.npz") as samples:
        obs, actions, lengths = samples["observations"], samples["actions"], samples["lengths"]
    cn, running = config["CN"], config["running"]
    variational = config["group"] == "VICRL"
    kwargs = dict(
        obs_dim=obs.shape[-1], acs_dim=actions.shape[-1],
        hidden_sizes=cn["cn_layers"], batch_size=cn["cn_batch_size"],
        lr_schedule=lambda progress: cn["cn_learning_rate"] * cn["anneal_clr_by_factor"] ** (running["n_iters"] * (1 - progress)),
        expert_obs=expert_obs, expert_acs=expert_actions, is_discrete=False,
        task=config["group"], regularizer_coeff=cn["cn_reg_coeff"],
        obs_select_dim=cn.get("obs_select_dim"), acs_select_dim=cn.get("acs_select_dim"),
        optimizer_kwargs={"eps": cn.get("optimizer_epsilon", 1e-5)},
        clip_obs=cn["clip_obs"], eps=cn["cn_eps"],
        action_low=-np.ones(actions.shape[-1]), action_high=np.ones(actions.shape[-1]),
        target_kl_old_new=cn["cn_target_kl_old_new"], target_kl_new_old=cn["cn_target_kl_new_old"],
        train_gail_lambda=cn["train_gail_lambda"], device="cpu", log_file=io.StringIO(),
    )
    if variational:
        kwargs.update(di_prior=cn["di_prior"], mode=cn["mode"])
    else:
        kwargs.update(no_importance_sampling=cn["no_importance_sampling"],
                      per_step_importance_sampling=cn["per_step_importance_sampling"])
    reference_cls = _reference_classes()[int(variational)]
    actual_cls = VariationalConstraintNet if variational else ConstraintNet
    models = [reference_cls(**kwargs), actual_cls(**kwargs)]
    losses, gradients = [], []
    for model in models:
        model.load_state_dict(snapshot["model"])
        model.optimizer.load_state_dict(snapshot["optimizer"])
        gradients.append(_record_gradients(model.optimizer, model.named_parameters()))
        torch.set_rng_state(snapshot["torch_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        with contextlib.redirect_stderr(io.StringIO()):
            losses.append(model.train_nn(
                iterations=cn["backward_iters"], nominal_obs=obs, nominal_acs=actions,
                obs_mean=snapshot.get("obs_mean"), obs_var=snapshot.get("obs_var"),
                episode_lengths=lengths, current_progress_remaining=1.0))
    result = dict(
        nominal_transitions=len(obs), expert_transitions=len(expert_obs),
        nominal_episodes=len(lengths), optimizer_steps=len(gradients[0]),
        loss_metrics_max_abs=_compare(losses[0], losses[1]),
        gradients_max_abs=_compare(gradients[0], gradients[1]),
        parameters_max_abs=_compare(models[0].state_dict(), models[1].state_dict()),
        optimizer_max_abs=_compare(models[0].optimizer.state_dict(), models[1].optimizer.state_dict()),
    )
    stored = run / "backward_000.json"
    if stored.exists():
        result["recorded_run_metrics_max_abs"] = _compare(losses[1], json.loads(stored.read_text()))
    return result


def _parameter_pairs(reference, actual, config):
    actual_params = dict(actual.named_parameters())
    branches = {
        "mlp_extractor.policy_net.": "actor.mean.",
        "mlp_extractor.value_net.": "reward_critic.critic_0.",
        "mlp_extractor.cost_value_net.": "cost_critic.critic_0.",
        "action_net.": f"actor.mean.{2 * len(config['policy_layers'])}.",
        "value_net.": f"reward_critic.critic_0.{2 * len(config['reward_vf_layers'])}.",
        "cost_value_net.": f"cost_critic.critic_0.{2 * len(config['cost_vf_layers'])}.",
    }
    for name, param in reference.named_parameters():
        translated = "actor.log_std" if name == "log_std" else name
        for prefix, destination in branches.items():
            if name.startswith(prefix):
                translated = destination + name[len(prefix):]
                break
        yield name, param, actual_params[translated]


def validate_ppo(run, config):
    checkpoint = torch.load(run / "policy_before_update.pt", map_location="cpu", weights_only=False)
    with np.load(run / "first_ppo_rollout.npz") as rollout:
        data = {key: rollout[key].copy() for key in rollout.files}
    steps, workers, obs_dim = data["obs"].shape
    act_dim = data["act"].shape[-1]
    env = types.SimpleNamespace(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_space=gym.spaces.Box(-1, 1, (act_dim,), np.float32),
    )
    c = config["PPO"]
    actual = LiuPPOLag(env, c, "cpu")
    actual.policy.load_state_dict(checkpoint["policy"])
    actual.optimizer.load_state_dict(checkpoint["optimizer"])
    original = _reference_ppo()
    policy = original.Policy(env.observation_space, env.action_space, lambda _: c["learning_rate"],
                             net_arch=[dict(pi=c["policy_layers"], vf=c["reward_vf_layers"], cvf=c["cost_vf_layers"])])
    pairs = list(_parameter_pairs(policy, actual.policy, c))
    with torch.no_grad():
        for _, reference_param, actual_param in pairs:
            reference_param.copy_(actual_param)
            if actual_param in actual.optimizer.state:
                policy.optimizer.state[reference_param] = copy.deepcopy(actual.optimizer.state[actual_param])
    for source, target in zip(actual.optimizer.param_groups, policy.optimizer.param_groups):
        target.update({key: value for key, value in source.items() if key != "params"})
    buffer = original.Buffer(
        steps, env.observation_space, env.action_space, n_envs=workers,
        reward_gamma=c["reward_gamma"], reward_gae_lambda=c["reward_gae_lambda"],
        cost_gamma=c["cost_gamma"], cost_gae_lambda=c["cost_gae_lambda"])
    for reference_name, actual_name in {
        "observations": "obs", "orig_observations": "obs", "actions": "act", "log_probs": "logp",
        "reward_values": "vr", "cost_values": "vc", "orig_costs": "raw_cost",
        "rewards": "reward", "costs": "cost",
    }.items():
        setattr(buffer, reference_name, data[actual_name].copy())
    # The reference collector passes PRE-final-step critic values and bool final
    # dones. Recompute the saved advantages, not just the optimizer that uses them.
    buffer.dones = np.concatenate((np.zeros((1, workers), np.float32), data["done"][:-1]))
    buffer.compute_returns_and_advantage(
        torch.from_numpy(data["vr"][-1]), torch.from_numpy(data["vc"][-1]),
        dones=data["done"][-1].astype(bool))
    gae_error = _compare(
        {"adv_r": buffer.reward_advantages, "adv_c": buffer.cost_advantages,
         "ret_r": buffer.reward_returns, "ret_c": buffer.cost_returns},
        {key: data[key] for key in ("adv_r", "adv_c", "ret_r", "ret_c")})
    buffer.full = True
    reference = original.PPO.__new__(original.PPO)
    for key in ("batch_size", "n_epochs", "clip_range_reward_vf", "clip_range_cost_vf", "target_kl",
                "ent_coef", "reward_vf_coef", "cost_vf_coef", "max_grad_norm"):
        value = c.get(key) if key in ("clip_range_reward_vf", "clip_range_cost_vf") else c[key]
        setattr(reference, key, value)
    reference.policy, reference.rollout_buffer = policy, buffer
    reference.action_space = env.action_space
    reference.clip_range = lambda _: c["clip_range"]
    reference._current_progress_remaining = 1.0
    reference._update_learning_rate = lambda optimizer: None
    reference._n_updates, reference.verbose, reference.use_sde = 0, 0, False
    reference.update_penalty_after = 1
    reference.dual = original.Dual(c["budget"], c["penalty_learning_rate"], c["penalty_initial_value"])
    if "dual" in checkpoint:
        with torch.no_grad():
            reference.dual.nu.log_nu.copy_(checkpoint["dual"])
            actual.dual.log_nu.copy_(checkpoint["dual"])
    if "dual_optimizer" in checkpoint:
        reference.dual.optimizer.load_state_dict(checkpoint["dual_optimizer"])
        actual.dual.optimizer.load_state_dict(checkpoint["dual_optimizer"])
    ref_gradients = _record_gradients(policy.optimizer, ((name, ref) for name, ref, _ in pairs))
    new_gradients = _record_gradients(actual.optimizer, ((name, new) for name, _, new in pairs))
    np.random.set_state(checkpoint["numpy_rng"])
    reference.train()
    np.random.set_state(checkpoint["numpy_rng"])
    metrics = actual.update(data)
    tolerances = dict(atol=2e-7, rtol=2e-5)
    parameters_error = max(_compare(ref, new, path=name, **tolerances) for name, ref, new in pairs)
    optimizer_error = max(_compare(policy.optimizer.state[ref], actual.optimizer.state[new], path=name,
                                   **tolerances) for name, ref, new in pairs)
    result = dict(
        rollout_transitions=steps * workers, workers=workers, optimizer_steps=len(ref_gradients),
        ppo_epochs=metrics["ppo_epochs"], gae_max_abs=gae_error,
        gradients_max_abs=_compare(ref_gradients, new_gradients, **tolerances),
        parameters_max_abs=parameters_error, optimizer_max_abs=optimizer_error,
        dual_max_abs=_compare(reference.dual.nu.log_nu, actual.dual.log_nu),
    )
    mapping = {"train/policy_gradient_loss": "policy_loss", "train/reward_value_loss": "reward_value_loss",
               "train/cost_value_loss": "cost_value_loss", "train/nu": "nu", "train/nu_loss": "dual_loss"}
    result["loss_metrics_max_abs"] = _compare({key: original.logger[key] for key in mapping},
                                             {key: metrics[value] for key, value in mapping.items()}, **tolerances)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--expert-data", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = yaml.safe_load((args.run / "config.yaml").read_text())
    result = {"reference": "ICRL-benchmarks-public unmodified classes on the same PyTorch runtime",
              "torch": torch.__version__, "constraint": validate_constraint(args.run, config, args.expert_data)}
    result['ppo'] = ({'status': 'not_applicable', 'solver': 'omnisafe',
                      'reason': 'Native PPO-Lag uses its own optimizer/collector semantics, not the source PPO implementation.'}
                     if config.get('run', {}).get('solver') == 'omnisafe' else validate_ppo(args.run, config))
    (args.run / "numerical_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
