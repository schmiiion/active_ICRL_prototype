"""Replay recorded VICRL/MECL constraint updates against Liu's original classes.

Usage: python -m ICRL.validate_run ICRL/runs/audit_VICRL_HC
The saved rollout inputs and expert demonstrations drive both implementations
on the same PyTorch runtime. Validation compares losses, gradients, parameters
and Adam states, and writes numerical_validation.json in the run directory.
It does not establish trajectory-level identity across simulator releases.
"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ICRL.constraint_models.constraint_net.constraint_net import ConstraintNet
from ICRL.constraint_models.constraint_net.variational_constraint_net import VariationalConstraintNet
from ICRL.demonstrations import load_trajectory
from ICRL.tests.test_constraint_equivalence import _reference_classes


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="recorded run directory containing constraint replay inputs")
    parser.add_argument("--expert-data", type=Path, help="override the expert demonstration directory")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = yaml.safe_load((args.run / "config.yaml").read_text())
    result = {"reference": "ICRL-benchmarks-public unmodified classes on the same PyTorch runtime",
              "torch": torch.__version__, "constraint": validate_constraint(args.run, config, args.expert_data)}
    (args.run / "numerical_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
