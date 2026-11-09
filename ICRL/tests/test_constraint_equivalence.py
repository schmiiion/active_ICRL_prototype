"""Numerical checks against Liu et al.'s unmodified constraint implementations.

Run from the prototype root with ``python -m unittest discover -s ICRL/tests``.
The sibling ICRL-benchmarks-public checkout is the reference; override its path
with LIU_ICRL_REFERENCE. Only dependency imports are isolated: the original
classes, MLP factory, learning-rate updater, and KL helper execute from source.
Neither Gym/MuJoCo nor the old Stable-Baselines package must be installed.
"""

import ast
import contextlib
import io
import os
from pathlib import Path
import sys
import types
import typing
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from ICRL.constraint_models.constraint_net.constraint_net import ConstraintNet
from ICRL.constraint_models.constraint_net.variational_constraint_net import VariationalConstraintNet


REFERENCE = Path(os.environ.get(
    "LIU_ICRL_REFERENCE",
    Path(__file__).resolve().parents[3] / "ICRL-benchmarks-public",
))


def _function_from_source(path, name):
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "th": torch, "np": np, "nn": nn,
                 "List": typing.List, "Type": typing.Type}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _module(name, **members):
    module = types.ModuleType(name)
    module.__dict__.update(members)
    return module


def _reference_classes():
    if not (REFERENCE / "constraint_models/constraint_net/constraint_net.py").is_file():
        raise unittest.SkipTest(f"Liu reference checkout not found: {REFERENCE}")
    create_mlp = _function_from_source(
        REFERENCE / "stable_baselines3/common/torch_layers.py", "create_mlp")
    update_learning_rate = _function_from_source(
        REFERENCE / "stable_baselines3/common/utils.py", "update_learning_rate")
    idx2vector = _function_from_source(REFERENCE / "utils/data_utils.py", "idx2vector")
    kl = _function_from_source(REFERENCE / "utils/model_utils.py", "dirichlet_kl_divergence_loss")
    replacements = {
        "stable_baselines3": _module("stable_baselines3"),
        "stable_baselines3.common": _module("stable_baselines3.common"),
        "stable_baselines3.common.torch_layers": _module("torch_layers", create_mlp=create_mlp),
        "stable_baselines3.common.utils": _module("utils", update_learning_rate=update_learning_rate),
        "utils": _module("utils"),
        "utils.data_utils": _module("data_utils", idx2vector=idx2vector),
        "utils.model_utils": _module("model_utils", dirichlet_kl_divergence_loss=kl),
        "constraint_models": _module("constraint_models"),
        "constraint_models.constraint_net": _module("constraint_net"),
        # The reference imports pyplot but never uses it in either class.
        "matplotlib": _module("matplotlib"),
        "matplotlib.pyplot": _module("matplotlib.pyplot"),
    }
    with patch.dict(sys.modules, replacements):
        modules = []
        for name in ("constraint_net", "variational_constraint_net"):
            path = REFERENCE / f"constraint_models/constraint_net/{name}.py"
            module = _module(f"constraint_models.constraint_net.{name}")
            exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
            sys.modules[module.__name__] = module
            # Disable progress display, leaving iteration order unchanged.
            module.tqdm = lambda iterable: iterable
            modules.append(module)
    return modules[0].ConstraintNet, modules[1].VariationalConstraintNet


class ConfiguredFeaturesTest(unittest.TestCase):
    def test_selected_actions_follow_full_observation_block(self):
        for cls in (ConstraintNet, VariationalConstraintNet):
            for obs_dims, action_dims, expected in (
                    ([2, 0], [1], [[17.0, 5.0, -0.7]]),
                    ([-1], None, [[0.4, -0.7]])):
                with self.subTest(model=cls.__name__, observations=obs_dims):
                    model = cls(obs_dim=3, acs_dim=2, hidden_sizes=[2], batch_size=None,
                                lr_schedule=None, expert_obs=None, expert_acs=None,
                                is_discrete=False, optimizer_class=None, clip_obs=None,
                                obs_select_dim=obs_dims, acs_select_dim=action_dims)
                    actual = model.prepare_data(np.array([[5.0, 11.0, 17.0]]),
                                                np.array([[0.4, -0.7]]))
                    torch.testing.assert_close(actual, torch.tensor(expected))


class RecordingAdam(torch.optim.Adam):
    """Record each real backward/update for differential comparison."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradients = []
        self.parameters_after_step = []

    def step(self, closure=None):
        params = [p for group in self.param_groups for p in group["params"]]
        self.gradients.append([p.grad.detach().clone() for p in params])
        result = super().step(closure)
        self.parameters_after_step.append([p.detach().clone() for p in params])
        return result


class ConstraintEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.references = _reference_classes()

    def setUp(self):
        rng = np.random.RandomState(37)
        self.expert_obs = rng.normal(size=(11, 18))
        self.expert_acs = rng.uniform(-2, 2, size=(11, 6))
        self.nominal_obs = rng.normal(size=(13, 18))
        self.nominal_acs = rng.uniform(-2, 2, size=(13, 6))
        self.mean = rng.normal(size=18)
        self.var = np.square(rng.normal(size=18))
        self.var[0] = 0.0

    def kwargs(self, variational, **overrides):
        kwargs = dict(
            obs_dim=18, acs_dim=6, hidden_sizes=(20,), batch_size=None,
            lr_schedule=lambda progress: 0.005 * 0.9 ** (30 * (1 - progress)),
            expert_obs=self.expert_obs, expert_acs=self.expert_acs,
            is_discrete=False, task="VICRL" if variational else "ICRL",
            regularizer_coeff=0.5, no_importance_sampling=variational,
            per_step_importance_sampling=not variational,
            clip_obs=20.0, eps=1e-5,
            action_low=-np.ones(6), action_high=np.ones(6),
            target_kl_old_new=10, target_kl_new_old=2.5,
            device="cpu", log_file=io.StringIO(),
        )
        if variational:
            kwargs.update(di_prior=[0.9, 0.1], mode="sample")
        kwargs.update(overrides)
        return kwargs

    def pair(self, variational, seed=123, **overrides):
        actual_cls = VariationalConstraintNet if variational else ConstraintNet
        reference_cls = self.references[int(variational)]
        torch.manual_seed(seed)
        reference = reference_cls(**self.kwargs(variational, **overrides))
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        actual = actual_cls(**self.kwargs(variational, **overrides))
        self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        return reference, actual

    def assert_equal_nested(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_equal_nested(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for lhs, rhs in zip(left, right):
                self.assert_equal_nested(lhs, rhs)
        else:
            self.assertEqual(left, right)

    def test_seeded_initialization_and_default_adam(self):
        for variational in (False, True):
            for seed in (123, 321, 666):
                for hidden_sizes in ((), (20,), (8, 5)):
                    with self.subTest(variational=variational, seed=seed, layers=hidden_sizes):
                        reference, actual = self.pair(variational, seed, hidden_sizes=hidden_sizes)
                        self.assert_equal_nested(reference.state_dict(), actual.state_dict())
                        self.assert_equal_nested(reference.optimizer.state_dict(), actual.optimizer.state_dict())
                        self.assertEqual(actual.optimizer.defaults["eps"], 1e-5)

    def test_preprocessing_clipping_normalization_and_selection(self):
        for variational in (False, True):
            for normalize in (False, True):
                for selection in (None, (0, 2, 5)):
                    with self.subTest(variational=variational, normalize=normalize, selection=selection):
                        reference, actual = self.pair(
                            variational, obs_select_dim=selection,
                            initial_obs_mean=self.mean if normalize else None,
                            initial_obs_var=self.var if normalize else None,
                        )
                        obs = self.nominal_obs.copy()
                        obs[0, 0], obs[1, 0] = -100, 100
                        prepared = actual.prepare_data(obs, self.nominal_acs)
                        expected_obs = obs if not normalize else (obs - self.mean) / np.sqrt(self.var + 1e-5)
                        expected = np.concatenate((np.clip(expected_obs, -20, 20),
                                                   np.clip(self.nominal_acs, -1, 1)), axis=-1)
                        if selection is None:
                            # Released experiments use all features: retain source parity.
                            self.assert_equal_nested(reference.prepare_data(obs, self.nominal_acs), prepared)
                            self.assertEqual(prepared[0, 0].item(), -20.0)
                            self.assertEqual(prepared[1, 0].item(), 20.0)
                        else:
                            # Source offsets actions by len(selection), incorrectly reading
                            # observation columns. Explicit selection uses the actual actions.
                            expected = expected[:, list(selection) + list(range(18, 24))]
                        self.assert_equal_nested(prepared, torch.tensor(expected, dtype=torch.float32))

    def test_cost_modes_and_beta_reparameterization(self):
        for variational in (False, True):
            reference, actual = self.pair(variational)
            modes = ("sample", "mean", "hard", "VaR", "CVaR") if variational else (None,)
            for mode in modes:
                with self.subTest(variational=variational, mode=mode):
                    torch.manual_seed(456)
                    np.random.seed(456)
                    expected = reference.cost_function(self.nominal_obs, self.nominal_acs, force_mode=mode)
                    expected_rng = torch.get_rng_state().clone()
                    torch.manual_seed(456)
                    np.random.seed(456)
                    result = actual.cost_function(self.nominal_obs, self.nominal_acs, force_mode=mode)
                    self.assert_equal_nested(expected, result)
                    self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            data = actual.prepare_data(self.nominal_obs, self.nominal_acs)
            torch.manual_seed(654)
            expected = reference(data)
            expected.sum().backward()
            torch.manual_seed(654)
            result = actual(data)
            result.sum().backward()
            self.assert_equal_nested(expected, result)
            self.assert_equal_nested([p.grad for p in reference.parameters()],
                                     [p.grad for p in actual.parameters()])

    def test_backward_losses_gradients_parameters_and_adam_states(self):
        for variational in (False, True):
            for batch_size in (None, 4):
                with self.subTest(variational=variational, batch_size=batch_size):
                    reference, actual = self.pair(
                        variational, batch_size=batch_size,
                        optimizer_class=RecordingAdam, optimizer_kwargs={"eps": 1e-5},
                    )
                    # Explicitly compare updates from identical supplied parameters as well.
                    actual.load_state_dict(reference.state_dict())
                    for progress, normalize in ((1.0, False), (0.6, True)):
                        args = dict(iterations=3, nominal_obs=self.nominal_obs,
                                    nominal_acs=self.nominal_acs, episode_lengths=np.array([6, 7]),
                                    obs_mean=self.mean if normalize else None,
                                    obs_var=self.var if normalize else None,
                                    current_progress_remaining=progress)
                        torch.manual_seed(789)
                        np.random.seed(789)
                        with contextlib.redirect_stderr(io.StringIO()):
                            expected = reference.train_nn(**args)
                        expected_rng = torch.get_rng_state().clone()
                        torch.manual_seed(789)
                        np.random.seed(789)
                        with contextlib.redirect_stderr(io.StringIO()):
                            result = actual.train_nn(**args)
                        self.assert_equal_nested(expected, result)
                        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                        self.assert_equal_nested(reference.optimizer.gradients, actual.optimizer.gradients)
                        self.assert_equal_nested(reference.optimizer.parameters_after_step,
                                                 actual.optimizer.parameters_after_step)
                        self.assert_equal_nested(reference.state_dict(), actual.state_dict())
                        self.assert_equal_nested(reference.optimizer.state_dict(), actual.optimizer.state_dict())
                        self.assertEqual(actual.optimizer.param_groups[0]["lr"],
                                         0.005 * 0.9 ** (30 * (1 - progress)))


if __name__ == "__main__":
    unittest.main()
