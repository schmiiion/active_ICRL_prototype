"""Differential checks of the PPO update against Liu's original SB3 source.

The policy, rollout buffer, distributions, and PPOLagrangian.train execute
unmodified reference code. Gymnasium Box spaces substitute for old Gym Box;
the unused SB3 algorithm base and logging infrastructure are isolated stubs.
No MuJoCo installation or original SB3 package import is needed.
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

import gymnasium as gym
import numpy as np
import torch

from ICRL.liu_ppo import DualVariable, LiuPolicy, LiuPPOLag, gae


REFERENCE = Path(os.environ.get(
    "LIU_ICRL_REFERENCE", Path(__file__).resolve().parents[3] / "ICRL-benchmarks-public",
))


def _module(name, **members):
    module = types.ModuleType(name)
    module.__dict__.update(members)
    return module


def _reference_ppo():
    if not (REFERENCE / "stable_baselines3/ppo_lag/ppo_lag.py").is_file():
        raise unittest.SkipTest(f"Liu reference checkout not found: {REFERENCE}")
    records = {}
    logger = _module("logger", record=lambda key, value, **kw: records.update({key: value}))
    callbacks = _module("callbacks", BaseCallback=type("BaseCallback", (), {}))
    utils = _module("stable_baselines3.common.utils", th=torch, np=np, gym=gym,
                    Callable=typing.Callable, Union=typing.Union)
    path = REFERENCE / "stable_baselines3/common/utils.py"
    names = {"get_device", "is_vectorized_observation", "explained_variance",
             "get_schedule_fn", "constant_fn", "update_learning_rate"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), utils.__dict__)
    replacements = {
        "gym": gym, "gym.spaces": gym.spaces,
        "stable_baselines3": _module("stable_baselines3"),
        "stable_baselines3.common": _module("stable_baselines3.common", logger=logger, callbacks=callbacks),
        "stable_baselines3.common.utils": utils,
        "stable_baselines3.common.vec_env": _module(
            "vec_env", VecEnv=type("VecEnv", (), {}), VecNormalize=type("VecNormalize", (), {}),
            VecTransposeImage=type("VecTransposeImage", (), {})),
        "stable_baselines3.common.on_policy_algorithm": _module(
            "on_policy_algorithm", OnPolicyWithCostAlgorithm=object),
    }
    modules = {}
    with patch.dict(sys.modules, replacements):
        for relative in ("common/preprocessing", "common/torch_layers", "common/distributions",
                         "common/policies", "common/type_aliases", "common/buffers",
                         "common/dual_variable", "ppo_lag/ppo_lag"):
            name = "stable_baselines3." + relative.replace("/", ".")
            path = REFERENCE / f"stable_baselines3/{relative}.py"
            module = _module(name)
            sys.modules[name] = module
            exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
            modules[relative] = module
    # Execute the source collector too: buffer tests alone cannot verify which
    # final value the caller passes into GAE.
    path = REFERENCE / "stable_baselines3/common/on_policy_algorithm.py"
    base = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "OnPolicyWithCostAlgorithm")
    collect = next(node for node in base.body
                   if isinstance(node, ast.FunctionDef) and node.name == "collect_rollouts")
    collector_globals = dict(th=torch, np=np, gym=gym, Union=typing.Union,
                             Callable=typing.Callable, VecEnv=object, BaseCallback=object,
                             RolloutBufferWithCost=object,
                             VecNormalize=type("VecNormalize", (), {}),
                             VecNormalizeWithCost=type("VecNormalizeWithCost", (), {}))
    exec(compile(ast.Module(body=[collect], type_ignores=[]), str(path), "exec"), collector_globals)
    return types.SimpleNamespace(
        Policy=modules["common/policies"].ActorTwoCriticsPolicy,
        Buffer=modules["common/buffers"].RolloutBufferWithCost,
        PPO=modules["ppo_lag/ppo_lag"].PPOLagrangian,
        Dual=modules["common/dual_variable"].DualVariable,
        logger=records,
        collect=collector_globals["collect_rollouts"],
    )


class PPOEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = _reference_ppo()
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.obs_space = gym.spaces.Box(-np.inf, np.inf, (18,), np.float32)
        self.act_space = gym.spaces.Box(-1, 1, (6,), np.float32)
        self.config = dict(
            policy_layers=[64, 64], reward_vf_layers=[64, 64], cost_vf_layers=[64, 64],
            learning_rate=0.0003, n_epochs=3, batch_size=64,
            clip_range=0.2, clip_range_reward_vf=None, clip_range_cost_vf=None,
            ent_coef=0.013, reward_vf_coef=0.5, cost_vf_coef=0.5,
            max_grad_norm=0.5, target_kl=None, penalty_initial_value=1,
            penalty_learning_rate=0.1, budget=0,
            reward_gamma=0.99, reward_gae_lambda=0.95,
            cost_gamma=0.99, cost_gae_lambda=0.95,
        )

    def policies(self, seed=123):
        torch.manual_seed(seed)
        reference = self.reference.Policy(
            self.obs_space, self.act_space, lambda _: self.config["learning_rate"],
            net_arch=[dict(pi=self.config["policy_layers"], vf=self.config["reward_vf_layers"],
                           cvf=self.config["cost_vf_layers"])],
        )
        rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        actual = LiuPolicy(self.obs_space, self.act_space, self.config)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()), "policy creation RNG differs")
        return reference, actual

    def parameter_pairs(self, reference, actual):
        actual_params = dict(actual.named_parameters())
        branches = {
            "mlp_extractor.policy_net.": "actor.mean.",
            "mlp_extractor.value_net.": "reward_critic.critic_0.",
            "mlp_extractor.cost_value_net.": "cost_critic.critic_0.",
            "action_net.": f"actor.mean.{2 * len(self.config['policy_layers'])}.",
            "value_net.": f"reward_critic.critic_0.{2 * len(self.config['reward_vf_layers'])}.",
            "cost_value_net.": f"cost_critic.critic_0.{2 * len(self.config['cost_vf_layers'])}.",
        }
        for name, param in reference.named_parameters():
            translated = "actor.log_std" if name == "log_std" else name
            for prefix, destination in branches.items():
                if name.startswith(prefix):
                    translated = destination + name[len(prefix):]
                    break
            yield name, param, actual_params[translated]

    def assert_parameters(self, reference, actual, *, exact=False):
        for name, lhs, rhs in self.parameter_pairs(reference, actual):
            with self.subTest(parameter=name):
                torch.testing.assert_close(lhs, rhs, rtol=0 if exact else 2e-6,
                                           atol=0 if exact else 2e-7)

    def test_seeded_policy_initialization_evaluation_and_actions(self):
        rng = np.random.RandomState(7)
        obs = torch.tensor(rng.normal(size=(13, 18)), dtype=torch.float32)
        actions = torch.tensor(rng.normal(size=(13, 6)), dtype=torch.float32)
        for seed in (123, 321, 666):
            with self.subTest(seed=seed):
                reference, actual = self.policies(seed)
                self.assert_parameters(reference, actual, exact=True)
                for lhs, rhs in zip(reference.evaluate_actions(obs, actions), actual.evaluate(obs, actions)):
                    torch.testing.assert_close(lhs.flatten(), rhs.flatten(), rtol=0, atol=0)
                # Evaluate sampling with nonzero means and non-unit standard deviations.
                with torch.no_grad():
                    reference.log_std.copy_(torch.linspace(-0.8, 0.5, 6))
                    actual.actor.log_std.copy_(reference.log_std)
                torch.manual_seed(31)
                with torch.no_grad():
                    expected = reference(obs)
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(31)
                result = actual.step(obs)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                for lhs, rhs in zip(expected, result):
                    torch.testing.assert_close(lhs.flatten(), rhs.flatten(), rtol=0, atol=0)

    def synthetic_rollout(self, policy, count):
        # Real HC dimensions, two parallel envs, transitions split by episode ends.
        steps, n_envs = count // 2, 2
        rng = np.random.RandomState(107)
        obs = rng.normal(size=(steps, n_envs, 18)).astype(np.float32)
        act = rng.normal(size=(steps, n_envs, 6)).astype(np.float32)
        with torch.no_grad():
            vr, vc, logp, _ = policy.evaluate_actions(
                torch.tensor(obs.reshape(-1, 18)), torch.tensor(act.reshape(-1, 6)))
        data = {
            "obs": obs, "act": act,
            "logp": logp.numpy().reshape(steps, n_envs) + rng.normal(0, 0.4, (steps, n_envs)).astype(np.float32),
            "vr": vr.numpy().reshape(steps, n_envs), "vc": vc.numpy().reshape(steps, n_envs),
            "adv_r": rng.normal(12, 3, (steps, n_envs)).astype(np.float32),
            "adv_c": rng.normal(9, 7, (steps, n_envs)).astype(np.float32),
            "ret_r": rng.normal(6, 4, (steps, n_envs)).astype(np.float32),
            "ret_c": rng.normal(4, 2, (steps, n_envs)).astype(np.float32),
            "raw_cost": rng.binomial(1, 0.3, (steps, n_envs)).astype(np.float32),
        }
        buffer = self.reference.Buffer(steps, self.obs_space, self.act_space, n_envs=n_envs)
        for original_name, new_name in {
            "observations": "obs", "orig_observations": "obs", "actions": "act",
            "log_probs": "logp", "reward_values": "vr", "cost_values": "vc",
            "reward_advantages": "adv_r", "cost_advantages": "adv_c",
            "reward_returns": "ret_r", "cost_returns": "ret_c", "orig_costs": "raw_cost",
        }.items():
            setattr(buffer, original_name, data[new_name].copy())
        buffer.full = True
        return data, buffer

    def original_solver(self, policy, buffer):
        solver = self.reference.PPO.__new__(self.reference.PPO)
        for key in ("batch_size", "n_epochs", "clip_range_reward_vf", "clip_range_cost_vf",
                    "target_kl", "ent_coef", "reward_vf_coef", "cost_vf_coef", "max_grad_norm"):
            setattr(solver, key, self.config[key])
        solver.policy, solver.rollout_buffer = policy, buffer
        solver.action_space = self.act_space
        solver.clip_range = lambda _: self.config["clip_range"]
        solver._current_progress_remaining = 1.0
        solver._update_learning_rate = lambda optimizer: None  # constant reference schedule
        solver._n_updates, solver.verbose, solver.use_sde = 0, 0, False
        solver.update_penalty_after = 1
        solver.dual = self.reference.Dual(self.config["budget"], self.config["penalty_learning_rate"],
                                          self.config["penalty_initial_value"])
        return solver

    def test_full_original_ppo_update_and_optimizer(self):
        # 130 exposes the short final minibatch; tiny KL verifies end-of-epoch stopping.
        for count, target_kl in ((128, None), (130, None), (130, 1e-8)):
            with self.subTest(count=count, target_kl=target_kl):
                self.config["target_kl"] = target_kl
                reference_policy, actual_policy = self.policies()
                # Transfer identical parameters explicitly before testing the optimizer.
                with torch.no_grad():
                    for _, source, destination in self.parameter_pairs(reference_policy, actual_policy):
                        destination.copy_(source)
                data, buffer = self.synthetic_rollout(reference_policy, count)
                reference = self.original_solver(reference_policy, buffer)
                env = types.SimpleNamespace(observation_space=self.obs_space, action_space=self.act_space)
                actual = LiuPPOLag(env, self.config)
                actual.policy = actual_policy
                actual.optimizer = torch.optim.Adam(actual.policy.parameters(), lr=self.config["learning_rate"], eps=1e-5)
                traces = [[], []]

                def capture(which, policy):
                    params = list(policy.named_parameters())
                    def hook(optimizer, args, kwargs):
                        traces[which].append({name: p.grad.detach().clone() for name, p in params})
                    return hook

                reference_policy.optimizer.register_step_pre_hook(capture(0, reference_policy))
                actual.optimizer.register_step_pre_hook(capture(1, actual_policy))
                np.random.seed(481)
                self.reference.logger.clear()
                reference.train()
                np.random.seed(481)
                metrics = actual.update(data)
                self.assertEqual(len(traces[0]), len(traces[1]))
                self.assert_parameters(reference_policy, actual_policy)
                # Compare clipped gradients and Adam moments by semantic parameter mapping.
                names = {id(param): name for name, param in actual_policy.named_parameters()}
                for reference_name, reference_param, actual_param in self.parameter_pairs(reference_policy, actual_policy):
                    actual_name = names[id(actual_param)]
                    for lhs, rhs in zip(traces[0], traces[1]):
                        torch.testing.assert_close(lhs[reference_name], rhs[actual_name], rtol=2e-5, atol=2e-7)
                    for state in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(reference_policy.optimizer.state[reference_param][state],
                                                   actual.optimizer.state[actual_param][state], rtol=2e-5, atol=2e-7)
                for reference_key, actual_key in {
                    "train/policy_gradient_loss": "policy_loss",
                    "train/reward_value_loss": "reward_value_loss",
                    "train/cost_value_loss": "cost_value_loss",
                    "train/nu": "nu", "train/nu_loss": "dual_loss",
                }.items():
                    self.assertAlmostEqual(float(self.reference.logger[reference_key]), metrics[actual_key], places=5)
                torch.testing.assert_close(reference.dual.nu.log_nu, actual.dual.log_nu, rtol=0, atol=0)
                for state in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(reference.dual.optimizer.state[reference.dual.nu.log_nu][state],
                                               actual.dual.optimizer.state[actual.dual.log_nu][state], rtol=0, atol=0)
                self.assertEqual(len(traces[1]), metrics["ppo_epochs"] * int(np.ceil(count / self.config["batch_size"])))

    def test_dual_adam_and_floor_match_original(self):
        self.config["budget"] = 0.4
        original = self.reference.Dual(0.4, 0.1, 1)
        actual = DualVariable(self.config)
        for cost in (0.0, 0.2, 1.0, 0.8, 0.0, 0.0, 0.0):
            original.update_parameter(cost)
            loss = actual.update(cost)
            self.assertEqual(original.loss.item(), loss)
            torch.testing.assert_close(original.nu.log_nu, actual.log_nu, rtol=0, atol=0)
            self.assertEqual(original.optimizer.defaults["eps"], actual.optimizer.defaults["eps"])

    def test_gae_matches_original_buffer_transition_alignment(self):
        rng = np.random.RandomState(49)
        rewards, values = [rng.normal(size=(8, 2)).astype(np.float32) for _ in range(2)]
        dones = np.zeros((8, 2), dtype=np.float32)
        dones[2, 0], dones[5, 1], dones[-1, 0] = 1, 1, 1
        last_value = rng.normal(size=2).astype(np.float32)
        buffer = self.reference.Buffer(8, self.obs_space, self.act_space, n_envs=2)
        # The reference stores previous-step done flags, ours stores current-transition flags.
        previous_dones = np.concatenate((np.zeros((1, 2), dtype=np.float32), dones[:-1]))
        returns, advantages = buffer._compute_returns_and_advantage(
            rewards, values, previous_dones, 0.99, 0.95, torch.tensor(last_value),
            dones[-1].astype(bool), np.zeros_like(values))
        actual_advantages, actual_returns = gae(rewards, values, dones, last_value, 0.99, 0.95)
        np.testing.assert_array_equal(advantages, actual_advantages)
        np.testing.assert_array_equal(returns, actual_returns)

    def test_collector_matches_original_including_final_bootstrap(self):
        self.config["n_steps"] = 8
        original_policy, actual_policy = self.policies()
        obs_space, act_space = self.obs_space, self.act_space

        class Environment:
            num_envs = 2
            observation_space, action_space = obs_space, act_space

            def __init__(self):
                self.index = 0

            def step(self, actions):
                self.index += 1
                # A conspicuous state change on every transition exposes a
                # bootstrap from next_obs rather than the reference's last obs.
                obs = np.full((2, 18), self.index * 0.4, np.float32)
                obs[:, :6] += actions
                reward = np.sum(actions, axis=-1).astype(np.float32)
                self.last_raw_cost = np.square(actions[:, 0])
                done = np.array([self.index % 5 == 0, self.index % 7 == 0])
                return obs, reward, self.last_raw_cost, done

        class OriginalEnvironment(Environment):
            def step(self, actions):
                obs, reward, cost, done = super().step(actions)
                return obs, reward, done, [{"cost": c} for c in cost]

        original_env, actual_env = OriginalEnvironment(), Environment()
        original_buffer = self.reference.Buffer(
            8, self.obs_space, self.act_space, n_envs=2,
            reward_gamma=0.99, reward_gae_lambda=0.95, cost_gamma=0.99, cost_gae_lambda=0.95)
        initial_obs = np.zeros((2, 18), np.float32)
        original = types.SimpleNamespace(
            _last_obs=initial_obs.copy(), _last_original_obs=initial_obs.copy(),
            _last_dones=np.zeros(2, bool), use_sde=False, device="cpu",
            policy=original_policy, num_timesteps=0, action_space=self.act_space,
            _update_info_buffer=lambda _: None)
        callback = types.SimpleNamespace(on_rollout_start=lambda: None, update_locals=lambda _: None,
                                         on_step=lambda: True, on_rollout_end=lambda: None)
        actual = LiuPPOLag(actual_env, self.config)
        actual.policy = actual_policy
        actual_obs = initial_obs.copy()
        # Continuing rollout checks a done boundary inherited from the previous
        # rollout as well as endings strictly inside one batch.
        for rollout in range(2):
            torch.manual_seed(71 + rollout)
            self.reference.collect(original, original_env, callback, original_buffer, 8, "cost")
            original_rng = torch.get_rng_state().clone()
            torch.manual_seed(71 + rollout)
            actual_obs, data = actual.collect(actual_obs)
            self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
            np.testing.assert_array_equal(original._last_obs, actual_obs)
            for old, new in {
                "observations": "obs", "actions": "act", "log_probs": "logp",
                "rewards": "reward", "costs": "cost", "orig_costs": "raw_cost",
                "reward_values": "vr", "cost_values": "vc",
                "reward_advantages": "adv_r", "cost_advantages": "adv_c",
                "reward_returns": "ret_r", "cost_returns": "ret_c",
            }.items():
                with self.subTest(rollout=rollout, field=new):
                    np.testing.assert_array_equal(getattr(original_buffer, old), data[new])

    def test_liu_execution_flag_preserves_raw_rollout_samples(self):
        config = dict(self.config, n_steps=2)
        obs_space, act_space = self.obs_space, self.act_space

        class Environment:
            num_envs = 1
            observation_space, action_space = obs_space, act_space

            def __init__(self, clip_actions):
                self.clip_actions = clip_actions
                self.actions = []

            def step(self, action):
                self.actions.append(action.copy())
                self.last_raw_cost = np.zeros(1, dtype=np.float32)
                return (np.zeros((1, 18), dtype=np.float32),
                        -.1 * np.square(action).sum(-1), self.last_raw_cost,
                        np.zeros(1, dtype=bool))

        for enabled in (False, True):
            with self.subTest(clip_actions=enabled):
                env = Environment(enabled)
                learner = LiuPPOLag(env, config)
                with torch.no_grad():
                    layers = [m for m in learner.policy.actor.mean.modules()
                              if isinstance(m, torch.nn.Linear)]
                    for layer in layers:
                        layer.weight.zero_()
                        layer.bias.zero_()
                    layers[-1].bias.fill_(2.0)
                    learner.policy.actor.log_std.fill_(-2.0)
                initial = np.zeros((1, 18), dtype=np.float32)
                np.testing.assert_array_equal(learner.predict(initial, deterministic=True),
                                              np.full((1, 6), 1.0 if enabled else 2.0))
                _, data = learner.collect(initial)
                self.assertTrue((data['act'] > 1).all())
                executed = np.asarray(env.actions)
                np.testing.assert_array_equal(executed,
                                              np.clip(data['act'], -1, 1) if enabled else data['act'])


if __name__ == "__main__":
    unittest.main()
