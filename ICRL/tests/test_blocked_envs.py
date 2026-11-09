"""Task checks and original-source equations on the installed MuJoCo engine.

The differential checks require Liu's sibling checkout (or LIU_SOURCE_ROOT).
"""

import ast
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from gymnasium.envs.mujoco import MujocoEnv

from ICRL.blocked_envs import ASSET_DIR, HORIZONS, LiuBlockedEnv, benchmark_random_state, make_raw_env, true_cost
from omnisafe.envs.wrapper import AutoReset, TimeLimit


SOURCE_ROOT = Path(os.environ.get('LIU_SOURCE_ROOT',
    str(Path(__file__).resolve().parents[3] / 'ICRL-benchmarks-public')))
SOURCE_ENVS = SOURCE_ROOT / 'mujuco_environment/custom_envs/envs'
SOURCE_CLASSES = {
    'HCWithPos-v0': ('half_cheetah.py', 'HalfCheetahWithPos'),
    'AntWall-v0': ('ant.py', 'AntWall'),
    'WalkerWithPos-v0': ('walker.py', 'WalkerWithPos'),
    'SwimmerWithPos-v0': ('swimmer.py', 'SwimmerWithPos'),
}


class _SimulatorShim:
    """Map source sim.data and sim.step calls onto the installed simulator."""

    def __init__(self, env):
        self.env = env
        self.sim = SimpleNamespace(data=env.data,
                                   step=lambda: mujoco.mj_step(env.model, env.data))

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def __getattr__(self, name):
        if name == 'done':
            return False  # AntWall explicitly disables unhealthy termination.
        return getattr(self.env, name)


def original_class(env_id):
    filename, name = SOURCE_CLASSES[env_id]
    tree = ast.parse((SOURCE_ENVS / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)
               and n.name in {'step', '_get_obs', 'old_reward', 'new_reward', 'reset_model'}]
    ref = ast.ClassDef(name='Original', bases=[ast.Name(id='_SimulatorShim', ctx=ast.Load())],
                       keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[ref], type_ignores=[]))
    namespace = {'np': np, '_SimulatorShim': _SimulatorShim, 'REWARD_TYPE': 'old'}
    exec(compile(module, str(SOURCE_ENVS / filename), 'exec'), namespace)
    return namespace['Original']


class BlockedEnvironmentTests(unittest.TestCase):
    def test_default_tasks_use_installed_mujoco(self):
        for env_id in HORIZONS:
            with self.subTest(env_id=env_id):
                env = LiuBlockedEnv(env_id)
                try:
                    self.assertIsInstance(env.raw_env, MujocoEnv)
                    obs, _ = env.reset(seed=123)
                    transition = env.step(torch.zeros(env.action_space.shape))
                    self.assertEqual(len(transition), 6)
                    self.assertEqual(transition[0].shape, obs.shape)
                    self.assertTrue(torch.isfinite(transition[0]).all())
                finally:
                    env.close()
        with self.assertRaisesRegex(ValueError, 'Only the installed MuJoCo backend'):
            LiuBlockedEnv('HCWithPos-v0', backend='unavailable')

    def test_dimensions_reset_distribution_and_seed_replay(self):
        for env_id, obs_dim, action_dim in (
                ('HCWithPos-v0', 18, 6), ('AntWall-v0', 113, 8),
                ('WalkerWithPos-v0', 18, 6), ('SwimmerWithPos-v0', 10, 2)):
            with self.subTest(env_id=env_id):
                env = make_raw_env(env_id)
                try:
                    observation, _ = env.reset(seed=123)
                    self.assertEqual(observation.shape, (obs_dim,))
                    self.assertEqual(env.action_space.shape, (action_dim,))
                    scale = 0.005 if env_id == 'WalkerWithPos-v0' else 0.1
                    rng = benchmark_random_state(123)
                    qpos = env.init_qpos + rng.uniform(-scale, scale, env.model.nq)
                    velocity_noise = (rng.randn(env.model.nv) * scale
                                      if env_id in ('HCWithPos-v0', 'AntWall-v0')
                                      else rng.uniform(-scale, scale, env.model.nv))
                    # set_state normalizes Ant's free-joint quaternion in MuJoCo.
                    if env_id == 'AntWall-v0':
                        qpos[3:7] /= np.linalg.norm(qpos[3:7])
                    np.testing.assert_allclose(env.data.qpos, qpos, atol=1e-14)
                    np.testing.assert_allclose(env.data.qvel, env.init_qvel + velocity_noise, atol=1e-14)
                    replay, _ = env.reset(seed=123)
                    np.testing.assert_array_equal(observation, replay)
                finally:
                    env.close()

    def test_strict_cost_boundaries(self):
        for env_id in HORIZONS:
            boundary = 0.5 if env_id == 'SwimmerWithPos-v0' else -3.0
            obs = np.array([[boundary - 1e-6], [boundary], [boundary + 1e-6]])
            expected = [0, 0, 1] if boundary == 0.5 else [1, 0, 0]
            np.testing.assert_array_equal(true_cost(env_id, obs), expected)

    def test_configured_boundary_and_horizon_reach_wrapped_environment(self):
        task = LiuBlockedEnv('HCWithPos-v0', backend='modern', cost_boundary=-1.0,
                             max_episode_steps=2)
        env = AutoReset(TimeLimit(task, task.max_episode_steps, torch.device('cpu')),
                        torch.device('cpu'))
        try:
            env.reset(seed=123)
            raw = task.raw_env
            position = raw.init_qpos.copy()
            position[0] = -1.1
            raw.set_state(position, raw.init_qvel * 0)
            _, _, cost, terminated, truncated, _ = env.step(torch.zeros(6))
            self.assertEqual(cost.item(), 1.0)
            self.assertFalse(terminated.item())
            self.assertFalse(truncated.item())
            _, _, _, _, truncated, info = env.step(torch.zeros(6))
            self.assertTrue(truncated.item())
            self.assertIn('final_observation', info)
        finally:
            env.close()

    def test_cost_uses_successor_and_does_not_terminate_hc(self):
        env = LiuBlockedEnv('HCWithPos-v0', backend='modern')
        try:
            env.reset(seed=123)
            qpos, qvel = env.raw_env.init_qpos.copy(), env.raw_env.init_qvel.copy()
            qpos[0], qvel[0] = -2.999, -1.0
            env.raw_env.set_state(qpos, qvel)
            obs, _, cost, terminated, truncated, info = env.step(torch.zeros(6))
            self.assertLess(obs[0].item(), -3.0)
            self.assertEqual(cost.item(), 1.0)
            self.assertEqual(info['lag_cost'], 1)
            self.assertFalse(terminated.item())
            self.assertFalse(truncated.item())
        finally:
            env.close()

    def test_full_walker_velocity_is_not_clipped(self):
        env = make_raw_env('WalkerWithPos-v0')
        try:
            env.reset(seed=7)
            velocity = np.linspace(-20, 20, env.model.nv)
            env.set_state(env.init_qpos, velocity)
            np.testing.assert_array_equal(env._get_obs()[-env.model.nv:], velocity)
        finally:
            env.close()

    def test_walker_body_frame_repair_preserves_generalized_dynamics(self):
        modern = mujoco.MjModel.from_xml_path(str(ASSET_DIR / 'walker2d.xml'))
        root = ET.fromstring((ASSET_DIR / 'walker2d.xml').read_text())
        for body in root.iter('body'):
            if body.get('name') in ('foot', 'foot_left'):
                body.set('pos', '0.1 0 0.1')  # literal numerical evaluation of 0.2/2
        repaired = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        rng = np.random.RandomState(42)
        for _ in range(100):
            qpos = modern.qpos0 + rng.uniform(-0.05, 0.05, modern.nq)
            qvel = rng.uniform(-0.2, 0.2, modern.nv)
            action = rng.uniform(-1, 1, modern.nu)
            outputs = []
            for model in (modern, repaired):
                data = mujoco.MjData(model)
                data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, action
                mujoco.mj_forward(model, data)
                mass = np.empty((model.nv, model.nv))
                mujoco.mj_fullM(model, mass, data.qM)
                acceleration = data.qacc.copy()
                mujoco.mj_step(model, data, nstep=4)
                outputs.append((mass, acceleration, data.qpos.copy(), data.qvel.copy()))
            for first, second in zip(*outputs):
                np.testing.assert_allclose(first, second, rtol=1e-11, atol=1e-9)

    def test_horizon_is_truncation_and_autoreset_preserves_terminal_state(self):
        for env_id, horizon in HORIZONS.items():
            env = LiuBlockedEnv(env_id, backend='modern')
            self.assertEqual(env.max_episode_steps, horizon)
            env.close()
        env = AutoReset(TimeLimit(LiuBlockedEnv('HCWithPos-v0', backend='modern'), 2, torch.device('cpu')),
                        torch.device('cpu'))
        try:
            env.reset(seed=1)
            env.step(torch.zeros(6))
            obs, _, _, terminated, truncated, info = env.step(torch.zeros(6))
            self.assertFalse(terminated.item())
            self.assertTrue(truncated.item())
            self.assertIn('final_observation', info)
            self.assertFalse(torch.equal(obs, info['final_observation']))
        finally:
            env.close()

    def test_removing_extra_contact_postprocessing_preserves_hc_trajectories(self):
        port, gymnasium_env = make_raw_env('HCWithPos-v0'), make_raw_env('HCWithPos-v0')
        # The old port inherited this method, including mj_rnePostConstraint.
        gymnasium_env.do_simulation = MethodType(MujocoEnv.do_simulation, gymnasium_env)
        try:
            port.reset(seed=87)
            gymnasium_env.reset(seed=87)
            rng = np.random.RandomState(31)
            for _ in range(100):
                action = rng.uniform(-1, 1, port.action_space.shape)
                actual, previous = port.step(action), gymnasium_env.step(action)
                np.testing.assert_array_equal(port.data.qpos, gymnasium_env.data.qpos)
                np.testing.assert_array_equal(port.data.qvel, gymnasium_env.data.qvel)
                np.testing.assert_array_equal(actual[0], previous[0])
                self.assertEqual(actual[1:], previous[1:])
        finally:
            port.close()
            gymnasium_env.close()

    def test_ant_preserves_original_zero_contact_fields(self):
        env = make_raw_env('AntWall-v0')
        try:
            env.reset(seed=87)
            self.assertEqual(env.model.nsensor, 0)
            rng = np.random.RandomState(31)
            for _ in range(100):
                observation, _, _, _, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
                np.testing.assert_array_equal(observation[-84:], np.zeros(84))
                np.testing.assert_array_equal(env.data.cfrc_ext, np.zeros_like(env.data.cfrc_ext))
                self.assertEqual(info['reward_contact'], 0.0)
        finally:
            env.close()

    @unittest.skipUnless(SOURCE_ENVS.exists(), 'Liu source checkout unavailable')
    def test_original_step_source_on_identical_modern_simulators(self):
        for env_id in HORIZONS:
            with self.subTest(env_id=env_id):
                port, reference_env = make_raw_env(env_id), make_raw_env(env_id)
                try:
                    port.reset(seed=87)
                    reference_env.reset(seed=87)
                    reference = original_class(env_id)(reference_env)
                    rng = np.random.RandomState(31)
                    for _ in range(20):
                        action = rng.uniform(-1, 1, port.action_space.shape)
                        obs, reward, terminated, truncated, info = port.step(action)
                        old_obs, old_reward, old_done, old_info = reference.step(action)
                        np.testing.assert_array_equal(obs, old_obs)
                        self.assertEqual(reward, old_reward)
                        self.assertEqual(terminated, old_done)
                        self.assertFalse(truncated)
                        self.assertEqual(info.keys(), old_info.keys())
                        for key in info:
                            np.testing.assert_array_equal(info[key], old_info[key])
                finally:
                    port.close()
                    reference_env.close()

    @unittest.skipUnless(SOURCE_ENVS.exists(), 'Liu source checkout unavailable')
    def test_original_hc_reset_source(self):
        port, reference_env = make_raw_env('HCWithPos-v0'), make_raw_env('HCWithPos-v0')
        try:
            reference = original_class('HCWithPos-v0')(reference_env)
            for seed in (0, 123, 999):
                actual, _ = port.reset(seed=seed)
                reference_env._np_random = benchmark_random_state(seed)
                expected = reference.reset_model()
                np.testing.assert_array_equal(actual, expected)
        finally:
            port.close()
            reference_env.close()

if __name__ == '__main__':
    unittest.main()
