"""Expert readers preserve raw state/action alignment and reference ordering."""

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from ICRL.demonstrations import load_demonstrations, load_trajectory, validate_demonstration_environment


class DemonstrationsTest(unittest.TestCase):
    def test_collector_manifest_rejects_different_physics_or_task(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'cost_boundary': -3.0, 'max_episode_steps': 1000}
            validate_demonstration_environment(directory, 'HCWithPos-v0', config, 'modern')
            manifest = {'env_id': 'HCWithPos-v0', 'env_cfgs': {'backend': 'modern', **config}}
            (Path(directory) / 'manifest.json').write_text(json.dumps(manifest))
            validate_demonstration_environment(directory, 'HCWithPos-v0', config, 'modern')
            with self.assertRaisesRegex(ValueError, 'backend'):
                validate_demonstration_environment(directory, 'HCWithPos-v0', config, 'other-runtime')
            for key, value in [('cost_boundary', -1.0), ('max_episode_steps', 500)]:
                with self.subTest(setting=key), self.assertRaisesRegex(ValueError, key):
                    validate_demonstration_environment(directory, 'HCWithPos-v0',
                                                       {**config, key: value}, 'modern')
            with self.assertRaisesRegex(ValueError, 'environment'):
                validate_demonstration_environment(directory, 'WalkerWithPos-v0', config, 'modern')

    def test_numpy_and_reference_pickles_load_the_same_raw_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays, pickles = root / 'numpy', root / 'liu'
            arrays.mkdir()
            pickles.mkdir()
            for name, value, length in [('scene-2', 2, 3), ('scene-10', 10, 2), ('scene-0', 0, 1)]:
                obs = np.full((length, 3), value, dtype=np.float64)
                actions = np.full((length, 2), value / 10, dtype=np.float32)
                np.savez(arrays / f'{name}.npz', observations=obs, actions=actions,
                         next_observations=obs + 1)
                (pickles / f'{name}.pkl').write_bytes(pickle.dumps({
                    'original_observations': obs, 'observations': obs + 100, 'actions': actions}))
            numpy_obs, numpy_actions, numpy_hashes = load_demonstrations(arrays, 2)
            liu_obs, liu_actions, liu_hashes = load_demonstrations(pickles, 2)
            np.testing.assert_array_equal(numpy_obs, liu_obs)
            np.testing.assert_array_equal(numpy_actions, liu_actions)
            np.testing.assert_array_equal(numpy_obs[:, 0], [0, 10, 10])
            self.assertEqual(list(numpy_hashes), ['scene-0.npz', 'scene-10.npz'])
            self.assertEqual(list(liu_hashes), ['scene-0.pkl', 'scene-10.pkl'])
            for name, digest in numpy_hashes.items():
                self.assertEqual(digest, hashlib.sha256((arrays / name).read_bytes()).hexdigest())

    def test_mixed_formats_and_insufficient_episodes_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_demonstrations(root, 1)
            np.savez(root / 'episode.npz', observations=np.zeros((1, 3)), actions=np.zeros((1, 2)))
            (root / 'episode.pkl').write_bytes(pickle.dumps({}))
            with self.assertRaisesRegex(ValueError, 'separate directories'):
                load_demonstrations(root, 1)

    def test_numpy_object_arrays_are_not_unpickled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.npz'
            np.savez(path, observations=np.array([{}], dtype=object), actions=np.zeros((1, 2)))
            with self.assertRaisesRegex(ValueError, 'Object arrays'):
                load_trajectory(path)


if __name__ == '__main__':
    unittest.main()
