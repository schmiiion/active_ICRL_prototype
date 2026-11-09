"""Replay eviction preserves complete, aligned trajectory records."""

import random
import unittest

import numpy as np

from ICRL.utils.memory_buffer import IRLDataQueue


class QueueBeforeLengthFix(IRLDataQueue):
    """Reproduce the previous eviction solely for fixed-length/RNG comparison."""

    def pop(self, index):
        del self.store_obs[index]
        del self.store_acts[index]
        del self.store_rs[index]


class MemoryBufferTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())

    @staticmethod
    def trajectories(lengths):
        observations = [np.full((length, 2), index, np.float32)
                        for index, length in enumerate(lengths)]
        actions = [np.full((length, 1), index + 100, np.float32)
                   for index, length in enumerate(lengths)]
        rewards = [np.arange(length) + index * 1000 for index, length in enumerate(lengths)]
        return observations, actions, rewards, lengths

    def assert_aligned(self, records):
        for observations, actions, rewards, length in zip(*records):
            identity = int(observations[0, 0])
            self.assertEqual(length, len(observations))
            self.assertEqual(length, len(actions))
            self.assertEqual(length, len(rewards))
            np.testing.assert_array_equal(observations, identity)
            np.testing.assert_array_equal(actions, identity + 100)
            np.testing.assert_array_equal(rewards, np.arange(length) + identity * 1000)

    def test_overflow_keeps_variable_length_records_aligned_and_bounded(self):
        records = self.trajectories(list(range(1, 13)))
        queue = IRLDataQueue(max_rollouts=3, seed=42)
        for index in range(12):
            queue.put(*[[values[index]] for values in records])
            stored = (queue.store_obs, queue.store_acts, queue.store_rs, queue.store_ls)
            self.assertEqual([len(values) for values in stored], [min(index + 1, 3)] * 4)
            self.assert_aligned(stored)
        self.assert_aligned(queue.get(50))

    def test_fixed_1000_step_hc_samples_and_rng_match_previous_behavior(self):
        records = self.trajectories([1000] * 17)
        for seed in (0, 42, 123):
            with self.subTest(seed=seed):
                previous = QueueBeforeLengthFix(max_rollouts=3, seed=seed)
                previous.put(*records)
                expected = previous.get(25)
                expected_rng = random.getstate()
                actual = IRLDataQueue(max_rollouts=3, seed=seed)
                actual.put(*records)
                sampled = actual.get(25)
                for old_values, new_values in zip(expected, sampled):
                    for old_value, new_value in zip(old_values, new_values):
                        np.testing.assert_array_equal(old_value, new_value)
                self.assertEqual(random.getstate(), expected_rng)
                self.assertEqual(len(actual.store_ls), 3)


if __name__ == '__main__':
    unittest.main()
