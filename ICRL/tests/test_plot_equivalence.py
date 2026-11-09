"""Worker ordering and monitor precision match the original Figure 2 reader."""

import ast
import csv
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ICRL.plot_results import read_runs


class PlotEquivalence(unittest.TestCase):
    def test_worker_ordinals_match_original_monitor_reader(self):
        reference = Path(os.environ.get('LIU_ICRL_REFERENCE',
                         Path(__file__).resolve().parents[3] / 'ICRL-benchmarks-public'))
        source = reference / 'utils/data_utils.py'
        if not source.exists():
            self.skipTest('Original source checkout unavailable')
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'read_running_logs')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workers = [[10.123, 20.456, 30.789], [40.123, 50.456]]
            paths = []
            for worker, rewards in enumerate(workers):
                path = root / f'{worker}.monitor.csv'
                paths.append(str(path))
                with path.open('w') as stream:
                    stream.write('# metadata\nreward,reward_nc,len,time,constraint\n')
                    for reward in rewards:
                        stream.write(f'{round(reward,2)},{round(reward,2)},10,1,0\n')
            expected = namespace['read_running_logs'](paths, ['reward_nc', 'constraint'],
                                                       1e6, -1e6, 1000)[0]
            path = root / 'episodes.csv'
            fields = ['method', 'env', 'seed', 'episode', 'worker', 'worker_episode',
                      'reward', 'reward_nc', 'constraint']
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for episode, (worker, index) in enumerate([(0,0),(0,1),(1,0),(0,2),(1,1)]):
                    reward = workers[worker][index]
                    writer.writerow(dict(method='VICRL', env='WalkerWithPos-v0', seed=123,
                                         episode=episode+1, worker=worker, worker_episode=index+1,
                                         reward=reward, reward_nc=reward, constraint=0))
            runs, _ = read_runs([path], ['VICRL'])
            actual = runs['Walker', 'VICRL', '123']
            np.testing.assert_array_equal(actual[:, 2], expected['reward_nc'])
            np.testing.assert_array_equal(actual[:, 3], expected['constraint'])


if __name__ == '__main__':
    unittest.main()
