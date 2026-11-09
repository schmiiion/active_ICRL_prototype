"""Train using OmniSafe's native configs or an explicit experiment YAML.

Run from the repository root::

    python -m examples.train_from_config --algo PPOLag --env-id HCWithPos-v0 \
        --env-import ICRL.blocked_envs

Native mode loads omnisafe/configs/<algorithm-type>/<algorithm>.yaml, including
the selected environment's overrides. Optional external YAML paths use the
working directory, as do paths inside those YAMLs.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Optional external experiment YAML.')
    parser.add_argument('--algo', help='Algorithm whose native config should be loaded.')
    parser.add_argument('--env-id', help='Environment section in the native config.')
    parser.add_argument('--env-import', action='append', default=[],
                        help='Import a custom environment registration module; repeatable.')
    args = parser.parse_args()

    if args.config is not None:
        if args.algo is not None or args.env_id is not None:
            parser.error('Use --config or --algo with --env-id, not both.')
        with args.config.open(encoding='utf-8') as file:
            config = yaml.safe_load(file)
        algo, env_id = config['algo'], config['env_id']
        imports = config.get('env_imports', []) + args.env_import
        agent_kwargs = {'custom_cfgs': config.get('custom_cfgs', {})}
    else:
        if args.algo is None or args.env_id is None:
            parser.error('Provide --algo and --env-id, or --config PATH.')
        algo, env_id = args.algo, args.env_id
        imports, agent_kwargs = args.env_import, {}
    for module in imports:
        importlib.import_module(module)

    import omnisafe

    agent = omnisafe.Agent(
        algo,
        env_id,
        **agent_kwargs,
    )
    agent.learn()


if __name__ == '__main__':
    main()
