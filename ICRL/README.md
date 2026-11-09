# MECL and VICRL on Liu et al.'s Blocked tasks

Upstream licenses and attribution are preserved in [third-party notices](third_party/NOTICE).

This module ports the MECL and VICRL experiments for Blocked Half-Cheetah,
Ant, Walker, and Swimmer from *Benchmarking Constraint Inference in Inverse
Reinforcement Learning* (ICLR 2023). `MECL` is the paper's label for the original
repository's `ICRL_Pos_with-action` method. The original source commit and exact
config/demo checksums are recorded in `data/liu_provenance.json`. All current
environments use the installed OmniSafe/Gymnasium MuJoCo runtime. This changes
physics relative to the original paper's simulator.

All task changes, normalization, cost replacement, and reproduction-specific
optimization live under `ICRL/`. No changes to the OmniSafe framework are needed.
Importing `ICRL.blocked_envs` registers four additional CMDP IDs; the ordinary
OmniSafe and Safety-Gymnasium tasks retain their original behavior.

## Run

From the repository root, in the existing `active_ICRL` conda environment:

```bash
python -m ICRL.interface.train_icrl --config ICRL/config/VICRL_HC.yaml
python -m ICRL.interface.train_icrl --config ICRL/config/MECL_HC.yaml
```

Edit one complete experiment YAML: `config/{MECL,VICRL}_{HC,Ant,Walker,Swimmer}.yaml`.
Its settings include seeds, device, worker count, simulator backend, input/output
paths, the wall boundary and horizon, normalization and clipping, PPO networks
and optimization, constraint networks and optimization, demonstrations, outer
iterations, sampling, evaluation and checkpoints. No hyperparameter arguments
are required. Relative paths in these files resolve from the YAML's directory.
The complete resolved configuration is saved in each run, along with hashes of
the selected config and reference config. Output directories must be new.

`clip_actions` is a top-level switch in each experiment YAML. The active presets
set it to `true`, following Liu's collector: clip a copy of each sampled action
to the policy-space bounds before `ActionScale` and environment execution. PPO
still stores the original sampled action and its original log probability.
Nominal trajectories and learned-cost inputs use the executed policy-coordinate
action. Setting `clip_actions: false` restores the earlier native runs' unclipped
execution. The switch applies consistently to ICRL training, sampling and
evaluation, entirely inside `ICRL/`; ordinary OmniSafe forward training is
unaffected. Existing run outputs and expert datasets are unchanged.

`run.solver: omnisafe` selects native OmniSafe `PPOLag` with persistent
actor/critics, optimizers, multiplier and normalizers across constraint updates.
The active `VICRL_HC.yaml` and `MECL_HC.yaml` use this solver and the same ten
newly generated expert trajectories. Their `omnisafe` blocks expose the native
optimizer's settings. The MECL HC preset matches VICRL's shared settings,
including replay (capacity 500), while retaining MECL's objective and per-step
importance sampling. This controlled comparison differs from Liu's released
MECL preset, which disables replay.
`run.solver: liu` retains the source-equivalent solver used by the other tasks' presets.
`run.backend: modern` identifies the only supported
simulator runtime; it does not select a PPO implementation.
The original copied YAMLs in `config/liu/` remain unchanged as provenance; they
are reference inputs rather than runnable experiment configs. Active configs
omit copied options that the implementation does not support, such as SDE and
PID multiplier updates. Task reward equations and XML assets remain part of the
environment definition.

The older `--method VICRL --task HC` syntax selects the corresponding active
config. Other CLI flags remain optional explicit overrides for diagnostics;
their defaults no longer override values in YAML. The native solver currently
uses one simulator and one process; the Liu presets use five independent
simulators. CPU PyTorch uses one thread by default. Environments execute
synchronously in the current Python process using the installed `mujoco` package
(currently MuJoCo 2.3.3 with Gymnasium 0.28.1). Each environment has its own reset
RNG. Observation/reward/cost statistics are
updated jointly after each batch, matching the reference subprocess layout.
Pass `--device cuda` on a supported machine to use CUDA neural networks.

Each native MECL/VICRL HC outer iteration runs exactly 200,000 training transitions.
The active VICRL HC config uses 60 iterations (12,000,000 training transitions,
12,000 episodes); MECL HC uses 30 iterations (6,000,000 transitions). Each native epoch collects
20,000 transitions, aligned with the 1,000-step episode horizon. PPO uses ten
update passes, subject to native KL early stopping. Both constraint networks
retain their respective source learning settings.
The native learned-cost budget is zero, matching Liu's target; this differs
from the expert policy's known-cost training budget of one. Native
observation/reward/cost normalization is enabled. Native normalization,
initialization, GAE and PPO-Lag update semantics are retained rather than
replaced with the Liu solver's implementations.

For the Liu solver, a 200,000-step budget rounds up to 20 complete rollouts of
5 x 2,048 transitions: 204,800 per iteration (6,144,000 over 30 iterations).
The sampler and evaluation add separate transitions. Other tasks have their own budgets
and hyperparameters in their active configs; these are not replaced with HC defaults.

The reference plots use seeds 123, 321, 456, 654, and 666. For the full requested
comparison, run each method/task/seed combination:

```bash
for task in HC Ant Walker Swimmer; do
  for method in MECL VICRL; do
    for seed in 123 321 456 654 666; do
      python -m ICRL.interface.train_icrl --method "$method" --task "$task" --seed "$seed"
    done
  done
done
```

Output directories must be new. `--output PATH` selects an alternative path.
For a short execution check (not a reproduction of the learning curves):

```bash
python -m ICRL.interface.train_icrl --method VICRL --task HC \
  --iterations 2 --forward-steps 20000 --output ICRL/diagnostics/quick_VICRL_HC
```

The bundled reference demonstrations are unmodified local copies of Liu's
released data. The loader accepts either original `.pkl` trajectories or numeric
`.npz` trajectories generated by the collector below, in separate directories.
It selects the first N lexicographically sorted files, exactly as the reference
does. File hashes are recorded in each run's `metadata.json`. For normal
experiments, edit `running.expert_path`; `--expert-data PATH` is an optional override.
`running.save_every` controls numbered outer-iteration checkpoints; the latest
`checkpoint.pt` is also refreshed after each iteration. Sampling and evaluation
are stochastic by default, as in Liu; their deterministic settings are explicit.

## Expert training with native OmniSafe

Expert policy training is ordinary forward constrained RL. It calls native
`omnisafe.Agent('PPOLag', ...)`, using the custom environment's true cost.
This standard training path currently retains native action execution without
additional clipping. The ICRL/collector `clip_actions` setting does not change
the expert's training objective; clipped collection alone does not retrain it.
There is no known-cost mode in the ICRL learner. Edit the `HCWithPos-v0` section
of `omnisafe/configs/on-policy/PPOLag.yaml` and launch it with the generic example:

```bash
python -m examples.train_from_config --algo PPOLag --env-id HCWithPos-v0 \
  --env-import ICRL.blocked_envs
```

The native loader inherits the `defaults` section and applies the `HCWithPos-v0`
overrides in the same file. There is no separate copy of PPO-Lag defaults to
keep synchronized. This task uses modern physics, a strict post-step wall at
**x < -3**, a 1,000-step horizon, and a mean raw episode-cost budget of **1.0**.
Native hyperparameter names are retained. In contrast,
the Liu solver's `PPO.budget` is a mean per-step raw cost. Logs and policy/observation
normalizer checkpoints go under `runs/experts/HC_modern/`.

The launcher lives in `examples/` and contains only config selection, custom task
registration and the native Agent call. It supplies no hyperparameter overrides
in native mode and adds no optimization implementation. The environment
definition is shared with ICRL through `ICRL.blocked_envs`.

## Collect expert trajectories

`collect_experts.py` is a standalone data exporter for native OmniSafe actor
snapshots. It does not train a policy or constraint model. Edit
`config/collect_experts_HC.yaml`, then run from the repository root:

```bash
python -m ICRL.collect_experts --config ICRL/config/collect_experts_HC.yaml
```

The config selects the training run, checkpoint, output directory, number of
accepted episodes, seed, stochastic/deterministic policy, cost/reward filters,
maximum attempts and Torch threads. Paths resolve relative to the YAML. The
output directory must be new; edit `output` when collecting another dataset.
The supplied config points to the 4M-step HC policy and requests ten stochastic
episodes with exactly zero raw true cost. The optional reward filter is disabled.
It explicitly enables `clip_actions` and writes a new `HC_seed123_clipped`
directory, leaving the existing `HC_seed123` dataset intact.

The collector restores the environment/backend/boundary/horizon and actor
architecture from the run's saved `config.json`. It normalizes policy inputs
with frozen checkpoint statistics, while storing raw observations. Its optional
top-level `clip_actions` overrides the saved ICRL run's action-execution setting;
if omitted, collection inherits that setting, or uses unclipped execution for
historical forward runs without it. Stored `actions` are the executed
policy-coordinate actions, before `ActionScale`. Single episodes are collected
without AutoReset, so their final next observation is the true terminal state.
Only complete episodes passing the filters are saved.

Each `trajectory-000000.npz` contains numeric NumPy arrays, loadable with
`np.load(path, allow_pickle=False)`:

| Array | Meaning | HC shape |
| --- | --- | --- |
| `observations` | Raw pre-action observation, s_t | (1000, 18) |
| `actions` | Executed policy-coordinate action, before ActionScale | (1000, 6) |
| `next_observations` | Raw successor observation, s_t+1 | (1000, 18) |
| `rewards`, `costs` | Raw transition reward and true cost | (1000,) each |
| `terminated`, `truncated` | Transition end flags | (1000,) each |

`manifest.json` records every attempted episode, including rejected episodes,
seeds, returns, costs, lengths, acceptance rate, checkpoint/config hashes,
environment settings, package versions and saved-file hashes. Exhausting
`max_attempts` leaves a truthful partial manifest and raises an error.

To use the dataset, set `running.expert_path` in the selected MECL/VICRL config
and choose matching environment settings. The runner checks the manifest's
backend, boundary and horizon to prevent accidental mismatches. The active
native MECL and VICRL HC configs still select the existing `HC_seed123` dataset;
collecting a new dataset does not automatically change that selection. Set
`running.expert_path` explicitly when choosing the newly collected trajectories.

## TensorBoard and Figure 2 data

```bash
python -m tensorboard.main --logdir ICRL/runs/VICRL_HC_omnisafe_seed123_60iters
```

The outer run's `tb/` logs raw episode reward, reward before the first true
violation, true/learned episode costs, episode violation flags and lengths.
`Figure2/*Raw` uses zero-based episode count. The smoothed
`Figure2/ConstraintViolationRate` and `Figure2/FeasibleReward` use exactly Liu's
forward 100-episode mean, two-decimal reward rounding and N-100 output points.
Each smoothed point is emitted after the next episode arrives so the source's
discarded final window is preserved. Full unsmoothed data live in `episodes.csv`.

`Outer/*` logs evaluation, sample counts, learned costs and parameter changes;
`Constraint/*` logs objective components, last-batch gradient norm and learning
rate, including MECL importance weights and KL stopping metrics. VICRL also logs
expert/nominal alpha, beta and expected cost statistics under `Posterior/*`.
`Forward/*` mirrors native PPO metrics against total training transitions at
each outer iteration. The native logger under `forward/.../tb/` additionally
logs every PPO epoch, including policy/critic losses, KL, entropy, policy standard
deviation, clipped gradient norms, multiplier and true safety metrics.

`config.yaml`, `metadata.json` and the native logger's `config.json` record
resolved settings, dataset hashes, runtime versions and code hashes. Outer
checkpoints include actor/critics, all optimizers, multiplier, normalizers,
constraint state, replay data and RNG state. A single native run is a transfer
experiment; it does not supply the paper's five-seed uncertainty band.

## Implementation and numerical checks

- `blocked_envs.py`: original reward equations, observations, reset noise,
  positional constraints, fall termination, and horizons. Original XML assets
  are bundled in `env_assets/`. Both the registered CMDPs and `make_raw_env`
  use the same installed MuJoCo runtime.
- `normalization.py`: Liu's float64 running moments and discounted-return
  reward/cost scaling, including clipping and reset behavior.
- `rollouts.py`: cached pre-action state/action inputs to the learned cost;
  post-action state for the true positional constraint; independent workers;
  original sampling/evaluation and per-episode metrics.
- `native_ppo.py`, `ICRL_PPOLag.py`, `native_sampling.py`, `learned_cost_adapter.py`:
  native PPO-Lag continuation, learned-cost substitution, raw-state sampling,
  frozen evaluation statistics and per-episode safety/Figure 2 logging.
- `liu_ppo.py`: OmniSafe Gaussian actor and value critics with original-equivalent
  initialization, rollout GAE, and PPO-Lagrangian loss/optimizer/dual updates.
  Using default OmniSafe `PPOLag` would change the cost surrogate, advantage
  normalization, optimizer, and multiplier update, so this solver is isolated.
- `constraint_models/`: original MECL/VICRL losses, Beta sampling, regularization,
  importance sampling, and learning-rate schedules. The MLP factory consumes the
  same RNG stream as the reference. Numerical quirks in the source are retained
  when changing them would alter the experiment.

The same config-first entry point selects either forward solver; the VICRL
constraint-network objective is shared by both.

**Liu solver's preserved collector quirk:** the original `OnPolicyWithCostAlgorithm`
bootstraps the final transition with the critic values evaluated before that
transition, rather than evaluating the returned next observation. This solver
deliberately retains that behavior. It also retains the original boolean final
done flags, which promote the final GAE intermediates to float64 before storage
as float32. The collector test executes the original method on two consecutive
rollouts and compares every stored field, including advantages and returns.
Changing either behavior would define a different experiment.

The differential tests execute the original source using the same PyTorch
runtime and identical tensors, rather than comparing independently drifting
learning curves. They check initialization, network outputs, losses, gradients,
parameter updates, Adam moments, GAE, clipping, and dual updates:

```bash
python -m unittest discover -s ICRL/tests -v
python -m ICRL.validate_run ICRL/runs/your_run
```

For validation only, the original repository is expected beside this repository
at `../ICRL-benchmarks-public`, or set `LIU_ICRL_REFERENCE` to its location.
Training does not import or otherwise depend on that checkout.

Every run records `config.yaml`, provenance, `episodes.csv`, `iterations.csv`,
constraint-loss metrics, and a model checkpoint. It also records actual first
rollout/sampler inputs and pre-update optimizer states so `validate_run` can
replay real updates against unmodified original classes. These files establish
numerical update parity on fixed data. The validator also reconstructs the saved
advantages and returns using the original buffer and collector bootstrap,
requiring exact agreement before it compares optimizer updates. A successful
optimizer-only replay is insufficient to establish collector parity. These
checks do not imply identical trajectories
across different MuJoCo or PyTorch versions. See `reproduction_audit.md` and
`env_assets/AUDIT.md` for the source comparison and simulator limitations.
For native runs, `validate_run` replays the constraint update against source
and explicitly marks source-PPO replay as not applicable: native PPO-Lag has
different collector/optimizer semantics. Persistence and native integration
are tested separately rather than advertised as SB3 numerical parity.

## Plot the requested panels

```bash
python -m ICRL.plot_results ICRL/runs --output ICRL/results/figure2.png
```

Pass only the intended full experiment directories if diagnostics share a
method/task/seed with full runs. For example:

```bash
python -m ICRL.plot_results ICRL/runs/VICRL_HC_seed123 \
  --envs HC --output ICRL/results/HC_seed123.png
```

The upper row is the fraction of episodes with any true constraint violation.
The lower row sums raw reward strictly before the first true violation. Neither
metric is the learned episode cost used for optimization. Smoothing uses Liu's
forward 100-episode moving average; bands use half the population standard
deviation across seeds, as in their plotting code. A single-seed curve has no
across-seed uncertainty estimate and cannot establish reproduction of Figure 2.
Episode CSVs retain raw rewards and worker ordinals. Plotting rounds rewards to
two decimals and interleaves worker episode ordinals, as the reference monitor
reader does; this matters when Walker workers have different episode lengths.

## Simulator validation

Environment tests compare task equations against the original source on the
same installed simulator. They cover reset distributions, observations, rewards,
position costs, termination, time limits and Ant contact conventions. Removing
the alternate runtime preserves the previous modern implementation's reset and
100-transition traces bit-exactly for all four tasks, so the 4M-step HC expert
and its collected trajectories remain compatible.

Installed MuJoCo physics are not identical to the original paper's engine.
Historical experiment artifacts retain the actual runtime in their metadata;
they are not evidence that current trajectories match the old simulator.

The learner uses the installed modern PyTorch runtime. Numerical comparisons
execute original and ported updates on that same runtime; they do not establish
bitwise equivalence to PyTorch 1.12 from the old environment. Long-run empirical
agreement remains an experimental result, assessed from completed multi-seed
curves rather than inferred from fixed-input tests.
