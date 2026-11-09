# Liu benchmark reproduction audit

Audit date: 2026-11-08. This document records source behavior and the intended
mapping into this prototype on the installed OmniSafe MuJoCo runtime.
Passing component checks is not evidence that the
published learning curves have been reproduced; full multi-seed training and
evaluation remain separate requirements.

## Reference and input provenance

- Reference repository: <https://github.com/Guiliang/ICRL-benchmarks-public>.
- Inspected source commit: `0d79d2c5bdf8e5fe87f89626036a423a806c4c90`.
- `config/liu/{MECL,VICRL}_{HC,Ant,Walker,Swimmer}.yaml` are byte-for-byte copies
  of the eight corresponding source configurations. MECL filenames map to
  source `train_ICRL_*` files and retain source `group: ICRL`.
- `data/liu_provenance.json` contains source-relative paths, SHA256 hashes and
  sizes for all eight configurations and 200 demonstration pickle files.
- The four copied demonstration directories are `BlockedHalf-cheetah`,
  `BlockedAnt`, `BlockedWalker`, and `BlockedSwimmer`: 50 trajectories each,
  39,414,584 bytes total. They were already present in the user's source
  checkout; their original download date and upstream archive checksum are
  unknown. The manifest identifies the actual local input bytes, not an
  independently authenticated upstream archive.
- Inspected examples contain 500 time steps. Observation/action widths are
  HC 18/6, Ant 113/8, Walker 18/6, Swimmer 10/2. HC and Ant examples have no
  per-step `reward` field; reward sums are present. Missing demonstration rewards
  are not grounds for generating substitute demonstrations.
- The original loader selects the first N **lexicographically sorted filenames**,
  reads `original_observations`, and retains all steps (`add_next_step=False`):
  source `interface/train_icrl.py:200` and `utils/data_utils.py:319`.

## Published configuration values represented by the copied YAMLs

| Setting | HC | Ant | Walker | Swimmer |
|---|---:|---:|---:|---:|
| Outer iterations | 30 | 50 | 100 | 100 |
| Forward steps requested per iteration | 200,000 | 200,000 | 200,000 | 50,000 |
| Expert / sampled trajectories | 10 / 10 | 50 / 50 | 50 / 50 | 50 / 50 |
| PPO learning rate | 3e-4 | 3e-5 | 1e-4 | 3e-4 |
| PPO epochs / minibatch | 10 / 64 | 20 / 128 | 20 / 128 | 20 / 128 |
| PPO clip range / target KL | .2 / .01 | .4 / .02 | .4 / .02 | .4 / .01 |
| Reward / cost GAE lambda | .95 / .95 | .9 / .9 | .9 / .9 | .9 / .95 |
| Constraint hidden layers | [20] | [40,40] | [64,64] | [20] |
| Constraint learning rate | .005 | .005 | .001 | .001 |
| Constraint regularizer | .5 | .6 | .6 | .6 |
| Constraint updates / iteration | 10 | 5 | 5 | 5 |
| Initial dual, MECL / VICRL | 1 / 1 | .1 / .1 | .1 / .1 | 1 / 5 |
| Dual learning rate | .1 | .05 | .05 | .01 |
| VICRL prior | [.9,.1] | [.9,.1] | [.009,.001] | [.9,.1] |

All four use 2,048 rollout steps per worker, actor and each critic `[64,64]`,
reward/cost discounts .99, value coefficients .5, gradient clipping .5,
entropy coefficient 0, and cost budget 0. Five workers are the prototype's
reference run setting, consistent with the source tutorial's `-n 5` command;
the source CLI itself defaults to one worker. Five workers produce 10,240
transitions per complete rollout, so requested 200,000/50,000 forward steps
actually collect 204,800/51,200. Avoid treating the requested count as exact.

MECL uses per-step importance sampling, without replay. VICRL samples its Beta
feasibility distribution. Only HC VICRL enables the replay queue, capped at
500 trajectories. Constraint learning rate is multiplied by `.9 ** iteration`.
Observation, reward and cost normalization are enabled in the training env;
constraint-network observation normalization is disabled. Full YAMLs remain
the source of truth for settings omitted from this table.

## Source-to-port semantic mapping

Source line numbers below refer to the pinned reference checkout.

| Reference source | Required behavior | Prototype location |
|---|---|---|
| `utils/model_utils.py:192`; `stable_baselines3/ppo_lag/ppo_lag.py:71` | `algo_type` defaults to `lagrangian`. Although PID coefficients appear in YAML and are passed as `pid_kwargs`, they do not activate PID. | `liu_ppo.py`, `LiuPPOLag` and `DualVariable` |
| `ppo_lag.py:216` | Normalize reward advantages per minibatch; center cost advantages without dividing by their standard deviation. | `liu_ppo.py:104`, `ppo_loss` |
| `ppo_lag.py:227` | Clip reward surrogate only; add the un-clipped cost importance-ratio term; divide policy objective by `1 + nu`. | `liu_ppo.py:104`, `ppo_loss` |
| `ppo_lag.py:288` | KL estimate is mean old log-probability minus new log-probability. Stop after the epoch when it exceeds `1.5 * target_kl`. | `liu_ppo.py:164`, `LiuPPOLag.update` |
| `ppo_lag.py:300`; `common/dual_variable.py:17` | Update dual after policy updates using **mean raw per-step rollout cost**, with Adam on a softplus parameter and the source clamp expression. This is not an episode-cost update. | `liu_ppo.py:67`, `DualVariable` |
| `common/on_policy_algorithm.py:390,424`; `common/buffers.py` | Bootstrap with the critic values evaluated **before the final rollout action**, as the source collector does, including on nonterminal rollout boundaries. Do not replace them with recomputed `V(next_obs)`. Preserve the boolean final done mask and float32 earlier masks, including the resulting float64 final-step intermediate. | `liu_ppo.py`, `collect` and `gae`; original-source collector regression in `tests/test_ppo_equivalence.py` |
| `common/base_class.py:337`; `common/on_policy_algorithm.py:447` | Every outer `.learn()` call resets timestep counters and environments, while retaining policy, dual, optimizers and running normalization statistics. | `liu_ppo.py:190`, `train_for`; `rollouts.py:43`, `reset` |
| `stable_baselines3/common/vec_env/vec_normalize.py` | Preserve running-moment update order and discounted-return scaling; reset updates return statistics but not observation statistics in this source version. | `normalization.py:28`, `LiuNormalize`; `tests/test_normalization_equivalence.py` |
| `stable_baselines3/common/vec_env/vec_cost_wrapper.py:64`; `utils/env_utils.py:98` | Learned cost sees pre-step state/action; true cost sees post-step state. Compute true cost before auto-reset. | `rollouts.py:55`, `step` |
| `common/cns_monitor.py:150` | Episode violation is an ever-violated 0/1 flag. Feasible reward excludes the first violating transition and everything afterward, even if the agent returns to safety. | `rollouts.py:55`, `step`; `episodes.csv` columns `constraint`, `reward_nc` |
| `interface/train_icrl.py:359` | Alternate forward policy optimization, fresh stochastic nominal trajectory sampling, then constraint-network update. | `interface/train_icrl.py`; `rollouts.py:96`, `sample_trajectories` |
| `constraint_models/constraint_net/` | Preserve MECL/VICRL objectives, feature selection, preprocessing, importance weights, prior and stochastic cost behavior. | copied `constraint_models/`; `tests/test_constraint_equivalence.py` |

The dedicated `LiuPPOLag` path is the reference reproduction solver. Merely
selecting the ordinary OmniSafe PPOLag solver with matching YAML values does
not preserve these numerical semantics.

## Initialization and random-number ordering

The source constructs environments, then the constraint network, then the PPO
agent. `common/cns_env.py:51` calls global `set_random_seed(seed)` during each
environment-factory construction. Sampling/evaluation factories use their
default seed 0. Consequently their construction can reset Python/NumPy/Torch
global RNGs to 0 **before constraint-network initialization**, even for a run
whose requested training seed differs. Later PPO `_setup_model()` calls
`set_random_seed(self.seed)` before policy construction and reseeds training
environments with worker offsets. See `common/on_policy_algorithm.py:321` and
`common/base_class.py:479`. A port must preserve or explicitly record a change
to this order; setting a single global seed once at program entry is different.

## Figure 2-style plotting

- `interface/generate_running_plots.py:55` uses **training episode monitor logs**,
  not an evaluation mean recorded once per outer iteration.
- The checked-in default HC map labels `ICRL_Pos_with-action` as MECL and
  `VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1_clr-5e-3` as VICRL.
- Those HC paths use seeds **123, 321, 456, 654, 666**:
  `interface/plot_results/plot_results_dirs.py:666` and `:810`. The checked-in
  mapping does not provide corresponding four-task log mappings; do not claim
  that those unavailable original logs were compared directly.
- Smoothing is a **forward arithmetic moving average of 100 episodes**, not
  an EWMA. `utils/data_utils.py:137` discards the padded tail and also the last
  otherwise-valid window, leaving `N - 100` plotted points.
- Seed mean and population standard deviation are computed before smoothing.
  The displayed multi-method band is **plus/minus half standard deviation**
  (`generate_running_plots.py:233`), not standard error or a confidence interval.
- Source unequal-length aggregation repeats available seed values after short
  runs end (`utils/data_utils.py:478`); it performs no timestep interpolation.
- `plot_results.py` consumes `episodes.csv` and defaults to that aggregation,
  smoothing and band. `--alignment common` explicitly truncates to the common
  seed prefix instead. `--window 1` disables smoothing for smoke runs. Every
  figure has an adjacent JSON file documenting inputs/checksums/settings.
- The source draws PPO-Lag as a constant reference from the final 100 episodes
  and uses its ordinary reward (`generate_running_plots.py:174`). The current
  prototype plot defaults to the MECL and VICRL learning curves only; supplying
  an oracle curve does not by itself reproduce that source baseline formatting.

## Validation scope and runtime limits

The component suites are `tests/test_constraint_equivalence.py`,
`tests/test_ppo_equivalence.py`, and `tests/test_normalization_equivalence.py`.
Use their actual execution results and the runner's `validate_run.py` report
when stating which checks passed. This audit does not assert a completed
five-seed learning-curve reproduction.

The plotting helper was exercised on synthetic four-task/two-method/two-seed
episode CSVs; forward smoothing and unequal-length aggregation were checked
numerically. Synthetic files are validation fixtures, not training results.

## Current simulator runtime

All executable environment paths use the installed OmniSafe/Gymnasium MuJoCo
package (currently Gymnasium 0.28.1 / MuJoCo 2.3.3). The interpreter bridge,
vendored old task runtime, compilation caches, alternate-engine CLI options,
and old-engine test fixtures have been removed. `run.backend: modern` is retained
as a declaration in configs and dataset provenance; other values are rejected.

The custom reset distribution and per-frame stepping are part of the task port.
They run entirely through the installed `mujoco` bindings. Refactoring to one
runtime preserves reset and 100-transition traces of all four former modern
paths bit-exactly. The 4M-step native OmniSafe HC expert and its NumPy trajectories
therefore retain the same task behavior.

The active VICRL HC experiment now uses native OmniSafe `PPOLag` inside the
ICRL loop as well as for expert training. Other reference presets retain
`LiuPPOLag`; `run.solver` records the selected implementation explicitly.
Fixed-batch numerical checks still compare the ported objectives/updates to
original source on the installed PyTorch version. Environment replay against
an old simulator is no longer supported. Historical run metadata and results
retain their original simulator provenance and must be interpreted accordingly.

## Expert demonstration counts

For MECL and VICRL, the released Figure 2 configurations load 10 Half-Cheetah
trajectories and 50 each for Ant, Walker, Swimmer and Biased Pendulum. The four
robot tasks are the subset implemented here. Each supplied dataset contains
50 files. HC, Ant and Swimmer demonstrations have 500 transitions; Pendulum
has 100. Walker has 41 full 500-step episodes and nine shorter episodes,
for 23,474 pairs in total.

Our newly collected HC dataset contains ten 1,000-step episodes: 10,000 pairs,
twice the released HC experiment's 5,000. Equal trajectory counts do not imply
equal demonstration volume. The active reference settings preserve the source
counts, while the collector explicitly records its selected episode horizon.

## Active configuration and collection

Editable experiment YAMLs live directly in `config/`, one per MECL/VICRL and
task. `config/liu/` retains unchanged reference copies. Config loading saves a
resolved YAML, source/reference hashes, runtime versions and dataset hashes.
Seeds, paths, architecture, optimization, normalization, sampling/evaluation
and checkpoint cadence are explicit; omitted CLI values never overwrite YAML.

Native expert training inherits defaults and `HCWithPos-v0` overrides from
`omnisafe/configs/on-policy/PPOLag.yaml`. `collect_experts.py` restores native
actor snapshots and frozen observation statistics and exports complete numeric
NPZ episodes. The collector's top-level `clip_actions` optionally overrides the
saved ICRL run's execution setting. Without an override it inherits that setting,
or defaults to unclipped execution for historical forward runs. Stored actions
are executed policy-coordinate actions before morphology scaling. The supplied
collector config enables clipping and targets a new `HC_seed123_clipped`
directory; existing trajectories are not rewritten or regenerated.
Dataset manifests record attempts, filtering, environment settings and hashes.
The ICRL loader rejects environment/backend/boundary/horizon mismatches.

## Recorded native OmniSafe VICRL HC experiment

The completed six-million-step VICRL HC seed-123 run selected ten generated NPZ
trajectories (10,000 pairs), one simulator, 30 outer iterations and 200,000
forward transitions per iteration. Its saved config and metadata retain the
settings actually used. The current `config/VICRL_HC.yaml` preset uses 60 outer
iterations; MECL HC uses 30. Native epochs contain 20,000 transitions. The
`omnisafe` block controls the
native PPO/model/normalization/Lagrange settings; resolved settings are saved
with the run. The learned-cost target remains zero, distinct from the expert's
known-cost training budget of one.

`ICRL_PPOLag` delegates all policy/critic/Lagrange updates to inherited native
methods and retains their state across outer iterations. Its additional
gradient measurements do not modify gradients. `LearnedCostAdapter` replaces
only the optimization cost, preserving raw true costs for evaluation/logging.
The native sampler freezes training observation statistics and records raw
pre-action states plus policy-coordinate actions.

Native initialization, normalization, KL stopping and advantage handling differ
from Liu's source pipeline. Native `ActionScale` is affine only. Earlier native
experiments also omitted Liu's collector-level action clipping, so their reward
penalty used raw Gaussian commands even when MuJoCo internally saturated the
motor control. This is an action-execution and reward-semantics difference, not
an unavoidable consequence of the simulator backend.

The active experiment configs now expose top-level `clip_actions: true`. ICRL
clips an environment-bound copy of each command before `ActionScale`, preserving
the original sampled action and log probability in PPO's buffer. Training,
nominal sampling, evaluation and expert collection use consistent execution
semantics. Setting the flag to `false` restores the prior native behavior.
These changes stay in `ICRL/` and do not alter ordinary OmniSafe forward CRL.
Historical outputs and datasets keep their original provenance and contents.

The recorded first 10-update VICRL
constraint fit matches original losses, gradients, parameters and Adam states
exactly on the same samples (`numerical_validation.json`). Source-PPO validation
is explicitly not applicable to the native solver. Fixed-data constraint parity
does not establish equivalence of environment rewards, collectors or complete
experiments.

TensorBoard and `episodes.csv` preserve raw returns, true/learned costs,
episode violations and feasible rewards. The Figure2 event series implement
the source's exact forward-100 smoothing, two-decimal reward rounding and N-100
output length. Additional logs cover PPO/critic losses, KL, policy spread,
clipped gradients, multiplier, constraint objectives/gradients, posterior
moments and dataset/replay counts. Checkpoints preserve all models, optimizers,
normalizers, random states and replay data. Single-seed performance is a
transfer experiment rather than a five-seed statistical reproduction.

Explicit observation/action selection fixes an inherited indexing bug: actions
are offset by the full observation dimension, since selection follows
concatenation. Released experiments select all features, so their defaults are
unchanged. Modified activation/feature/normalization settings define modified
experiments; source-parity results apply to original-equivalent settings.
