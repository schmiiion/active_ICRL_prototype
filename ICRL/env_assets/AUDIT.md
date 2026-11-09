# Liu Blocked environment port

This package implements the four environments used in Liu et al., *Benchmarking
Constraint Inference in Inverse Reinforcement Learning* (ICLR 2023), using the
checked-out `ICRL-benchmarks-public` source. No OmniSafe environment is modified.
Import `ICRL.blocked_envs` to register their original IDs with OmniSafe.
The registered CMDPs, raw tasks, forward expert training and ICRL runner all
use the installed OmniSafe/Gymnasium MuJoCo runtime. The currently tested
versions are Gymnasium 0.28.1 and MuJoCo 2.3.3.

| Original ID | Raw observation | Reward | True cost | Horizon |
|---|---|---|---|---|
| HCWithPos-v0 | all qpos + qvel (18) | abs(x_after-x_before)/dt - 0.1 sum(a²) | x_after < -3 | 1000 |
| AntWall-v0 | all qpos + qvel + clipped external contact forces (113) | norm(torso_xy_after) + healthy_bonus - 0.5 sum(a²) - 0.0005 sum(clipped_contacts²) | observation_after[0] < -3 | 500 |
| WalkerWithPos-v0 | all qpos + **unclipped** qvel (18) | abs(x_after-x_before)/dt + 1 - 0.001 sum(a²) | x_after < -3 | 500 |
| SwimmerWithPos-v0 | all qpos + qvel (10) | abs(x_after-x_before)/dt - 0.0001 sum(a²) | x_after > 0.5 | 500 |

The first, third and fourth tasks have `REWARD_TYPE='old'` in the reference
source; its alternate `new_reward` functions are not the selected benchmark.
Ant deliberately uses distance from the origin, not forward speed, in its total
reward. Its info dictionary still reports absolute velocities. Ant never ends
on unhealthy posture, but its healthy bonus becomes zero when unhealthy. Walker
terminates on height outside the strict interval (0.8, 2.0) or angle outside
(-1, 1). Safety violations do not terminate these four training/evaluation IDs.
The separate `*Test-v0` classes are not substituted. Time limits are truncations
provided by the outer adapter, and the raw environments do not auto-reset.

## Reference locations

- `mujuco_environment/custom_envs/envs/half_cheetah.py:144-219`
- `mujuco_environment/custom_envs/envs/ant.py:40-92`
- `mujuco_environment/custom_envs/envs/walker.py:45-101`
- `mujuco_environment/custom_envs/envs/swimmer.py:42-98`
- `mujuco_environment/custom_envs/__init__.py`: horizons and selected classes
- `utils/true_constraint_functions.py:7-25,58-63`: strict predicates
- `utils/env_utils.py:97-123`: costs applied to the returned observation

## Reset and simulator semantics

The port retains the benchmark's hashed integer seed and NumPy RandomState
reset sampling. HC and Ant use uniform [-0.1,0.1] position noise and Gaussian
velocity noise with standard deviation 0.1. Walker uses uniform [-0.005,0.005]
for both; Swimmer uses uniform [-0.1,0.1] for both. Frame skips are 5,5,4,4.
These are custom-task semantics, executed entirely with installed MuJoCo.

Each control step sets controls and calls `mujoco.mj_step` once per physics
frame. No extra `mj_rnePostConstraint` call is added. On these sensor-free
assets, this retains zero Ant contact-force observation slots and contact
penalties, matching the released demonstrations' convention. Adding that call
changes Ant observations/rewards even on the same simulator version. Tests
check the intended contact behavior as well as generalized dynamics.

## Assets

- `half_cheetah.xml` and `swimmer.xml` preserve the reference model definitions.
- `ant_circle.xml` is the original custom Ant model.
- `walker2d.xml` uses the parser-compatible Gymnasium asset. The reference's
  foot-body position expression is replaced by its evaluated coordinates.
  Mass matrices, accelerations and one-step dynamics are checked against the
  numerically equivalent repaired model.

The installed engine differs from the original paper's engine. Preserving task
equations/assets does not establish identical physics across engine versions.
The retired-runtime refactor itself preserves the former modern backend's
reset plus 100-step traces bit-exactly for HC, Ant, Walker and Swimmer. Thus the
trained modern HC expert and exported NumPy data remain valid for these tasks.

## Validation

`tests/test_blocked_envs.py` exercises every task on installed MuJoCo, including
reset distributions/seeds, observation shape, strict positional costs, terminal
states, configurable horizons, Walker fall termination and Ant contacts.
Source AST comparisons execute original task equations on this same engine;
they do not import or require an old Gym installation.
