"""Plot Liu-style Figure 2 panels from the prototype's per-episode CSV files.

Example:
    python plot_results.py runs --output results/figure2.png

The default reproduces the reference plot's forward 100-episode moving average
and mean +/- one half population standard deviation. This is not an EWMA or a
confidence interval. Use --alignment common to avoid the reference code's
replication of available seed values when run lengths differ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ENV_ALIASES = {
    "HC": "HC", "HCWithPos-v0": "HC", "BlockedHalfCheetah": "HC",
    "Ant": "Ant", "AntWall-v0": "Ant", "BlockedAnt": "Ant",
    "Walker": "Walker", "WalkerWithPos-v0": "Walker", "BlockedWalker": "Walker",
    "Swimmer": "Swimmer", "SwimmerWithPos-v0": "Swimmer", "BlockedSwimmer": "Swimmer",
}
TITLES = {"HC": "Blocked Half-Cheetah", "Ant": "Blocked Ant",
          "Walker": "Blocked Walker", "Swimmer": "Blocked Swimmer"}
COLORS = {"MECL": "#2ca02c", "VICRL": "#d62728", "PPO_lag": "#9467bd"}
REQUIRED = {"method", "env", "seed", "episode", "reward", "reward_nc", "constraint"}


def read_runs(inputs: list[Path], methods: list[str]):
    """Read each seed as an independent run; reject ambiguous duplicate runs."""
    paths = sorted({p.resolve() for root in inputs for p in
                    ([root] if root.is_file() else root.rglob("episodes.csv"))})
    if not paths:
        raise ValueError("No episodes.csv files found in the selected inputs.")
    runs, provenance = {}, []
    for path in paths:
        file_runs = {}
        with path.open(newline="") as stream:
            reader = csv.DictReader(line for line in stream if not line.startswith("#"))
            missing = REQUIRED - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")
            for row in reader:
                method = "MECL" if row["method"] == "ICRL" else row["method"]
                if method not in methods:
                    continue
                env = ENV_ALIASES.get(row["env"])
                if env is None:
                    raise ValueError(f"{path}: unsupported environment {row['env']!r}")
                key = (env, method, row["seed"])
                values = [float(row['episode']), round(float(row['reward']), 2),
                          round(float(row['reward_nc']), 2), float(row['constraint'])]
                if not np.isfinite(values).all():
                    raise ValueError(f"{path}: nonfinite episode data in {key}")
                if values[3] not in (0.0, 1.0):
                    raise ValueError(f"{path}: constraint must be an episode-level 0/1 indicator")
                order = ((int(row['worker_episode']), int(row['worker']))
                         if 'worker_episode' in row else (values[0], 0))
                file_runs.setdefault(key, []).append((order, values))
        for key, records in file_runs.items():
            if key in runs:
                raise ValueError(f"Multiple files for {key}; select only one run per seed.")
            # Original read_running_logs interleaves monitor files by each
            # worker's episode ordinal, rather than wall-clock completion.
            rows = np.asarray([values for _, values in sorted(records, key=lambda item: item[0])])
            if len(np.unique(rows[:, 0])) != len(rows):
                raise ValueError(f"{path}: duplicate episode numbers for {key}")
            runs[key] = rows
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return runs, provenance


def aggregate(values: list[np.ndarray], alignment: str):
    """Reference aggregation uses ordinal episode indices, not interpolation."""
    if alignment == "common":
        length = min(map(len, values))
        matrix = np.asarray([run[:length] for run in values])
    else:
        # Match utils/data_utils.py:478-509. For an index with fewer seed
        # values, repeat the available values in their original order.
        length = max(map(len, values))
        columns = []
        for index in range(length):
            present = [run[index] for run in values if index < len(run)]
            padded = present.copy()
            for j in range(len(values) - len(present)):
                padded.append(padded[j % len(padded)])
            columns.append(padded)
        matrix = np.asarray(columns).T
    return matrix.mean(axis=0), matrix.std(axis=0, ddof=0)


def paper_smooth(values: np.ndarray, window: int):
    """Exact retained segment of reference compute_moving_average (window>1).

    Its random padding is discarded before plotting, so the surviving values
    are deterministic forward window means, with the final valid window also
    discarded. A window of one explicitly disables smoothing for short checks.
    """
    if window == 1:
        return values.copy()
    if len(values) <= window:
        return np.array([], dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")[:-1]


def plot(runs, output: Path, methods, envs, window, alignment, band, max_episodes):
    fig, axes = plt.subplots(2, len(envs), figsize=(3.5 * len(envs), 5.2), squeeze=False)
    plotted, metadata = 0, []
    for column, env in enumerate(envs):
        for method in methods:
            matching = [(key, rows) for key, rows in runs.items() if key[:2] == (env, method)]
            if not matching:
                continue
            for row_index, data_column in [(0, 3), (1, 2)]:
                mean, std = aggregate([rows[:, data_column] for _, rows in matching], alignment)
                mean, std = paper_smooth(mean, window), paper_smooth(std, window)
                if max_episodes is not None:
                    mean, std = mean[:max_episodes], std[:max_episodes]
                if not len(mean):
                    continue
                scale = {"paper": 0.5, "std": 1.0, "sem": 1.0 / np.sqrt(len(matching)), "none": 0.0}[band]
                x = np.arange(len(mean))
                ax = axes[row_index, column]
                color = COLORS.get(method)
                ax.plot(x, mean, label=method, color=color, linewidth=1.5)
                if scale:
                    ax.fill_between(x, mean - scale * std, mean + scale * std,
                                    color=color, alpha=0.18, linewidth=0)
                plotted += 1
            metadata.append({"env": env, "method": method, "seeds": [key[2] for key, _ in matching],
                             "episodes_per_seed": [len(rows) for _, rows in matching]})
        for row_index in range(2):
            ax = axes[row_index, column]
            ax.set_title(TITLES[env], fontsize=10)
            ax.set_xlabel("Episode")
            ax.grid(alpha=0.15)
            if not ax.lines:
                ax.text(0.5, 0.5, "No completed curves\nfor this panel", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
        axes[0, column].set_ylim(-0.03, 1.03)
    axes[0, 0].set_ylabel("Constraint Violation Rate")
    axes[1, 0].set_ylabel("Feasible Reward")
    handles, labels = [], []
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle); labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if not plotted:
        print(f"No run exceeds the smoothing window ({window}); use --window 1 for a short smoke run.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--runs", nargs="+", type=Path, help="Alternative spelling for input directories")
    parser.add_argument("--output", type=Path, default=Path("results/figure2.png"))
    parser.add_argument("--methods", nargs="+", default=["MECL", "VICRL"])
    parser.add_argument("--envs", nargs="+", choices=list(TITLES), default=list(TITLES))
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--alignment", choices=["paper", "common"], default="paper")
    parser.add_argument("--band", choices=["paper", "std", "sem", "none"], default="paper")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    try:
        runs, inputs = read_runs(args.inputs + (args.runs or []) or [Path("runs")], args.methods)
        curves = plot(runs, args.output, args.methods, args.envs, args.window, args.alignment,
                      args.band, args.max_episodes)
    except ValueError as error:
        parser.error(str(error))
    report = {"source_commit": "0d79d2c5bdf8e5fe87f89626036a423a806c4c90",
              "inputs": inputs, "curves": curves, "window": args.window,
              "smoothing": "forward arithmetic moving average; final window omitted (window>1)",
              "alignment": args.alignment, "band": args.band,
              "band_note": "paper = +/- half population standard deviation; not a confidence interval"}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
