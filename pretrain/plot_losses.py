"""Draw loss curves from training printouts.

Save a run's printout, then plot it (several logs go on one chart, for comparisons):
    python -m pretrain.train > runs/my_run.log
    python -m pretrain.plot_losses runs/my_run.log --out docs/curves/my_run.png --ymax 1.8
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to a file; no window needed
import matplotlib.pyplot as plt  # must come after choosing the backend

# A progress line: "step   500 | train 1.546 | val book 1.701 |    21s"
PROGRESS = re.compile(r"^step\s+(\d+) \| (.+) \|\s+\d+s$")


def parse_log(text: str) -> dict[str, list[tuple[int, float]]]:
    """Every loss series in a training printout: {name: [(step, loss), ...]}."""
    series: dict[str, list[tuple[int, float]]] = {}
    for line in text.splitlines():
        match = PROGRESS.match(line.strip())
        if not match:
            continue
        step = int(match.group(1))
        for field in match.group(2).split(" | "):
            name, value = field.rsplit(" ", 1)
            name = name.removesuffix(" loss")  # older printouts said "train loss", "val loss"
            series.setdefault(name, []).append((step, float(value)))
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path, help="training printouts to plot")
    parser.add_argument("--out", type=Path, required=True, help="where to save the PNG")
    parser.add_argument(
        "--ymax", type=float, help="top of the chart; hides the very high first-step losses"
    )
    parser.add_argument("--title", default="Loss curves")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5))
    for log in args.logs:
        for name, points in parse_log(log.read_text(encoding="utf-8")).items():
            steps, losses = zip(*points, strict=True)
            label = name if len(args.logs) == 1 else f"{log.stem}: {name}"
            # Training loss dashed, validation solid: the gap between them is the story.
            ax.plot(steps, losses, label=label, linestyle="--" if name == "train" else "-")
    if args.ymax:
        ax.set_ylim(top=args.ymax)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (lower is better)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
