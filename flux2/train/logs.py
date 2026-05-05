from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_graph(
    fork,
    project_dir: Path,
) -> None:
    graph_path = project_dir / "models" / "graph.png"
    losses = np.array([step.loss for step in fork.steps], dtype=float)
    moving_averages = np.array([step.moving_average for step in fork.steps], dtype=float)

    if losses.size == 0 or moving_averages.size == 0:
        return
    if losses.size != moving_averages.size:
        raise ValueError(
            f"Cannot plot log graph with mismatched loss and moving average counts: "
            f"{losses.size} vs {moving_averages.size}"
        )

    x_values = np.arange(1, losses.size + 1)
    y_values = np.concatenate([losses, moving_averages])
    finite_y_values = y_values[np.isfinite(y_values)]
    if finite_y_values.size == 0:
        return

    y_min = float(np.min(finite_y_values))
    y_max = float(np.max(finite_y_values))
    y_span = y_max - y_min
    y_padding = max(y_span * 0.05, abs(y_max) * 0.025, 1e-6)

    figure, axis = plt.subplots(figsize=(10, 5), dpi=150)
    axis.plot(x_values, losses, color="lightskyblue", linewidth=1.4, label="loss")
    axis.plot(x_values, moving_averages, color="orange", linewidth=1.8, label="moving average")
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    if losses.size == 1:
        axis.set_xlim(0.5, 1.5)
    else:
        axis.set_xlim(1, losses.size)
    axis.set_ylim(y_min - y_padding, y_max + y_padding)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(graph_path)
    plt.close(figure)
