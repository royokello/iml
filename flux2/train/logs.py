from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_graph(
    steps,
    project_dir: Path,
) -> None:
    graph_path = project_dir / "models" / "graph.png"
    losses = np.array([step.loss for step in steps], dtype=float)
    ma16_vals = np.array([step.ma16 for step in steps], dtype=float)
    ma64_vals = np.array([step.ma64 for step in steps], dtype=float)

    if losses.size == 0 or ma16_vals.size == 0 or ma64_vals.size == 0:
        return
    if not (losses.size == ma16_vals.size == ma64_vals.size):
        raise ValueError(
            f"Cannot plot log graph with mismatched loss and moving average counts: "
            f"{losses.size} losses, {ma16_vals.size} ma16, {ma64_vals.size} ma64"
        )

    x_values = np.arange(1, losses.size + 1)
    y_values = np.concatenate([losses, ma16_vals, ma64_vals])
    finite_y_values = y_values[np.isfinite(y_values)]
    if finite_y_values.size == 0:
        return

    y_min = float(np.min(finite_y_values))
    y_max = float(np.max(finite_y_values))
    y_span = y_max - y_min
    y_padding = max(y_span * 0.05, abs(y_max) * 0.025, 1e-6)

    figure, axis = plt.subplots(figsize=(10, 5), dpi=150)
    axis.plot(x_values, losses, color="lightskyblue", linewidth=1.4, label="loss")
    axis.plot(x_values, ma16_vals, color="orange", linewidth=1.8, label="ma16")
    axis.plot(x_values, ma64_vals, color="darkgreen", linewidth=1.8, alpha=0.8, label="ma64")
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
