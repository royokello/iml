from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from flux2.train.run import ERA_STEPS

MOVING_AVERAGE_INTERVAL = 16
STEP_CSV_HEADER = "datetime,step,epoch,sample,era,learning rate,loss,moving_average"


class Checkpoint:
    era: int
    steps: int
    moving_average: float


class Step:
    datetime: str
    epoch: int
    sample: int
    era: int
    step: int
    learning_rate: float
    loss: float
    moving_average: float

    def __init__(self, datetime_value: str | None = None):
        self.datetime = datetime_value or datetime.now().isoformat()

    def to_csv_row(self) -> str:
        return (
            f"{self.datetime},{self.step},{self.epoch},{self.sample},"
            f"{self.era},{self.learning_rate},{self.loss},{self.moving_average}"
        )


class Fork:
    checkpoints: List[Checkpoint]
    steps: List[Step]

    def __init__(self):
        self.checkpoints = []
        self.steps = []

    @classmethod
    def load_from_csv(cls, models_dirpath: Path) -> Fork:
        fork = cls()
        steps_csv_path = models_dirpath / "steps.csv"

        with steps_csv_path.open("r", encoding="utf-8", newline="") as steps_handle:
            steps_reader = csv.reader(steps_handle)
            for row in steps_reader:
                if not row:
                    continue
                if row[0] == "datetime":
                    continue
                if len(row) != 8:
                    raise ValueError(f"Unexpected step row in {steps_csv_path}: {row}")

                step = Step(row[0])
                step.step = int(row[1])
                step.epoch = int(row[2])
                step.sample = int(row[3])
                step.era = int(row[4])
                step.learning_rate = float(row[5])
                step.loss = float(row[6])
                step.moving_average = float(row[7])
                fork.steps.append(step)

        for checkpoint_path in sorted(
            (path for path in models_dirpath.glob("*.safetensors") if path.stem.isdigit()),
            key=lambda path: int(path.stem),
        ):
            checkpoint = Checkpoint()
            checkpoint.steps = int(checkpoint_path.stem)
            matching_step = next(
                (
                    step
                    for step in fork.steps
                    if step.step == checkpoint.steps
                ),
                None,
            )
            if matching_step is None:
                raise ValueError(
                    f"Missing step {checkpoint.steps} for checkpoint "
                    f"{checkpoint_path.name} in {steps_csv_path}"
                )
            checkpoint.era = matching_step.era
            checkpoint.moving_average = matching_step.moving_average
            fork.checkpoints.append(checkpoint)

        return fork

    def generate_next_step(
        self,
        total_step: int,
        epoch: int,
        sample: int,
        era: int,
        learning_rate: float,
        loss,
    ) -> Step:
        "Calculate the ma from the configured interval across eras"
        step = Step()
        step.step = total_step
        step.epoch = epoch
        step.sample = sample
        step.era = era
        step.learning_rate = float(learning_rate)
        step.loss = float(loss)
        previous_loss_count = max(MOVING_AVERAGE_INTERVAL - 1, 0)
        recent_losses = [
            previous_step.loss
            for previous_step in self.steps[-previous_loss_count:]
        ] if previous_loss_count else []
        recent_losses.append(step.loss)
        step.moving_average = float(np.mean(recent_losses))
        self.steps.append(step)
        return step

    def rollback(self, max_step: int) -> None:
        
        self.steps = [step for step in self.steps if step.step <= max_step]
        self.checkpoints = [
            checkpoint for checkpoint in self.checkpoints if checkpoint.steps <= max_step
        ]

    def last_era_best_avg_loss(self) -> float:
            if not self.steps:
                return float("inf")

            latest_era = self.steps[-1].era
            latest_era_steps = [step for step in self.steps if step.era == latest_era]
            return min(step.moving_average for step in latest_era_steps)

    def learning_rate_index(
        self,
        learning_rate: float,
        available_learning_rates: list[float],
    ) -> int:
        return min(
            range(len(available_learning_rates)),
            key=lambda index: abs(available_learning_rates[index] - learning_rate),
        )


    def lower_learning_rate(
        self,
        learning_rate: float,
        available_learning_rates: list[float],
        hard: bool = False,
    ) -> float:
        index = self.learning_rate_index(learning_rate, available_learning_rates)
        offset = 2 if hard else 1
        return available_learning_rates[
            min(index + offset, len(available_learning_rates) - 1)
        ]


    def higher_learning_rate(
        self,
        learning_rate: float,
        available_learning_rates: list[float],
    ) -> float:
        index = self.learning_rate_index(learning_rate, available_learning_rates)
        return available_learning_rates[max(index - 1, 0)]


    def checkpoint_at_or_before(self, step: int, era: int) -> Checkpoint | None:
        candidates = [
            checkpoint
            for checkpoint in self.checkpoints
            if checkpoint.era == era and checkpoint.steps <= step
        ]

        if not candidates:
            return None

        return max(candidates, key=lambda checkpoint: checkpoint.steps)


    def smoothed_loss_points(
        self,
        window_steps: list[Step],
        group_size: int = MOVING_AVERAGE_INTERVAL,
    ) -> list[tuple[int, float]]:
        points = []

        for index in range(0, len(window_steps), group_size):
            group = window_steps[index:index + group_size]

            if not group:
                continue

            points.append(
                (
                    group[-1].step,
                    sum(step.moving_average for step in group) / len(group),
                )
            )

        return points


    def loss_slope(self, points: list[tuple[int, float]]) -> float:
        if len(points) < 2:
            return 0.0

        x = np.array([point[0] for point in points], dtype=np.float64)
        y = np.array([point[1] for point in points], dtype=np.float64)

        x = x - x.mean()
        y = y - y.mean()

        denominator = np.dot(x, x)

        if denominator == 0:
            return 0.0

        return float(np.dot(x, y) / denominator)


    def recommend_training_action(
        self,
        learning_rate: float,
        available_learning_rates: list[float],
    ) -> tuple[str, float, int | None]:
        if not self.steps:
            return "continue", learning_rate, None

        latest_era = self.steps[-1].era
        era_steps = sorted(
            [step for step in self.steps if step.era == latest_era],
            key=lambda step: step.step,
        )

        if len(era_steps) < ERA_STEPS:
            return "continue", learning_rate, None

        window_steps = era_steps[-ERA_STEPS:]

        if len(window_steps) < MOVING_AVERAGE_INTERVAL * 2:
            return "continue", learning_rate, None

        points = self.smoothed_loss_points(window_steps)

        if len(points) < 2:
            return "continue", learning_rate, None

        slope = self.loss_slope(points)

        current_ma = window_steps[-1].moving_average
        start_ma = window_steps[0].moving_average

        best_step = min(window_steps, key=lambda step: step.moving_average)
        best_ma = best_step.moving_average

        gap_from_best = current_ma - best_ma
        window_delta = current_ma - start_ma

        window_span = max(points[-1][0] - points[0][0], 1)

        flat_factor = 0.01          # 1% of best loss
        slow_factor = 0.05          # 5% of best loss

        flat_delta = best_ma * flat_factor
        slow_delta = best_ma * slow_factor

        flat_slope = flat_delta / window_span
        slow_slope = slow_delta / window_span

        rollback_gap = best_ma * 0.03
        blowup_gap = best_ma * 0.08

        rollback_checkpoint = self.checkpoint_at_or_before(best_step.step, latest_era)

        if gap_from_best > blowup_gap:
            if rollback_checkpoint is None:
                return (
                    "lower_continue_hard",
                    self.lower_learning_rate(learning_rate, available_learning_rates, hard=True),
                    None,
                )

            return (
                "rollback_lower_hard",
                self.lower_learning_rate(learning_rate, available_learning_rates, hard=True),
                rollback_checkpoint.steps,
            )

        if slope < -slow_slope:
            return "continue", learning_rate, None

        if slope < -flat_slope:
            return (
                "lower_continue",
                self.lower_learning_rate(learning_rate, available_learning_rates),
                None,
            )

        if abs(slope) <= flat_slope:
            if gap_from_best <= rollback_gap:
                if abs(window_delta) <= flat_delta:
                    return (
                        "increase_continue",
                        self.higher_learning_rate(learning_rate, available_learning_rates),
                        None,
                    )

                return (
                    "lower_continue",
                    self.lower_learning_rate(learning_rate, available_learning_rates),
                    None,
                )

            if rollback_checkpoint is None:
                return (
                    "lower_continue",
                    self.lower_learning_rate(learning_rate, available_learning_rates),
                    None,
                )

            return (
                "rollback_lower",
                self.lower_learning_rate(learning_rate, available_learning_rates),
                rollback_checkpoint.steps,
            )

        if rollback_checkpoint is None:
            return (
                "lower_continue",
                self.lower_learning_rate(learning_rate, available_learning_rates),
                None,
            )

        return (
            "rollback_lower",
            self.lower_learning_rate(learning_rate, available_learning_rates),
            rollback_checkpoint.steps,
        )

    def is_best_era_loss(self, era, loss) -> bool:
        era_checkpoint_losses = [
            checkpoint.moving_average
            for checkpoint in self.checkpoints
            if checkpoint.era == era and checkpoint.steps <= 128
        ]
        if not era_checkpoint_losses:
            return True

        return float(loss) <= min(era_checkpoint_losses)
