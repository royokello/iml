from __future__ import annotations

from datetime import datetime

STEP_CSV_HEADER = "datetime,step,epoch,sample_index,sample_name,learning rate,loss,grad_norm"
DEFAULT_MAX_GRAD_NORM = 1.0

class Step:
    datetime: str
    epoch: int
    sample_index: int
    sample_name: str
    step: int
    learning_rate: float
    loss: float
    grad_norm: float

    def __init__(self, datetime_value: str | None = None):
        self.datetime = datetime_value or datetime.now().isoformat()

    def to_csv_row(self) -> str:
        return (
            f"{self.datetime},{self.step},{self.epoch},{self.sample_index},{self.sample_name},"
            f"{self.learning_rate},{self.loss},{self.grad_norm}"
        )
