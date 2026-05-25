from __future__ import annotations

from datetime import datetime

STEP_CSV_HEADER = "datetime,step,epoch,sample,learning rate,loss,ma16,ma64"


class Checkpoint:
    steps: int
    ma16: float


class Step:
    datetime: str
    epoch: int
    sample: int
    step: int
    learning_rate: float
    loss: float
    ma16: float
    ma64: float

    def __init__(self, datetime_value: str | None = None):
        self.datetime = datetime_value or datetime.now().isoformat()

    def to_csv_row(self) -> str:
        return (
            f"{self.datetime},{self.step},{self.epoch},{self.sample},"
            f"{self.learning_rate},{self.loss},{self.ma16},{self.ma64}"
        )
