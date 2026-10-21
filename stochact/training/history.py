"""Serializable, framework-independent epoch history."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingHistory:
    """Epoch logs. All records contain numeric Python scalars; no tensors are held."""

    records: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int | None = None
    best_value: float | None = None
    stopped_early: bool = False

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: int | str) -> dict[str, float] | list[float]:
        if isinstance(key, int):
            return self.records[key]
        return [record[key] for record in self.records if key in record]

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": [dict(row) for row in self.records],
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "stopped_early": self.stopped_early,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "TrainingHistory":
        return cls(
            records=[dict(row) for row in state["records"]],
            best_epoch=state.get("best_epoch"),
            best_value=state.get("best_value"),
            stopped_early=state.get("stopped_early", False),
        )
