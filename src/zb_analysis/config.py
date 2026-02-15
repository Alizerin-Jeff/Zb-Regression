"""Configuration for ZB analysis."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class AnalysisConfig:
    """Static analysis configuration."""

    symbol: str = "ZB=F"
    start_date: date = date(2024, 2, 13)
    end_date: date = date(2026, 2, 13)
    local_timezone: str = "Asia/Dubai"
    session_start_hour: int = 1
    session_end_hour: int = 16
    tick_size: float = 1.0 / 32.0
    ticks_to_close_threshold: int = 4
    ticks_per_point: int = 32
    dollars_per_tick: float = 31.25

