from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

@dataclass
class Vehicle:
    """Data class representing a vehicle in the simulation."""

    id: int
    capacity: float  # payload, tonnes
    node: int  # current OSM node id
    status: str  # 'idle' | 'en_route' | 'on_break'
    free_at: float  # simulation time at which the vehicle becomes available
    drive_time: float  # driving time since the last break (seconds)
    busy_time: float  # cumulative time on assignments, incl. service (seconds)
