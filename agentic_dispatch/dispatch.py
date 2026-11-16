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

class Order(NamedTuple):
    """Represents a transport order in the system.

    Attributes:
        id (int): The unique identifier for the order, assigned after sorting by arrival time.
        arrival (float): The simulation timestamp when the order arrives.
        load_node (int): The node ID where the load originates.
        dest_node (int): The destination node ID.
        tonnage (float): The weight or volume of the cargo.
    """

    id: int  # assigned after sorting by arrival time
    arrival: float  # simulation time
    load_node: int
    dest_node: int
    tonnage: float


