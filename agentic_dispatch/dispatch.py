from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, NamedTuple


SEED: int = 42
DRIVE_LIMIT: int = 16200
BREAK_DURATION: int = 2700
SERVICE_TIME: int = 1800
TICK: int = 300
BIG: float = 1e9
EPS_CAP: float = 1e-3
EPS_ID: float = 1e-6
ESCALATION_WAIT: int = 7200
HORIZON: int = 28800
DRAIN_LIMIT: int = 14400
MAX_EVENTS: int = 1000000
MAX_SPEED_KPH: int = 90


@dataclass
class Vehicle:
    """Data class representing a vehicle in the simulation.

    Attributes:
        id (int): Unique identifier for the vehicle.
        capacity (float): Payload capacity of the vehicle in tonnes.
        node (int): The current OSM node ID where the vehicle is located.
        status (str): Current operational status ('idle', 'en_route',
            or 'on_break').
        free_at (float): Simulation time at which the vehicle becomes
            available for a new task.
        drive_time (float): Cumulative driving time since the last break
            in seconds.
        busy_time (float): Total cumulative time spent on assignments,
            including service time, in seconds.
    """

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
        id (int): The unique identifier for the order, assigned after sorting
            by arrival time.
        arrival (float): The simulation timestamp when the order arrives.
        load_node (int): The node ID where the load originates.
        dest_node (int): The destination node ID.
        tonnage (float): Cargo weight in tonnes.
    """

    id: int  # assigned after sorting by arrival time
    arrival: float  # simulation time
    load_node: int
    dest_node: int
    tonnage: float


class Jam(NamedTuple):
    """Represents a traffic jam with spatial and temporal attributes.

    Attributes:
        id (int): Unique identifier for the jam.
        centroid (tuple[float, float]): The geographic center of the
            jam as (lat, lon).
        radius_m (float): The radius of the affected area in meters.
        speed_factor (float): A multiplier where 0 < factor <= 1;
            travel time is divided by it.
        start (float): The timestamp when the jam begins.
        end (float): The timestamp when the jam ends.
    """

    id: int
    centroid: tuple[float, float]  # (lat, lon)
    radius_m: float
    speed_factor: float  # 0 < factor <= 1; travel time is divided by it
    start: float
    end: float


class Assignment(NamedTuple):
    """A data structure representing a vehicle assignment for an order.

    Attributes:
        vehicle_id (int): The unique identifier of the assigned vehicle.
        order_id (int): The unique identifier of the associated order.
        issued_at (float): The timestamp when the assignment was issued.
    """

    vehicle_id: int
    order_id: int
    issued_at: float


@dataclass
class FleetState:
    """The shared state of the simulation and single source of truth.

    Attributes:
        vehicles (dict[int, Vehicle]): Mapping of vehicle IDs to Vehicle
            objects.
        orders (dict[int, Order]): Mapping of order IDs to Order objects.
        pending (set[int]): Set order IDs awaiting a vehicle.
        in_progress (set[int]): Set of order IDs currently being processed.
        completed (dict[int, float]): Mapping of order IDs to their completion
            timestamps.
        escalated (set[int]): Set of order IDs that have been escalated.
        active_jams (dict[int, Jam]): Mapping of jam identifiers to
            Jam objects.
        cycle_times (list[float]): Computation time of each policy call,
            in milliseconds.
        log (list[tuple[float, str, dict[str, Any]]]): A chronological
            log of events containing timestamps, types, and metadata.
        end_time (float): The final timestamp of the simulation run.
    """

    vehicles: dict[int, Vehicle]
    orders: dict[int, Order] = field(default_factory=dict)
    pending: set[int] = field(default_factory=set)
    in_progress: set[int] = field(default_factory=set)
    completed: dict[int, float] = field(default_factory=dict)
    escalated: set[int] = field(default_factory=set)
    active_jams: dict[int, Jam] = field(default_factory=dict)
    cycle_times: list[float] = field(default_factory=list)
    log: list[tuple[float, str, dict[str, Any]]] = field(default_factory=list)
    end_time: float = 0.0
