# Benelux Fleet Dispatch Demonstrator — Decomposition

*Section 6.4 of Chapter N — Possibilities of AI in Planning and Optimization*

---

## Script Architecture and Decomposition Plan

This document translates the nine roadmap milestones of the Constitution into concrete script components: their inputs, outputs, responsibilities, interface contracts, and task allocation between local and frontier models. Read alongside the Constitution and `data_action_draft.ipynb`.

---

## Component 1 — Data Structures

`Vehicle` is a `@dataclass` (mutable): its state changes on every assignment and after every break. `Order`, `Jam` and `Assignment` are `NamedTuple` (immutable): they are records created once and never modified in place, consistent with the LSCM pattern.

```python
@dataclass
class Vehicle:
    id: int
    capacity: float          # tonnes
    node: int                # current OSM node id
    status: str              # 'idle' | 'en_route' | 'on_break'
    free_at: float           # simulation time at which vehicle becomes available
    drive_time: float        # cumulative driving time since last break (seconds)

class Order(NamedTuple):
    id: int
    arrival: float           # simulation time
    load_node: int           # OSM node id of loading point
    dest_node: int           # OSM node id of delivery point
    tonnage: float

class Jam(NamedTuple):
    id: int
    centroid: tuple[float, float]   # (lat, lon)
    radius_m: float
    speed_factor: float      # 0 < factor <= 1; multiplies free-flow travel time
    start: float             # simulation time
    end: float               # simulation time

class Assignment(NamedTuple):
    vehicle_id: int
    order_id: int
    issued_at: float         # simulation time of the dispatch tick
```

**Constraints and notes.**

- `Vehicle.drive_time` resets to 0 after a 45-minute break (`BREAK_DURATION = 2700` seconds). The break is triggered when `drive_time >= DRIVE_LIMIT` where `DRIVE_LIMIT = 16200` seconds (4.5 hours).
- `Order.tonnage` is capped at the maximum vehicle capacity in the order generator; no splitting in this version.
- `Jam.speed_factor` modifies edge travel times in the affected area; a factor of 0.5 doubles the travel time through the jam.
- All four structures are imported from `dataclasses` and `typing`. `NamedTuple` is imported from `typing`.

**Task allocation:** locally coded (pure boilerplate, fully specified here).

## Component 2 — Road Network

**Function:** `build_network(place: str, graphml_path: str) -> nx.MultiDiGraph`

Downloads the drivable road network for `place` from OpenStreetMap via OSMnx, enriches it with edge speeds and travel times, saves it to GraphML for reuse, and returns the graph. If the GraphML file already exists, it is loaded directly without a download.

```python
def build_network(place: str, graphml_path: str) -> nx.MultiDiGraph:
    """Download or load the road network for place.

    Args:
        place: OSMnx place name, e.g. 'Antwerp, Belgium'.
        graphml_path: Path to cache the enriched graph.

    Returns:
        Directed multigraph with speed_kph and travel_time edge attributes.
    """
    if Path(graphml_path).exists():
        return ox.load_graphml(graphml_path)
    G = ox.graph_from_place(place, network_type='drive')
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    ox.save_graphml(G, graphml_path)
    return G
```

**Notes.**

- `ox.add_edge_speeds` imputes free-flow speeds from the mean maximum speed per road class. Override with a `hwy_speeds` dict for Benelux-specific limits (e.g. `{'motorway': 120, 'primary': 90, 'residential': 30}`).
- `ox.add_edge_travel_times` derives `travel_time` (seconds) from `length` (metres) and `speed_kph`.
- Save once, reload every subsequent run — the download for a city-scale area takes tens of seconds; the full Benelux motorway network takes longer. Start with a bounding box around Antwerp for the first test.
- The graph is passed to Components 3, 5 and 6; it is never modified after creation.

**Task allocation:** frontier model (OSMnx API details and caching logic).

## Component 3 — Travel-Time Matrix

**Function:** `build_travel_time_matrix(G: nx.MultiDiGraph, key_nodes: list[int]) -> tuple[np.ndarray, dict[int, int]]`

Computes the shortest-path travel time (seconds) between every pair of key nodes: the base, all loading nodes and all delivery nodes. Returns the matrix and an index map from OSM node id to matrix row/column.

```python
def build_travel_time_matrix(
    G: nx.MultiDiGraph, key_nodes: list[int]
) -> tuple[np.ndarray, dict[int, int]]:
    """Precompute travel times between all key nodes.

    Args:
        G: Road network with travel_time edge attributes.
        key_nodes: Base, loading and delivery node ids (no duplicates).

    Returns:
        (tt, index): tt[i, j] = travel time in seconds from key_nodes[i]
        to key_nodes[j]; index maps node id -> matrix position.
    """
```

**Method.** For each key node, run `nx.single_source_dijkstra_path_length(G, source, weight='travel_time')` once and read off the lengths to all other key nodes. One Dijkstra run per key node is far cheaper than one per pair.

**Why key nodes are enough.** Vehicles only ever stand at the base, a loading node or a delivery node, so every travel time the simulation needs is between two key nodes. The full graph is not needed after this step, except for recomputation under jams.

**Unreachable pairs.** If no path exists (a one-way street at the edge of the study area, for example), the entry is `np.inf`. The order generator must reject orders whose loading or delivery node is unreachable from the base.

**Jams.** When a jam starts, multiply the `travel_time` of every edge within `radius_m` of the jam centroid by `1 / speed_factor` on a copy of the graph, and recompute the matrix. When it ends, restore the original matrix. For a city-scale study area with a few hundred key nodes this is acceptable; for larger areas, recompute only the rows of sources near the jam.

**Task allocation:** frontier model (graph algorithms, jam recomputation, index bookkeeping).

## Component 4 — Scenario Generator: Orders and Fleet

Two functions produce the input scenario. Both take a seed and use their own `random.Random(seed)` instance, never the global `random` module, so the same seed always gives the same scenario.

```python
def generate_orders(
    candidate_nodes: list[int], n_orders: int, horizon: float,
    min_tonnage: float, max_tonnage: float, seed: int,
) -> list[Order]:
    """Generate a reproducible list of orders, sorted by arrival time."""

def make_fleet(
    n_vehicles: int, base_node: int, capacities: list[float], seed: int,
) -> list[Vehicle]:
    """Create the fleet, every vehicle idle at the base at time 0."""
```

**Order rules.**

- `load_node` and `dest_node` are drawn from `candidate_nodes` and must differ.
- `arrival` is drawn uniformly in `[0, horizon)`; the list is sorted by arrival.
- `tonnage` is drawn uniformly in `[min_tonnage, max_tonnage]`, where `max_tonnage` must not exceed the largest vehicle capacity (no splitting in this version).
- An order is rejected and redrawn if its loading or delivery node is unreachable from the base, or if the trip from loading point to destination alone exceeds `DRIVE_LIMIT`, since no driver could ever serve it.

**Fleet rules.**

- Each vehicle's capacity is drawn from `capacities` (for example `[3.5, 7.5, 12.0, 24.0]` tonnes for a mixed van and truck fleet).
- At time 0 every vehicle is at `base_node`, with `status='idle'`, `free_at=0.0` and `drive_time=0.0`.

**Candidate nodes.** The candidate set is a seeded sample of nodes from the road network. The base node is the network node nearest to the depot coordinates (`ox.distance.nearest_nodes`).

**Task allocation:** locally coded (self-contained; inputs and outputs fully specified here, no dependency on other components' behaviour).

## Component 5 — Feasibility Filter

**Function:** `is_feasible(vehicle: Vehicle, order: Order, tt: np.ndarray, index: dict[int, int]) -> bool`

Returns `True` only if the vehicle can legally and physically serve the order now. Both the dispatcher and the baseline call this same function, so the two policies are judged under identical rules.

**Conditions, all required.**

1. `vehicle.status == 'idle'`. Vehicles en route or on break are not considered in this version.
2. `vehicle.capacity >= order.tonnage`.
3. `vehicle.drive_time + t_to_load + t_load_to_dest <= DRIVE_LIMIT`, where `t_to_load = tt[index[vehicle.node], index[order.load_node]]` and `t_load_to_dest = tt[index[order.load_node], index[order.dest_node]]`. The whole trip must fit before the mandatory break.
4. Both travel times are finite.

```python
def is_feasible(
    vehicle: Vehicle, order: Order, tt: np.ndarray, index: dict[int, int]
) -> bool:
    """Check capacity and driving-time constraints for one vehicle-order pair."""
```

**Simplification, stated openly.** Only idle vehicles are candidates, so the cost of a pair in Component 6 reduces to the travel time to the loading point. Considering vehicles that will become free shortly (adding their remaining busy time to the cost) is an extension for a later version.

**Break rule.** If an idle vehicle is infeasible for every pending order only because of condition 3, it starts its 45-minute break (handled in Component 7). Otherwise a tired driver would stay idle forever.

**Task allocation:** frontier model (the rules must match the state update and both policies exactly).

## Component 6 — Cost Matrix and Assignment (the Dispatcher Policy)

**Function:** `dispatch_policy(state: FleetState, now: float, tt: np.ndarray, index: dict[int, int]) -> list[Assignment]`

The regional coordinator's decision step. It builds a cost matrix over idle vehicles (rows) and pending orders (columns) and solves it as one assignment problem per dispatch cycle.

**Steps.**

1. Collect the idle vehicles and the pending orders from the state. If either list is empty, return `[]`.
2. Build `cost` with shape `(n_idle, n_pending)`. For a feasible pair (Component 5), the cost is the travel time from the vehicle's node to the order's loading node. For an infeasible pair, the cost is `BIG`, a large finite constant (for example `1e9`).
3. Solve with `row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)`. The matrix may be rectangular; the solver then leaves some vehicles or orders unmatched.
4. Discard every matched pair whose cost is `>= BIG`; those pairs are infeasible and only filled the matrix.
5. Return one `Assignment(vehicle_id, order_id, issued_at=now)` per remaining pair.

**Why a finite BIG and not `np.inf`.** With `np.inf` entries, `linear_sum_assignment` raises an error when no complete assignment of the smaller side exists. A finite `BIG` always yields a solution, and step 4 removes the false matches.

**Interface contract.** `dispatch_policy` and `baseline_policy` (Component 9) have identical signatures, so the event loop can run either one without change.

**Link to earlier chapters.** This is the transportation problem with unit supplies and unit demands. The LSCM method is its greedy heuristic counterpart, and `baseline_policy` behaves similarly.

**Task allocation:** frontier model (most algorithmically subtle; the BIG handling and the shared interface are easy to get wrong).

## Component 7 — Shared State and State Update

**The shared state.** One `FleetState` object holds the current view and the time-ordered log. It is the single source of truth required by the Constitution; every component reads from it or writes to it.

```python
@dataclass
class FleetState:
    vehicles: dict[int, Vehicle]        # current view, keyed by vehicle id
    pending: dict[int, Order]           # arrived, not yet assigned
    in_progress: dict[int, Order]       # assigned, not yet delivered
    completed: dict[int, float]         # order id -> delivery time
    log: list[tuple[float, str, dict]]  # (time, record type, details)
```

Record types in the log: `order_arrived`, `assignment_issued`, `vehicle_free`, `break_started`, `break_ended`, `jam_started`, `jam_ended`, `escalation`.

**Function:** `apply_assignment(state: FleetState, a: Assignment, tt: np.ndarray, index: dict[int, int], now: float) -> float`

Applies one assignment and returns the time at which the vehicle becomes free, so the event loop can schedule a `VEHICLE_FREE` event.

1. Read `t_to_load` and `t_load_to_dest` from the matrix.
2. Vehicle: `status='en_route'`, `node=order.dest_node`, `drive_time += t_to_load + t_load_to_dest`, `free_at = now + t_to_load + SERVICE_TIME + t_load_to_dest`.
3. Order: move from `pending` to `in_progress`; on `VEHICLE_FREE`, move to `completed` with delivery time `free_at`.
4. Append an `assignment_issued` record to the log.

`SERVICE_TIME` is a constant for loading and unloading together (for example 1800 seconds). It is not driving time, so it does not count towards `DRIVE_LIMIT`.

**Function:** `on_vehicle_free(state: FleetState, vehicle_id: int, now: float) -> float | None`

When a vehicle finishes a delivery: if `drive_time >= DRIVE_LIMIT`, set `status='on_break'` and return `now + BREAK_DURATION` for a `BREAK_END` event; otherwise set `status='idle'` and return `None`. On `BREAK_END`, reset `drive_time` to 0 and set `status='idle'`. The break rule of Component 5 (idle but infeasible for every order) also calls the break branch.

**Escalation.** Whenever an order has waited longer than `ESCALATION_WAIT` (for example 2 hours) without a feasible vehicle, write an `escalation` record. This is the strategic tier reduced to logging, as stated in the Constitution.

**Task allocation:** frontier model (interdependent logic; must match the event loop and the feasibility rules exactly).

## Component 8 — Event Loop

**Function:** `run_simulation(state: FleetState, orders: list[Order], jams: list[Jam], policy: Callable, G: nx.MultiDiGraph, tt: np.ndarray, index: dict[int, int], horizon: float) -> FleetState`

A discrete-event simulation. Events wait in a priority queue (`heapq`) and are processed in time order; the clock jumps from one event to the next.

**Event tuple:** `(time, kind_priority, seq, kind, payload)`. `seq` is an increasing counter, so two events never compare by `payload` (which would raise an error).

| Event | Priority | Action |
| --- | --- | --- |
| `JAM_START` / `JAM_END` | 0 | Recompute or restore the travel-time matrix (Component 3) |
| `VEHICLE_FREE` | 1 | Complete the order; idle or start break (Component 7) |
| `BREAK_END` | 1 | Reset driving time; vehicle idle |
| `ORDER_ARRIVAL` | 2 | Add order to `pending` |
| `DISPATCH_TICK` | 3 | Call `policy`, apply each assignment, schedule `VEHICLE_FREE` events, schedule the next tick |

At the same timestamp, the priority puts state changes before the dispatch tick, so the policy always sees an up-to-date state.

**Initialisation.** Push one `ORDER_ARRIVAL` per order, one `JAM_START` and one `JAM_END` per jam, and the first `DISPATCH_TICK` at time 0. Ticks repeat every `TICK = 300` seconds.

**Termination.** Stop when the clock passes `horizon` and no orders are pending or in progress, or when the queue is empty. As a safety guard against infinite loops, stop after `MAX_EVENTS` events and log an error.

**Fair comparison.** The dispatcher and the baseline each run on a fresh copy of the same initial state and the same orders and jams. Only `policy` differs.

**Task allocation:** frontier model (orchestrates every other component).

## Component 9 — Metrics, Baseline and Entry Point

**Baseline policy.** `baseline_policy(state, now, tt, index) -> list[Assignment]`, with the same signature as `dispatch_policy`. Orders are taken in arrival order; each one gets the idle vehicle with sufficient capacity that passes `is_feasible` and has the shortest travel time to the loading point. That vehicle is then removed from the candidates for the rest of the tick. This is the greedy rule a human dispatcher would apply.

**Metrics.** `compute_metrics(state: FleetState, horizon: float, cycle_times: list[float]) -> dict[str, float]`

| Metric | Definition | Unit |
| --- | --- | --- |
| Mean time in system | Mean of (delivery time − arrival time) over completed orders | minutes |
| Unserved orders | Orders still pending or in progress at the end | count |
| Fleet utilization | Total en-route time of all vehicles / (number of vehicles × horizon) | % |
| Computation time per cycle | Mean and maximum wall-clock time of one `policy` call | milliseconds |

Cycle times are measured in the event loop with `time.perf_counter()` around each `policy` call.

**Entry point.** `main()` sets the constants and the seed, builds the network and the matrix, generates the scenario, runs `run_simulation` once with each policy on identical copies of the input, and logs a comparison table. Results are exported to CSV with `pandas`; a Folium map of the base, loading points and delivery points is optional. The script ends with the `if __name__ == "__main__":` guard.

**Constants**, defined once at the top of the script: `SEED`, `DRIVE_LIMIT = 16200`, `BREAK_DURATION = 2700`, `SERVICE_TIME`, `TICK = 300`, `BIG = 1e9`, `ESCALATION_WAIT`, `MAX_EVENTS`, `HORIZON`.

**Task allocation:** frontier model (the baseline must share the dispatcher's interface and feasibility rules; `main` orchestrates the whole script).

## Task Allocation List

The rule, as in the LSCM demonstrator: a component goes to the local model when this document specifies it completely and it depends on no other component's behaviour; it goes to the frontier model when getting it right requires reasoning about the whole script.

| Component | Model | Reason |
| --- | --- | --- |
| 1 — Data structures | Local (Gemma) | Pure boilerplate, fields fully specified |
| 4 — Scenario generator | Local (Gemma) | Self-contained; inputs and outputs fully specified |
| 2 — Road network | Frontier | OSMnx API details and caching |
| 3 — Travel-time matrix | Frontier | Graph algorithms, jam recomputation, index bookkeeping |
| 5 — Feasibility filter | Frontier | Must match the state update and both policies exactly |
| 6 — Dispatcher policy | Frontier | Most algorithmically subtle; BIG handling, shared interface |
| 7 — Shared state and update | Frontier | Interdependent logic with the event loop |
| 8 — Event loop | Frontier | Orchestrates every other component |
| 9 — Metrics, baseline, entry point | Frontier | Baseline shares the dispatcher's interface; `main` orchestrates |

The local components are generated first, since every other component imports them.
