# Benelux Fleet Dispatch Demonstrator — Decomposition

*Section 6.4 of Chapter N — Possibilities of AI in Planning and Optimization*

---

## Script Architecture and Decomposition Plan

This document translates the nine roadmap milestones of the Constitution into concrete script components: their inputs, outputs, responsibilities, interface contracts, and task allocation between local and frontier models. Read alongside the Constitution and `data_action_draft.ipynb`.

**Revision history.** *Revision 2 (9 November 2026)* incorporates the resolutions of an independent specification review by Gemini 4 Argon (seven ambiguities, four items of missing information, six internal contradictions). *Revision 3 (9 November 2026)* resolves three minor issues found in a second review: the stopping rule of the event loop, the functions that log arrivals and jams, and the depot coordinates. Both reviews are kept in the project repository. Every resolution was decided by the human author.

---

## Component 1 — Data Structures and Shared State

`Vehicle` and `FleetState` are `@dataclass` (mutable): their contents change throughout the simulation. `Order`, `Jam` and `Assignment` are `NamedTuple` (immutable): records created once and never modified, consistent with the LSCM pattern. All identifiers are integers starting at 0.

```python
@dataclass
class Vehicle:
    id: int
    capacity: float          # payload, tonnes
    node: int                # current OSM node id
    status: str              # 'idle' | 'en_route' | 'on_break'
    free_at: float           # simulation time at which the vehicle becomes available
    drive_time: float        # driving time since the last break (seconds)
    busy_time: float         # cumulative time on assignments, incl. service (seconds)

class Order(NamedTuple):
    id: int                  # assigned after sorting by arrival time
    arrival: float           # simulation time
    load_node: int
    dest_node: int
    tonnage: float

class Jam(NamedTuple):
    id: int
    centroid: tuple[float, float]   # (lat, lon)
    radius_m: float
    speed_factor: float      # 0 < factor <= 1; travel time is divided by it
    start: float
    end: float

class Assignment(NamedTuple):
    vehicle_id: int
    order_id: int
    issued_at: float

@dataclass
class FleetState:
    vehicles: dict[int, Vehicle]
    orders: dict[int, Order] = field(default_factory=dict)        # every arrived order
    pending: set[int] = field(default_factory=set)                # order ids awaiting a vehicle
    in_progress: set[int] = field(default_factory=set)            # order ids being served
    completed: dict[int, float] = field(default_factory=dict)     # order id -> delivery time
    escalated: set[int] = field(default_factory=set)              # order ids already escalated
    active_jams: dict[int, Jam] = field(default_factory=dict)
    cycle_times: list[float] = field(default_factory=list)        # ms per policy call
    log: list[tuple[float, str, dict]] = field(default_factory=list)
    end_time: float = 0.0                                         # set when the simulation stops
```

**Notes.**

- `FleetState` is the shared state of Section 6.3 and the single source of truth required by the Constitution. It is defined here, not in Component 7, because Components 5 to 9 all refer to it.
- `orders` keeps every order that has arrived, so arrival times remain available for the metrics after delivery. `pending`, `in_progress` and `completed` refer to orders by id only.
- Mutable defaults must use `field(default_factory=...)`, never a bare `[]`, `{}` or `set()`.
- `DRIVE_LIMIT = 16200` s (4.5 h) and `BREAK_DURATION = 2700` s (45 min) apply to `drive_time`; see Components 5 and 7.

**Task allocation:** locally coded (pure boilerplate, fully specified here).

## Component 2 — Road Network

**Function:** `build_network(place: str, graphml_path: str) -> nx.MultiDiGraph`

Downloads the drivable road network for `place`, keeps only its largest strongly connected component, sets edge speeds and travel times, saves the result to GraphML and returns it. If the GraphML file already exists, it is loaded instead of downloaded.

```python
def build_network(place: str, graphml_path: str) -> nx.MultiDiGraph:
    """Download or load the road network for place."""
    if Path(graphml_path).exists():
        return ox.load_graphml(graphml_path)
    G = ox.graph_from_place(place, network_type='drive')
    G = ox.truncate.largest_component(G, strongly=True)
    G = ox.add_edge_speeds(G)
    for _, _, data in G.edges(data=True):
        data['speed_kph'] = min(data['speed_kph'], MAX_SPEED_KPH)
    G = ox.add_edge_travel_times(G)
    ox.save_graphml(G, graphml_path)
    return G
```

**Rules, in this order.**

1. **Strongly connected component.** On a directed road network, a node can be reachable from the base but have no path back (a one-way stub at the edge of the study area). Keeping only the largest strongly connected component makes every node reachable from every other, so every travel time in Component 3 is finite and no reachability check is needed anywhere else.
2. **Speeds.** `ox.add_edge_speeds(G)` uses OpenStreetMap `maxspeed` tags where present and imputes the rest from the road class. Every edge is then capped at `MAX_SPEED_KPH = 90`, the speed-limiter limit for heavy goods vehicles. One cap for the whole fleet is a stated simplification: vans are assumed to drive like trucks.
3. **Travel times.** `ox.add_edge_travel_times(G)` must run after the cap, because it derives `travel_time` (seconds) from `length` and `speed_kph`.

**Version note.** OSMnx function names differ between releases (for example `ox.truncate.largest_component` in 2.x versus `ox.utils_graph.get_largest_component` in 1.x). Check them against the installed version.

**Where the graph goes.** The graph is used by Component 3 (travel-time matrix and jam recomputation) and passed through Component 8 for jam events. It is never modified after creation; jams work on a copy. The base node is found in `main()` with `ox.distance.nearest_nodes`.

**Task allocation:** frontier model (OSMnx API details and caching logic).

## Component 3 — Travel-Time Matrix and Jams

**Function:** `build_travel_time_matrix(G: nx.MultiDiGraph, key_nodes: list[int]) -> tuple[np.ndarray, dict[int, int]]`

Computes the shortest-path travel time (seconds) between every pair of key nodes and returns the matrix with an index map from OSM node id to matrix position.

```python
def build_travel_time_matrix(
    G: nx.MultiDiGraph, key_nodes: list[int]
) -> tuple[np.ndarray, dict[int, int]]:
    """tt[i, j] = travel time in seconds from key_nodes[i] to key_nodes[j]."""
```

**Key nodes.** `key_nodes = [base_node] + candidate_nodes`, fixed before any order is generated (Component 4). Orders draw their loading and delivery nodes from the candidates, so the matrix covers every trip the simulation can need. This order of construction removes the circular dependency between the matrix and the order generator.

**Method.** For each key node, run `nx.single_source_dijkstra_path_length(G, source, weight='travel_time')` once and read the lengths to all other key nodes. Because the graph is strongly connected (Component 2), every entry is finite; the function raises `ValueError` if it finds a missing value, since that would indicate a bug.

**Jams.** **Function:** `jammed_graph(G: nx.MultiDiGraph, active_jams: dict[int, Jam]) -> nx.MultiDiGraph`

Returns a copy of `G` in which every edge affected by an active jam has its `travel_time` divided by the jam's `speed_factor`.

- An edge is affected if **either endpoint** lies within `radius_m` of the jam centroid. Distances use `ox.distance.great_circle` on the node coordinates (`y` = latitude, `x` = longitude); check the function name against the installed OSMnx version.
- If several active jams cover the same edge, the **strongest** applies (the smallest `speed_factor`).

**Recomputation rule.** On every `JAM_START` or `JAM_END` event, the event loop rebuilds the matrix from the **original** graph and the **current set of active jams**: `build_travel_time_matrix(jammed_graph(G, state.active_jams), key_nodes)`. When no jam is active, it reuses the original matrix. Rebuilding from the active set, instead of undoing one jam, keeps overlapping jams correct.

**Cost.** A rebuild is one Dijkstra run per key node. With about 200 candidates on a city network this takes seconds, which is acceptable for a few jams per simulated shift.

**Task allocation:** frontier model (graph algorithms, jam geometry, index bookkeeping).

## Component 4 — Scenario Generator: Candidates, Orders, Fleet and Jams

Four functions produce the input scenario. Each uses its own `random.Random` instance, never the global `random` module, with a seed derived from `SEED` (see Component 9), so the same `SEED` always gives the same scenario.

**Order of construction** (in `main()`): candidate nodes → travel-time matrix over `[base_node] + candidates` (Component 3) → orders, fleet and jams.

```python
def sample_candidates(
    G: nx.MultiDiGraph, n_candidates: int, base_node: int, seed: int,
) -> list[int]:
    """Sample distinct candidate nodes from G, excluding the base."""

def generate_orders(
    candidate_nodes: list[int], tt: np.ndarray, index: dict[int, int],
    base_node: int, n_orders: int, horizon: float,
    min_tonnage: float, capacities: list[float], seed: int,
) -> list[Order]:
    """Generate a reproducible list of orders, sorted by arrival time."""

def make_fleet(
    n_vehicles: int, base_node: int, capacities: list[float], seed: int,
) -> list[Vehicle]:
    """Create the fleet, every vehicle idle at the base at time 0."""

def generate_jams(
    G: nx.MultiDiGraph, candidate_nodes: list[int], n_jams: int,
    horizon: float, seed: int,
) -> list[Jam]:
    """Generate a reproducible list of jams, sorted by start time."""
```

**Order rules.**

- `load_node` and `dest_node` are distinct nodes drawn from `candidate_nodes`.
- `arrival` is drawn uniformly in `[0, horizon)`.
- **Tonnage:** first draw a capacity class `c` from `capacities`, then draw the tonnage uniformly in `[min_tonnage, c]`. Demand is thus spread across vehicle classes; a plain uniform draw up to the largest capacity would make most orders servable only by the heaviest trucks. Orders never exceed the largest capacity, so no splitting is needed.
- An order is rejected and redrawn if `tt[base, load] + tt[load, dest] > DRIVE_LIMIT`, so a rested vehicle at the base can always serve it. On a city-scale network this check never binds; it matters when the study area grows.
- After generation, sort by `arrival` and assign `id` = 0, 1, 2, … in that order.

**Fleet rules.**

- Vehicle ids are 0, 1, 2, …; each capacity is drawn from `capacities`.
- At time 0 every vehicle is at `base_node`, `status='idle'`, `free_at=0.0`, `drive_time=0.0`, `busy_time=0.0`.

**Jam rules.**

- The centroid is the coordinate pair `(y, x)` of a randomly chosen candidate node.
- `radius_m` uniform in `[500, 2000]`; `speed_factor` uniform in `[0.3, 0.7]`.
- `start` uniform in `[0, horizon)`; duration uniform in `[1800, 7200]` s; `end = start + duration`.
- After generation, sort by `start` and assign ids 0, 1, 2, ….

**Task allocation:** locally coded. The functions are self-contained, and their one dependency, the `tt`/`index` pair, is a data interface fully specified in Component 3.

## Component 5 — Feasibility Filter and Break Rule

**Function:** `is_feasible(vehicle: Vehicle, order: Order, tt: np.ndarray, index: dict[int, int]) -> bool`

Returns `True` only if the vehicle can legally and physically serve the order now. The dispatcher and the baseline both call this function, so the two policies are judged under identical rules.

**Conditions, all required.**

1. `vehicle.status == 'idle'`.
2. `vehicle.capacity >= order.tonnage`.
3. `vehicle.drive_time + t_to_load + t_load_to_dest <= DRIVE_LIMIT`, where `t_to_load = tt[index[vehicle.node], index[order.load_node]]` and `t_load_to_dest = tt[index[order.load_node], index[order.dest_node]]`. The whole trip must fit before the mandatory break.

All travel times are finite by construction (Component 2), so no finiteness check is needed.

**Function:** `fails_only_on_drive_time(vehicle: Vehicle, order: Order, tt: np.ndarray, index: dict[int, int]) -> bool`

Returns `True` if conditions 1 and 2 hold and condition 3 fails. It is used only by the break rule.

**Break rule.** At each `DISPATCH_TICK`, after the policy's assignments have been applied, the event loop checks every vehicle that is still idle. The vehicle starts its break (`start_break`, Component 7) if both hold:

- no pending order is feasible for it, and
- at least one pending order fails for it only on condition 3.

In words: the vehicle goes on break when rest is the only thing preventing it from serving waiting work. A vehicle whose capacity is too small for every waiting order does not take a break on that account. When no orders are pending, no break is started.

**Simplifications, stated openly.**

- Only idle vehicles are candidates, so the cost of a pair in Component 6 is simply the travel time to the loading point. Including vehicles about to become free is a later extension.
- Idle waiting time does not count as rest; only a `start_break` resets `drive_time`.

**Task allocation:** frontier model (the rules must match the state update, the event loop and both policies exactly).

## Component 6 — Dispatcher Policy (Cost Matrix and Assignment)

**Function:** `dispatch_policy(state: FleetState, now: float, tt: np.ndarray, index: dict[int, int]) -> list[Assignment]`

The regional coordinator's decision step. It builds a cost matrix over idle vehicles (rows) and pending orders (columns) and solves one assignment problem per dispatch cycle.

**Steps.**

1. Collect the idle vehicles, sorted by id, and the pending orders, sorted by id. If either list is empty, return `[]`.
2. Build `cost` with shape `(n_idle, n_pending)`:
   - feasible pair (Component 5): `tt[vehicle node, load node] + EPS_CAP * vehicle.capacity + EPS_ID * vehicle.id`;
   - infeasible pair: `BIG`.
3. Solve with `row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)`. The matrix may be rectangular; the solver leaves the surplus vehicles or orders unmatched.
4. Discard every matched pair whose cost is `>= BIG`; those pairs are infeasible and only fill the matrix.
5. Return one `Assignment(vehicle_id, order_id, issued_at=now)` per remaining pair, sorted by `order_id`.

**Tie-breaking.** At time 0 all vehicles stand at the base, so many pairs have the same travel time. The two small terms make the choice deterministic and sensible: `EPS_CAP = 1e-3` (seconds per tonne) prefers the smallest sufficient vehicle, keeping heavy trucks free for heavy orders; `EPS_ID = 1e-6` then prefers the lowest vehicle id. Both are far too small to override a real difference in travel time.

**Why a finite BIG and not `np.inf`.** With `np.inf` entries, `linear_sum_assignment` raises an error when the smaller side cannot be fully assigned. A finite `BIG = 1e9` always yields a solution, and step 4 removes the false matches.

**Interface contract.** `dispatch_policy` and `baseline_policy` (Component 9) have identical signatures, so the event loop runs either one without change. Policies only decide; they never modify the state. The event loop applies the assignments and measures the computation time.

**Link to earlier chapters.** This is the transportation problem with unit supplies and unit demands. The LSCM method is its greedy heuristic counterpart, and `baseline_policy` behaves similarly.

**Task allocation:** frontier model (most algorithmically subtle; the BIG handling, the tie-breaking terms and the shared interface are easy to get wrong).

## Component 7 — State Update

The functions below are the only code that changes `FleetState` (defined in Component 1). Each one appends a record to `state.log`.

**`apply_assignment(state: FleetState, a: Assignment, tt: np.ndarray, index: dict[int, int], now: float) -> float`**

Applies one assignment and returns the time at which the vehicle becomes free. The event loop then schedules `VEHICLE_FREE` with payload `(vehicle_id, order_id)`.

1. Read `t_to_load` and `t_load_to_dest` from the matrix.
2. Vehicle: `status='en_route'`, `node=order.dest_node`, `drive_time += t_to_load + t_load_to_dest`, `busy_time += t_to_load + SERVICE_TIME + t_load_to_dest`, `free_at = now + t_to_load + SERVICE_TIME + t_load_to_dest`.
3. Order: move its id from `pending` to `in_progress`.

`SERVICE_TIME = 1800` s covers loading and unloading together. It counts as busy time but not as driving time.

**`on_vehicle_free(state: FleetState, vehicle_id: int, order_id: int, now: float) -> float | None`**

Moves the order from `in_progress` to `completed[order_id] = now`. If the vehicle's `drive_time >= DRIVE_LIMIT`, it calls `start_break` and returns its result; otherwise it sets `status='idle'` and returns `None`.

**`start_break(state: FleetState, vehicle_id: int, now: float) -> float`**

Sets `status='on_break'` and `free_at = now + BREAK_DURATION`, and returns `free_at` so the event loop can schedule `BREAK_END`. Called from `on_vehicle_free` and from the break rule (Component 5).

**`on_break_end(state: FleetState, vehicle_id: int, now: float) -> None`**

Resets `drive_time` to 0 and sets `status='idle'`.

**`escalate_waiting(state: FleetState, now: float) -> None`**

For each pending order not yet in `state.escalated`: if `now - arrival > ESCALATION_WAIT` (7200 s), write an `escalation` record and add the id to `state.escalated`. Each order is escalated at most once. This is the strategic tier reduced to logging, as stated in the Constitution.

**`on_order_arrival(state: FleetState, order: Order, now: float) -> None`**

Adds the order to `state.orders` and its id to `state.pending`; logs `order_arrived`.

**`on_jam_start(state: FleetState, jam: Jam, now: float) -> None`**

Adds the jam to `state.active_jams`; logs `jam_started`. The travel-time matrix is not part of `FleetState`; the event loop rebuilds it (Component 8).

**`on_jam_end(state: FleetState, jam_id: int, now: float) -> None`**

Removes the jam from `state.active_jams`; logs `jam_ended`.

With these three helpers, every one of the eight log record types has exactly one function that writes it, and no code outside this component changes `FleetState`, except that the event loop sets `end_time` and appends to `cycle_times`.

**Log schema.** Every record is `(time, record_type, details)`:

| Record type | Keys in `details` |
| --- | --- |
| `order_arrived` | `order_id` |
| `assignment_issued` | `vehicle_id`, `order_id`, `t_to_load`, `t_load_to_dest`, `free_at` |
| `vehicle_free` | `vehicle_id`, `order_id` |
| `break_started` | `vehicle_id`, `until` |
| `break_ended` | `vehicle_id` |
| `jam_started` | `jam_id` |
| `jam_ended` | `jam_id` |
| `escalation` | `order_id`, `waited` |

**Task allocation:** frontier model (interdependent logic; must match the event loop and the feasibility rules exactly).

## Component 8 — Event Loop

**Function:** `run_simulation(state: FleetState, orders: list[Order], jams: list[Jam], policy: Callable, G: nx.MultiDiGraph, tt0: np.ndarray, index: dict[int, int], key_nodes: list[int], horizon: float) -> FleetState`

A discrete-event simulation. Events wait in a priority queue (`heapq`); the clock jumps from one event to the next. `tt0` is the jam-free matrix; the loop keeps a current matrix `tt` that changes with jams.

**Event tuple:** `(time, priority, seq, kind, payload)`. `seq` is an increasing counter, so two events never compare by `payload`.

| Event | Priority | Payload | Action |
| --- | --- | --- | --- |
| `JAM_START` | 0 | `jam_id` | `on_jam_start`; rebuild `tt` from `G` and the active set (Component 3) |
| `JAM_END` | 0 | `jam_id` | `on_jam_end`; rebuild `tt`, or reuse `tt0` when no jam is active |
| `VEHICLE_FREE` | 1 | `(vehicle_id, order_id)` | `on_vehicle_free`; if it returns a time, schedule `BREAK_END`; then check the stopping rule |
| `BREAK_END` | 1 | `vehicle_id` | `on_break_end` |
| `ORDER_ARRIVAL` | 2 | `order_id` | `on_order_arrival` |
| `DISPATCH_TICK` | 3 | none | See below |

At the same timestamp, the priorities put state changes before the dispatch tick, so the policy always sees an up-to-date state.

**A dispatch tick, in order.**

1. Call `policy(state, now, tt, index)`, timing the call with `time.perf_counter()`; append the time in milliseconds to `state.cycle_times`.
2. Apply each assignment with `apply_assignment` and schedule its `VEHICLE_FREE` event.
3. Apply the break rule (Component 5) to every vehicle still idle; for each break started, schedule `BREAK_END`.
4. Call `escalate_waiting(state, now)`.
5. Schedule the next tick at `now + TICK` **only if** orders can still arrive (`now < horizon`) or orders are pending or in progress. Otherwise no further tick is scheduled.

**Initialisation.** Push one `ORDER_ARRIVAL` per order, one `JAM_START` and one `JAM_END` per jam, and the first `DISPATCH_TICK` at time 0.

**Termination.** Orders arrive during the shift `[0, horizon)`; the simulation then continues until every order is delivered.

- **Normal stop:** immediately after the `VEHICLE_FREE` event that completes the last order (`len(state.completed) == len(orders)`). Leftover `JAM_END`, `BREAK_END` or `DISPATCH_TICK` events still in the queue are discarded, so they cannot inflate `end_time` or add empty cycles to `cycle_times`.
- **Drain cap:** if the clock passes `horizon + DRAIN_LIMIT`, stop and log a warning.
- **Safety stop:** if more than `MAX_EVENTS` events have been processed, stop and log an error.

At any stop, set `state.end_time` to the current clock. Orders not in `completed` at that moment are **unserved** (zero after a normal stop; non-zero only after the drain cap or the safety stop).

**Fair comparison.** `main()` builds two fresh states with `make_fleet` and the same seed, one for each policy. The `orders` and `jams` lists are immutable and shared. Only `policy` differs. No `copy` module is needed.

**Task allocation:** frontier model (orchestrates every other component).

## Component 9 — Baseline, Metrics and Entry Point

**Baseline policy.** `baseline_policy(state, now, tt, index) -> list[Assignment]`, with the same signature as `dispatch_policy`. Pending orders are taken in id order (which is arrival order). For each order, among the idle vehicles not yet used in this tick that pass `is_feasible`, choose the one with the shortest travel time to the loading point; ties go to the smallest capacity, then the lowest id, as in the dispatcher. This is the greedy rule a human dispatcher would apply.

**Metrics.** `compute_metrics(state: FleetState) -> dict[str, float]` returns five keys:

| Key | Definition | Unit |
| --- | --- | --- |
| `mean_time_in_system_min` | Mean of `completed[id] − orders[id].arrival` over completed orders | minutes |
| `unserved_orders` | Orders in `state.orders` not in `completed` | count |
| `fleet_utilization_pct` | Sum of `busy_time` over all vehicles ÷ (number of vehicles × `end_time`) × 100 | % |
| `mean_cycle_ms` | Mean of `state.cycle_times` | milliseconds |
| `max_cycle_ms` | Maximum of `state.cycle_times` | milliseconds |

Everything the metrics need is stored in `FleetState`, so `compute_metrics` takes no other argument.

**Constants**, defined once at the top of the script:

| Constant | Value | Meaning |
| --- | --- | --- |
| `SEED` | 42 | Master seed |
| `DRIVE_LIMIT` | 16200 | 4.5 h of driving before a break (s) |
| `BREAK_DURATION` | 2700 | 45-minute break (s) |
| `SERVICE_TIME` | 1800 | Loading plus unloading (s) |
| `TICK` | 300 | Dispatch interval (s) |
| `BIG` | 1e9 | Cost of an infeasible pair |
| `EPS_CAP` | 1e-3 | Tie-break per tonne of capacity (s) |
| `EPS_ID` | 1e-6 | Tie-break per vehicle id (s) |
| `ESCALATION_WAIT` | 7200 | Waiting time before escalation (s) |
| `HORIZON` | 28800 | Order arrival window, one 8-hour shift (s) |
| `DRAIN_LIMIT` | 14400 | Extra time to finish open orders, 4 h (s) |
| `MAX_EVENTS` | 1 000 000 | Safety stop |
| `MAX_SPEED_KPH` | 90 | Speed cap for every edge (km/h) |

**Scenario parameters (first test).**

- `PLACE = 'Antwerp, Belgium'`; `GRAPHML_PATH = 'data/antwerp_drive.graphml'`.
- `DEPOT = (51.2600, 4.3580)` as `(lat, lon)`, a starting value in the Antwerp port area proposed by the reviewer; confirm it on a map before the first run. `ox.distance.nearest_nodes(G, X=DEPOT[1], Y=DEPOT[0])` snaps it to the network. Note the argument order: `X` is longitude, `Y` is latitude, the reverse of the `(lat, lon)` convention used for `DEPOT` and `Jam.centroid`.
- `N_CANDIDATES = 200`, `N_ORDERS = 300`, `MIN_TONNAGE = 0.2`, `N_JAMS = 3`.
- `N_VEHICLES = 50` for the first test, scaled to 200–300 once the script runs.
- `CAPACITIES = [1.2, 3.5, 7.0, 24.0]` tonnes of payload (approximate figures, to be checked against a reliable source).
- Seeds: candidates `SEED`, orders `SEED + 1`, fleet `SEED + 2`, jams `SEED + 3`.

**Entry point.** `main()` builds the network, finds the base node, samples the candidates, builds `tt0`, generates the orders, fleet and jams, and runs `run_simulation` once per policy, each on its own fresh state. It logs a comparison table and writes the outputs: `results/metrics.csv` (one row per policy, via `pandas`), `results/run.log`, and optionally `results/map.html` (Folium map of the base, loading points and delivery points). The script ends with the `if __name__ == "__main__":` guard.

**Task allocation:** frontier model (the baseline must share the dispatcher's interface and rules; `main` orchestrates the whole script).

## Task Allocation List

The rule, as in the LSCM demonstrator: a component goes to the local model when this document specifies it completely and it depends on no other component's behaviour; it goes to the frontier model when getting it right requires reasoning about the whole script.

| Component | Model | Reason |
| --- | --- | --- |
| 1 — Data structures and shared state | Local (Gemma) | Pure boilerplate, every field specified |
| 4 — Scenario generator | Local (Gemma) | Self-contained; its only dependency is the fully specified `tt`/`index` interface |
| 2 — Road network | Frontier | OSMnx API details, version differences, caching |
| 3 — Travel-time matrix and jams | Frontier | Graph algorithms, jam geometry, index bookkeeping |
| 5 — Feasibility filter and break rule | Frontier | Must match the state update, the event loop and both policies exactly |
| 6 — Dispatcher policy | Frontier | Most algorithmically subtle; BIG handling, tie-breaking, shared interface |
| 7 — State update | Frontier | Interdependent logic with the event loop |
| 8 — Event loop | Frontier | Orchestrates every other component |
| 9 — Baseline, metrics, entry point | Frontier | Baseline shares the dispatcher's interface; `main` orchestrates |

Component 1 is generated first, since every other component imports it. Component 4 follows once Component 3's interface is in place.
