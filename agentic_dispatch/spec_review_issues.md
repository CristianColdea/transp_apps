# Specification Review: Ambiguities, Missing Information, and Internal Contradictions

*Review of `agentic_dispatch_constitution.md` and `agentic_dispatch_decomposition.md` before code generation.*

---

## 1. Ambiguities

### 1.1 Interpretation of the Idle-Vehicle Break Rule (Decomposition, Component 5 & 7)
- **Specification Reference:** Component 5 (line 179): *"If an idle vehicle is infeasible for every pending order only because of condition 3, it starts its 45-minute break (handled in Component 7). Otherwise a tired driver would stay idle forever."*
- **Issues:**
  1. **Empty pending queue:** If `state.pending` is empty, "every pending order" is vacuously true. Presumably, an idle vehicle should only be evaluated for a break when `len(state.pending) > 0` and `vehicle.drive_time > 0`.
  2. **Multiple pending orders with mixed failure reasons:** Suppose there are two pending orders: Order A fails Condition 2 (tonnage exceeds a small van's capacity) and Order B passes Conditions 1, 2, and 4 but fails Condition 3 (remaining drive time is insufficient). Does *"infeasible for every pending order only because of condition 3"* mean:
     - **Option A:** For *every* pending order, the vehicle passes Conditions 1, 2, and 4 and fails Condition 3? (Under this rule, a small van could never take a break as long as a single heavy order is waiting in the queue.)
     - **Option B:** The vehicle is infeasible for *all* pending orders, and for *at least one* pending order it passes Conditions 1, 2, and 4 and fails only Condition 3?
  3. **Where in the event loop is this triggered?** Policies (`dispatch_policy` and `baseline_policy`) only return `list[Assignment]` and do not schedule `BREAK_END` events. Should `run_simulation` check remaining idle vehicles during `DISPATCH_TICK` *after* applying the returned assignments (or *before* calling `policy`), transition qualifying idle vehicles to `on_break`, and schedule their `BREAK_END` events?
  4. **Updating `free_at` on break:** When a vehicle enters `status = 'on_break'`, should `vehicle.free_at` also be set to `now + BREAK_DURATION`?

### 1.2 Definition of "Total En-Route Time" in Fleet Utilization (Decomposition, Component 7 & 9)
- **Specification Reference:** In `apply_assignment` (Component 7, line 226), `status='en_route'` lasts until `free_at = now + t_to_load + SERVICE_TIME + t_load_to_dest`, whereas `drive_time` only increments by `t_to_load + t_load_to_dest`. In Component 9 (line 276), Fleet Utilization is defined as:
  $$\frac{\text{Total en-route time of all vehicles}}{\text{number of vehicles} \times \text{horizon}} \quad (\%)$$
- **Issues:**
  1. Does "en-route time" mean `t_to_load + SERVICE_TIME + t_load_to_dest` (the entire duration the vehicle has `status == 'en_route'`) or `t_to_load + t_load_to_dest` (pure driving time)?
  2. If the simulation continues past `horizon` to complete in-progress orders, is the denominator `n_vehicles * horizon` or `n_vehicles * simulation_end_time`, and is en-route time after `horizon` included or clipped at `horizon`?

### 1.3 Jam Spatial Filtering and Overlapping Jams (Decomposition, Component 3 & 8)
- **Specification Reference:** Component 3 (line 120): *"When a jam starts, multiply the `travel_time` of every edge within `radius_m` of the jam centroid by `1 / speed_factor` on a copy of the graph, and recompute the matrix. When it ends, restore the original matrix."*
- **Issues:**
  1. **Edge-within-radius criterion:** How is an edge `(u, v, k)` determined to be within `radius_m` of `jam.centroid` `(lat, lon)`?
     - Endpoint `u` is within `radius_m`?
     - Both endpoints `u` and `v` are within `radius_m`?
     - Either endpoint `u` or `v` is within `radius_m`?
     - Edge midpoint is within `radius_m`?
  2. **Overlapping jams:** If Jam 1 (`[1000, 3000]`) and Jam 2 (`[2000, 4000]`) overlap in time, simply "restoring the original matrix" when Jam 1 ends at $t = 3000$ will wipe out Jam 2 while it is still active. Should the travel-time matrix instead be recomputed from the set of all currently active jams whenever any `JAM_START` or `JAM_END` event occurs (restoring the original matrix only when the active jam set becomes empty)?

### 1.4 Escalation Trigger and Deduplication (Decomposition, Component 7 & 8)
- **Specification Reference:** Component 7 (line 236): *"Whenever an order has waited longer than `ESCALATION_WAIT` (for example 2 hours) without a feasible vehicle, write an `escalation` record."*
- **Issues:**
  1. **Timing and condition:** Is this checked at each `DISPATCH_TICK` after assignments are applied for all orders still in `state.pending` where `now - order.arrival > ESCALATION_WAIT`? Or does it specifically require checking that `not any(is_feasible(v, order, tt, index) for v in state.vehicles.values())`?
  2. **Deduplication:** Should an `escalation` log record be emitted **once** per order (the first time its wait exceeds `ESCALATION_WAIT`), or repeatedly on every subsequent `DISPATCH_TICK` (every 300 seconds) while it remains pending?

### 1.5 Tie-Breaking in `baseline_policy` and `dispatch_policy` (Decomposition, Component 6 & 9)
- **Specification Reference:** At $t = 0$, all vehicles start idle at `base_node`. Consequently, every vehicle with sufficient capacity has the exact same travel time `t_to_load` to any given order's loading node.
- **Issues:**
  - In `baseline_policy` (and `dispatch_policy`), when multiple feasible vehicles are tied for the shortest `t_to_load`, should ties be broken by:
    - Smallest sufficient `vehicle.capacity` (preserving heavy trucks for heavier orders), or
    - Lowest `vehicle.id`?
  - Similarly, are `Vehicle.id` and `Order.id` 0-indexed (`0 .. n-1`) or 1-indexed (`1 .. n`), and are `Order.id` values assigned *after* sorting by `arrival` time?

### 1.6 Highway Speeds Override in `build_network` (Decomposition, Component 2)
- **Specification Reference:** The code snippet in Component 2 (line 77) calls `G = ox.add_edge_speeds(G)` without arguments, whereas the Note immediately below (line 85) says: *"Override with a `hwy_speeds` dict for Benelux-specific limits (e.g. `{'motorway': 120, 'primary': 90, 'residential': 30}`)."*
- **Issue:** Should `build_network` call `ox.add_edge_speeds(G)` as shown in the code block, or pass a specific `hwy_speeds` dictionary?

### 1.7 Reachability Direction in `generate_orders` (Decomposition, Component 3 & 4)
- **Specification Reference:** Component 3 (line 118) & Component 4 (line 146): *"An order is rejected and redrawn if its loading or delivery node is unreachable from the base, or if the trip from loading point to destination alone exceeds `DRIVE_LIMIT`."*
- **Issue:** Because the road network `G` is a directed multigraph (`nx.MultiDiGraph`), a node `n` can be reachable *from* the base (`tt[base, n] < inf`) while having no path *back* to the base or other nodes (`tt[n, base] == inf`, e.g., on a one-way boundary stub). Once a vehicle finishes delivering at `dest_node`, it remains at `dest_node` for future assignments. Should candidate/order nodes require **bidirectional** reachability with `base_node` (`tt[base, n] < inf` and `tt[n, base] < inf`)?

---

## 2. Missing Information

### 2.1 Missing Link Between `vehicle_id` and `order_id` on `VEHICLE_FREE` (Decomposition, Component 1, 7 & 8)
- `Vehicle` (Component 1, line 18) has fields `(id, capacity, node, status, free_at, drive_time)` — it does **not** store the currently assigned `order_id`.
- `FleetState.in_progress` (Component 7, line 214) is `dict[int, Order]`, keyed by `order_id`.
- `on_vehicle_free(state: FleetState, vehicle_id: int, now: float) -> float | None` (Component 7, line 232) only accepts `vehicle_id`.
- **Missing Specification:** When a `VEHICLE_FREE` event occurs, how does `on_vehicle_free` (or the event loop) know **which order** in `state.in_progress` was just completed by `vehicle_id` so that it can be moved from `state.in_progress` to `state.completed`?
  - For example: Does the `VEHICLE_FREE` event payload carry `(vehicle_id, order_id)` and pass `order_id` into `on_vehicle_free`?

### 2.2 No Jam Generator or Jam Scenario Specification (Decomposition, Component 4 & 9)
- Component 1 defines `Jam` and Component 8 (`run_simulation`) accepts `jams: list[Jam]`, but Component 4 (*"Scenario Generator: Orders and Fleet"*) only defines `generate_orders` and `make_fleet`.
- **Missing Specification:** How are `jams` created? Is there a `generate_jams` function or a fixed list of `Jam` instances in `main()`, and what are their centroids, radii (`radius_m`), speed factors (`speed_factor`), and `start`/`end` times?

### 2.3 Concrete Values for Constants, Scenario Parameters, and File Paths (Decomposition, Component 9)
- **Constants at top of script (line 283):**
  - Specified: `DRIVE_LIMIT = 16200`, `BREAK_DURATION = 2700`, `TICK = 300`, `BIG = 1e9`.
  - Unspecified or only parenthetical examples:
    - `SEED` (no value specified)
    - `SERVICE_TIME` (*"for example 1800 seconds"* in Component 7)
    - `ESCALATION_WAIT` (*"for example 2 hours"*, i.e., `7200` seconds, in Component 7)
    - `MAX_EVENTS` (no value specified)
    - `HORIZON` (no value specified)
- **Scenario parameters in `main()`:**
  - `place` string (e.g., `'Antwerp, Belgium'`) and `graphml_path`
  - Depot coordinates `(lat, lon)` in Antwerp used to find `base_node` via `ox.distance.nearest_nodes`
  - Number of `candidate_nodes` sampled from `G`
  - `n_orders`, `min_tonnage`, `max_tonnage`
  - `n_vehicles` and `capacities` (*"for example `[3.5, 7.5, 12.0, 24.0]`"*)
  - Output file paths for the CSV export, log file, and optional Folium HTML map.

### 2.4 Schema of `FleetState.log` Details and `compute_metrics` Dictionary Keys (Decomposition, Component 7 & 9)
- `FleetState.log` is `list[tuple[float, str, dict]]` with record types: `order_arrived`, `assignment_issued`, `vehicle_free`, `break_started`, `break_ended`, `jam_started`, `jam_ended`, `escalation`.
- **Missing Specification:**
  1. What are the exact dictionary keys inside `details` (`dict`) for each of the 8 log record types?
  2. What are the exact dictionary keys returned by `compute_metrics(state, horizon, cycle_times) -> dict[str, float]`? (Note that the metrics table in Component 9 has 4 rows, but row 4 — *"Computation time per cycle: Mean and maximum wall-clock time of one `policy` call"* — comprises two numeric values.)

---

## 3. Internal Contradictions

### 3.1 `generate_orders` Signature vs. Order Rejection Rules and Task Allocation (Decomposition, Component 3 & 4)
- **Contradiction:**
  - Component 4 (lines 129–132) specifies the signature:
    ```python
    def generate_orders(
        candidate_nodes: list[int], n_orders: int, horizon: float,
        min_tonnage: float, max_tonnage: float, seed: int,
    ) -> list[Order]:
    ```
    and assigns Component 4 to the local model because it is *"self-contained; inputs and outputs fully specified here, no dependency on other components' behaviour"* (line 155 & 294).
  - However, Component 3 (line 118) and Component 4 (line 146) require `generate_orders` to reject and redraw any order *"if its loading or delivery node is unreachable from the base, or if the trip from loading point to destination alone exceeds `DRIVE_LIMIT`."*
  - **Why this is contradictory:** `generate_orders` does not receive `tt`, `index`, `base_node`, or `G` in its parameter list, making it impossible for `generate_orders` to check reachability from the base or compare travel time `t_load_to_dest` against `DRIVE_LIMIT`.
  - Additionally, Component 3 (line 106) says `key_nodes` in `build_travel_time_matrix(G, key_nodes)` are *"Base, loading and delivery node ids (no duplicates)"*. If `key_nodes` comes from the generated orders, `build_travel_time_matrix` depends on `generate_orders` finishing first; yet `generate_orders` depends on `build_travel_time_matrix` to filter out unreachable/over-limit node pairs (unless `key_nodes` is defined as `[base_node] + candidate_nodes` and `tt`, `index`, `base_node` are passed into `generate_orders`).

### 3.2 `compute_metrics` Signature vs. Data Stored in `FleetState` (Decomposition, Component 7 & 9)
- **Contradiction:**
  - `compute_metrics(state: FleetState, horizon: float, cycle_times: list[float]) -> dict[str, float]` only takes `state`, `horizon`, and `cycle_times`.
  - **Missing order arrival times:** In `FleetState` (Component 7, lines 210–216 & 227), when an order is delivered, it is moved from `in_progress` to `completed: dict[int, float]`, which only stores `order_id -> delivery_time`. The `Order` instance (and thus `order.arrival`) is no longer in `pending`, `in_progress`, or `completed`. Unless `compute_metrics` parses `order_arrived` events from `state.log` (requiring a strict log schema), or `Order` objects are retained in `FleetState`, `delivery_time - arrival_time` cannot be computed.
  - **Missing cumulative en-route times:** `Vehicle.drive_time` resets to `0` after every break (`BREAK_DURATION`) and excludes `SERVICE_TIME`. Neither `Vehicle` nor `FleetState` has a cumulative en-route time field, so `compute_metrics` cannot compute Fleet Utilization from `state.vehicles` unless it reconstructs durations from `state.log` or a field is added.

### 3.3 `run_simulation` Return Type vs. `cycle_times` Parameter of `compute_metrics` (Decomposition, Component 8 & 9)
- **Contradiction:**
  - Component 9 (lines 270 & 279) states: *"Cycle times are measured in the event loop with `time.perf_counter()` around each `policy` call"* and passes `cycle_times: list[float]` to `compute_metrics(state: FleetState, horizon: float, cycle_times: list[float])`.
  - However, `run_simulation` in Component 8 (line 242) returns only `FleetState` (`-> FleetState`), and `FleetState` (Component 7, line 210) has no `cycle_times` field. Thus, `cycle_times` measured inside `run_simulation` never reach `main()` or `compute_metrics`.

### 3.4 `on_vehicle_free` Condition vs. Feasibility Filter Condition 3 and Idle Break Rule (Decomposition, Component 5 & 7)
- **Contradiction:**
  - Component 5 (Condition 3, line 167) enforces `vehicle.drive_time + t_to_load + t_load_to_dest <= DRIVE_LIMIT` before an assignment can be made. Therefore, when a vehicle finishes a delivery, its `drive_time` can never exceed `DRIVE_LIMIT` (it is strictly `< DRIVE_LIMIT` except in the rare case of exact equality `drive_time == DRIVE_LIMIT`).
  - Component 7 (line 234) specifies `on_vehicle_free`: *"if `drive_time >= DRIVE_LIMIT`, set `status='on_break'` and return `now + BREAK_DURATION` for a `BREAK_END` event; otherwise set `status='idle'` and return `None`... The break rule of Component 5 (idle but infeasible for every order) also calls the break branch."*
  - When an idle vehicle triggers the Component 5 break rule, its `drive_time` is strictly **less** than `DRIVE_LIMIT` (it is only `drive_time + t_to_load + t_load_to_dest` that exceeds `DRIVE_LIMIT`). Calling `on_vehicle_free` as written (`if drive_time >= DRIVE_LIMIT`) will evaluate to `False` and fail to put the vehicle on break, unless `on_vehicle_free` accepts a `force_break: bool = False` parameter or a separate `start_break` helper is defined.

### 3.5 Event Loop Termination vs. Infinite `DISPATCH_TICK` Scheduling and `Unserved orders` Metric (Decomposition, Component 8 & 9)
- **Contradiction:**
  - Component 8 (line 254) says `DISPATCH_TICK` always schedules the next tick, and line 260 says: *"Stop when the clock passes `horizon` and no orders are pending or in progress, or when the queue is empty. As a safety guard against infinite loops, stop after `MAX_EVENTS` events and log an error."*
  - Because `DISPATCH_TICK` reschedules itself unconditionally every `TICK = 300` seconds, the event queue is **never** empty.
  - Furthermore, if the simulation only terminates after `horizon` once *"no orders are pending or in progress"*, then the Component 9 metric **Unserved orders** (*"Orders still pending or in progress at the end"*, line 275) will **always be 0** unless the simulation hits `MAX_EVENTS` and logs an infinite-loop error.
  - Conversely, if the simulation is meant to stop at `horizon` so that remaining `pending` or `in_progress` orders are counted as `Unserved orders`, then the termination rule in Component 8 should stop when `now > horizon` (rather than waiting for `pending` and `in_progress` to empty).

### 3.6 Component Ordering, Graph Passing, and Allowed Libraries (Constitution vs. Decomposition)
- **`FleetState` placement vs. generation order:** Decomposition line 303 states that Components 1 and 4 are generated first because other components import their data structures. However, `FleetState` is defined in Component 7 (line 210) while referenced in the type annotations of Component 6 (`dispatch_policy(state: FleetState, ...)`, line 185). Moving `FleetState` to Component 1 resolves this.
- **Passing `G` to Components 5 and 6:** Component 2 (line 88) states *"The graph is passed to Components 3, 5 and 6"*, whereas Components 5 and 6 take `(tt, index)` rather than `G`, and Component 8 (`run_simulation`) takes `G`.
- **Copying state vs. allowed imports:** Constitution (line 19) restricts imports strictly to: `osmnx`, `networkx`, `scipy`, `numpy`, `pandas`, `folium`, `logging`, `dataclasses`, `heapq`, `random`, `typing`, `pathlib`, `time`. `copy` is not in the list, so running both policies on *"a fresh copy of the same initial state"* (Component 8, line 262; Component 9, line 281) must be done by constructing a fresh `FleetState` via `make_fleet` (or dict comprehensions creating fresh `Vehicle` instances) rather than `copy.deepcopy`.
