# Benelux Fleet Dispatch Demonstrator — Constitution

*Section 6.4 of Chapter N — Possibilities of AI in Planning and Optimization*

---

**Project Name:** Benelux Fleet Dispatch Demonstrator **Motto:** *"From architecture to allocation."*

---

## Mission

To implement a simplified but structurally honest fleet dispatch demonstrator for a mixed van/truck fleet operating in the Benelux region, based on the three-tier multi-agent architecture of Section 6.3. The demonstrator serves as a pedagogical illustration of agentic, time-oriented dispatch optimization, not as a production system.

---

## Tech Stack and Constraints

- **Libraries:** `osmnx`, `networkx`, `scipy`, `numpy`, `pandas`, `folium`, `logging`, `dataclasses`, `heapq`, `random`, `typing`, `pathlib`, `time`.
- **Imports:** `from __future__ import annotations`.
- All interaction is programmatic (no terminal input); outputs are logged and written to file or rendered as a Folium map.
- Output must be reproducible: a fixed random seed governs all stochastic elements.
- The listed libraries are the only ones to be used. No machine learning frameworks in this version.

---

## Principles

- Prefer clarity and readability over cleverness.
- Every public function and class must have a docstring and type hints.
- The state is the single source of truth: agents read from and write to it; they do not communicate peer-to-peer.
- Vehicles receive assignments and route updates only through their regional coordinator (one boss rule).
- The dispatch objective is time-oriented: minimize total time orders spend in the system.
- Simplifications must be stated explicitly, not hidden.

---

## Roadmap

Nine milestones in dependency order:

1. **Data structures and shared state:** `Vehicle` and `FleetState` as `dataclass`; `Order`, `Jam` and `Assignment` as `NamedTuple`.
2. **Road network:** download the study area from OpenStreetMap via OSMnx, add edge speeds and travel times, save to GraphML for reuse.
3. **Travel-time matrix:** precompute shortest-path travel times (Dijkstra, weight = `travel_time`) between the base and all candidate nodes, and rebuild it from the set of active jams whenever a jam starts or ends.
4. **Scenario generator:** produce a random but reproducible scenario: candidate nodes, orders (loading node, delivery node, tonnage, arrival time), the fleet, and traffic jams.
5. **Feasibility filter:** given the current vehicle states and an order, return the list of feasible vehicle–order pairs (capacity and driving-time constraints).
6. **Cost matrix and assignment:** build the time-to-loading cost matrix for feasible pairs and solve with `scipy.optimize.linear_sum_assignment`.
7. **State update:** apply an assignment to the shared state (vehicle position, free time, driving hours, order queue).
8. **Event loop:** priority queue of events (order arrival, vehicle free, break end, jam start/end, dispatch tick); advance clock, update state, run dispatch cycle.
9. **Metrics and baseline:** compute mean time in system, fleet utilization and computation time per cycle for both the dispatcher and the nearest-feasible-vehicle baseline; log and export results.

---

## Design Decisions (confirmed)

| Decision | Choice | Reason |
| --- | --- | --- |
| Objective | Minimize total time orders spend in the system | Time-oriented, as required |
| Allocation rule | Batch assignment per cycle: filter infeasible pairs, then minimize total time to reach loading points | Optimal within each cycle; no order gets the best vehicle only because it was processed first |
| Baseline | Each order, in arrival order, takes the nearest feasible free vehicle with sufficient capacity | Simple rule a human dispatcher would apply |
| Metrics | Mean time in system; fleet utilization; computation time per cycle | Service quality, efficiency, scalability |
| Uncertainty | Known traffic jams only | Probabilistic projection is a later extension |
| Scale | One base (Antwerp), subset of road network, precomputed travel times | Tractability |
| Reproducibility | Fixed random seed; same event timeline for dispatcher and baseline | Fair comparison |

---

## Explicit Simplifications

- Orders are capped at the largest vehicle capacity (no splitting in this version).
- Driver rules are limited to the 4.5-hour driving limit and 45-minute break (not full daily/weekly limits).
- Idle waiting does not count as rest; only a scheduled break resets the driving time.
- One speed cap of 90 km/h applies to the whole fleet; vans are assumed to drive like trucks.
- Only idle vehicles are considered at each dispatch tick.
- One regional coordinator handles the whole study area in this version.
- The strategic tier is reduced to logging situations it would escalate to the human dispatcher.
- No machine learning components in this version; the decision step uses a classical assignment solver.

---

*Next step: decomposition document.*
