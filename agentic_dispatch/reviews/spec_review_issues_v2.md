# Specification Review (Revision 2): Assessment of Resolutions and Remaining Edge Cases

*Second review of `agentic_dispatch_constitution.md` and `agentic_dispatch_decomposition.md` (Revision 2, 9 November 2026) prior to code generation.*

---

## Executive Summary

Revision 2 of `agentic_dispatch_constitution.md` and `agentic_dispatch_decomposition.md` resolves **all 17 issues** identified in the first specification review (7 ambiguities, 4 items of missing information, and 6 internal contradictions). The data structures, function signatures, construction sequence, and task allocation between the local and frontier models are now mutually consistent.

No major contradictions remain. Below are **three minor edge cases** introduced by the new resolutions that should be clarified or handled explicitly so that every component behaves consistently.

---

## 1. Internal Contradictions / Edge Cases in Revision 2

### 1.1 Lingering `JAM_END`, `BREAK_END`, and Trailing `DISPATCH_TICK` Events After the Final Delivery (Decomposition, Component 4, 8 & 9)
- **Specification Reference:**
  - Component 4 (line 184): Jam `start` is drawn in `[0, horizon)` and `duration` in `[1800, 7200]` s, so `end = start + duration` can be as late as `horizon + 7200` (`36000` s).
  - Component 8 (line 317): *"Schedule the next tick at `now + TICK` only if orders can still arrive (`now < horizon`) or orders are pending or in progress. Otherwise no further tick is scheduled."*
  - Component 8 (lines 321–326): *"Orders arrive during the shift `[0, horizon)`. The simulation then continues until every order is delivered, so the queue empties naturally once the last delivery is done and ticks stop rescheduling... At the stop, set `state.end_time` to the current clock."*
  - Component 9 (line 342): `fleet_utilization_pct` divides total `busy_time` by `(number of vehicles * end_time)`.
- **Issue:**
  1. **Leftover `JAM_END` and `BREAK_END` events:** Suppose the last order is delivered (`VEHICLE_FREE`) at $t = 29\,200\text{ s}$, while a `JAM_END` event is still sitting in the priority queue at $t = 35\,000\text{ s}$ (or another vehicle started a 45-minute break at $t = 29\,100\text{ s}$ with `BREAK_END` at $t = 31\,800\text{ s}$). If the event loop runs until the priority queue is empty, it will pop those leftover `JAM_END` / `BREAK_END` events after all orders have already been completed, advancing the simulation clock to $t = 35\,000\text{ s}$ and inflating `state.end_time` (which reduces `fleet_utilization_pct`).
  2. **One trailing idle `DISPATCH_TICK`:** While the final order is `in_progress` (say at $t = 29\,100\text{ s}$), `DISPATCH_TICK` Step 5 sees `state.in_progress` is non-empty and schedules the next tick at $t = 29\,400\text{ s}$. When the final order completes via `VEHICLE_FREE` at $t = 29\,200\text{ s}$, that already-scheduled `DISPATCH_TICK` at $t = 29\,400\text{ s}$ is still in the queue; when popped, it runs `policy` on an empty pending set (recording an extra near-zero cycle time in `state.cycle_times`) and advances `state.end_time` from $29\,200\text{ s}$ to $29\,400\text{ s}$.
- **Recommended Resolution:**
  - Stop the event loop immediately when all orders have been delivered (`len(state.completed) == len(orders)`), setting `state.end_time = now` at that `VEHICLE_FREE` event, rather than draining remaining `JAM_END`, `BREAK_END`, or trailing `DISPATCH_TICK` events from the priority queue.

---

### 1.2 Responsibility for Appending `order_arrived`, `jam_started`, and `jam_ended` to `state.log` (Decomposition, Component 7 vs. Component 8)
- **Specification Reference:**
  - Component 7 (line 249): *"The functions below are the only code that changes `FleetState` (defined in Component 1). Each one appends a record to `state.log`."*
  - Component 7 Log Schema table (lines 279–288): Lists 8 record types: `order_arrived`, `assignment_issued`, `vehicle_free`, `break_started`, `break_ended`, `jam_started`, `jam_ended`, and `escalation`.
  - Component 8 Event Table (lines 300–308): Handles `JAM_START`, `JAM_END`, and `ORDER_ARRIVAL` by mutating `state.active_jams`, `state.orders`, and `state.pending`, without mentioning `state.log`.
- **Issue:**
  - Component 7 defines 5 functions (`apply_assignment`, `on_vehicle_free`, `start_break`, `on_break_end`, `escalate_waiting`) covering 5 of the 8 log record types, but provides no functions for `order_arrived`, `jam_started`, and `jam_ended`. Meanwhile, Component 8's action descriptions for `JAM_START`, `JAM_END`, and `ORDER_ARRIVAL` do not mention logging those three events to `state.log`.
- **Recommended Resolution:**
  - Either specify in Component 8's event table (lines 302, 303, 306) that `JAM_START`, `JAM_END`, and `ORDER_ARRIVAL` also append `jam_started`, `jam_ended`, and `order_arrived` records to `state.log`, or add three corresponding state-update helpers to Component 7.

---

## 2. Missing Information in Revision 2

### 2.1 Concrete Coordinates for `DEPOT` and Coordinate Order in `nearest_nodes` (Decomposition, Component 9)
- **Specification Reference:**
  - Component 9 (line 369): `DEPOT = (lat, lon) in the Antwerp port area, set by the author; ox.distance.nearest_nodes snaps it to the network.`
- **Issue:**
  - `DEPOT = (lat, lon)` is the only remaining placeholder without a concrete numeric value in Component 9.
  - Additionally, while `DEPOT` and `Jam.centroid` follow `(lat, lon)` order, `ox.distance.nearest_nodes(G, X, Y)` expects longitude (`X = DEPOT[1]`) as its second argument and latitude (`Y = DEPOT[0]`) as its third argument.
- **Recommended Resolution:**
  - Specify concrete numeric coordinates for `DEPOT` (e.g. `DEPOT = (51.2600, 4.3580)` in the Port of Antwerp) and note the `(X=lon, Y=lat)` argument order for `ox.distance.nearest_nodes`.
