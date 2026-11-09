# Fleet Dispatch Demonstrator: Initial Design Notes

> **Status.** These are the initial design notes of the dispatch demonstrator, kept as a record of how the design began. They are no longer the specification. The binding documents are `agentic_dispatch_constitution.md` and `agentic_dispatch_decomposition.md` (Revision 3, 9 November 2026). The last section shows where each open question below was resolved.

## Objective

The dispatch optimization is **time-oriented**: decisions aim to minimize time (e.g., travel time to the loading point, total time to fulfill an order), not distance.

# DATA

## Vehicle Data

1. Available capacity.
2. Available time, i.e., time until the mandatory break.
3. Time until free: the vehicle is currently on the move to finish an order.
4. Position on the map.
5. Status (idle, en route, on break), so that a null value does not carry two meanings.

## Orders

1. Estimated travel time to fulfill the order (deliver the order tonnage from the loading point to the destination point).
2. Order loading point on the map.
3. Order destination (delivery) point on the map.
4. Order tonnage.

## Bottlenecks / Traffic jams

1. Bottleneck / jam centroid on the map.
2. Bottleneck / jam spread on the map.
3. Estimated average travel speed to traverse the bottleneck / jam (depends on the route crossing the jam).
4. Probability of the bottleneck / jam taking place (projection of the jam over the coming hours or days).

# ACTIONS

## For each order loading point

1. A ranking of vehicles by travel time between the vehicle position on the map and the loading point.
2. A ranking of vehicles by time until the mandatory break.
3. A ranking of vehicles by available capacity.

## Allocation

This is to be made based on the above rankings and on bottlenecks / jams.

## Update data after each allocation

1. Update vehicle state.
2. Update orders list.

## What isn't clear yet

1. A specialized function to properly assess travel time over a combination of urban / non-urban roads (OSMnx could be the key here).
2. A specialized function to apply bottlenecks.
3. A specialized function to decompose each order tonnage into sub-orders that fit the available capacities.
4. A simulation clock: the state changes with time as well as with allocations (vehicles finish orders, driving hours are consumed, jams appear and clear).
5. How to combine the three rankings: which criteria act as hard constraints (e.g., available capacity, time until the mandatory break) and which one is the objective (travel time).

## How to start? A proposal

1. Choose a presumable fleet base (e.g., Antwerp).
2. Assume each vehicle is initially located at the base.
3. Generate a random orders list. Randomness in three dimensions: a) loading points; b) delivery points and, thus, by applying the travel time function, the travel times are determined (orders as pairs of loading-destination points are natural for humans); c) tonnages.
4. A list / state of current bottlenecks at region level.
5. (Optional, later) A projection of bottlenecks into the future.

---

# How the Open Questions Were Resolved

Component numbers refer to `agentic_dispatch_decomposition.md` (Revision 3).

| Open question | Resolution | Where |
| --- | --- | --- |
| 1. Travel time on urban and non-urban roads | OSMnx sets edge speeds by road class (or from `maxspeed` tags), capped at 90 km/h for the whole fleet; Dijkstra computes a travel-time matrix between the base and candidate nodes | Components 2–3 |
| 2. Applying bottlenecks | Each jam slows the edges within its radius; the matrix is rebuilt from the set of active jams whenever a jam starts or ends | Components 3, 8 |
| 3. Splitting order tonnage | Not implemented in this version: orders never exceed the largest vehicle capacity. Splitting remains a later extension | Component 4; Constitution, simplifications |
| 4. Simulation clock | Discrete-event simulation with a priority queue of events (order arrivals, vehicles becoming free, breaks, jams, dispatch ticks) | Component 8 |
| 5. Combining the three rankings | Capacity and driving time are hard constraints (feasibility filter); travel time to the loading point is the objective; capacity survives only as a tie-breaker (smallest sufficient vehicle) | Components 5–6 |

| Starting proposal | Outcome |
| --- | --- |
| 1. Fleet base in Antwerp | Kept: a depot in the Antwerp port area |
| 2. All vehicles start at the base | Kept |
| 3. Random orders (loading points, delivery points, tonnages) | Kept, with a fixed seed for reproducibility; tonnage is drawn per vehicle class |
| 4. Current bottlenecks | Kept as known jams with a start and end time |
| 5. Projection of bottlenecks | Deferred to a later version |

Two items of the original data model also changed form: the *time until the mandatory break* became the driving time accumulated since the last break, checked against the 4.5-hour limit, and the *estimated travel time to fulfill an order* is read from the travel-time matrix instead of being stored with the order.
