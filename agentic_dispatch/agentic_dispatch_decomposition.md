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
