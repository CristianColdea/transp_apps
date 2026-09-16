import heapq
import math


"""
Simple Python script to exemplify code implementation of A* algorithm.
"""

# -------------------------------------------------------------------
# Exemplification of 'heapq' in action
# -------------------------------------------------------------------

# Define an unordered list
l_unord = [23, 12, 27, 9, 32]
# Turn the initial list into a heapifyed one
heapq.heapify(l_unord)
# prints [9, 12, 27, 23, 32]

# -------------------------------------------------------------------
# Graph definition
# Each node has (x, y) coordinates (e.g. approximate GPS-like coords)
# Edges are stored as adjacency list: node -> [(neighbour, weight)]
# -------------------------------------------------------------------

nodes = {
    'A': (0.0, 0.0),   # Depot
    'B': (1.0, 2.0),
    'C': (3.0, 1.0),
    'D': (4.0, 3.0),
    'E': (6.0, 0.5),   # Destination
    'F': (2.0, 4.0),
    'G': (5.0, 3.5),
}

edges = {
    'A': [('B', 2.2), ('C', 3.2)],
    'B': [('A', 2.2), ('C', 2.3), ('F', 2.8)],
    'C': [('A', 3.2), ('B', 2.3), ('D', 2.5), ('E', 3.6)],
    'D': [('B', 3.0), ('C', 2.5), ('F', 1.5), ('G', 1.6)],
    'E': [('C', 3.6), ('G', 3.2)],
    'F': [('B', 2.8), ('D', 1.5)],
    'G': [('D', 1.6), ('E', 3.2)],
}

# -------------------------------------------------------------------
# Heuristic: straight-line (Euclidean) distance between two nodes
# This is admissible because straight-line <= road distance
# -------------------------------------------------------------------

def heuristic(node, goal):
    x1, y1 = nodes[node]
    x2, y2 = nodes[goal]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# -------------------------------------------------------------------
# A* algorithm
# Returns: (total_cost, path) or (inf, []) if no path exists
# -------------------------------------------------------------------

def astar(start, goal):
    # Priority queue entries: (f_score, node, g_score, path)
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), start, 0, [start]))

    # Best known g_score for each visited node
    visited = {}

    while open_set:
        f, current, g, path = heapq.heappop(open_set)

        # Goal reached
        if current == goal:
            return g, path

        # Skip if we already found a better path to this node
        if current in visited and visited[current] <= g:
            continue
        visited[current] = g

        # Expand neighbours
        for neighbour, weight in edges.get(current, []):
            g_new = g + weight
            f_new = g_new + heuristic(neighbour, goal)
            heapq.heappush(open_set, (f_new, neighbour, g_new, path + [neighbour]))

    return float('inf'), []  # No path found

# -------------------------------------------------------------------
# Example: find shortest path from Depot (A) to Destination (E)
# -------------------------------------------------------------------

cost, path = astar('A', 'E')
print(f"Shortest path: {' -> '.join(path)}")
print(f"Total cost:    {cost:.2f}")
