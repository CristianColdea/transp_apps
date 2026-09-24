"""Least Unit Cost Matrix Allocation Method.

Generates a Basic Feasible Solution (BFS) for a balanced transportation
problem by iteratively allocating on the least-cost cell.
"""
from __future__ import annotations

import ast
import logging

import numpy as np
from typing import List, Tuple, NamedTuple


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: parse cost matrix (locally coded)
# ---------------------------------------------------------------------------
def parse_cost_matrix(input_str: str) -> list[list[int]]:
    """
    Parses a string representation of a cost matrix into a list of lists of integers.
    Supports various nesting styles (tuples, lists) and whitespace.
    A bare row like (1, 2, 3) is normalised into a single-row matrix [[1, 2, 3]].
    """
    # ast.literal_eval handles nested structures like [[1, 2], [3, 4]] or [(1, 2), (3, 4)]
    data = ast.literal_eval(input_str.strip())

    if not isinstance(data, (list, tuple)):
        raise ValueError("Input must represent a list of rows.")

    # Convert outer tuple to list
    if isinstance(data, tuple):
        data = list(data)

    # Detect bare row: a flat sequence of ints with no nested lists/tuples
    # e.g. (1, 2, 3) parsed as [1, 2, 3] — wrap into single-row matrix
    if data and all(isinstance(item, int) for item in data):
        return [data]

    processed_matrix = []
    for row in data:
        if isinstance(row, (list, tuple)):
            # Validate that all elements are integers — reject floats
            for item in row:
                if not isinstance(item, int):
                    raise ValueError(
                        f"Cost matrix elements must be integers, "
                        f"got {type(item).__name__}: {item}"
                    )
            processed_matrix.append(list(row))
        else:
            raise ValueError("Each row in the matrix must be a list or tuple.")

    return processed_matrix


# ---------------------------------------------------------------------------
# Data structures (locally coded)
# ---------------------------------------------------------------------------
class RawInput(NamedTuple):
    """A container for raw transportation problem inputs.

    Attributes:
        cost (List[List[int]]): A 2D list representing the cost matrix.
        supply (List[int]): A list of supply quantities at each source.
        demand (List[int]): A list of demand requirements at each destination.
    """

    cost: List[List[int]]
    supply: List[int]
    demand: List[int]


class ComputationalData(NamedTuple):
    """A container for processed numerical arrays used in calculations.

    Attributes:
        cost_array (np.ndarray): A 2D NumPy array of costs.
        supply_array (np.ndarray): A 1D NumPy array of supply values.
        demand_array (np.ndarray): A 1D NumPy array of demand values.
    """

    cost_array: np.ndarray
    supply_array: np.ndarray
    demand_array: np.ndarray


# ---------------------------------------------------------------------------
# get_user_input (frontier model coded)
# ---------------------------------------------------------------------------
def get_user_input() -> RawInput:
    """Prompt the user for cost matrix, supply, and demand values.

    Calls parse_cost_matrix for the cost matrix, and uses ast.literal_eval
    for supply and demand lists.  Raises ValueError on parse failure or
    invalid format.

    Returns:
        RawInput containing the parsed cost matrix, supply, and demand.
    """
    # --- Cost matrix ---
    try:
        cost_str = input("Enter the cost matrix (e.g. [[2,3],[5,1]]): ")
        cost = parse_cost_matrix(cost_str)
    except (ValueError, SyntaxError) as exc:
        logger.error("Failed to parse cost matrix: %s", exc)
        raise ValueError(f"Failed to parse cost matrix: {exc}") from exc

    # --- Supply ---
    try:
        supply_str = input("Enter the supply list (e.g. [20, 30]): ")
        supply_raw = ast.literal_eval(supply_str.strip())
    except (ValueError, SyntaxError) as exc:
        logger.error("Failed to parse supply input: %s", exc)
        raise ValueError(f"Failed to parse supply input: {exc}") from exc

    if not isinstance(supply_raw, (list, tuple)):
        logger.error("Supply input has invalid format: not a list or tuple.")
        raise ValueError("Supply must be a list or tuple of integers.")
    if not all(isinstance(x, int) for x in supply_raw):
        logger.error("Supply input has invalid format: non-integer element found.")
        raise ValueError("All supply values must be integers.")
    supply = list(supply_raw)

    # --- Demand ---
    try:
        demand_str = input("Enter the demand list (e.g. [10, 25, 15]): ")
        demand_raw = ast.literal_eval(demand_str.strip())
    except (ValueError, SyntaxError) as exc:
        logger.error("Failed to parse demand input: %s", exc)
        raise ValueError(f"Failed to parse demand input: {exc}") from exc

    if not isinstance(demand_raw, (list, tuple)):
        logger.error("Demand input has invalid format: not a list or tuple.")
        raise ValueError("Demand must be a list or tuple of integers.")
    if not all(isinstance(x, int) for x in demand_raw):
        logger.error("Demand input has invalid format: non-integer element found.")
        raise ValueError("All demand values must be integers.")
    demand = list(demand_raw)

    return RawInput(cost=cost, supply=supply, demand=demand)


# ---------------------------------------------------------------------------
# assertions (frontier model coded)
# ---------------------------------------------------------------------------
def assertions(raw: RawInput) -> None:
    """Validate shape consistency and balance of the transportation problem.

    Checks:
        1. Cost matrix is not ragged (all rows have equal length).
        2. Supply length matches the number of cost matrix rows.
        3. Demand length matches the number of cost matrix columns.
        4. Total supply equals total demand (balanced problem).

    Operates on a RawInput instance.  Raises ValueError on any failure.
    """
    # Check ragged matrix
    row_lengths = [len(row) for row in raw.cost]
    if len(set(row_lengths)) != 1:
        raise ValueError(
            f"Cost matrix is ragged: row lengths are {row_lengths}."
        )

    m = len(raw.cost)        # rows  (sources)
    n = row_lengths[0]       # columns (destinations)

    # Supply dimension
    if len(raw.supply) != m:
        raise ValueError(
            f"Supply length ({len(raw.supply)}) does not match "
            f"the number of cost matrix rows ({m})."
        )

    # Demand dimension
    if len(raw.demand) != n:
        raise ValueError(
            f"Demand length ({len(raw.demand)}) does not match "
            f"the number of cost matrix columns ({n})."
        )

    # Balanced check
    total_supply = sum(raw.supply)
    total_demand = sum(raw.demand)
    if total_supply != total_demand:
        raise ValueError(
            f"Problem is not balanced: total supply ({total_supply}) "
            f"!= total demand ({total_demand})."
        )


# ---------------------------------------------------------------------------
# tie_breaking (frontier model coded)
# ---------------------------------------------------------------------------
def tie_breaking(
    candidates: List[Tuple[int, int]],
    supply: np.ndarray,
    demand: np.ndarray,
) -> Tuple[int, int] | None:
    """Break ties among equally-cheapest cells.

    Rule 1 — Prefer the cell with the maximum allocatable quantity
             (min(supply[i], demand[j])).
    Rule 2 — Among cells with equal maximum quantity, prefer a cell
             where supply[i] >= demand[j].

    Args:
        candidates: list of (row, col) positions sharing the minimum cost.
        supply: current remaining supply array.
        demand: current remaining demand array.

    Returns:
        (row, col) of the selected cell, or None when neither rule
        resolves the tie (caller applies rule 3: index order).
    """
    # Rule 1: maximum allocatable amount
    amounts = [min(int(supply[i]), int(demand[j])) for i, j in candidates]
    max_amount = max(amounts)
    max_candidates = [
        (i, j)
        for (i, j), amt in zip(candidates, amounts)
        if amt == max_amount
    ]

    if len(max_candidates) == 1:
        return max_candidates[0]

    # Rule 2: prefer cell where supply >= demand
    preferred = [
        (i, j) for i, j in max_candidates if supply[i] >= demand[j]
    ]

    if preferred:
        # Multiple qualifying cells — pick first in row-major order
        return sorted(preferred)[0]

    # Neither rule resolved the tie
    return None


# ---------------------------------------------------------------------------
# allocation (frontier model coded)
# ---------------------------------------------------------------------------
def allocation(data: ComputationalData) -> np.ndarray:
    """Run the Least Unit Cost Matrix allocation algorithm.

    Iteratively allocates on the cheapest cell, applying tie-breaking
    rules when necessary.  Blocked cells receive a sentinel cost equal
    to max(cost) + 1.

    Args:
        data: ComputationalData with cost, supply, and demand arrays.

    Returns:
        2-D allocation matrix (the Basic Feasible Solution).
    """
    m, n = data.cost_array.shape
    cost_work = data.cost_array.astype(int).copy()
    remaining_supply = data.supply_array.astype(int).copy()
    remaining_demand = data.demand_array.astype(int).copy()
    alloc = np.zeros((m, n), dtype=int)

    BLOCK_COST = int(np.max(data.cost_array)) + 1
    max_iterations = m * n          # safety ceiling
    iteration = 0

    while int(np.sum(remaining_supply)) > 0 and iteration < max_iterations:
        iteration += 1

        min_cost = int(np.min(cost_work))
        if min_cost >= BLOCK_COST:
            logger.warning("All cells blocked — exiting allocation loop.")
            break

        # All cells sharing the current minimum cost
        rows, cols = np.where(cost_work == min_cost)
        positions = list(zip(rows.tolist(), cols.tolist()))

        if len(positions) == 1:
            # Unique minimum — normal allocation
            i, j = positions[0]
        else:
            # Attempt tie-breaking (rules 1 & 2)
            result = tie_breaking(positions, remaining_supply, remaining_demand)
            if result is not None:
                i, j = result
            else:
                # Rule 3: index order (left-to-right, top-to-bottom)
                i, j = sorted(positions)[0]

        # Allocate the maximum feasible quantity
        amount = min(int(remaining_supply[i]), int(remaining_demand[j]))
        alloc[i, j] = amount
        remaining_supply[i] -= amount
        remaining_demand[j] -= amount

        # Block exhausted rows / columns
        if remaining_supply[i] == 0:
            cost_work[i, :] = BLOCK_COST
        if remaining_demand[j] == 0:
            cost_work[:, j] = BLOCK_COST

    if iteration >= max_iterations:
        logger.error(
            "Safety limit reached: allocation loop hit %d iterations.",
            max_iterations,
        )

    return alloc


# ---------------------------------------------------------------------------
# feasibility_cost (frontier model coded)
# ---------------------------------------------------------------------------
def feasibility_cost(
    allocation_matrix: np.ndarray, cost_array: np.ndarray
) -> Tuple[bool, int]:
    """Check BFS feasibility and compute transportation cost.

    Feasibility criterion: allocated_cells == m + n - 1.
    Cost: numpy.sum(allocation_matrix * cost_array).

    Logs degeneracy messages via logger.info and raises ValueError when
    the solution is degenerate.

    Returns:
        (True, total_cost) when the solution is feasible.
    """
    m, n = allocation_matrix.shape
    allocated_cells = int(np.count_nonzero(allocation_matrix))
    required = m + n - 1

    if allocated_cells != required:
        logger.info(
            "The solution is not feasible, has %d allocation(s).",
            allocated_cells,
        )
        logger.info(
            "Should be %d allocation(s) for feasibility.", required
        )
        logger.info(
            "The subsequent optimization (e.g., MODI method) "
            "cannot proceed."
        )
        logger.info(
            "The cost of a basic degenerate solution makes no sense."
        )
        raise ValueError(
            f"Degenerate solution: {allocated_cells} allocation(s), "
            f"need {required}."
        )

    total_cost = int(np.sum(allocation_matrix * cost_array))
    return (True, total_cost)


# ---------------------------------------------------------------------------
# main (frontier model coded)
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Least Unit Cost Matrix transportation solver."""
    try:
        raw = get_user_input()
    except ValueError as exc:
        logger.error("Exiting due to input error: %s", exc)
        return

    try:
        assertions(raw)
    except ValueError as exc:
        logger.error("Exiting due to validation error: %s", exc)
        return

    data = ComputationalData(
        cost_array=np.array(raw.cost, dtype=int),
        supply_array=np.array(raw.supply, dtype=int),
        demand_array=np.array(raw.demand, dtype=int),
    )

    alloc = allocation(data)
    logger.info("Determined allocation (BFS):\n%s", alloc)

    try:
        is_feasible, total_cost = feasibility_cost(alloc, data.cost_array)
        logger.info("The solution is feasible: %s", is_feasible)
        logger.info("Total transportation cost: %d", total_cost)
    except ValueError as exc:
        logger.error("Exiting due to feasibility error: %s", exc)
        return


if __name__ == "__main__":
    main()
