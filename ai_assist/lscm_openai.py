from __future__ import annotations

"""Least Unit Cost Matrix Allocation (LSCM) implementation for OpenAI model.

The script follows the Specs‑Driven Development documents *constitution.md* and
*decomposition.md*. It implements the nine milestones listed in the decomposition
road‑map and adheres to the clarified conventions in *response3.txt*.
"""

import ast  # noqa: E402
import logging  # noqa: E402
from typing import List, NamedTuple, Tuple  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Logging configuration – all user‑visible output goes through the logger.
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class RawInput(NamedTuple):
    """Raw representation of the problem as entered by the user.

    Attributes
    ----------
    cost : List[List[int]]
        2‑D list where ``cost[i][j]`` is the unit cost from source *i* to demand *j*.
    supply : List[int]
        Available quantity at each source.
    demand : List[int]
        Required quantity at each destination.
    """

    cost: List[List[int]]
    supply: List[int]
    demand: List[int]

class ComputationalData(NamedTuple):
    """NumPy‑based container used during the algorithm.

    Attributes
    ----------
    cost_array : np.ndarray
    supply_array : np.ndarray
    demand_array : np.ndarray
    """

    cost_array: np.ndarray
    supply_array: np.ndarray
    demand_array: np.ndarray

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def parse_matrix(input_str: str) -> List[List[int]]:
    """Parse a string into a 2‑D list of integers.

    The function accepts any nesting style understood by ``ast.literal_eval`` –
    e.g. ``[[1, 2], (3, 4)]`` or ``((1,2),(3,4))``. It validates that the top‑level
    object is a list (or tuple) of rows and that each row contains only integer‑
    convertible items.

    Parameters
    ----------
    input_str: str
        Raw user input.

    Returns
    -------
    List[List[int]]
        The parsed matrix.

    Raises
    ------
    ValueError
        If the structure is not a proper nested list of numbers.
    """
    try:
        data = ast.literal_eval(input_str.strip())
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid literal structure provided: {e}")

    # Normalise tuples to lists for internal consistency.
    if isinstance(data, (list, tuple)):
        rows = list(data)
    else:
        raise ValueError("Input must represent a nested list/tuple structure.")

    # Handle bare single‑row case where top‑level is a flat list of ints
    if rows and all(isinstance(item, int) and not isinstance(item, bool) for item in rows):
        return [list(rows)]

    processed: List[List[int]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            raise ValueError("Each row in the matrix must be a list or tuple.")
        # Convert each element to int – any failure raises ValueError.
        try:
            int_row = [int(item) for item in row]
        except Exception as e:
            raise ValueError(f"Matrix elements must be integers: {e}")
        processed.append(int_row)
    return processed

def get_user_input() -> RawInput:
    """Interactively obtain cost matrix, supply list and demand list from the user.

    The function uses ``input()`` only (no ``argparse`` or ``sys.stdin``) as per the spec.
    Each line is parsed with :func:`parse_matrix` (cost) or a simple ``ast.literal_eval``
    for the one‑dimensional lists.
    """
    cost_str = input("Please enter the cost matrix (e.g. [[1,2],[3,4]]): ").strip()
    cost = parse_matrix(cost_str)

    supply_str = input("Please enter the supply list (e.g. [30,20]): ").strip()
    try:
        supply_raw = ast.literal_eval(supply_str)
    except Exception as e:
        raise ValueError(f"Invalid supply list: {e}")
    if not isinstance(supply_raw, (list, tuple)):
        raise ValueError("Supply must be a list or tuple of integers.")
    supply = []
    for v in supply_raw:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"Supply element {v} is not a valid integer.")
        supply.append(v)

    demand_str = input("Please enter the demand list (e.g. [25,25]): ").strip()
    try:
        demand_raw = ast.literal_eval(demand_str)
    except Exception as e:
        raise ValueError(f"Invalid demand list: {e}")
    if not isinstance(demand_raw, (list, tuple)):
        raise ValueError("Demand must be a list or tuple of integers.")
    demand = []
    for v in demand_raw:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"Demand element {v} is not a valid integer.")
        demand.append(v)

    return RawInput(cost=cost, supply=supply, demand=demand)



# ---------------------------------------------------------------------------
# Validation (assertions)
# ---------------------------------------------------------------------------
def assertions(raw: RawInput) -> None:
    """Validate dimensions, non‑negativity and balance of the problem.

    Raises
    ------
    ValueError
        If any check fails.
    """
    # Basic shape checks
    rows = len(raw.cost)
    if rows == 0:
        raise ValueError("Cost matrix must have at least one row.")
    cols = len(raw.cost[0])
    for idx, row in enumerate(raw.cost):
        if len(row) != cols:
            raise ValueError(f"Row {idx} length {len(row)} differs from expected {cols} (ragged matrix).")
    if len(raw.supply) != rows:
        raise ValueError("Supply length does not match number of cost matrix rows.")
    if len(raw.demand) != cols:
        raise ValueError("Demand length does not match number of cost matrix columns.")

    # Non‑negative values
    for i, row in enumerate(raw.cost):
        for j, val in enumerate(row):
            if val < 0:
                raise ValueError(f"Cost at ({i},{j}) is negative ({val}).")
    if any(v < 0 for v in raw.supply):
        raise ValueError("Supply contains negative values.")
    if any(v < 0 for v in raw.demand):
        raise ValueError("Demand contains negative values.")

    # Balance check
    if sum(raw.supply) != sum(raw.demand):
        raise ValueError("Problem is unbalanced: sum(supply) != sum(demand).")

# ---------------------------------------------------------------------------
# Tie‑breaking logic
# ---------------------------------------------------------------------------
def tie_breaking(
    candidates: List[Tuple[int, int]],
    supply: np.ndarray,
    demand: np.ndarray,
) -> Tuple[int, int] | None:
    """Apply Rules 2 and 3 to a list of candidate cells.

    Parameters
    ----------
    candidates : List[Tuple[int, int]]
        Coordinates of cells that share the minimal cost.
    supply, demand : np.ndarray
        Remaining supply and demand vectors (after Rule 1).

    Returns
    -------
    Tuple[int, int] | None
        The chosen ``(row, col)`` or ``None`` if no candidate satisfies Rule 2.
    """
    # Rule 2: prefer cells where remaining_supply >= remaining_demand
    rule2_candidates = [c for c in candidates if supply[c[0]] >= demand[c[1]]]
    if len(rule2_candidates) == 1:
        return rule2_candidates[0]
    if rule2_candidates:
        # Multiple still – pick the first in row‑major order
        return sorted(rule2_candidates)[0]
    # If Rule 2 yields nothing, the caller will apply Rule 3 (left‑to‑right).
    return None

# ---------------------------------------------------------------------------
# Allocation / orchestrator
# ---------------------------------------------------------------------------
def allocation(data: ComputationalData) -> np.ndarray:
    """Compute the basic feasible solution using the Least Unit Cost method.

    Parameters
    ----------
    data: ComputationalData
        Pre‑converted NumPy arrays for cost, supply and demand.

    Returns
    -------
    np.ndarray
        Allocation matrix of the same shape as the cost matrix (dtype=int).
    """
    cost = data.cost_array.copy()
    supply = data.supply_array.copy()
    demand = data.demand_array.copy()
    rows, cols = cost.shape
    allocation_mat = np.zeros((rows, cols), dtype=int)

    BLOCK_COST = int(np.max(cost)) + 1
    max_iters = rows * cols  # safety guard per spec
    it = 0
    while supply.sum() > 0 and demand.sum() > 0:
        it += 1
        if it > max_iters:
            logger.warning("Maximum iterations reached – possible infinite loop.")
            break
        # Find minimal cost among unblocked cells
        min_cost = np.min(cost)
        candidate_coords = [(i, j) for i in range(rows) for j in range(cols) if cost[i, j] == min_cost]
        # Rule 1 – choose cell with maximum allocatable quantity
        max_qty = -1
        rule1_winners: List[Tuple[int, int]] = []
        for i, j in candidate_coords:
            qty = min(supply[i], demand[j])
            if qty > max_qty:
                max_qty = qty
                rule1_winners = [(i, j)]
            elif qty == max_qty:
                rule1_winners.append((i, j))
        if len(rule1_winners) == 1:
            chosen = rule1_winners[0]
        else:
            # Apply Rule 2 via tie_breaking
            tb = tie_breaking(rule1_winners, supply, demand)
            if tb is not None:
                chosen = tb
            else:
                # Rule 3 – left‑to‑right, top‑to‑bottom
                chosen = sorted(rule1_winners)[0]
        i, j = chosen
        alloc_qty = min(supply[i], demand[j])
        allocation_mat[i, j] = alloc_qty
        supply[i] -= alloc_qty
        demand[j] -= alloc_qty
        # Block exhausted rows/cols
        if supply[i] == 0:
            cost[i, :] = BLOCK_COST
        if demand[j] == 0:
            cost[:, j] = BLOCK_COST
    return allocation_mat

# ---------------------------------------------------------------------------
# Feasibility and cost evaluation
# ---------------------------------------------------------------------------
def feasibility_cost(allocation_mat: np.ndarray, cost_arr: np.ndarray) -> int:
    """Validate BFS feasibility and return total transportation cost.

    Raises
    ------
    ValueError
        If the basic feasible solution is degenerate (allocated cells != m + n - 1).
    """
    rows, cols = cost_arr.shape
    allocated_cells = np.count_nonzero(allocation_mat)
    expected_cells = rows + cols - 1
    if allocated_cells != expected_cells:
        # Log detailed info before raising per spec
        logger.info("Feasibility check failed: allocated cells %d != expected %d", allocated_cells, expected_cells)
        logger.info("Allocation matrix:\n%s", allocation_mat)
        logger.info("Cost matrix:\n%s", cost_arr)
        logger.info("Expected number of basic variables is rows + cols - 1")
        raise ValueError("Degenerate basic feasible solution.")
    total_cost = int(np.sum(allocation_mat * cost_arr))
    logger.info("Feasibility check passed. Total cost: %d", total_cost)
    return total_cost

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the script.

    It reads user input, validates it, runs the allocation algorithm and reports
    the result using the configured logger.
    """
    try:
        raw = get_user_input()
        assertions(raw)
        data = ComputationalData(
            cost_array=np.array(raw.cost, dtype=int),
            supply_array=np.array(raw.supply, dtype=int),
            demand_array=np.array(raw.demand, dtype=int),
        )
        alloc = allocation(data)
        total = feasibility_cost(alloc, data.cost_array)
        logger.info("Allocation successful. Total transportation cost: %d", total)
        logger.info("Allocation matrix:\n%s", alloc)
    except ValueError as ve:
        logger.error("Input validation error: %s", ve)
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)

if __name__ == "__main__":
    main()
