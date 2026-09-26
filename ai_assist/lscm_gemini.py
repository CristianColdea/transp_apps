"""Least Unit Cost Matrix Allocation Method.

Generates a Basic Feasible Solution (BFS) for a balanced transportation
problem by iteratively allocating on the least-cost cell.
"""
from __future__ import annotations

import ast
import logging
from typing import List, Tuple, NamedTuple
import numpy as np

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component 1: Helper function to parse unit cost matrix user input (locally coded)
# ---------------------------------------------------------------------------
def parse_matrix(input_data: str) -> List[List[int]]:
    """Parse a string representation of a nested numeric structure into a 2D list of integers.

    Args:
        input_data: A string containing nested lists or tuples (e.g., "[[1, 2], [3, 4]]").

    Returns:
        A list of lists of integers representing the unit cost matrix.

    Raises:
        ValueError: If input cannot be parsed or has invalid format/element types.
    """
    if not input_data or not input_data.strip():
        raise ValueError("Cost matrix input string cannot be empty.")

    try:
        parsed = ast.literal_eval(input_data.strip())
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid input format for cost matrix: {e}") from e

    if not isinstance(parsed, (list, tuple)):
        raise ValueError("Input must represent a list or tuple of rows.")

    data = list(parsed) if isinstance(parsed, tuple) else parsed

    if not data:
        raise ValueError("Cost matrix cannot be empty.")

    # Handle flat single row: e.g. (1, 2, 3) or [1, 2, 3] where all elements are integers
    if all(isinstance(item, int) and not isinstance(item, bool) for item in data):
        return [data]

    processed_matrix: List[List[int]] = []
    for row in data:
        if not isinstance(row, (list, tuple)):
            raise ValueError("Each row in the matrix must be a list or tuple.")

        processed_row: List[int] = []
        for item in row:
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError(
                    f"Cost matrix elements must be integers, got {type(item).__name__}: {item}"
                )
            processed_row.append(int(item))

        processed_matrix.append(processed_row)

    return processed_matrix


# ---------------------------------------------------------------------------
# Component 2: Function to parse user input (frontier model coded)
# ---------------------------------------------------------------------------
def get_user_input() -> RawInput:
    """Prompt user for cost matrix, supply, and demand values from terminal.

    Uses parse_matrix for cost matrix, and ast.literal_eval for supply
    and demand lists. Employs try/except (with raising) and isinstance validation.

    Returns:
        RawInput containing parsed cost matrix, supply list, and demand list.

    Raises:
        ValueError: On parse failure or invalid input format.
    """
    # 1. Cost matrix
    try:
        cost_str = input("Enter unit cost matrix (e.g., [[3, 1, 7, 4], [2, 6, 5, 9]]): ")
        cost = parse_matrix(cost_str)
    except (ValueError, SyntaxError) as exc:
        logger.error("Failed to parse cost matrix: %s", exc)
        raise ValueError(f"Failed to parse cost matrix: {exc}") from exc

    # 2. Supply vector
    try:
        supply_str = input("Enter supply list (e.g., [250, 300]): ")
        parsed_supply = ast.literal_eval(supply_str.strip())
    except (ValueError, SyntaxError) as exc:
        logger.error("Failed to parse supply input: %s", exc)
        raise ValueError(f"Failed to parse supply input: {exc}") from exc

    if not isinstance(parsed_supply, (list, tuple)):
        logger.error("Supply input has invalid format: not a list or tuple.")
        raise ValueError("Supply must be a list or tuple of integers.")

    if not all(isinstance(x, int) and not isinstance(x, bool) for x in parsed_supply):
        logger.error("Supply input has invalid format: non-integer element found.")
        raise ValueError("All supply values must be integers.")

    supply = [int(x) for x in parsed_supply]

    # 3. Demand vector
    try:
        demand_str = input("Enter demand list (e.g., [200, 225, 125]): ")
        parsed_demand = ast.literal_eval(demand_str.strip())
    except (ValueError, SyntaxError) as exc:
        logger.error("Failed to parse demand input: %s", exc)
        raise ValueError(f"Failed to parse demand input: {exc}") from exc

    if not isinstance(parsed_demand, (list, tuple)):
        logger.error("Demand input has invalid format: not a list or tuple.")
        raise ValueError("Demand must be a list or tuple of integers.")

    if not all(isinstance(x, int) and not isinstance(x, bool) for x in parsed_demand):
        logger.error("Demand input has invalid format: non-integer element found.")
        raise ValueError("All demand values must be integers.")

    demand = [int(x) for x in parsed_demand]

    return RawInput(cost=cost, supply=supply, demand=demand)


# ---------------------------------------------------------------------------
# Component 3: Data structures for raw user input and computational format (locally coded)
# ---------------------------------------------------------------------------
class RawInput(NamedTuple):
    """Container for raw transportation problem user inputs.

    Attributes:
        cost: A 2D list where cost[i][j] represents unit cost from source i to destination j.
        supply: Available supply quantities at each source.
        demand: Required demand quantities at each destination.
    """

    cost: List[List[int]]
    supply: List[int]
    demand: List[int]


class ComputationalData(NamedTuple):
    """Container for processed numerical NumPy arrays used in calculation.

    Attributes:
        cost_array: 2D NumPy array of costs.
        supply_array: 1D NumPy array of supply values.
        demand_array: 1D NumPy array of demand values.
    """

    cost_array: np.ndarray
    supply_array: np.ndarray
    demand_array: np.ndarray


# ---------------------------------------------------------------------------
# Component 4: Assertions check function (frontier model coded)
# ---------------------------------------------------------------------------
def assertions(raw: RawInput) -> None:
    """Ensure shape match, cost matrix non-raggedness, non-negative values, and balanced problem.

    Operates on RawInput structure and raises ValueError on any violation. Employs
    try/except to convert unexpected runtime exceptions into ValueError.

    Args:
        raw: RawInput instance to validate.

    Raises:
        ValueError: If shape mismatch, ragged rows, negative values, or unbalanced problem.
    """
    try:
        # 1. Non-ragged check
        row_lengths = [len(row) for row in raw.cost]
        if not row_lengths or len(set(row_lengths)) != 1:
            raise ValueError(f"Cost matrix is ragged or empty: row lengths are {row_lengths}.")

        m = len(raw.cost)
        n = row_lengths[0]

        # 2. Shape matching
        if len(raw.supply) != m:
            raise ValueError(
                f"Supply length ({len(raw.supply)}) does not match cost matrix rows ({m})."
            )
        if len(raw.demand) != n:
            raise ValueError(
                f"Demand length ({len(raw.demand)}) does not match cost matrix columns ({n})."
            )

        # 3. Non-negative checks
        for i, row in enumerate(raw.cost):
            for j, val in enumerate(row):
                if val < 0:
                    raise ValueError(f"Cost matrix value at cell ({i}, {j}) is negative: {val}")

        for i, val in enumerate(raw.supply):
            if val < 0:
                raise ValueError(f"Supply value at index {i} is negative: {val}")

        for j, val in enumerate(raw.demand):
            if val < 0:
                raise ValueError(f"Demand value at index {j} is negative: {val}")

        # 4. Balance check
        total_supply = sum(raw.supply)
        total_demand = sum(raw.demand)
        if total_supply != total_demand:
            raise ValueError(
                f"Problem is unbalanced: total supply ({total_supply}) != total demand ({total_demand})."
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unexpected error during assertions check: {exc}") from exc


# ---------------------------------------------------------------------------
# Component 5: Tie-breaking function (frontier model coded)
# ---------------------------------------------------------------------------
def tie_breaking(
    candidates: List[Tuple[int, int]],
    supply: np.ndarray,
    demand: np.ndarray,
) -> Tuple[int, int] | None:
    """Break ties among equally cheapest matrix cells using Rules 1 and 2.

    Rule 1: Maximum allocation quantity (min(supply[i], demand[j])).
    Rule 2: Prefer cell where supply[i] >= demand[j].

    Args:
        candidates: List of (row, col) tuples sharing minimum cost.
        supply: Current remaining supply array.
        demand: Current remaining demand array.

    Returns:
        Selected (row, col) position, or None if Rules 1 & 2 fail to resolve tie.
    """
    if not candidates:
        return None

    # Rule 1: Max allocatable quantity
    amounts = [min(int(supply[i]), int(demand[j])) for i, j in candidates]
    max_amount = max(amounts)
    max_candidates = [
        (i, j) for (i, j), amt in zip(candidates, amounts) if amt == max_amount
    ]

    if len(max_candidates) == 1:
        return max_candidates[0]

    # Rule 2: Prefer cell where supply >= demand
    preferred = [(i, j) for i, j in max_candidates if supply[i] >= demand[j]]

    if len(preferred) == 1:
        return preferred[0]
    elif len(preferred) > 1:
        return sorted(preferred)[0]

    return None


# ---------------------------------------------------------------------------
# Component 6: Allocation function (frontier model coded)
# ---------------------------------------------------------------------------
def allocation(data: ComputationalData) -> np.ndarray:
    """Run Least Unit Cost Matrix allocation algorithm to generate Basic Feasible Solution (BFS).

    Blocked cells use BLOCK_COST = max(cost_matrix) + 1. Exhausted rows and columns
    are fully blocked. Main loop runs while total remaining supply > 0 with a safety limit.

    Args:
        data: ComputationalData instance.

    Returns:
        2D NumPy array representing the allocation matrix (BFS).
    """
    m, n = data.cost_array.shape
    cost_work = data.cost_array.astype(int).copy()
    remaining_supply = data.supply_array.astype(int).copy()
    remaining_demand = data.demand_array.astype(int).copy()
    alloc = np.zeros((m, n), dtype=int)

    BLOCK_COST = int(np.max(data.cost_array)) + 1
    max_iterations = m * n
    iteration = 0

    while int(np.sum(remaining_supply)) > 0 and iteration < max_iterations:
        iteration += 1

        min_cost = int(np.min(cost_work))
        if min_cost >= BLOCK_COST:
            logger.warning("All cells blocked before allocation completed.")
            break

        # Find all cells matching minimum unblocked cost
        rows, cols = np.where(cost_work == min_cost)
        candidates = list(zip(rows.tolist(), cols.tolist()))

        if len(candidates) == 1:
            i, j = candidates[0]
        else:
            # Apply tie breaking rules 1 & 2
            chosen = tie_breaking(candidates, remaining_supply, remaining_demand)
            if chosen is not None:
                i, j = chosen
            else:
                # Rule 3: Left-to-right, top-to-bottom index order
                i, j = sorted(candidates)[0]

        # Allocate max possible amount
        amount = min(int(remaining_supply[i]), int(remaining_demand[j]))
        alloc[i, j] = amount
        remaining_supply[i] -= amount
        remaining_demand[j] -= amount

        # Block exhausted row and/or column completely
        if remaining_supply[i] == 0:
            cost_work[i, :] = BLOCK_COST
        if remaining_demand[j] == 0:
            cost_work[:, j] = BLOCK_COST

    if iteration >= max_iterations:
        logger.error("Safety mechanism triggered: allocation loop exceeded max iterations.")

    return alloc


# ---------------------------------------------------------------------------
# Component 7: Feasibility & transportation cost check (frontier model coded)
# ---------------------------------------------------------------------------
def feasibility_cost(
    allocation_matrix: np.ndarray, cost_array: np.ndarray
) -> Tuple[bool, int]:
    """Check feasibility (total allocated cells == m + n - 1) and compute BFS cost.

    Logs degeneracy notification messages via logger.info and raises ValueError
    if the solution is degenerate.

    Args:
        allocation_matrix: 2D NumPy allocation array (BFS).
        cost_array: Original 2D NumPy cost array.

    Returns:
        Tuple of (True, BFS_cost) if solution is feasible.

    Raises:
        ValueError: If solution is degenerate.
    """
    m, n = allocation_matrix.shape
    allocated_cells = int(np.count_nonzero(allocation_matrix))
    required_allocations = m + n - 1

    if allocated_cells != required_allocations:
        logger.info("The solution is not feasible, has %d allocations.", allocated_cells)
        logger.info("Should be %d allocations for feasibility.", required_allocations)
        logger.info("The subsequent optimization (e.g., MODI method) cannot proceed.")
        logger.info("The cost of a basic degenerate solution makes no sense.")
        raise ValueError(
            f"Degenerate solution: {allocated_cells} allocation(s), required {required_allocations}."
        )

    bfs_cost = int(np.sum(allocation_matrix * cost_array))
    return (True, bfs_cost)


# ---------------------------------------------------------------------------
# Component 9 & 8: Main orchestrator (frontier model coded)
# ---------------------------------------------------------------------------
def main() -> None:
    """Orchestrate Least Unit Cost Matrix allocation pipeline."""
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
    logger.info("Basic Feasible Solution (Allocation Matrix):\n%s", alloc)

    try:
        is_feasible, total_cost = feasibility_cost(alloc, data.cost_array)
        logger.info("Solution feasibility: %s", is_feasible)
        logger.info("Total transportation cost: %d", total_cost)
    except ValueError as exc:
        logger.error("Exiting due to feasibility error: %s", exc)
        return


if __name__ == "__main__":
    main()
