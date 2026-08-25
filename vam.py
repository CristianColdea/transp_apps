"""
Script to determine a Basic Feasible Solution (BFS) with Vogel alloc method.
It takes as inputs the cost matrix, the supply and demand lists. The length of
supply must match the number of cost matrix rows and that of demand the
columns, respectively.
Returns the allocation table, whether the feasible solution is possible, i.e.,
bool, and the allocation total cost.
In case of equal deltas on rows/cols the tie is broken on minimum unit cost
associated on rows/cols. If the minimum unit costs are equal this second tie
is broken by choosing the greatest amount possible to allocate. Otherwise the
allocation goes on min unit cost got by rows analysis.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, NamedTuple
import ast  # Required for safe string evaluation
import logging

logging.basicConfig(
    level=logging.DEBUG,  # switch to logging.INFO to silence the .debug() calls
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_matrix_literal(raw: str) -> List[List[int]]:
    """
    Parse a cost-matrix string as a single Python literal.

    Accepts any nesting style ast.literal_eval understands -- rows as
    tuples or lists, with or without an outer wrapper, with or without
    spaces -- rather than guessing the structure via string replacement.
    A single bare row (e.g. "(1, 2, 3)") is normalized into a one-row
    matrix so the return shape is always List[List[int]].
    """
    parsed = ast.literal_eval(raw.strip())

    if parsed and not isinstance(parsed[0], (list, tuple)):
        parsed = (parsed,)

    return [list(row) for row in parsed]


# +++++ USER INTERFACE AND INPUT HANDLING SECTION +++++

def get_user_input() -> tuple[List[List[int]], List[int], List[int]]:
    """
    Prompts the user for cost matrix, supply, and demand lists.
    Validates and safely converts the string inputs to required Python lists.
    """
    print("""
The following section provides the means of inputting data for the transportation problem.
Required data: **cost matrix (C)**, **supply list (S)**, and **demand list (D)**.

Input format example:
- Cost Matrix C (by row): (1, 2, 3), (4, 5, 6) or [[1, 2, 3], [4, 5, 6]]
- Supply List S: (10, 15) or [10, 15]
- Demand List D: (8, 8, 9) or [8, 8, 9]

The transportation plan must be **balanced**: sum of supplies = sum of demands.
""")

    try:
        # Get input strings
        c_str = input("Enter the matrix cost C:\n> ")
        s_str = input("Enter the list of supply S:\n> ")
        d_str = input("Enter the list of demand D:\n> ")

        # --- Safe Parsing using ast.literal_eval ---
        # This safely evaluates a string containing a Python literal structure (list/tuple).

        # Parsing Supply and Demand lists.
        # Evaluate once, then check isinstance against BOTH list and tuple --
        # the input format we advertise above accepts either, e.g. (10, 15)
        # or [10, 15], and ast.literal_eval returns a `list` for the bracket
        # form, not a `tuple`, so checking only `== tuple` silently mishandles
        # the bracket form (it gets wrapped as a single-element list instead
        # of unpacked).
        parsed_supply = ast.literal_eval(s_str.strip())
        if isinstance(parsed_supply, (list, tuple)):
            s_lst: List[int] = list(parsed_supply)
        else:
            s_lst = [parsed_supply]

        parsed_demand = ast.literal_eval(d_str.strip())
        if isinstance(parsed_demand, (list, tuple)):
            d_lst: List[int] = list(parsed_demand)
        else:
            d_lst = [parsed_demand]

        # Parsing Cost Matrix. Delegated to parse_matrix_literal() -- see
        # its docstring -- instead of manually splitting/replacing on
        # bracket characters, which only matched a couple of anticipated
        # spacing/punctuation patterns.
        c_lst = parse_matrix_literal(c_str)

        return c_lst, s_lst, d_lst

    except ValueError as exc:
        # Don't log or exit() here -- this function's job is to report the
        # problem, not decide the program's fate. Raise it and let the
        # caller (main(), at the bottom of the script) decide whether to
        # log, retry, or exit.
        raise ValueError(f"Failed to parse input. Please check your formatting. Details: {exc}") from exc
    except SyntaxError as exc:
        raise ValueError(
            "Input format is invalid. Ensure you use proper list/tuple syntax "
            "(e.g., (10, 15) or [10, 15])."
        ) from exc


# --- Execution Start ---

# Create the first structure for handling user input
class RawInput(NamedTuple):
    cost: List[List[int]]
    supply: List[int]
    demand: List[int]


# The assertions function is called after successful parsing.

def assertions(entries: RawInput) -> None:
    # Using the standard type hint List instead of list for compatibility with python versions < 3.9

    """
    Function to check the dimensional 'sanity', i.e., compatibility,
    of the cost matrix with supply and demand lists/arrays.
    Takes as input the cost matrix, supply and demand arrays from the
    first structure.
    Signals problems and aborts execution.
    """
    if len(entries.cost) != len(entries.supply):
        raise ValueError(f"Dimensional error: Cost matrix has \
                        {len(entries.cost)} rows, but Supply list has \
                        {len(entries.supply)} elements.")

    # Check that every row has the same length. The next check below only
    # looks at entries.cost[0] as a stand-in for "the matrix's column
    # count" -- that's only valid once we know the matrix is rectangular,
    # so this has to run first. Without it, a ragged matrix like
    # [[1, 2, 3], [4, 5]] passes every other check here (row count matches
    # supply, row 0's length matches demand, sums balance) and only fails
    # later, deep inside np.array(), with a raw numpy traceback instead of
    # a clean validation message.
    row_lengths = [len(row) for row in entries.cost]
    if len(set(row_lengths)) != 1:
        raise ValueError(f"Dimensional error: Cost matrix rows have \
                        inconsistent lengths {row_lengths}. \
                        Every row must have the same number of columns.")

    if len(entries.cost[0]) != len(entries.demand):
        raise ValueError(f"Dimensional error: Cost matrix has \
        {len(entries.cost[0])} columns, but Demand list has \
        {len(entries.demand)} elements.")

    if sum(entries.supply) != sum(entries.demand):
        raise ValueError(f"Transportation problem is **unbalanced**: \
                Sum of Supply \
                ({sum(entries.supply)}) != Sum of Demand \
                ({sum(entries.demand)}).")


# Create matrices and arrays for working data
# Bundle the computational arrays into a second structure

class ComputationalData(NamedTuple):
    cost_array: np.ndarray
    supply_array: np.ndarray
    demand_array: np.ndarray


# Bundle per-dimension allocation candidate: the chosen cell (row, col),
# the minimum unit cost available on the max-delta row/col, and the overall
# max delta for that dimension (used to compare rows vs cols).

class AllocCandidate(NamedTuple):
    row: int
    col: int
    min_unit_cost: int
    delta: int


# Sentinel returned when no valid candidate could be determined for a dimension
# (e.g. all unit costs in that dimension are already blocked).
_INVALID_CANDIDATE = AllocCandidate(row=-2, col=-2, min_unit_cost=-2, delta=-2)


def select_diff(uc_array: np.ndarray) -> int:
    """
    Analyzes the Unit Cost array (i.e., row or column) passed as arg.

    Returns the difference between the two Least Positive Unit Costs.
    """

    # 1. Ensure unit cost array copy for unwanted side effects
    uc_cp = uc_array.copy()

    # 2. Extract the difference between two of the least unit costs
    # 2.a. Extract true (not zeroed out by previous allocs) unit cost list
    diffs: list[int] = [l_uc for l_uc in np.sort(uc_cp) if l_uc > -1]
    if len(diffs) > 1:  # there are at least two positive least unit costs
        diff: int = diffs[1] - diffs[0]
    if len(diffs) == 1:  # only one positive unit cost
        diff = -1
    if len(diffs) == 0:  # no positive unit cost
        diff = -2

    return diff


def get_uc_min(ddiffs: dict[int, int], c_cp: np.ndarray) -> tuple[int, int, int]:
    """
    Selects the minimum Unit Cost index out of deltas dict.

    Returns the index of the value of minimum unit cost
    on max delta row/col.
    """

    max_delta: int = max(ddiffs.values())
    # get the index of row/col of max_delta
    max_ind: list[int] = [k for k, v in ddiffs.items() if v == max_delta]

    store_ind: list[int] = []
    store_uc_min: list[int] = []
    for ind in max_ind:  # for each max_delta row/col
        # get available unit cost list on max_delta row/col
        w_lst: list[int] = [uc for uc in c_cp[ind] if uc > -1]
        store_ind.append(ind)
        store_uc_min.append(min(w_lst))  # append the min uc available

    ind_uc_min: int = store_uc_min.index(min(store_uc_min))
    true_uc: list[int] = [x for x in list(c_cp[store_ind[ind_uc_min]]) if x >
                          -1]
    j: int = list(c_cp[store_ind[ind_uc_min]]).index(min(true_uc))

    return (store_ind[ind_uc_min], j, store_uc_min[ind_uc_min])


def detect_false_delta(delta_ind: int, uc_array: np.ndarray) -> tuple[int, int, int]:
    """
    Detects whether a row/col has only one true unit cost (uc).

    Returns the position and value of the only true unit cost
    (if it's the case) as a tuple (i, j, val) or (-2, -2, -2) if there are
    more or less than one true unit costs on row/col.
    """

    # 1. Extract a list with negative unit cost
    neg_uc: list[int] = [uc for uc in uc_array if uc == -1]

    # 2. Compare lengths of passed array and extracted list
    uc_list: list[int] = list(uc_array)
    uc_ind: int = uc_list.index(max(uc_list))
    if (len(uc_array) - len(neg_uc)) == 1:  # precisely one true uc
        return (delta_ind, uc_ind, max(uc_list))
    else:
        return (-2, -2, -2)


def alloc_vam(arrays: ComputationalData) -> np.ndarray:
    """
    Determines a Basic Feasible Solution (BFS) using the Vogel Alloc Method.

    Takes as input the ComputationalData structure (supply, demand, unit cost arrays).

    Returns the allocation matrix (decision variables).
    """
    # 1. Ensure Copies for Side-Effect-Free Operation
    s_cp = arrays.supply_array.copy()   # Working copy of Supply
    d_cp = arrays.demand_array.copy()   # Working copy of Demand
    c_cp = arrays.cost_array.copy()     # Working copy of Cost matrix (to '-1' out satisfied rows/cols)

    # Initialize the Allocation matrix (X_ij)
    # Using np.zeros_like is cleaner and more NumPy idiomatic
    allocation_matrix = np.zeros_like(arrays.cost_array, dtype=int)

    # A unit cost equal to -1 to effectively block satisfied sources/destinations
    BLOCK_COST = -1

    # Core loop continues until all supply is exhausted (which means demand is also zero,
    # due to the balancing assertion)

    t = 0  # set a counter for main loop

    while np.sum(s_cp) > 0:

        # 2. Call the specialized function to extract the difference between
        #    the least and next-to-the-least unit costs on rows and columns of
        #    the Unit Cost Matrix (UCM).
        #    Store the least unit cost on row/column pair indexes
        #    (as key) and difference (as value) in dicts, on rows/cols.
        #    The two dicts are necessary due to two perspectives regarding
        #    deltas (i.e., on rows and cols), being possible to have the same
        #    pair of indices (i, j) (the dict keys) where the Least Cost Unit
        #    is located, on rows and cols.

        ddiffs_r = {}    # dict to store {r: diff} on rows
        for r in range(len(c_cp)):    # iterate over rows of UCM
            diff = select_diff(c_cp[r])
            ddiffs_r[r] = diff

        ddiffs_c = {}    # dict to store {c: diff} on cols
        for c in range(len(c_cp.T)):    # iterate over columns of UCM
            diff = select_diff(c_cp.T[c])
            ddiffs_c[c] = diff

        # 3. Handle Ties and Allocation. The differentiation is either on
        #    equal max deltas or equal min unit costs.
        # 3.a. Determine the AllocCandidate for the row perspective.
        #      detect_false_delta() is called first for rows that have only one
        #      remaining unit cost (delta == -1); get_uc_min() handles all other cases.
        #      Note on index semantics:
        #        detect_false_delta(k, c_cp[k])   -> returns (row, col, uc)
        #        get_uc_min(ddiffs_r, c_cp)        -> returns (row, col, uc)

        is_uc_r_neg: bool = all(d <= -2 for d in ddiffs_r.values())
        is_fake_delta_r: bool = -1 in ddiffs_r.values()

        row_pick: AllocCandidate = _INVALID_CANDIDATE
        if not is_uc_r_neg:
            uc_r0: int = int(np.max(c_cp)) + 1  # running best uc for false-delta scan
            if is_fake_delta_r:
                for k, v in ddiffs_r.items():
                    if v == -1:  # suspect row: only one remaining unit cost
                        fd_row, fd_col, uc_r = detect_false_delta(k, c_cp[k])
                        if uc_r <= uc_r0:
                            uc_r0 = uc_r
                            row_pick = AllocCandidate(
                                row=fd_row, col=fd_col,
                                min_unit_cost=uc_r,
                                delta=max(ddiffs_r.values()),
                            )
                            logger.debug("row_pick from false delta: %s", row_pick)

            if is_fake_delta_r and row_pick is _INVALID_CANDIDATE:  # no valid false delta
                row, col, uc_min = get_uc_min(ddiffs_r, c_cp)
                row_pick = AllocCandidate(
                    row=row, col=col,
                    min_unit_cost=uc_min,
                    delta=max(ddiffs_r.values()),
                )

            if not is_fake_delta_r:  # no suspect delta row encountered
                row, col, uc_min = get_uc_min(ddiffs_r, c_cp)
                row_pick = AllocCandidate(
                    row=row, col=col,
                    min_unit_cost=uc_min,
                    delta=max(ddiffs_r.values()),
                )

        logger.debug("row_pick: %s", row_pick)

        # 3.b. Determine the AllocCandidate for the column perspective.
        #      Note on index semantics (transposed matrix means indices are flipped):
        #        detect_false_delta(k, c_cp.T[k]) -> returns (col, row, uc)
        #        get_uc_min(ddiffs_c, c_cp.T)      -> returns (col, row, uc)
        #      In both cases the first element is the column index and the
        #      second is the row index, so we assign them accordingly when
        #      constructing AllocCandidate(row=..., col=...).

        is_uc_c_neg: bool = all(d <= -2 for d in ddiffs_c.values())
        is_fake_delta_c: bool = -1 in ddiffs_c.values()

        col_pick: AllocCandidate = _INVALID_CANDIDATE
        if not is_uc_c_neg:
            uc_c0: int = int(np.max(c_cp)) + 1  # running best uc for false-delta scan
            if is_fake_delta_c:
                for k, v in ddiffs_c.items():
                    if v == -1:  # suspect col: only one remaining unit cost
                        fd_col, fd_row, uc_c = detect_false_delta(k, c_cp.T[k])
                        if uc_c <= uc_c0:
                            uc_c0 = uc_c
                            col_pick = AllocCandidate(
                                row=fd_row, col=fd_col,
                                min_unit_cost=uc_c,
                                delta=max(ddiffs_c.values()),
                            )
                            logger.debug("col_pick from false delta: %s", col_pick)

            if is_fake_delta_c and col_pick is _INVALID_CANDIDATE:  # no valid false delta
                col, row, uc_min = get_uc_min(ddiffs_c, c_cp.T)
                col_pick = AllocCandidate(
                    row=row, col=col,
                    min_unit_cost=uc_min,
                    delta=max(ddiffs_c.values()),
                )

            if not is_fake_delta_c:  # no suspect delta col encountered
                col, row, uc_min = get_uc_min(ddiffs_c, c_cp.T)
                col_pick = AllocCandidate(
                    row=row, col=col,
                    min_unit_cost=uc_min,
                    delta=max(ddiffs_c.values()),
                )

        logger.debug("col_pick: %s", col_pick)

        # Break early when all unit costs in both dimensions are blocked.
        if is_uc_r_neg and is_uc_c_neg:
            break

        # 3.c. Select the preferred candidate based on deltas and min unit cost.
        #      row_pick.delta / col_pick.delta hold the max delta for each dimension,
        #      so a direct comparison drives the tiebreak cascade correctly.
        chosen: AllocCandidate
        if row_pick.delta > col_pick.delta:
            chosen = row_pick
        elif col_pick.delta > row_pick.delta:
            chosen = col_pick
        else:  # equal deltas: fall back to min unit cost
            if col_pick.min_unit_cost > row_pick.min_unit_cost:
                chosen = row_pick
            elif row_pick.min_unit_cost > col_pick.min_unit_cost:
                chosen = col_pick
            else:  # equal min unit costs: prefer the greater allocatable quantity
                qty_r = min(s_cp[row_pick.row], d_cp[row_pick.col])
                qty_c = min(s_cp[col_pick.row], d_cp[col_pick.col])
                chosen = row_pick if qty_r >= qty_c else col_pick

        logger.debug("chosen: %s", chosen)

        # 4. Allocate to the chosen cell.
        allocation_quantity = min(s_cp[chosen.row], d_cp[chosen.col])
        allocation_matrix[chosen.row, chosen.col] = allocation_quantity
        logger.debug("alloc_quantity: %s", allocation_quantity)
        # Update remaining supply and demand
        s_cp[chosen.row] -= allocation_quantity
        d_cp[chosen.col] -= allocation_quantity

        # --- Block Satisfied Rows/Columns (Setting cost to BLOCK_COST) ---

        # If the supply source is exhausted, block the entire row
        if s_cp[chosen.row] == 0:
            c_cp[chosen.row, :] = BLOCK_COST

        # If the demand destination is satisfied, block the entire column
        if d_cp[chosen.col] == 0:
            c_cp[:, chosen.col] = BLOCK_COST

        logger.debug("alloc_matrix:\n%s", allocation_matrix)
        logger.debug("c_cp:\n%s", c_cp)

        t += 1
        logger.debug("iteration is: %s\n", t)

        # Safety: guard against an infinite loop if blocking logic has a gap.
        if allocation_quantity == 0 and np.sum(s_cp) > 0:
            logger.warning("Warning: Allocation loop stuck. Check data.")
            break

    return allocation_matrix


# ... (rest of the script follows: sum_check, feasibility_cost) ...
def feasibility_cost(allocation_matrix: np.ndarray, arrays: ComputationalData) -> Tuple[bool, int]:
    """
    Checks for non-degenerate Basic Feasible Solution (BFS) and computes the total cost.

    A BFS is non-degenerate if the number of basic (allocated) variables
    is equal to m + n - 1, where m is the number of rows (supply sources)
    and n is the number of columns (demand destinations).

    Args:
        allocation_matrix (np.ndarray): The matrix of decision variables (X_ij).
        arrays (ComputationalData): The structure containing the unit cost array (C_ij).

    Returns:
        Tuple[bool, int]: A tuple containing (is_feasible_bfs, total_cost).

    Raises:
        ValueError: If the basic solution is degenerate.
    """

    # 1. Calculate Total Cost (using NumPy for efficiency)
    # The element-wise multiplication of allocation * cost gives the total cost
    # for each cell, and then we sum the entire matrix.
    total_cost = int(np.sum(allocation_matrix * arrays.cost_array))

    # 2. Check Feasibility (Non-Degeneracy)
    # The shape gives us the dimensions: (m, n) -> (rows, columns)
    m, n = allocation_matrix.shape

    # Count the number of positive allocations (basic variables)
    num_basic_variables = np.count_nonzero(allocation_matrix)

    # Feasibility check: must have exactly m + n - 1 basic variables
    required_basic_vars = m + n - 1

    is_feasible_bfs = num_basic_variables == required_basic_vars

    # 3. Handle Degeneracy (Raise Exception)
    if not is_feasible_bfs:
        # Instead of printing and exiting, we raise an exception.
        # The main script can catch this and handle the termination or logging.
        raise ValueError(
            f"The basic solution is degenerate. Required basic variables: {required_basic_vars}. "
            f"Found: {num_basic_variables}. "
            "Optimization (e.g., MODI/Stepping Stone) cannot proceed directly."
        )

    # 4. Return Result
    # If the solution is feasible, return the check and the cost
    return (is_feasible_bfs, total_cost)


# --- How to use it in the main script ---

# sum_check function is also simplified:
def sum_check(allocation_matrix: np.ndarray) -> int:
    """
    Function to check whether the sum of allocated quantities matches the total supply/demand.
    """
    # Use NumPy's built-in sum() for a direct, efficient calculation.
    # Wrapped in int() -- ndarray.sum() is typed to return Any by numpy's
    # stubs, so without this mypy --strict flags an implicit Any leaking
    # out of a function declared to return int.
    return int(allocation_matrix.sum())


# Final output section uses a try/except block to catch the new ValueError

def main() -> None:
    # Get the basic user input. Parsing errors from get_user_input() now
    # arrive as a raised ValueError instead of the function calling exit()
    # itself -- main() is the only place that decides to log-and-exit.
    try:
        user_cost, user_supply, user_demand = get_user_input()
    except ValueError as exc:
        logger.error("\n❌ Error: %s", exc)
        exit(1)

    # Bundle the user input into the first structure for future usage
    entries = RawInput(cost=user_cost, supply=user_supply, demand=user_demand)

    try:
        assertions(entries)
    except ValueError as exc:
        logger.error("\n🛑 Validation Error: %s", exc)
        exit(1)

    arrays = ComputationalData(cost_array=np.array(entries.cost),
                               supply_array=np.array(entries.supply),
                               demand_array=np.array(entries.demand))

    # The core function call
    zrs_alloc_array = alloc_vam(arrays)

    logger.info("\n### Allocation Results ###")
    logger.info("Alloc matrix (Decision Variables) with VAM:\n%s", zrs_alloc_array)

    logger.info("\n### Final Cost and Feasibility Check ###")

    # Check total quantity matches
    sum_z = sum_check(zrs_alloc_array)
    logger.info("Sum of allocated quantities checks the total supply/demand:%s", sum_z == sum(arrays.supply_array))

    try:
        is_feasible, total_cost = feasibility_cost(zrs_alloc_array, arrays)

        logger.info("Basic solution is feasible:%s", is_feasible)
        logger.info("Vogel Allocation Method total allocation cost:%s", total_cost)

    except ValueError as exc:
        # Gracefully handle the degeneracy error raised by the function
        logger.error("\n🛑 Error: %s", exc)


if __name__ == "__main__":
    main()
