"""
Script to determine a Basic Feasible Solution (BFS) with Least Unit Cost
allocation method.
It takes as inputs the cost matrix, the supply and demand lists. The length of
supply must match the number of cost matrix rows and that of demand the
columns, respectively.
Returns the allocation table, whether the feasible solution is possible, i.e.,
bool, and the allocation total cost.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, NamedTuple
import ast # Required for safe string evaluation
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


# The assertions function it is called after successful parsing.

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

"""
If the search operation for the least unit cost yields a Tie, there is a 
preferred allocation according to the following two criteria (in this order):

1) allocate to the least cost unit that allows to move the higher quantity,
2) allocate where the supply exceeds demand.
For this purpose a specialized function is to be coded.
"""

def alloc_pref(arrays: ComputationalData, min_indices: np.ndarray) ->\
        Tuple[int, int]:
    """
    Determines the preferred allocation if there is Least Unit Cost Tie.
    Takes as inputs the supply, demand, unit cost and min_indices arrays.
    Returns a tuple with indexes (i_pref, j_pref) of preferred allocation.
    Raises value error if the array of minimum indices doesn't have at least
    two pair of values.
    """

    if min_indices.shape[0] < 2:
        raise ValueError(
            f"alloc_pref requires at least two tied indices, got {min_indices.shape[0]}."
        )
    
    # 1. Ensure Copies for Side-Effect-Free Operation
    s_cp = arrays.supply_array.copy()
    d_cp = arrays.demand_array.copy()
    min_indices_cp = min_indices.copy()
    
    #dict to store the indices and masses associated with least cost
    masses = {}

    # 2. Loop the least unit cost indices array and store the associated masses
    for i, j in min_indices_cp:
        masses[(i, j)] = min(s_cp[i], d_cp[j])

    # 3. Get the indices of least unit cost with greatest tonnage
    max_ton = max(masses.values())   #extract the Max value out of dict
    min_indsQmax = [k for k in masses if masses[k] == max_ton] #list of inds

    # 4. Allocate when there is only one max tonnage
    if(len(min_indsQmax) < 2):
        return min_indsQmax[0]
    # 5. More than one max tonnage, check if S > D
    else:
        for i,j in min_indsQmax:
            if s_cp[i] >= d_cp[j]:
                return(i,j)
        return min_indsQmax[0]
     
def alloc_luc(arrays: ComputationalData) -> np.ndarray:
    """
    Determines a Basic Feasible Solution (BFS) using the Least Unit Cost Method.
    Takes as inputs the supply, demand, and unit cost arrays.
    Returns the allocation matrix (decision variables).
    """

    # 1. Ensure Copies for Side-Effect-Free Operation
    s_cp = arrays.supply_array.copy()  # Working copy of Supply
    d_cp = arrays.demand_array.copy()  # Working copy of Demand
    c_cp = arrays.cost_array.copy()  # Working copy of Cost matrix (to 'inf' out satisfied rows/cols)
    
    # Initialize the Allocation matrix (X_ij)
    # Using np.zeros_like is cleaner and more NumPy idiomatic
    allocation_matrix = np.zeros_like(c_cp, dtype=int)
    
    # A value larger than any possible cost to effectively block satisfied sources/destinations
    BLOCK_COST = np.max(c_cp) + 1 
    
    # Core loop continues until all supply is exhausted (which means demand is also zero,
    # due to the balancing assertion)
    while np.sum(s_cp) > 0:
        
        # 2. Find the minimum cost in the *current* cost matrix
        min_cost = np.min(c_cp)
        
        # Get all (row, column) indices where the cost equals the current minimum
        # np.argwhere returns a list of [row, col] arrays
        min_indices = np.argwhere(c_cp == min_cost)
        
        allocated_in_cycle = False    #safety for while loop ...
                               
        # 3. Allocate to preferred position, if this is the case
        #    by calling alloc_pref() function
        if min_indices.shape[0] != 1:
            (i_pref, j_pref) = alloc_pref(arrays, min_indices)
            allocation_quantity = min(s_cp[i_pref], d_cp[j_pref])
            allocation_matrix[i_pref, j_pref] = allocation_quantity
            # Update remaining supply and demand
            s_cp[i_pref] -= allocation_quantity
            d_cp[j_pref] -= allocation_quantity
            
            # --- Block Satisfied Rows/Columns (Setting cost to a high value) ---
            
            # If the supply source 'i_pref' is exhausted, block the entire row
            if s_cp[i_pref] == 0:
                c_cp[i_pref, :] = BLOCK_COST
            
            # If the demand destination 'j_pref' is satisfied, block the entire column
            if d_cp[j_pref] == 0:
                c_cp[:, j_pref] = BLOCK_COST
            
            allocated_in_cycle = True

            continue
        
        
        # 4. Allocate normaly ...
        for i, j in min_indices:
            # i = source row index (supply)
            # j = destination column index (demand)
            
            
            # Determine the maximum possible allocation (min of remaining supply and demand)
            allocation_quantity = min(s_cp[i], d_cp[j])
            
            # Skip if this cell has zero remaining supply/demand (shouldn't happen 
            # often if blocking works, but provides a guard)
            if allocation_quantity == 0:
                continue

            # --- Make the Allocation ---
            
            # Store the allocation quantity in the result matrix
            allocation_matrix[i, j] = allocation_quantity
            
            # Update remaining supply and demand
            s_cp[i] -= allocation_quantity
            d_cp[j] -= allocation_quantity
            
            # --- Block Satisfied Rows/Columns (Setting cost to a high value) ---
            
            # If the supply source 'i' is exhausted, block the entire row
            if s_cp[i] == 0:
                c_cp[i, :] = BLOCK_COST
            
            # If the demand destination 'j' is satisfied, block the entire column
            if d_cp[j] == 0:
                c_cp[:, j] = BLOCK_COST
                
            allocated_in_cycle = True
            break # Essential: Restart the while loop to find the NEW minimum cost
                  # (it might be in a different row/col now)
        
        # This check should ideally not be needed if the blocking logic is perfect, 
        # but acts as a safety against infinite loops in unusual edge cases.
        if not allocated_in_cycle and np.sum(s_cp) > 0:
             # If we couldn't allocate anything despite remaining supply/demand, 
             # something is fundamentally wrong (e.g., all remaining costs were blocked).
             logger.warning("Warning: Allocation loop stuck. Check data.")
             break
        
    return allocation_matrix

# Example usage within the main script structure:

# ... (rest of the script follows: sum_check, feasibility_cost) ...
def feasibility_cost(allocation_matrix: np.ndarray, arrays: ComputationalData) -> Tuple[bool, int]:
    """
    Checks for non-degenerate Basic Feasible Solution (BFS) and computes the total cost.

    A BFS is non-degenerate if the number of basic (allocated) variables 
    is equal to m + n - 1, where m is the number of rows (supply sources) 
    and n is the number of columns (demand destinations).

    Args:
        allocation_matrix (np.ndarray): The matrix of decision variables (X_ij).
        cost_matrix (np.ndarray): The matrix of unit costs (C_ij).

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
    zrs_alloc_array = alloc_luc(arrays)

    logger.info("\n### Allocation Results ###")
    logger.info("Alloc matrix (Decision Variables) with LUCM:\n%s", zrs_alloc_array)

    logger.info("\n### Final Cost and Feasibility Check ###")

    # Check total quantity matches
    sum_z = sum_check(zrs_alloc_array)
    logger.info("Sum of allocated quantities checks the total supply/demand:%s", sum_z == sum(arrays.supply_array))

    try:
        is_feasible, total_cost = feasibility_cost(zrs_alloc_array, arrays)

        logger.info("Basic solution is feasible:%s", is_feasible)
        logger.info("Least Unit Cost Method total allocation cost:%s", total_cost)

    except ValueError as exc:
        # Gracefully handle the degeneracy error raised by the function
        logger.error("\n🛑 Error: %s", exc)


if __name__ == "__main__":
    main()
