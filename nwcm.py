"""
Script to determine a Basic Feasible Solution (BFS) with North_West Corner
allocation method.
It takes as inputs the cost matrix, the supply and demand lists. The length of
supply must match the number of cost matrix rows and that of demand the
columns, respectively.
Returns the allocation table, whether the feasible solution is possible, i.e.,
bool, and the allocation total cost.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Any, NamedTuple
import ast # Required for safe string evaluation
import logging
from math import e

logging.basicConfig(
    level=logging.DEBUG,  # switch to logging.INFO to silence the .debug() calls
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

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

        # Parsing Supply and Demand lists
        if type(ast.literal_eval(s_str.strip())) == tuple:
            s_lst: List[int] = list(ast.literal_eval(s_str.strip()))
        else:
            s_lst = [ast.literal_eval(s_str.strip())]

        if type(ast.literal_eval(d_str.strip())) == tuple:
            d_lst: List[int] = list(ast.literal_eval(d_str.strip()))

        else:
            d_lst = [ast.literal_eval(d_str.strip())]


        # Parsing Cost Matrix (handling multiple rows/tuples)
        c_lst: List[List[int]] = []
        # Normalizing input: replace outer brackets/parentheses, then split by row separator
        c_rows_str = c_str.strip().strip('[]()')
        
        # Determine the row delimiter based on common matrix input styles
        # If it's a list of lists, split by the comma outside of inner lists/tuples
        if any(char in c_rows_str for char in ['[', '(']):
             # If input is like (1, 2), (3, 4) - we split by '), ('
             c_rows_str = c_rows_str.replace('], [', ')|(').replace('), (', ')|(') 
             row_list = c_rows_str.split('|')
        else:
             # Assuming input is comma-separated tuples like: (1, 2, 3), (4, 5, 6)
             # The original example implies splitting by a comma followed by a space. 
             # We will try to parse each row separately.
             row_list = c_rows_str.split('), ') # A simple heuristic for specific format

        # Final list comprehension for parsing each row
        for r_str in row_list:
            # Clean up residual characters and evaluate
            cleaned_r_str = r_str.strip(' )([]')
            # If the string is empty after cleaning, skip (e.g., from extra commas)
            if not cleaned_r_str:
                continue
            
            # Reconstruct the tuple/list structure before evaluation for safety/compatibility
        
            evaluated_row = ast.literal_eval(f"({cleaned_r_str})") 
            c_lst.append(list(evaluated_row))
        
        return c_lst, s_lst, d_lst

    except ValueError as e:
        logger.error(f"\n❌ Error: Failed to parse input. Please check your formatting.")
        logger.error(f"Details: {e}")
        exit(1)
    except SyntaxError:
        logger.error(f"\n❌ Error: Input format is invalid. Ensure you use proper list/tuple syntax (e.g., (10, 15) or [10, 15]).")
        exit(1)


# --- Execution Start ---

# Create the first structure for handling user input
class RawInput(NamedTuple):
    cost: List[List[int]]
    supply: List[int]
    demand: List[int]

# Get the basic user input
user_cost, user_supply, user_demand = get_user_input()

# Bundle the user input into the first structure for future usage
entries = RawInput(cost=user_cost, supply=user_supply, demand=user_demand)

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
    if len(entries.cost[0]) != len(entries.demand):

        raise ValueError(f"Dimensional error: Cost matrix has \
        {len(entries.cost[0])} columns, but Demand list has \
        {len(entries.demand)} elements.")
    if sum(entries.supply) != sum(entries.demand):
        raise ValueError(f"Transportation problem is **unbalanced**: \
                Sum of Supply \
                ({sum(entries.supply)}) != Sum of Demand \
                ({sum(entries.demand)}).")

try:
    assertions(entries)
except ValueError as e:
    logger.error(f"\n🛑 Validation Error: {e}")
    exit(1)


# Create matrices and arrays for working data
# Bundle the computational arrays into a second structure

class ComputationalData(NamedTuple):
    cost_array: np.ndarray
    supply_array: np.ndarray
    demand_array: np.ndarray
    
arrays = ComputationalData(cost_array=np.array(entries.cost),
                            supply_array=np.array(entries.supply),
                            demand_array=np.array(entries.demand))

# the core code of the script
def allocNW(arrays: ComputationalData) -> np.ndarray:
    """
    Function to determine a BFS with NW Corner Method.
    Takes as inputs the supply and demand arrays and the matrix of zeros with
    the same shape as the cost matrix.
    Returns the modified matrix of zeros with certain zeros replaced by the
    decision variables (positive integers) in proper positions.
    """

    # 1. Ensure Copies for Side-Effect-Free Operation
    s_cp = arrays.supply_array.copy()  # Working copy of Supply
    d_cp = arrays.demand_array.copy()  # Working copy of Demand
    
    # 2. Generate the properly shaped zeros array
    allocation_matrix = np.zeros_like(arrays.cost_array, dtype=int) 

    # 3. Make the allocation according to the NW method
    for s in range(len(s_cp)):
        if s_cp[s] != 0:
            for d in range(len(d_cp)):
                if d_cp[d] != 0:
                    allocation_matrix[s, d] = min(s_cp[s], d_cp[d])
                    s_cp[s]-= allocation_matrix[s, d]    #update supply after alloc
                    d_cp[d]-= allocation_matrix[s, d]    #update demand after alloc
    return allocation_matrix

# Example usage within the main script structure:

# The core function call
zrs_alloc_array = allocNW(arrays)

print()
logger.info("### Allocation Results ###")
logger.info("Alloc matrix (Decision Variables) with NWCM:\n%s", zrs_alloc_array) 

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
    total_cost = np.sum(allocation_matrix * arrays.cost_array)
    
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
    # Use NumPy's built-in sum() for a direct, efficient calculation
    return allocation_matrix.sum()

# Final output section uses a try/except block to catch the new ValueError

logger.info("\n### Final Cost and Feasibility Check ###")

# Check total quantity matches
sum_z = sum_check(zrs_alloc_array)
logger.info("Sum of allocated quantities checks the total supply/demand:%s", sum_z == sum(arrays.supply_array))


try:
    # Function call is simplified (fewer arguments needed)
    is_feasible, total_cost = feasibility_cost(zrs_alloc_array, arrays)

    logger.info("Basic solution is feasible:%s", is_feasible)
    logger.info("North West Corner Method total allocation cost:%s", total_cost)

except ValueError as e:
    # Gracefully handle the degeneracy error raised by the function
    logger.error(f"\n🛑 Error: {e}")
