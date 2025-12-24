import numpy as np
from typing import List, Tuple, Any
import ast # Required for safe string evaluation

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
            s_lst: List[int] = [ast.literal_eval(s_str.strip())]

        if type(ast.literal_eval(d_str.strip())) == tuple:
            d_lst: List[int] = list(ast.literal_eval(d_str.strip()))

        else:
            d_lst: List[int] = [ast.literal_eval(d_str.strip())]

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
            if type(evaluated_row) == tuple:
                c_lst.append(list(evaluated_row))
            else:
                c_lst.append([evaluated_row])
        
        return c_lst, s_lst, d_lst

    except ValueError as e:
        print(f"\n❌ Error: Failed to parse input. Please check your formatting.")
        print(f"Details: {e}")
        exit(1)
    except SyntaxError:
        print(f"\n❌ Error: Input format is invalid. Ensure you use proper list/tuple syntax (e.g., (10, 15) or [10, 15]).")
        exit(1)


# --- Execution Start ---

c_lst, s_lst, d_lst = get_user_input()

sum_s = sum(s_lst)
sum_d = sum(d_lst)

# The assertions function remains the same, but it is called after successful parsing.

def assertions(c: List[List[int]], s: List[int], d: List[int]) -> None:
    # Using the standard type hint List instead of list for compatibility with python versions < 3.9
    
    """
    Function to check the dimensional 'sanity', i.e., compatibility,
    of the cost matrix with supply and demand lists/arrays.
    Takes as input the cost matrix, supply and demand arrays.
    Signals problems and aborts execution.
    """
    if len(c) != len(s):
        raise ValueError(f"Dimensional error: Cost matrix has {len(c)} rows, but Supply list has {len(s)} elements.")
    if len(c[0]) != len(d):
        raise ValueError(f"Dimensional error: Cost matrix has {len(c[0])} columns, but Demand list has {len(d)} elements.")
    if sum(s) != sum(d):
        raise ValueError(f"Transportation problem is **unbalanced**: Sum of Supply ({sum(s)}) != Sum of Demand ({sum(d)}).")


    try:
        assertions(c_lst, s_lst, d_lst)
    except ValueError as e:
        print(f"\n🛑 Validation Error: {e}")
        exit(1)


# create matrices and arrays for working data
c_array = np.array(c_lst)
s_array = np.array(s_lst)
d_array = np.array(d_lst)

# The rest of the script (allocLUC, etc.) follows here...

"""
If the search operation for the least unit cost yields a Tie, there is a 
preferred allocation according to the following two criteria (in this order):

1) allocate to the least cost unit that allows to move the higher quantity,
2) allocate where the supply exceeds demand.
For this purpose a specialized function is to be coded.
"""

def allocPREF(s_array: np.ndarray, d_array: np.ndarray,
              c_array: np.ndarray, min_indices: np.array) -> Tuple:
    """
    Determines the preferred allocation if there is Least Unit Cost Tie.
    
    Takes as inputs the supply, demand, unit cost and min_indices arrays.

    Returns a tuple with indexes (i_pref, j_pref) of preferred allocation.

    Raises value error if the array of minimum indices doesn't have at least
    two pair of values.
    """

    # 1. Ensure Copies for Side-Effect-Free Operation
    s_cp = s_array.copy()
    d_cp = d_array.copy()
    c_cp = c_array.copy()
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
        # print(f"min_indsQmax for only one alloc possible: {min_indsQmax}.")
        return min_indsQmax[0]
    # 5. More than one max tonnage, check if S > D
    else:
        for i,j in min_indsQmax:
            if s_cp[i] >= d_cp[j]:
                # print(f"More Qmax, S > D: {i,j}.")
                return(i,j)
        # print(f"More Qmax, S == D: {min_indsQmax[0]}.")
        return min_indsQmax[0]
    
    
    # 5. Raises error if least unit cost indices not a Tie
    if min_indices.shape[0] < 2:
        raise ValueError("Min_indices of least unit costs doesn't yield a Tie."
                f"There are vaues: {min_indices.shape[0]}")

    try:
        allocPREF(s_array, d_array, c_array, min_indices)

    except ValueError as e:
        print(f"\n🛑 Validation Error: {e}")
        exit(1)


def allocLUC(s_array: np.ndarray, d_array: np.ndarray,
            c_array: np.ndarray) -> np.ndarray:
    """
    Determines a Basic Feasible Solution (BFS) using the Least Unit Cost Method.

    Takes as inputs the supply, demand, and unit cost arrays.
    Returns the allocation matrix (decision variables).
    """
    # 1. Ensure Copies for Side-Effect-Free Operation
    s_cp = s_array.copy()  # Working copy of Supply
    d_cp = d_array.copy()  # Working copy of Demand
    c_cp = c_array.copy()  # Working copy of Cost matrix (to 'inf' out satisfied rows/cols)
    
    # Initialize the Allocation matrix (X_ij)
    # Using np.zeros_like is cleaner and more NumPy idiomatic
    allocation_matrix = np.zeros_like(c_array, dtype=int)
    
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
        #    by calling allocPREF() function
        if min_indices.shape[0] != 1:
            (i_pref, j_pref) = allocPREF(s_array, d_array, c_array, min_indices)
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
             # For a professional script, you might raise an error here.
             print("Warning: Allocation loop stuck. Check data.")
             break
        
    return allocation_matrix

# Example usage within the main script structure:

# ... (data input and assertions section from previous answer) ...


# The core function call is simplified
zrs_alloc_array = allocLUC(s_array, d_array, c_array)

print("\n### Allocation Results ###")
print("Alloc matrix (Decision Variables) with LUCM:") 
print(zrs_alloc_array)

# ... (rest of the script follows: sum_check, feasibility_cost) ...

def feasibility_cost(allocation_matrix: np.ndarray, cost_matrix: np.ndarray) -> Tuple[bool, int]:
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
    total_cost = np.sum(allocation_matrix * cost_matrix)
    
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

# ... (Previous code: c_array, s_array, d_array, zrs_alloc_array defined) ...

# Final output section uses a try/except block to catch the new ValueError

print("\n### Final Cost and Feasibility Check ###")

# Check total quantity matches
sum_z = sum_check(zrs_alloc_array)
print("Sum of allocated quantities checks the total supply/demand:", sum_z == sum(s_array))


try:
    # Function call is also simplified (fewer arguments needed)
    is_feasible, total_cost = feasibility_cost(zrs_alloc_array, c_array)

    print("Basic solution is feasible:", is_feasible)
    print(f"Least Unit Cost Method total allocation cost: {total_cost:,}")

except ValueError as e:
    # Gracefully handle the degeneracy error raised by the function
    print(f"\n🛑 Error: {e}")
    # You can now choose to exit, log, or prompt the user for other action here.
    # For now, we will just print the error and let the program naturally end.
