"""
Script to determine a Basic Feasible Solution (BFS) with Vogel alloc method.
It takes as inputs the cost matrix, the supply and demand lists. The length of
supply must match the number of cost matrix rows and that of demand the
columns, respectively.
Returns the allocation table, whether the feasible solution is possible, i.e.,
bool, and the allocation total cost.
"""

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
        s_lst: List[int] = list(ast.literal_eval(s_str.strip()))
        d_lst: List[int] = list(ast.literal_eval(d_str.strip()))
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

#c_lst, s_lst, d_lst = get_user_input()
c_lst = [[7, 6, 4, 3], [9, 5, 2, 6], [4, 8, 5, 3]]
s_lst = [20, 30, 50]
d_lst = [15, 37, 23, 25]

sum_s = sum(s_lst)
sum_d = sum(d_lst)

# The assertions function remains the same, but it is called after successful parsing.


def assertions(c: List[List[int]], s: List[int], d: List[int]) -> None:
    # ... (Your original assertions function)
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
# More Pythonic way to create the zero array
# zrs_array = np.zeros_like(c_array, dtype=int) 


def select_diff(uc_array: np.ndarray) -> int:
    """
    Analyzes the Unit Cost array (i.e., row or column) passed as arg.

    Returns the difference between the two Least Positive Unit Costs.
    """

    #1. Ensure unit cost array copy for unwanted side effects
    uc_cp = uc_array.copy()

    # 2. Extract the difference between two of the least unit costs
    # 2.a. Extract true (not zeroed out by previous allocs) unit cost list
    diffs: list[int] = [l_uc for l_uc in np.sort(uc_cp) if l_uc > -1]
    if len(diffs) > 1: #there are at least two positive least unit costs
        diff: int = diffs[1] - diffs[0]
    else: # only one positive unit cost
        diff = -1

    return diff


def get_uc_min(ddiffs: dict, c_cp: np.ndarray) -> tuple:
    """
    Selects the minimum Unit Cost index out of deltas dict.

    Returns the index of the value of minimum unit cost
    on max delta row/col.
    """
    
    import numpy as np

    max_delta: int = max(ddiffs.values())
    # get the index of row/col of max_delta
    max_ind: list[int] = [k for k, v in ddiffs.items() if v == max_delta]
    print('\n')
    print(f"max_delta {max_delta}")
    print(f"max_ind {max_ind}")
    
    store_ind: list[int] = []
    store_uc_min: list[int] = []
    for ind in max_ind:  # for each max_delta row/col
        # get available unit cost list on max_delta row/col
        w_lst: list[int] = [uc for uc in c_cp[ind] if uc > -1]
        print(f"w_lst {w_lst}")
        store_ind.append(ind)
        store_uc_min.append(min(w_lst))  # append the min uc available
    
    print(f"store_ind {store_ind}")
    print(f"store_uc_min {store_uc_min}")
    # Needs attention here!!!
    ind_uc_min: int = store_uc_min.index(min(store_uc_min))
    print(f"ind_uc_min {ind_uc_min}")
    print(f"c_cp[store_ind[ind_uc_min]]: {c_cp[store_ind[ind_uc_min]]}")
    #print(f"c_cp[store_ind[ind_uc_min]]: {type(c_cp[store_ind[ind_uc_min]])}")
    true_uc: list[int] = [x for x in list(c_cp[store_ind[ind_uc_min]]) if x > 0]
    j: int = list(c_cp[store_ind[ind_uc_min]]).index(min(true_uc))
    print(f"j: {j}")
    print('\n')
        
    return (store_ind[ind_uc_min], j, store_uc_min[ind_uc_min])

"""
dctCHK = {0:1, 1:3, 2:0, 3:3}
c_cp = c_array.copy()

print(f"Check returned {get_ucmin(dctCHK, c_cp.T)}")
"""


def detect_false_delta(delta_ind: int, uc_array: np.array) -> tuple:
    """
    Detects wheter a row/col has only one true unit cost (uc).

    Returns the position and value of the only true unit cost
    (if it's the case) as a tuple (i, j, val) or (-2, -2, -2) if there are
    more or less than one true unit costs on row/col.
    """
    
    # 1. Extract a list with negative unit cost
    neg_uc: list[int] = [uc for uc in uc_array if uc == -1]
    print(f"neg_uc: {neg_uc}")
    
    # 2. Compare lenghts of passed array and extracted list
    uc_list: list[int] = list(uc_array)
    uc_ind: int = uc_list.index(max(uc_list))
    if (len(uc_array) - len(neg_uc)) == 1:  # precisely one true uc
        return (delta_ind, uc_ind, max(uc_list))
    else:
        return (-2, -2, -2)
    
def alloc_vam(s_array: np.ndarray, d_array: np.ndarray,
            c_array: np.ndarray) -> np.ndarray:
    """
    Determines a Basic Feasible Solution (BFS) using the Vogel Alloc Method.

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
    
    # A unit cost equal to -1 to effectively block satisfied sources/destinations
    BLOCK_COST = -1 
    
    # Core loop continues until all supply is exhausted (which means demand is also zero,
    # due to the balancing assertion)
    
    t = 0  # set a counter for main loop

    while np.sum(s_cp) > 0:
        
        # 2. Call the speciliazed function to extract the difference between
        #    the least and next-to-the-least unit costs on rows and columns of
        #    the Unit Cost Matrix (UCM).
        #    Store the least unit cost on row/column pair indexes
        #    (as key) and difference (as value) in dicts, on rows/cols.
        #    The two dicts are necessary due to two perspectives regarding
        #    deltas (i.e., on rows and cols), being possible to have the same
        #    pair of indices (i, j) (the dict keys) where the Least Cost Unit
        #    is located, on rows and cols.
 
        ddiffs_r = {}    #dict to store {r: diff} on rows
        for r in range(len(c_cp)):    #iterate over rows of UCM
            #print("row, ", c_cp[r])
            diff = select_diff(c_cp[r])
            #print("diffR, ", diff)
            ddiffs_r[r] = diff
        print("Diffs dict after rows, ", ddiffs_r)

        ddiffs_c = {}    #dict to store {c: diff} on cols
        for c in range(len(c_cp.T)):    #iterate over columns of UCM
            #print("col, ", c_cp.T[c])
            diff = select_diff(c_cp.T[c])
            #print("diffC, ", diff)
            ddiffs_c[c] = diff
        print("Diffs dict after cols, ", ddiffs_c)
 
        # 3. Handle Ties and Allocation.The differentiation is either on
        #    equal max deltas or equal min unit costs
        # 3.a. Get the indexes of min unit cost on max_delta rows/cols
        #      by calling get_uc_min or detect_false_delta func
        uc_r0: int = np.max(c_cp) + 1  # set the uc_r initial value
        is_fake_delta_r: bool = -1 in list(ddiffs_r.values())
        print(f"is_fake_delta_r: {is_fake_delta_r}")
        if is_fake_delta_r:
            for k,v in ddiffs_r.items():
                if v == -1:  # suspect row here ...
                    i_r, j_r, uc_r = detect_false_delta(k, c_cp[k])
                    if uc_r <= uc_r0:
                        uc_r0 = uc_r
                        i = i_r
                        j = j_r
              
                        print(f"i_(-1r): {i}")
                        print(f"j_(-1r): {j}")

        if not is_fake_delta_r:  # no suspect delta row encountered
            i_r,j_r, uc_min_r = get_uc_min(ddiffs_r, c_cp)
        print(f"i_r {i_r}")
        print(f"j_r {j_r}")
        print(f"uc_min_r: {uc_min_r}")
        
        uc_c0: int = np.max(c_cp) + 1  # set the uc_c initial value
        print(f"uc_c0: {uc_c0}")
        is_fake_delta_c: bool = -1 in list(ddiffs_c.values())
        print(f"is_fake_delta_c: {is_fake_delta_c}")
        if is_fake_delta_c:
            for k,v in ddiffs_c.items():
                if v == -1:  # suspect col here ...
                    i_c, j_c, uc_c = detect_false_delta(k, c_cp.T[k])
                    if uc_c <= uc_c0:
                        uc_c0 = uc_c
                        j = i_c
                        i = j_c
              
                        print(f"i_(-1c): {i}")
                        print(f"j_(-1c): {j}")

        if not is_fake_delta_c:  # no suspect delta col encountered
            j_c, i_c, uc_min_c = get_uc_min(ddiffs_c, c_cp.T)
        print(f"i_c {i_c}")
        print(f"j_c {j_c}")
        print(f"uc_min_c: {uc_min_c}")
        
        # 3.b. Select where to allocate based on deltas and min unit cost
        max_delta_row: int = max(ddiffs_r.values())
        max_delta_col: int = max(ddiffs_c.values())

        if not is_fake_delta_r and not is_fake_delta_c:
            # delta on rows is greater than the one on cols 
            if max_delta_row > max_delta_col:
                i: int = i_r
                j: int = j_r
            # delta on cols is greater than the one on rows
            if max_delta_col > max_delta_row:
                i = i_c
                j = j_c
            # deltas are equal
            if max_delta_row == max_delta_col:
                if uc_min_c >= uc_min_r:
                    i = i_r
                    j = j_r
                else:
                    i = i_c
                    j = j_c

        print(f"i: {i}")
        print(f"j: {j}")
           
        allocated_in_cycle = False    #safety for while loop ...
       
        # 4. Allocate 
        # 4.a. Get the alloc quantity
        allocation_quantity = min(s_cp[i], d_cp[j])
        allocation_matrix[i, j] = allocation_quantity
        # Update remaining supply and demand
        s_cp[i] -= allocation_quantity
        d_cp[j] -= allocation_quantity
            
        # --- Block Satisfied Rows/Columns (Setting cost to a high value) ---
            
        # If the supply source 'i' is exhausted, block the entire row
        if s_cp[i] == 0:
            c_cp[i, :] = BLOCK_COST
            
        # If the demand destination 'j_pref' is satisfied, block the entire column
        if d_cp[j] == 0:
            c_cp[:, j] = BLOCK_COST
            
        allocated_in_cycle = True
        
        t += 1
        print(f"t: {t}")
        if t == 4:
            break

        continue # resume while loop      
        
        if not allocated_in_cycle and np.sum(s_cp) > 0:
             # If we couldn't allocate anything despite remaining supply/demand, 
             # something is fundamentally wrong (e.g., all remaining costs were blocked).
             # For a professional script, you might raise an error here.
             print("Warning: Allocation loop stuck. Check data.")
             break
        
    return allocation_matrix

# Allocation function call
zrs_alloc_array = alloc_vam(s_array, d_array, c_array)

print("\n### Allocation Results ###")
print("Alloc matrix (Decision Variables) with NWCM:") 
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
    print(f"North West Corner Method total allocation cost: {total_cost:,}")

except ValueError as e:
    # Gracefully handle the degeneracy error raised by the function
    print(f"\n🛑 Error: {e}")
    # You can now choose to exit, log, or prompt the user for other action here.
    # For now, we will just print the error and let the program naturally end.
