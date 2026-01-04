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


def allocPREF(s_array: np.ndarray, d_array: np.ndarray,
                  c_array: np.ndarray, min_indices: np.ndarray) -> Tuple:
    """
    Determines the preferred allocation if there are least unit cost Tie.
    
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
    #print("s_cp, ", s_cp)
    #print("d_cp, ", d_cp)
    #print("c_cp, ", c_cp)
    #print("min_indices_cp, ", min_indices_cp)
    
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

def selectDIFF(uc_array: np.ndarray) -> int:
    """
    Analyzes the Unit Cost array (i.e., row or column) passed as arg.

    Returns the difference between the two Least Positive Unit Costs.
    """

    #1. Ensure array copy for unwanted side effects
    uc_cp = uc_array.copy()

    #2. Extract the difference between two of the Least Unit Costs
    diffs = [luc for luc in np.sort(uc_cp) if luc > 0]
    if len(diffs) > 1: #there are at least two Positive Least Unit Costs
        diff = diffs[1] - diffs[0]
    else: # only one Positive Unit Cost
        diff = -1

    return diff

def getUCMIN(*arr_lst):
    """
    Selects the minimum Unit Cost (minUC) out of one or many np array
    (i.e., rows or cols of max delta).

    Returns the index and the value of minUC.
    """

    items = arr_lst[0]  #select the first element of *args tuple
    print(f"items {items}")
    print(f"items type {type(items)}")
    arr_inds = []
    if type(items) == list:  #if a list of arrays is passed as arg
        minARR = np.min(items, axis=0) #get the min val array across arrays
        print(f"minARR is {minARR}")
        minUC = np.min(minARR) #get the min val out of minARR
        print(f"minUC is {minUC}")
        for i in range(len(items)):
            print("items[", i, "] is ", items[i])
            ind = [ind for ind in range(len(items[i])) if items[i][ind] == minUC]
            for local_ind in ind:
                arr_inds.append((i, local_ind))
    else:   #or just a single array is passed as arg
        minUC = np.min(items)
        print("minUC, ", minUC)
        local_ind = np.argwhere(items == minUC)
        print(f"local_ind type {type(local_ind[0])}")
        arr_inds.append(local_ind)

    return (arr_inds, minUC)

a = np.array([3, 1, 4])
b = np.array([2, 4, 5])
lst = []
lst.append(a)
lst.append(b)
d = np.array([2, 1, 3])
#print(getUCMIN(lst))
print(getUCMIN(d))

def allocVAM(s_array: np.ndarray, d_array: np.ndarray,
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
    while np.sum(s_cp) > 0:
        
        # 2. Call the speciliased function to extract the difference between
        #    the two least unit cost on rows and columns of the Unit Cost
        #    Matrix (UCM). Store the least unit cost on row/column pair indexes
        #    (as key) and difference (as value) in dicts, on rows/cols.
        #    The two dicts are necessary due to two perspectives regarding
        #    deltas (i.e., on rows and cols), being possible to have the same
        #    pair of indices (i, j) (the dict keys) where the Least Cost Unit
        #    is located, on rows and cols.

        ddiffsR = {}    #dict to store {r: diff} on rows
        for r in range(len(c_cp)):    #iterate over rows of UCM
            print("row, ", c_cp[r])
            diff = selectDIFF(c_cp[r])
            print("diffR, ", diff)
            ddiffsR[r] = diff
        print("Diffs dict after rows, ", ddiffsR)

        ddiffsC = {}    #dict to store {c: diff} on cols
        for c in range(len(c_cp.T)):    #iterate over columns of UCM
            print("col, ", c_cp.T[c])
            diff = selectDIFF(c_cp.T[c])
            print("diffC, ", diff)
            ddiffsC[c] = diff
        print("Diffs dict after cols, ", ddiffsC)
 
        # 3. Handle Ties and Allocation
        
        allocated_in_cycle = False    #safety for while loop ...

        maxRows = max(ddiffsR.values())
        maxesR = [k for k, v in ddiffsR.items() if v == maxRows]
        print("maxesR, ", maxesR)
        print("maxRows, ", maxRows)
       
        maxCols = max(ddiffsC.values())
        maxesC = [k for k, v in ddiffsC.items() if v == maxCols]
        print("maxesC, ", maxesC)
        print("maxCols, ", maxCols)

        # 4. Search for preferred allocs. The differentiation is either on
        #    equal max deltas or equal min unit costs
        
        # 4..a. Select the max delta
        if maxRows >= maxCols: #delta(s) on row are greater ...
            if len(maxesR) > 1: #more max deltas on rows
                maxesALLR = [] #initialize to append to
                for item in maxesR:
                    maxesALLR.append(c_cp[item])
                print("maxesALLR, ", maxesALLR)
                indxR = getUCMIN(maxesALLR)
                print("indxR of tuples, ", indxR)
                (i_pref, j_pref) = allocPREF(s_cp, d_cp, c_cp)
            else: #only one max delta on rows
                i = maxesR[0]
                (indxs, minUC) = getUCMIN(c_cp[i])
                print("indxs, ", indxs, " and minUC, ", minUC)
                j = indxs[0]
                allocation_quantity = min(s_cp[i], d_cp[j])
                allocation_matrix[i, j] = allocation_quantity
                print(f"alloc_quantity {allocation_quantity}")
                print(f"i = {i},", f"j = {j}")
                # update Supply/Demand after allocation
                s_cp[i] -= allocation_quantity
                d_cp[j] -= allocation_quantity
        else: #delta(s) on row are less than those on cols
            if maxRows < maxCols:
                if len(maxesC > 1): #more max deltas on cols
                    inC = [0] #initialize to append to
                    for item in maxesC:
                        maxesALLC = np.append(inC, c_cp.T[item])
                    maxesALLC = np.trim_zeros(maxesALLC)
                    minUCC = np.min(minUCC)


        # 4.a. If only one unit cost is positive on row/col

        



        # 4. Allocate to preferred position, if possible
        if is_preferred == True:
            allocation_quantity = d_cp[j_pref]
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

            continue # resume while loop
        
        
        # 5. Allocate normaly ...
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

# Allocation function call
zrs_alloc_array = allocVAM(s_array, d_array, c_array)

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
