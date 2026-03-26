"""
Script to optimize the Assignment Problem by implementing the Hungarian
Method/Algorithm. Takes as input the square matrix of assignment costs,
'sources' and 'destinations' being each equal to one. Assignment costs
must be natural numbers, even 0.
Returns the optimal assignment solution found with the Hungarian Method.
"""

from typing import List, Tuple, Any
import ast # Required for safe string evaluation
import numpy as np



# +++++ USER INTERFACE AND INPUT HANDLING SECTION +++++

def get_user_input() -> list[list[int]]:
    """
    Prompts the user for assignment cost matrix.
    Validates and safely converts the string inputs to required Python lists.
    """
    print("""
The following section provides the means of inputting data for the transportation problem.
Required data: **assignment cost matrix (C)**.

Input format example:
- Cost Matrix C (by row): (1, 2, 3), (4, 5, 6), (7, 8, 9) or
  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

The assignment problem must be **balanced**: the matrix of assignment
costs must be a squared one.
""")

    try:
        # Get input strings
        c_str = input("Enter the matrix cost C:\n> ")
       
        # --- Safe Parsing using ast.literal_eval ---
        # This safely evaluates a string containing a Python literal structure (list/tuple).

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
        
        return c_lst

    except ValueError as e:
        print(f"\n❌ Error: Failed to parse input. Please check your formatting.")
        print(f"Details: {e}")
        exit(1)
    except SyntaxError:
        print(f"\n❌ Error: Input format is invalid. Ensure you use proper list/tuple syntax (e.g., (10, 15) or [10, 15]).")
        exit(1)


# --- Execution Start ---

c_lst = get_user_input()
# print(f"c_lst: {c_lst}")

def assertions(c: list[list[int]]) -> None:
    # ... (Your original assertions function)
    # Using the standard type hint List instead of list for compatibility with python versions < 3.9
    
    """
    Function to check the dimensional 'sanity', i.e. assignment cost
    matrix must be a squared one.
    Signals problems and aborts execution.
    """
    if len(c) != len(c[0]):
        raise ValueError(f"Dimensional error: Cost matrix is not squared.")

try:
    assertions(c_lst)
except ValueError as e:
    print(f"\n🛑 Validation Error: {e}")
    exit(1)


# create a matrix from costs list of lists
c_array = np.array(c_lst)
    
# print(f"\nInput data numpy array: \n{c_array}")

# 1. Function to reduce cost matrix on rows or columns
def reduce_matrix(c:np.ndarray) -> np.ndarray:
    """
    Function to reduce the costs on row or columns.
    Takes as input the assignment cost numpy matrix.
    Returns reduced matrix on rows.
    """

    # make copy to combat unwanted side effects
    c_cp = c.copy()
    # check if passed arg is a square matrix
    if c_cp.shape[0] != c_cp.shape[1]:
        print(f"\n🛑Matrix Shape Error: {c_cp.shape}")
        exit(1)

    return (c_cp - np.min(c_cp, axis=1, keepdims=True))

#2. Function to efficiently cross out zeros in reduced costs matrix
def cross_out_nulls(c_red:np.ndarray) -> list[int]:
    """
    Takes as input the reduced costs matrix.
    Counts the zeros on each row.
    Returns a list with number of zeros on each row.
    """

    # make copy to combat unwanted side effects
    c_red_cp = c_red.copy()
    nulls = []
    for row in c_red_cp:
        nulls.append(np.count_nonzero(row == 0))

    return nulls


#3. Finding the correct sequences of zeros (one zero/row-col) 
def assign_opt(c_red:np.ndarray) -> list[list[tuple]]:
    """
    Finds all sequences of zeros in the reduced costs matrix such
    there is exactly one zero in each row and each column.
    Returns a list with found sequences.
    """

    # make copy to combat unwanted side effects
    c_red_cp = c_red.copy()
 
    n = c_red_cp.shape[0]
    
    # Pre-compute the column indices of zeros for each row to speed up lookup
    zero_positions = [np.where(c_red_cp[r] == 0)[0] for r in range(n)]
    
    all_sequences = []

    def backtrack(row, used_cols, current_seq):
        if row == n:
            all_sequences.append(list(current_seq))
            return

        for col in zero_positions[row]:
            if col not in used_cols:
                # Explore this branch
                used_cols.add(col)
                current_seq.append((row, col))
                
                backtrack(row + 1, used_cols, current_seq)
                
                # Backtrack
                current_seq.pop()
                used_cols.remove(col)

    backtrack(0, set(), [])
    return all_sequences

#4. Optimization function
def optimize(c_nulls:np.ndarray) -> None:
    """
    Optimizes the reduced costs matrix with nout enough zeros.
    Returns the optimized assignment
    """

    # make a copy to combat unwanted effects
    c_nulls_cp = c_nulls.copy()

    return None

def best_zeros(seq:list[tuple], c_array: np.ndarray) -> tuple:
    """
    Gets the assignment solution and its associated total cost.
    Takes as input the assignment sequence and the initial cost
    matrix.
    Returnes a tuple with total cost and the numpy array of assignment.
    """

    # make copies to combat unwanted side effects
    seq_cp = seq.copy()
    c_cp = c_array.copy()

    # create a matrix with zeros for assignment

    assignment_matrix = np.zeros_like(c_array, dtype=int)

    for tpl in seq_cp:
        assignment_matrix[tpl[0], tpl[1]] = c_cp[tpl[0], tpl[1]]

    assignment_cost = np.sum(assignment_matrix)

    return (assignment_cost, assignment_matrix)

# print(best_zeros([(0,3), (1,2), (2,0), (3,1)], c_array)[1])

#5. Functions call, assignment and total cost
# Testing the assignment on zeros function
c_red = reduce_matrix(reduce_matrix(c_array).T).T

print(f"\nReduced cost matrix: \n{c_red}")

BLOCK_COST = -1
# crossed = 0 #start iterating from zero crossed out rows/cols
not_allocated = True

(1, 3, 4), (1, 2, 5), (3, 2, 6)
(3, 2, 5), (5, 5, 3), (5, 4, 2)
(7, 6, 4, 3), (9, 5, 2, 6), (4, 8, 5, 3), (6, 2, 5, 8)

while (not_allocated):
    c_work = c_red.copy() #get a working copy of reduced costs array
    # print(f"c_copy: \n{c_red.copy()}")
    crossed = 0
    
    # 5.1. Cross out zeros 'efficiently'
    while(np.count_nonzero(c_work == 0) != 0): #there are still zeros ...
        nulls_on_rows = cross_out_nulls(c_work)  #check the nulls on rows
        nulls_on_cols = cross_out_nulls(c_work.T) #check the nulls on cols

        crossed_rows = [] #store the indexes of crossed out rows
        crossed_cols = [] #store the indexes of crossed out cols
        
        if (max(nulls_on_rows) >= max(nulls_on_cols)): #more nulls on rows ...
            to_cross_out = nulls_on_rows.index(max(nulls_on_rows))
            c_work[to_cross_out] = BLOCK_COST #replace crossed outs
            crossed_rows.append(nulls_on_rows)
            crossed += 1 #count crossed outs
        else: #more nulls on cols
            to_cross_out = nulls_on_cols.index(max(nulls_on_cols))
            c_work.T[to_cross_out] = BLOCK_COST #replace crossed outs
            crossed_cols.append(nulls_on_cols)
            crossed += 1 #count crossed outs

        print(f"\nAfter: {crossed} cross out: \n{c_work}")
        print(f"crossed: {crossed}")

        print(f"\nCrossed out rows: {crossed_rows}")
        print(f"\nCrossed out cols: {crossed_cols}")


    if crossed == len(c_red): #optimum solution is possible
        # 5.2. Allocate on zeros in c_red array
        zero_seqs = assign_opt(c_red)
        print(f"zero_seqs: {zero_seqs}")
        possible_assignments = {}
        for seq in zero_seqs:
            print(f"sequence: {seq}")
            possible_assignments[best_zeros(seq, c_array)[0]] = \
                                 best_zeros(seq, c_array)[1]

        delivered_assignment = possible_assignments[min(possible_assignments)]

        print(f"\nDelivered assignment solution: \n{delivered_assignment}")
        print(f"\nTotal cost of assignment: {max(possible_assignments)}")

        not_allocated = False
    else:
        # 5.3. Get the indexes of intersection of crossed out rows/cols
        intersections = []
        for indxr in crossed_rows:
            for indxc in crossed_cols:
                instersection.append((indxr, indxc))
        print(f"\nIntersections: {intersections}")
        not_allocated = False   
