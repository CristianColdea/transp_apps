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

def get_user_input() -> tuple[List[List[int]]]:
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

def assertions(c: List[List[int]]) -> None:
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
    
print(f"Input data numpy array: {c_array}")

# 1. Function to reduce cost matrix on rows or columns
def reduce_matrix(c:np.ndarray) -> np.ndarray:
    """
    Function to reduce the costs on row or columns.
    Takes as input the assignment cost numpy matrix.
    Returns reduced matrix on rows.
    """
    # check if passed arg is a square matrix
    if c.shape[0] != c.shape[1]:
        print(f"\n🛑 Matrix Shape Error: {e}")
        exit(1)

    return (c - np.min(c, axis=1, keepdims=True))

#2. Function to efficiently cross out zeros in reduced costs matrix
#def cross_out_nulls(c_red:np.ndarray) -> Tuple(bool, int)

#3. Functions call, assignment and total cost
c_red = reduce_matrix(reduce_matrix(c_array).T).T

print(f"\n Reduced cost matrix: {c_red}")
