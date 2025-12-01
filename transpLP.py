"""
Script to optimize a transportation plan with Linear Programming method.
It takes as inputs the cost matrix, the supply and demand lists. The length of
supply must match the number of cost matrix rows and that of demand the
columns, respectively.
Returns the optimal transportation plan (i.e. the optimal combination of
decision variables) and the total cost associated.
Requires the PuLP Python library.
"""

from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from typing import List, Tuple, Any
import ast # Required for safe string evaluation
import numpy as np



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
            c_lst.append(list(evaluated_row))
        
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

# Create the problem
problem = LpProblem("Transportation_Problem", LpMinimize)

# Decision variables
routes = [(i, j) for i in range(len(s_lst)) for j in range(len(d_lst))]
x = LpVariable.dicts("Route", routes, lowBound=0)

# Objective function
problem += lpSum(c_lst[i][j] * x[(i, j)] for (i, j) in routes)

# Constraints
for i in range(len(s_lst)):
    problem += lpSum(x[(i, j)] for j in range(len(d_lst))) <= s_lst[i]

for j in range(len(d_lst)):
    problem += lpSum(x[(i, j)] for i in range(len(s_lst))) >= d_lst[j]

# Solve the problem
problem.solve()

# print("Dict is, ",x) 

opt_flat = []
for v in problem.variables():
    opt_flat.append(v.varValue)

opt_matrix = [opt_flat[i:i+len(d_lst)] for i in range(0, len(opt_flat),
                                                     len(d_lst))]

opt_matrix_array = np.array(opt_matrix)

print("Optimal alloc matrix, ")
print(opt_matrix_array)
