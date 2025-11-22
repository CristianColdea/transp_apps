"""
This is a first version of a script to solve a transportation plan for minimum
using Linear Programing, via pulp library.
"""



from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from typing import Any
import numpy as np

#+++++ USER INTERFACE SECTION +++++

print("""The following section provides the user with the mean of inputing data.
The required data consists of cost matrix and the supply and demand
lists. The matrix of cost is to be inputed by row in the form of (a1, ..., an),
(b1, ..., bn), ..., (z1, ..., zn), and the supply list must match the
number of cost matrix rows in the form of (s1, ..., sz). The demand list
must match the number of cost matrix columns in the form (d1, ..., dn).\n
A complete example data of cost, supply and demand: cost (1, 2, 3),
(4, 5, 6), supply (10, 15), demand (8, 8, 9). The transportation plan
must be balanced, i.e., sum of supplies = sum of demands.\n
Here goes the input data part.\n""")

c = input("Enter here the matrix cost:\n> ")
s = input("Enter here the list of supply:\n> ")
d = input("Enter here the list of demand:\n> ")

s_lst = []
for supply in s.replace("(","").replace(")","").split(","):
    s_lst.append(int(supply))

#print("s_lst, ", s_lst)

d_lst = []
for supply in d.replace("(","").replace(")","").split(","):
    d_lst.append(int(supply))

#print("d_lst, ", d_lst)

cstock = []
for r in c.split(", "):
    r = r.replace("(","").replace(")","")
    for cost in r.split(", "):
        cstock.append(int(cost))


c_lst = [cstock[i:i+len(d_lst)] for i in range(0, len(cstock), len(d_lst))]

print("cstock, ", cstock)
print("c_lst, ", c_lst)

# Define supply and demand
#supply = [20, 30, 50]  # Supply at factories
#demand = [15, 37, 23, 25]  # Demand at distribution centers
#costs = [[7, 6, 4, 3], [9, 5, 2, 6], [4, 8, 5, 3]]  # Cost matrix

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

# print("Flat list, ", opt_flat)

opt_matrix = [opt_flat[i:i+len(d_lst)] for i in range(0, len(opt_flat),
                                                     len(d_lst))]

opt_matrix_array = np.array(opt_matrix)

print("Optimal alloc matrix, ")
print(opt_matrix_array)
