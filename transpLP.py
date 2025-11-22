"""
This is a first version of a script to solve a transportation plan for minimum
using Linear Programing, via pulp library.
"""



from pulp import LpProblem, LpMinimize, LpVariable, lpSum

# Define supply and demand
supply = [20, 30, 50]  # Supply at factories
demand = [15, 37, 23, 25]  # Demand at distribution centers
costs = [[7, 6, 4, 3], [9, 5, 2, 6], [4, 8, 5, 3]]  # Cost matrix

# Create the problem
problem = LpProblem("Transportation_Problem", LpMinimize)

# Decision variables
routes = [(i, j) for i in range(len(supply)) for j in range(len(demand))]
x = LpVariable.dicts("Route", routes, lowBound=0)

# Objective function
problem += lpSum(costs[i][j] * x[(i, j)] for (i, j) in routes)

# Constraints
for i in range(len(supply)):
    problem += lpSum(x[(i, j)] for j in range(len(demand))) <= supply[i]

for j in range(len(demand)):
    problem += lpSum(x[(i, j)] for i in range(len(supply))) >= demand[j]

# Solve the problem
problem.solve()

# print("Dict is, ",x) 

for v in problem.variables():
    print(v.name, "=", v.varValue)

