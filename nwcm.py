"""
Script to determine a basic feasible solution (BFS) with North_West Corner
allocation method.
It takes as inputs the cost matrix, the supply and demand lists. The length of
supply must match the number of cost matrix rows and that of demand the
columns, respectively.
Returns the allocation table, wether the feasible solution is possible, i.e.,
bool, and the allocation total cost.
"""

# Enter here the cost matrix
# and supply and demand quantities
c = [[2, 5, 1],
     [7, 3, 2],
     [1, 5, 3]]

s = [25, 23, 26]
d = [31, 22, 21]

assert len(c) == len(s)   #rows must be = to number of supply sources
assert len(c[0]) == len(d)    #cols must be = to number of demands
assert sum(s) == sum(d)    #sum of supply must be = to sum of demand


