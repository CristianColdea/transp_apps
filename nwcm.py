"""
Script to determine a basic feasible solution (BFS) with North_West Corner
allocation method.
It takes as inputs the cost matrix, the supply and demand lists. The length of
supply must match the number of cost matrix rows and that of demand the
columns, respectively.
Returns the allocation table, wether the feasible solution is possible, i.e.,
bool, and the allocation total cost.
"""
