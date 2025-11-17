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
sum_s = sum(s)
sum_d = sum(d)

assert len(c) == len(s)   #rows must be = to number of supply sources
assert len(c[0]) == len(d)    #cols must be = to number of demands
assert sum(s) == sum(d)    #sum of supply must be = to sum of demand

# create a matrix of zeros for decision variables
zrs = [ [0] * len(c[0]) for _ in range(len(c))]
print("Zeroed 2D list, ",zrs)

# the core code of the script
for ss in range(len(s)):
    if s[ss] != 0:
        for ds in range(len(d)):
            if d[ds] != 0:
                zrs[ss][ds] = min(s[ss], d[ds])
                s[ss] = s[ss] - zrs[ss][ds]    #update supply after alloc
                d[ds] = d[ds] - zrs[ss][ds]    #update demand after alloc

print("Alloc matrix with NWCM, ", zrs)

sumz = 0
for row in zrs:
    sumz = sumz + sum(row)

print("Sum of decision variables checks the sum of supply & demand, ", sumz == sum_s,",", sumz == sum_d)


