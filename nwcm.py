"""
Script to determine a Basic Feasible Solution (BFS) with North_West Corner
allocation method.
It takes as inputs the cost matrix, the supply and demand lists. The length of
supply must match the number of cost matrix rows and that of demand the
columns, respectively.
Returns the allocation table, whether the feasible solution is possible, i.e.,
bool, and the allocation total cost.
"""

import numpy as np

# Enter here the cost matrix
# and supply and demand quantities

#+++++ START DATA SECTION +++++
#+++++ CASE 1 +++++
c1 = [[2, 5, 1],
     [7, 3, 2],
     [1, 5, 3]]

s1 = [25, 23, 26]
d1 = [31, 22, 21]
sum_s1 = sum(s1)
sum_d1 = sum(d1)

#+++++ CASE 2 +++++
c2 = [[2, 5, 1],
      [7, 3, 2]]

s2 = [25, 23]
d2 = [11, 22, 15]
sum_s2 = sum(s2)
sum_d2 = sum(d2)

#+++++ CASE 3 +++++
c3 = [[2, 5, 1],
     [7, 3, 2],
     [1, 5, 3]]

s3 = [25, 23, 26]
d3 = [31, 22, 21]
sum_s3 = sum(s3)
sum_d3 = sum(d3)
#+++++ END DATA SECTION +++++

def assertions(c, s, d):
    """
    Function to check the dimensional 'sanity', i.e., compatibility,
    of the cost matrix with supply and demand lists/arrays.
    Takes as input the cost matrix, supply and demand arrays.
    Signals problems and aborts execution.
    """
    assert len(c) == len(s)   #rows must be = to number of supply sources
    assert len(c[0]) == len(d)    #cols must be = to number of demands
    assert sum(s) == sum(d)    #sum of supply must be = to sum of demand

# create a matrix of zeros for decision variables
zrs = [ [0] * len(c[0]) for _ in range(len(c))]
# print("Zeroed 2D list, ",zrs)

# numpy list conversions to arrays
c_array = np.array(c)
s_array = np.array(s)
d_array = np.array(d)
zrs_array = np.array(zrs)

# the core code of the script
def allocNW(s_array, d_array, zrs_array):
    """
    Function to determine a BFS with NW Corner Method.
    Takes as inputs the supply and demand arrays and the matrix of zeros with
    the same shape as the cost matrix.
    Returns the modified matrix of zeros with certain zeros replaced by the
    decision variables (positive integers) in proper positions.
    """
    for s in range(len(s_array)):
        if s_array[s] != 0:
            for d in range(len(d_array)):
                if d_array[d] != 0:
                    zrs_array[s, d] = min(s_array[s], d_array[d])
                    s_array[s] = s_array[s] - zrs_array[s, d]    #update supply after alloc
                    d_array[d] = d_array[d] - zrs_array[s, d]    #update demand after alloc
    return zrs_array

zrs_array = allocNW(s_array, d_array, zrs_array)

print("Alloc matrix with NWCM, ") 
print(zrs_array)

# check the allocated quantities to match total
def sum_check(zrs_array):
    """
    Function to check whether the sum of allocated quantities match the total.
    Takes as input the allocation matrix.
    Returns the sum derived from the allocation matrix.
    """
    sumz = 0
    for indx in range(len(zrs_array)):
        sumz = sumz + sum(zrs_array[indx])
    return sumz

sum_z = sum_check(zrs_array)

print("Sum of decision variables checks the sum of supply & demand, ", sum_z == sum_s)

# check feasibility and compute cost
def feasibility_cost(zrs_array, c_array, s_array, d_array):
    """
    Function to check feasibility and total cost if the basic solution is not a
    degenerate one.
    Takes as inputs the allocation and cost matrices, and supply and demand
    arrays.
    Returns a tuple with a bool if feasibility is checked and the total cost.
    If feasibility is not checked prints a warning and ends execution.
    """
    fn_cost = 0
    dec_variables = 0
    for r_inx in range(len(zrs_array)):
        for c_inx in range(len(zrs_array[r_inx])):
            if zrs_array[r_inx, c_inx] != 0:
                fn_cost = fn_cost + zrs_array[r_inx, c_inx] * c[r_inx][c_inx]
                dec_variables += 1

    fbool = dec_variables == len(s_array) + len(d_array) - 1 
    if fbool:
        return (fbool, fn_cost)
    else:
        print("The basic solution is degenerate. Exiting function!")
        exit()

print("Basic solution is feasible, ", feasibility_cost(zrs_array, c_array,
                                                       s_array, d_array)[0])

print("NW Corner Method total allocation cost, ", feasibility_cost(zrs_array, c_array,
                                                       s_array, d_array)[1])
