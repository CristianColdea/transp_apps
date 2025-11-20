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
from typing import Any

#+++++ USER INTERFACE SECTION +++++
"""
print(The following section provides the user with the mean of inputin data.
      The required data consists of cost matrix and the supply and demand
      lists. The matrix of cost is to be inputed by row in the form of (a1, ..., an),
      (b1, ..., bn), ..., (z1, ..., zn), and the supply list must match the
      number of cost matrix rows in the form of (s1, ..., sz). The demand list
      must match the number of cost matrix columns in the form (d1, ..., dn).\n
      A complete example data of cost, supply and demand: cost (1, 2, 3),
      (4, 5, 6), supply (10, 15), demand (8, 8, 9). The transportation plan
      must be balanced, i.e., sum of supplies = sum of demands.)
"""

#c = input("Enter here the matrix cost:\n> ")
#s = input("Enter here the list of supply:\n> ")
#d = input("Enter here the list of demand:\n> ")

c = "(2, 5, 1), (7, 3, 2), (1, 5, 3)"
s = "(25, 23, 26)"
d = "(31, 22, 21)"
print(c, s, d)

cstock = []
for r in c.strip(", "):
    r = r.replace("(","").replace(")","")
    for cost in r.strip(", "):
        cstock.append(int(cost))

s_lst = []
s = s.replace("(","").replace(")","")
print("s, ", s)
print("striped, ", s.strip(", "))
#print("len(s), ", len(s))
for supply in s.strip(","):
    print("supply, ", supply)
    #print(int(supply))
    #s_lst.append(int(supply))

print("s_lst, ", s_lst)

c_lst = [cstock[i:i+3] for i in range(0, len(cstock), 3)]
print("cstock, ", cstock)
print("c_lst, ", c_lst)

"""
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
c3 = [[2, 5, 1, 7, 3, 2]]

s3 =  [148]
d3 = [25, 23, 26, 31, 22, 21]

sum_s3 = sum(s3)
sum_d3 = sum(d3)

#+++++ CASE 4 +++++
c4 = [[2],
      [5],
      [1],
      [7],
      [3],
      [2]]

s4 = [25, 23, 26, 31, 22, 21]
d4 = [148]
sum_s4 = sum(s4)
sum_d4 = sum(d4)
#+++++ END DATA SECTION +++++
"""

def assertions(c: list, s: list[int], d: list[int]) -> None:
    """
    Function to check the dimensional 'sanity', i.e., compatibility,
    of the cost matrix with supply and demand lists/arrays.
    Takes as input the cost matrix, supply and demand arrays.
    Signals problems and aborts execution.
    """
    assert len(c) == len(s)   #rows must be = to number of supply sources
    assert len(c[0]) == len(d)    #cols must be = to number of demands
    assert sum(s) == sum(d)    #sum of supply must be = to sum of demand

# create matrices of zeros for decision variables
zrs1 = [ [0] * len(c1[0]) for _ in range(len(c1))]
zrs2 = [ [0] * len(c2[0]) for _ in range(len(c2))]
zrs3 = [ [0] * len(c3[0]) for _ in range(len(c3))]
zrs4 = [ [0] * len(c4[0]) for _ in range(len(c4))]

# numpy lists conversions to arrays
c1_array = np.array(c1)
s1_array = np.array(s1)
d1_array = np.array(d1)
zrs1_array = np.array(zrs1)

c2_array = np.array(c2)
s2_array = np.array(s2)
d2_array = np.array(d2)
zrs2_array = np.array(zrs2)

c3_array = np.array(c3)
s3_array = np.array(s3)
d3_array = np.array(d3)
zrs3_array = np.array(zrs3)

c4_array = np.array(c4)
s4_array = np.array(s4)
d4_array = np.array(d4)
zrs4_array = np.array(zrs4)

"""
print("Entry dataset, CASE 1 (costs, supply, demand):")
print(c1_array)
print(s1_array)
print(d1_array)
print()

print("Entry dataset, CASE 2 (costs, supply, demand):")
print(c2_array)
print(s2_array)
print(d2_array)
print()

print("Entry dataset, CASE 3 (costs, supply, demand):")
print(c3_array)
print(s3_array)
print(d3_array)
print()

print("Entry dataset, CASE 4 (costs, supply, demand):")
print(c4_array)
print(s4_array)
print(d4_array)

print("++++++++++"'\n')
"""

# the core code of the script
def allocNW(s_array: np.ndarray, d_array: np.ndarray,
            zrs_array: np.ndarray) -> np.ndarray:
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

zrs1_alloc_array = allocNW(s1_array, d1_array, zrs1_array)
zrs2_alloc_array = allocNW(s2_array, d2_array, zrs2_array)
zrs3_alloc_array = allocNW(s3_array, d3_array, zrs3_array)
zrs4_alloc_array = allocNW(s4_array, d4_array, zrs4_array)

"""
print("Alloc matrix one with NWCM, ") 
print(zrs1_alloc_array)

print("Alloc matrix two with NWCM, ") 
print(zrs2_alloc_array)

print("Alloc matrix three with NWCM, ") 
print(zrs3_alloc_array)

print("Alloc matrix four with NWCM, ") 
print(zrs4_alloc_array)

print("++++++++++"'\n')
"""

# check the allocated quantities to match total
def sum_check(zrs_array: np.ndarray) -> int:
    """
    Function to check whether the sum of allocated quantities match the total.
    Takes as input the allocation matrix.
    Returns the sum derived from the allocation matrix.
    """
    sumz = 0
    for indx in range(len(zrs_array)):
        sumz = sumz + sum(zrs_array[indx])
    return sumz

sum_z1 = sum_check(zrs1_alloc_array)
sum_z2 = sum_check(zrs2_alloc_array)
sum_z3 = sum_check(zrs3_alloc_array)
sum_z4 = sum_check(zrs4_alloc_array)

"""
print("Sum of decision variables checks the sum of supply & demand matrix one, ", sum_z1
      == sum_s1)
print("Sum of decision variables checks the sum of supply & demand matrix two, ", sum_z2
      == sum_s2)
print("Sum of decision variables checks the sum of supply & demand matrix three, ", sum_z3
      == sum_s3)
print("Sum of decision variables checks the sum of supply & demand matrix four, ", sum_z4
      == sum_s4)

print("++++++++++"'\n')
"""

# check feasibility and compute cost
def feasibility_cost(zrs_array: np.ndarray, c_array: np.ndarray,
                     s_array: np.ndarray,
                     d_array: np.ndarray) -> tuple[bool, int]|str:
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
                fn_cost = fn_cost + zrs_array[r_inx, c_inx] * c_array[r_inx, c_inx]
                dec_variables += 1

    fbool = dec_variables == len(s_array) + len(d_array) - 1 
    if fbool:
        return (fbool, fn_cost)
    else:
        print("The basic solution is degenerate. Exiting function!")
        exit()

"""
print("Basic solution is feasible (one), ",
      feasibility_cost(zrs1_alloc_array, c1_array, s1_array, d1_array)[0])
print("NW Corner Method total allocation cost (one), ",
      feasibility_cost(zrs1_alloc_array, c1_array, s1_array, d1_array)[1])

print("Basic solution is feasible (two), ",
      feasibility_cost(zrs2_alloc_array, c2_array, s2_array, d2_array)[0])
print("NW Corner Method total allocation cost (two), ",
      feasibility_cost(zrs2_alloc_array, c2_array, s2_array, d2_array)[1])

print("Basic solution is feasible (three), ",
      feasibility_cost(zrs3_alloc_array, c3_array, s3_array, d3_array)[0])
print("NW Corner Method total allocation cost (three), ",
      feasibility_cost(zrs3_alloc_array, c3_array, s3_array, d3_array)[1])

print("Basic solution is feasible (four), ",
      feasibility_cost(zrs4_alloc_array, c4_array, s4_array, d4_array)[0])
print("NW Corner Method total allocation cost (four), ",
      feasibility_cost(zrs4_alloc_array, c4_array, s4_array, d4_array)[1])
"""
