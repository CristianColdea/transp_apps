"""
Script to determine a Basic Feasible Solution (BFS) with Least Unit Cost
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

d_lst = []
for supply in d.replace("(","").replace(")","").split(","):
    d_lst.append(int(supply))

cstock = []
for r in c.split(", "):
    r = r.replace("(","").replace(")","")
    for cost in r.split(", "):
        cstock.append(int(cost))


c_lst = [cstock[i:i+len(d_lst)] for i in range(0, len(cstock), len(d_lst))]

sum_s = sum(s_lst)
sum_d = sum(d_lst)

# the working data

#s_lst = [20, 30, 50]  # supply
#d_lst = [15, 37, 23, 25]  # demand
#c_lst = [[7, 6, 4, 3], [9, 5, 2, 6], [4, 8, 5, 3]]  # cost matrix

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

assertions(c_lst, s_lst, d_lst)

# create matrices of zeros for decision variables
zrs = [ [0] * len(c_lst[0]) for _ in range(len(c_lst))]

# numpy lists conversions to arrays
c_array = np.array(c_lst)
s_array = np.array(s_lst)
d_array = np.array(d_lst)
zrs_array = np.array(zrs)

# the core code of the script
def allocLUC(s_array: np.ndarray, d_array: np.ndarray,
            c_array: np.ndarray, zrs_array: np.ndarray) -> np.ndarray:
    """
    Function to determine a BFS with Least Unit Cost Method.
    Takes as inputs the supply, demand and cost unit arrays and the matrix
    of zeros with the same shape as the cost matrix.
    Returns the modified matrix of zeros with certain zeros replaced by the
    decision variables (positive integers) in proper positions.
    """
    c_array_cp = c_array.copy()
    vmax = np.max(c_array_cp)
    
    while(sum(s_array) != 0):
        imin = np.argwhere(c_array_cp == np.min(c_array_cp))
        i = sum(c_array_cp.shape)
        j = sum(c_array_cp.shape)
        flg = False

        for r in imin:
            i = r[0]
            j = r[1]
            if(s_array[i] >= d_array[j]):
                flg = True
                if(s_array[i] != 0 and d_array[j] != 0):
                    zrs_array[i, j] = min(s_array[i], d_array[j])
                    s_array[i] = s_array[i] - zrs_array[i, j]
                    d_array[j] = d_array[j] - zrs_array[i, j]
                    if(s_array[i] == 0):
                       c_array_cp[i] = vmax + 1
                    if(d_array[j] == 0):
                        c_array_cp[:, j] = vmax + 1
                    
                    break
          
               
        imin = np.argwhere(c_array_cp == np.min(c_array_cp))

        if flg == True:
            continue
         
        # if supply less than demand
        i0 = imin[0, 0]
        j0 = imin[0, 1]
        zrs_array[i0, j0] = min(s_array[i0], d_array[j0])
        s_array[i0] = s_array[i0] - zrs_array[i0, j0]
        d_array[j0] = d_array[j0] - zrs_array[i0, j0]
        
        if(s_array[i0] == 0):
            c_array_cp[i0] = vmax + 1
        if(d_array[j0] == 0):
            c_array_cp[:, j0] = vmax + 1

        imin = np.argwhere(c_array_cp == np.min(c_array_cp))
                
    return zrs_array

zrs_alloc_array = allocLUC(s_array, d_array, c_array, zrs_array)

print("Alloc matrix one with LUCM, ") 
print(zrs_alloc_array)

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

sum_z = sum_check(zrs_alloc_array)

print("Sum of decision variables checks the sum of supply & demand, ", sum_z
      == sum(s_lst))

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
        print("The basic solution is degenerate. Exiting script!")
        exit()


print("Basic solution is feasible, ",
      feasibility_cost(zrs_alloc_array, c_array, s_array, d_array)[0])
print("Least Unit Cost Method total allocation cost, ",
      feasibility_cost(zrs_alloc_array, c_array, s_array, d_array)[1])
