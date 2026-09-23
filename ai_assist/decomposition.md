* Script Architecture and Decomposition Plan

### The Allocation Algorithm
The Least Unit Cost Matrix allocation method (LSCM) is used to generate a Basic
Feasible Solution (BFS) for a transportation plan by allocating quantities on
the least unit cost cell of the unit cost matrix.
There are two possible ties that must be broken in the case of more than one
least cost:
1. The allocation must be made on the maximum amount.
2. If there are two or more equal maximum amounts on the same least unit cost
   cells, the tie must be broken by allocating where supply is greater or equal
to demand; in this way the algorithm is preserved more logical and
anthropomorphic.
3. If in the tie case no 2 there are no supply greater or equal than demand the
   allocation must be made in indexes logic, i.e., left to right and down to
bottom.

### The Script Components
1. A helper function to process the unit cost matrix as user input. It must use
   ast library and accept any nesting style ast.literal\_eval understands,
namely rows as tuples/lists, with/without wrappers, spaces. The return should
be in the form of List[List[int]].
2. A function to parse the user input. ast library is to be used here too.
   try/except must be employed, not exit. The role of the function is to report
a problem, not to stop the execution. Should report 1) the fail to parse input,
and 2) input invalid format. The use of 'isinstance' is desired within this
function.
3. Two data structure to store raw user input and computational formats in the
   form of:

class RawInput(NamedTuple):
    cost: List[List[int]]
    supply: List[int]
    demand: List[int]

class ComputationalData(NamedTuple):
    cost\_array: np.ndarray
    supply\_array: np.ndarray
    demand\_array: np.ndarray
The structure RawInput is to be created after user input proper parsing, before
assertions (see no 4.). After shape inputs shape and matrix integrity check and
pass, ComputationalData is to be created to provide proper args to following
functions.
4. An assertions check function which ensures the shape of supply and
   demand lists are matching rows and columns of unit cost matrix, the cost
matrix isn't a ragged one, and the transportation problem is a balanced one.
The 'assertions' function must operate on RawInput structure. try/except is to
be employed.
5. The tie breaking functions, one for the maximum amount, the other for
   allocation where the supply is greater or equal to demand, and the third
logically normal allocation (see 'constituion.md' for more details).
There is also a possibility to consolidate all the tie-breakings
into one function, too; yet to be seen as the script is built. This function(s)
must return the precise preferrd allocation position when called from the
allocation function (see no 6).
6. The allocation function which makes use of the previously defined functions,
   with proper data formats as args. Allocated cells must have the unit cost
blocked in the form of 'BLOCK\_COST = max(cost\_matrix) + 1'. The allocation
sum must be checked after each allocation against the supply/demand total (no
matter which since the transportation problem is a balanced one, i.e., Sum of
Supply = Sum of Demand. Also a safety mechanism for the unit cost matrix loop
must exists, not decisively necessary, but exists as safety against infinite
looping. This function returns the allocation matrix, i.e., the basic feasible
solution.
7. The feasibility\_cost function, first for feasibilty check of the basic
   solution with the relation 'total allocated cells = m + n -1', where m is
the number of cost unit matrix rows (supplies), and n is the number
of columns of the same matrix (demands), and second for total basic feasible
solution (BFS) transportation cost; using numpy.sum(BFS * cost\_array) is
desired. Here too try/except is to be employed, returning a tuple of
(is\_feasible, BFS\_cost).
8. main() function to manage entry point, with try/excepts where are required.
9. Final __name__ == "__main__" check.

### Tasks allocation list
To address the known limitations of the local models the following
clarification follows:
1. Locally coded - parse\_matrix, RawInput, ComputationData, feasibility\_cost.
2. Frontier model coded - get\_user\_input, assertions, tie\_breaking,
   allocation, main.
