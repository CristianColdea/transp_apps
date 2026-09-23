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

2. As assertion/shape check function which ensures the shape of supply and
   demand lists are matching rows and columns of unit cost matrix,
respectively.
3. The tie breaking functions, one for the maximum amount, and the other for
   allocation where the supply is greater or equal to demand; the third tie is
managed within the orchestrator based on the returns from the tie breaking
functions.
4. The orchestrator function which calls the helpers, receives the returns, and
   make the proper allocation.
