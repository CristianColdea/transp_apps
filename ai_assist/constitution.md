# Constitution

**Project Name:** Least Unit Cost Matrix Allocation with AI code assistance
**Motto:** *"Adding Efficiency To Allocation."*

### Mission
To code The Least Unit Cost Matrix allocation method using solely AI code
assistance.

### Tech Stack and Constraints
* The following libraries are to be used: numpy, typing, ast, logging.
* Imports: annotations.
* The script is meant to interact with the user solely in terminal.
* Output must be reproducible.
* The indicated libraries are the only ones to be used for coding, nothing more.

From typing List, Tuple and NamedTuple is to be imported. Two data structures
are to be created as NamedTuple, first as bundler for raw user input, second as
a computational format, i.e., numpy arrays, bundle. See the 'decomposition.md'
document for more details.

### Principles
* Prefer clarity and readability over cleverness.
* Every public function must have a doctring and type hints.
* Tie-breaking must be deterministic and explicable.
* The allocation/orchestrator function is the only entry point for the algorithm.

### Roadmap
Decomposition of the script according to nine milestones:
1. Code a helper function for solid processing of unit cost matrix input from
  user.
2. Code the parser to process all the input from the user.
3. Code two data structures of NamedTuple type to store input and computational
data.
4. Code a function to check the shape match between cost unit matrix, and supply
and demand lists.
5. Code helper function for the first two tie-breaking rules (see 'decomposition.md'
for more details on those rules).
6. Code the allocation/orchestrator function within which is the allocation algorithm.
7. Code a function to check if the determined basic solution if feasible and, if it is,
compute the transportation solution cost.
8. Testing the script.
9. Wrap the execution into 'main()'.
