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
a computational format, i.e., numpy arrays, bundle. The general form of bundles
is:

class RawInput(NamedTuple):
    cost: List[List[int]]
    supply: List[int]
    demand: List[int]



### Principles
* Prefer clarity and readability over cleverness.
* Every public function must have a doctring and type hints.
* Tie-breaking must be deterministic and explicable.
* The orchestrator function is the only entry point.

### Roadmap
Decomposition of the script according to four milestones:
* Code the parser to process input from user.
* Code helper functions for each tie-breaking rule.
* Code the orchestrator function within which is the allocation algorithm.
* Wrap the execution into 'main'.
* Test the code.
