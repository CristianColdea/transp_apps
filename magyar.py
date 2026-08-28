"""
Script to optimize the Assignment Problem by implementing the Hungarian
Method/Algorithm. Takes as input the square matrix of assignment costs,
'sources' and 'destinations' being each equal to one. Assignment costs
must be natural numbers, even 0.
Returns the optimal assignment solution found with the Hungarian Method.

Refactored for better architecture:
  - All execution wrapped in main() / ``if __name__ == "__main__"`` guard.
  - All print() calls replaced with structured logging.
  - exit() calls replaced with exceptions.
  - Duplicate best_zeros() call eliminated.
  - Dead bare-literal lines removed.
  - Old-style typing imports replaced with modern built-in generics.
  - Type annotations added throughout (including nested backtrack function).
  - Bug fix: final output now correctly reports the minimum total cost.
"""

from __future__ import annotations

import ast
import logging

import numpy as np

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,  # switch to logging.INFO to silence .debug() calls
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)


# ── User interface and input handling ────────────────────────────────────────

def parse_matrix_literal(raw: str) -> list[list[int]]:
    """
    Parse a cost-matrix string as a single Python literal.

    Accepts any nesting style ast.literal_eval understands -- rows as
    tuples or lists, with or without an outer wrapper, with or without
    spaces -- rather than guessing the structure via string replacement.
    A single bare row (e.g. "(1, 2, 3)") is normalized into a one-row
    matrix so the return shape is always list[list[int]].
    """
    parsed = ast.literal_eval(raw.strip())

    if parsed and not isinstance(parsed[0], (list, tuple)):
        parsed = (parsed,)

    return [list(row) for row in parsed]


def get_user_input() -> list[list[int]]:
    """
    Prompt the user for the assignment cost matrix.

    Validates and safely converts the string input to a Python list of lists.
    Accepts any format ast.literal_eval understands, for example:
      - Tuple rows:  (1, 2, 3), (4, 5, 6), (7, 8, 9)
      - List rows:   [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    Raises
    ------
    ValueError
        On any parse or syntax error.  The caller (main()) decides whether
        to log and exit -- this function does not call exit() itself.
    """
    logger.info(
        "\nThe following section provides the means of inputting data for the"
        " assignment problem.\n"
        "Required data: assignment cost matrix (C).\n\n"
        "Input format example:\n"
        "  - Cost Matrix C (by row): (1, 2, 3), (4, 5, 6), (7, 8, 9)  or\n"
        "    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n\n"
        "The assignment problem must be balanced: the cost matrix must be square."
    )

    try:
        c_str = input("Enter the matrix cost C:\n> ")
        # Parsing delegated to parse_matrix_literal() -- see its docstring --
        # instead of manually splitting/replacing on bracket characters, which
        # only matched a couple of anticipated spacing/punctuation patterns.
        c_lst = parse_matrix_literal(c_str)
        return c_lst

    except ValueError as exc:
        # Don't log or exit() here -- this function's job is to report the
        # problem, not decide the program's fate.  Raise it and let the
        # caller (main()) decide whether to log, retry, or exit.
        raise ValueError(
            f"Failed to parse input. Please check your formatting. Details: {exc}"
        ) from exc
    except SyntaxError as exc:
        raise ValueError(
            "Input format is invalid. Ensure you use proper list/tuple syntax "
            "(e.g., (1, 2, 3), (4, 5, 6) or [[1, 2, 3], [4, 5, 6]])."
        ) from exc


# ── Validation ───────────────────────────────────────────────────────────────

def assertions(c: list[list[int]]) -> None:
    """
    Check dimensional 'sanity': the assignment cost matrix must be square.

    Raises
    ------
    ValueError
        If the matrix is not square.
    """
    if len(c) != len(c[0]):
        raise ValueError("Dimensional error: cost matrix is not square.")


# ── Algorithm functions ───────────────────────────────────────────────────────

def reduce_matrix(c: np.ndarray) -> np.ndarray:
    """
    Subtract the row minimum from every element of each row.

    Parameters
    ----------
    c : np.ndarray
        Square assignment-cost matrix.

    Returns
    -------
    np.ndarray
        Row-reduced matrix.

    Raises
    ------
    ValueError
        If the matrix is not square.
    """
    c_cp = c.copy()
    if c_cp.shape[0] != c_cp.shape[1]:
        raise ValueError(f"Matrix shape error: {c_cp.shape} is not square.")

    row_mins: np.ndarray = np.min(c_cp, axis=1, keepdims=True)
    result: np.ndarray = c_cp - row_mins
    return result


def cross_out_nulls(c_red: np.ndarray) -> list[int]:
    """
    Count the zeros on each row of the reduced cost matrix.

    Parameters
    ----------
    c_red : np.ndarray
        Reduced cost matrix.

    Returns
    -------
    list[int]
        Number of zeros in each row.
    """
    c_red_cp = c_red.copy()
    return [int(np.count_nonzero(row == 0)) for row in c_red_cp]


def assign_opt(c_red: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Find all zero-assignment sequences with exactly one zero per row and column.

    Uses backtracking over the pre-computed zero positions in each row.

    Parameters
    ----------
    c_red : np.ndarray
        Reduced cost matrix.

    Returns
    -------
    list[list[tuple[int, int]]]
        All valid assignment sequences (each a list of (row, col) pairs).
    """
    c_red_cp = c_red.copy()
    n = c_red_cp.shape[0]

    # Pre-compute column indices of zeros for each row.
    zero_positions = [np.where(c_red_cp[r] == 0)[0] for r in range(n)]

    all_sequences: list[list[tuple[int, int]]] = []

    def backtrack(
        row: int,
        used_cols: set[int],
        current_seq: list[tuple[int, int]],
    ) -> None:
        if row == n:
            all_sequences.append(list(current_seq))
            return

        for col in zero_positions[row]:
            if col not in used_cols:
                used_cols.add(col)
                current_seq.append((row, int(col)))

                backtrack(row + 1, used_cols, current_seq)

                current_seq.pop()
                used_cols.remove(col)

    backtrack(0, set(), [])
    return all_sequences


def best_zeros(
    seq: list[tuple[int, int]],
    c_array: np.ndarray,
) -> tuple[int, np.ndarray]:
    """
    Compute the total assignment cost and the assignment matrix for a sequence.

    Parameters
    ----------
    seq : list[tuple[int, int]]
        Assignment sequence — one (row, col) pair per task.
    c_array : np.ndarray
        Original (unreduced) cost matrix.

    Returns
    -------
    tuple[int, np.ndarray]
        ``(total_cost, assignment_matrix)`` where *assignment_matrix* has the
        original costs at assigned positions and zeros elsewhere.
    """
    c_cp = c_array.copy()
    assignment_matrix = np.zeros_like(c_array, dtype=int)

    for row, col in seq:
        assignment_matrix[row, col] = c_cp[row, col]

    assignment_cost = int(np.sum(assignment_matrix))
    return (assignment_cost, assignment_matrix)


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """Run the Hungarian Method assignment optimiser."""

    # ── Input & validation ──
    # Parsing errors from get_user_input() arrive as a raised ValueError
    # instead of the function calling exit() itself -- main() is the only
    # place that decides to log-and-exit.
    try:
        c_lst = get_user_input()
    except ValueError as exc:
        logger.error("\n\u274c Error: %s", exc)
        raise SystemExit(1) from exc

    try:
        assertions(c_lst)
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        raise SystemExit(1) from exc

    c_array = np.array(c_lst)

    # ── Reduce costs on rows then columns (two-pass Hungarian reduction) ──
    # First pass:  subtract row minima  →  reduce_matrix(c_array)
    # Second pass: subtract column minima →  reduce_matrix(...T).T
    c_red = reduce_matrix(reduce_matrix(c_array).T).T
    logger.info("Reduced cost matrix:\n%s", c_red)

    BLOCK_COST = -1   # sentinel used to mark crossed-out rows/columns
    not_allocated = True

    while not_allocated:
        c_work = c_red.copy()
        crossed = 0
        crossed_rows: list[int] = []
        crossed_cols: list[int] = []

        # ── Step 1: cross out zeros efficiently ──
        # Greedily cover all zeros with the minimum number of lines by always
        # crossing out the row or column that contains the most zeros.
        while np.count_nonzero(c_work == 0) != 0:
            nulls_on_rows = cross_out_nulls(c_work)
            nulls_on_cols = cross_out_nulls(c_work.T)

            if max(nulls_on_rows) >= max(nulls_on_cols):
                to_cross = nulls_on_rows.index(max(nulls_on_rows))
                c_work[to_cross] = BLOCK_COST
                crossed_rows.append(to_cross)
            else:
                to_cross = nulls_on_cols.index(max(nulls_on_cols))
                c_work.T[to_cross] = BLOCK_COST
                crossed_cols.append(to_cross)

            crossed += 1

        if crossed == len(c_red):
            # ── Step 2: optimal solution reachable — assign on zeros ──
            zero_seqs = assign_opt(c_red)
            possible_assignments: dict[int, np.ndarray] = {}

            for seq in zero_seqs:
                cost, matrix = best_zeros(seq, c_array)
                possible_assignments[cost] = matrix

            best_cost = min(possible_assignments)
            delivered = possible_assignments[best_cost]

            logger.info("Delivered assignment solution:\n%s", delivered)
            logger.info("Total cost of assignment: %s", best_cost)

            not_allocated = False

        else:
            # ── Step 3: not yet optimal — adjust the reduced cost matrix ──

            # Find intersections of crossed rows and columns.
            intersections = [
                (r, c) for r in crossed_rows for c in crossed_cols
            ]

            # Subtract the smallest uncovered value from all uncovered cells.
            min_opt = int(np.min(c_work[c_work > -1]))
            c_work = np.where(c_work > -1, c_work - min_opt, c_work)

            # Add it back to the intersection cells.
            for r, c in intersections:
                c_work[r, c] = c_red[r, c] + min_opt

            # Restore the blocked (crossed-out) cells to their reduced values.
            c_red = np.where(c_work == -1, c_red, c_work)

            logger.debug("Updated reduced cost matrix:\n%s", c_red)


if __name__ == "__main__":
    main()
