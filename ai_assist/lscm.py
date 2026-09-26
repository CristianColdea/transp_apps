"""Least Unit Cost Matrix (LSCM) Allocation Method.

Generates a Basic Feasible Solution (BFS) for a balanced transportation
problem by iteratively allocating on the least-cost cell.
"""
from __future__ import annotations

import ast
import logging
from typing import List, Tuple, NamedTuple
import numpy as np

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component 1: Helper function to parse unit cost matrix user input
# ---------------------------------------------------------------------------
def parse_cost_matrix(input_str: str) -> List[List[int]]:
    """Parse a string representation of a unit cost matrix into a 2D list of integers.

    Accepts nested lists or tuples with or without outer wrappers, spaces, and
    arbitrary nesting styles supported by ast.literal_eval.
    For example: '[[1, 2], [3, 4]]', '[(1, 2), (3, 4)]', or flat single row '(1, 2)'.

    Args:
        input_str: Raw string input entered by the user.

    Returns:
        List of lists containing integer unit costs.

    Raises:
        ValueError: If input cannot be parsed or has invalid element types/structure.
    """
    if not input_str or not input_str.strip():
        raise ValueError("Cost matrix input string cannot be empty.")

    try:
        parsed = ast.literal_eval(input_str.strip())
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Failed to parse cost matrix input: {exc}") from exc

    if not isinstance(parsed, (list, tuple)):
        raise ValueError("Cost matrix input must represent a sequence of rows.")

    data = list(parsed) if isinstance(parsed, tuple) else parsed

    if not data:
        raise ValueError("Cost matrix cannot be empty.")

    # Handle flat single row: e.g. (1, 2, 3) or [1, 2, 3] where all elements are integers
    if all(isinstance(item, int) and not isinstance(item, bool) for item in data):
        return [data]

    processed_matrix: List[List[int]] = []
    for row in data:
        if not isinstance(row, (list, tuple)):
            raise ValueError("Each row in the cost matrix must be a list or tuple.")

        processed_row: List[int] = []
        for item in row:
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError(
                    f"Cost matrix elements must be integers, got {type(item).__name__}: {item}"
                )
            processed_row.append(int(item))

        processed_matrix.append(processed_row)

    return processed_matrix
