"""
import ast


def parse_cost_matrix(input_str: str) -> list[list[int]]:
    """
    Parses a string representation of a cost matrix into a list of lists of integers.
    Supports various nesting styles (tuples, lists) and whitespace.
    """
    # ast.literal_eval handles nested structures like [[1, 2], [3, 4]] or [(1, 2), (3, 4)]
    data = ast.literal_eval(input_str.strip())

    if not isinstance(data, list):
        # Handle cases where the outer structure might be a tuple
        if isinstance(data, tuple):
            data = list(data)
        else:
            raise ValueError("Input must represent a list of rows.")

    processed_matrix = []
    for row in data:
        if isinstance(row, (list, tuple)):
            # Convert inner elements to integers and the row to a list
            processed_matrix.append([int(item) for item in row])
        else:
            raise ValueError("Each row in the matrix must be a list or tuple.")

    return processed_matrix

from typing import List
import numpy as np

# Two data structure to store raw user input and computational formats in the
# form of:


class RawInput(NamedTuple):
    """A container for raw transportation problem inputs.

    Attributes:
        cost (List[List[int]]): A 2D list representing the cost matrix.
        supply (List[int]): A list of supply quantities at each source.
        demand (List[int]): A list of demand requirements at each destination.
    """

    cost: List[List[int]]
    supply: List[int]
    demand: List[int]


class ComputationalData(NamedTuple):
    """A container for processed numerical arrays used in calculations.

    Attributes:
        cost_array (np.ndarray): A 2D NumPy array of costs.
        supply_array (np.ndarray): A 1D NumPy array of supply values.
        demand_array (np.ndarray): A 1D NumPy array of demand values.
    """

    cost_array: np.ndarray
    supply_array: np.ndarray
    demand_array: np.ndarray

