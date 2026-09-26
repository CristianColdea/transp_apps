import ast


def parse_unit_cost_matrix(input_data: str) -> list[list[int]]:
    """
    Parses a string representation of a nested numeric structure into a 2D list.

    Args:
        input_data: A string containing nested lists or tuples (e.g., "[[1, 2], [3, 4]]").

    Returns:
        A list of lists of integers representing the unit cost matrix.

    Raises:
        ValueError: If the input is not a valid nested structure of numbers.
    """
    try:
        # ast.literal_eval handles various spacings and nesting styles (tuples/lists)
        parsed = ast.literal_eval(input_data.strip())
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid input format for cost matrix: {e}")

    if not isinstance(parsed, list):
        # Handle cases where the outer structure might be a tuple or other sequence
        if not hasattr(parsed, "__getitem__"):
            raise ValueError("Input must represent a list of lists.")
        data = list(parsed)
    else:
        data = parsed

    processed_matrix = []
    for row in data:
        if not isinstance(row, (list, tuple)):
            raise ValueError("Each row in the matrix must be a list or tuple.")

        # Ensure all elements are integers
        row_list = [int(item) for item in row]
from typing import List
from typing import NamedTuple


class RawInput(NamedTuple):
    """
    A data structure representing the raw input parameters for a transportation problem.

    Attributes:
        cost (List[List[int]]): A 2D list where cost[i][j] represents the unit cost of
            transporting goods from source i to destination j.
        supply (List[int]): A list where supply[i] is the total available quantity at source i.
        demand (List[int]): A list where demand[j] is the total required quantity at destination j.
    """

    cost: List[[int]]
    supply: List[int]
    demand: List[int]
import numpy as np

class ComputationalData(NamedTuple):
    """
    A container for transportation problem data arrays.

    Attributes:
        cost_array (np.ndarray): The cost matrix associated with transporting goods.
        supply_array (np.ndarray): The available supply quantities at each source.
        demand_array (np.ndarray): The required demand quantities at each destination.
    """

    cost_array: np.ndarray
    supply_array: np.ndarray
    demand_array: np.ndarray
