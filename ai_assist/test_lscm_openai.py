import numpy as np
import pytest

from lscm_openai import RawInput, ComputationalData, allocation, feasibility_cost

@pytest.fixture
def example_data() -> RawInput:
    cost = [[7, 6, 4, 3],
            [9, 5, 2, 6],
            [4, 8, 5, 3]]
    supply = [20, 30, 50]
    demand = [15, 37, 23, 25]
    return RawInput(cost=cost, supply=supply, demand=demand)

def test_allocation_basic_feasible_solution(example_data: RawInput) -> None:
    data = ComputationalData(
        cost_array=np.array(example_data.cost),
        supply_array=np.array(example_data.supply),
        demand_array=np.array(example_data.demand),
    )
    alloc = allocation(data)
    expected = np.array([[0, 20, 0, 0],
                         [0, 7, 23, 0],
                         [15, 10, 0, 25]])
    assert np.array_equal(alloc, expected)

def test_total_transportation_cost(example_data: RawInput) -> None:
    data = ComputationalData(
        cost_array=np.array(example_data.cost),
        supply_array=np.array(example_data.supply),
        demand_array=np.array(example_data.demand),
    )
    alloc = allocation(data)
    total = feasibility_cost(alloc, np.array(example_data.cost))
    assert total == 416
