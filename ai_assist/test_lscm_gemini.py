"""Unit tests for lscm_gemini.py module."""

import unittest
import numpy as np
from lscm_gemini import (
    parse_matrix,
    RawInput,
    ComputationalData,
    assertions,
    tie_breaking,
    allocation,
    feasibility_cost,
)


class TestLSCMGemini(unittest.TestCase):
    """Test suite for Least Unit Cost Matrix Allocation components."""

    # -----------------------------------------------------------------------
    # 1. parse_matrix tests
    # -----------------------------------------------------------------------
    def test_parse_matrix_valid(self):
        """Test valid matrix syntax parsing."""
        self.assertEqual(
            parse_matrix("[[1, 2], [3, 4]]"), [[1, 2], [3, 4]]
        )
        self.assertEqual(
            parse_matrix("[(10, 20), (30, 40)]"), [[10, 20], [30, 40]]
        )
        self.assertEqual(
            parse_matrix("(5, 10, 15)"), [[5, 10, 15]]
        )

    def test_parse_matrix_invalid(self):
        """Test invalid syntax and non-integer rejection."""
        with self.assertRaises(ValueError):
            parse_matrix("")
        with self.assertRaises(ValueError):
            parse_matrix("[[1.5, 2], [3, 4]]")
        with self.assertRaises(ValueError):
            parse_matrix("[[True, 2], [3, 4]]")
        with self.assertRaises(ValueError):
            parse_matrix("not a matrix")

    # -----------------------------------------------------------------------
    # 2. assertions tests
    # -----------------------------------------------------------------------
    def test_assertions_valid(self):
        """Test valid balanced problem assertions."""
        raw = RawInput(
            cost=[[2, 3], [5, 1]], supply=[20, 30], demand=[10, 40]
        )
        # Should not raise
        assertions(raw)

    def test_assertions_ragged(self):
        """Test ragged cost matrix detection."""
        raw = RawInput(
            cost=[[2, 3], [5]], supply=[20, 30], demand=[10, 40]
        )
        with self.assertRaises(ValueError):
            assertions(raw)

    def test_assertions_shape_mismatch(self):
        """Test supply/demand length mismatch detection."""
        raw = RawInput(
            cost=[[2, 3], [5, 1]], supply=[20], demand=[10, 40]
        )
        with self.assertRaises(ValueError):
            assertions(raw)

    def test_assertions_unbalanced(self):
        """Test total supply != total demand detection."""
        raw = RawInput(
            cost=[[2, 3], [5, 1]], supply=[20, 30], demand=[10, 50]
        )
        with self.assertRaises(ValueError):
            assertions(raw)

    def test_assertions_negative(self):
        """Test negative value detection."""
        raw = RawInput(
            cost=[[-2, 3], [5, 1]], supply=[20, 30], demand=[10, 40]
        )
        with self.assertRaises(ValueError):
            assertions(raw)

    # -----------------------------------------------------------------------
    # 3. tie_breaking tests
    # -----------------------------------------------------------------------
    def test_tie_breaking_rule1(self):
        """Test Rule 1: Max allocatable amount."""
        candidates = [(0, 3), (2, 3)]
        supply = np.array([20, 7, 50])
        demand = np.array([15, 37, 0, 25])
        # (0,3) max alloc = min(20, 25) = 20
        # (2,3) max alloc = min(50, 25) = 25 -> Rule 1 chooses (2,3)
        self.assertEqual(tie_breaking(candidates, supply, demand), (2, 3))

    def test_tie_breaking_rule2(self):
        """Test Rule 2: Prefer cell where supply >= demand when max alloc is equal."""
        candidates = [(0, 0), (1, 1)]
        supply = np.array([10, 20])
        demand = np.array([15, 10])
        # (0,0) max alloc = min(10, 15) = 10, supply 10 < demand 15
        # (1,1) max alloc = min(20, 10) = 10, supply 20 >= demand 10 -> Rule 2 chooses (1,1)
        self.assertEqual(tie_breaking(candidates, supply, demand), (1, 1))

    # -----------------------------------------------------------------------
    # 4. allocation and feasibility_cost tests
    # -----------------------------------------------------------------------
    def test_allocation_and_feasibility(self):
        """Test full allocation algorithm and cost calculation."""
        raw = RawInput(
            cost=[[3, 1, 7, 4], [2, 6, 5, 9], [8, 3, 3, 2]],
            supply=[250, 300, 400],
            demand=[200, 225, 275, 250],
        )
        data = ComputationalData(
            cost_array=np.array(raw.cost, dtype=int),
            supply_array=np.array(raw.supply, dtype=int),
            demand_array=np.array(raw.demand, dtype=int),
        )
        alloc = allocation(data)
        expected_alloc = np.array(
            [[0, 225, 25, 0], [200, 0, 100, 0], [0, 0, 150, 250]]
        )
        np.testing.assert_array_equal(alloc, expected_alloc)

        is_feasible, cost = feasibility_cost(alloc, data.cost_array)
        self.assertTrue(is_feasible)
        self.assertEqual(cost, 2250)

    def test_feasibility_cost_degenerate(self):
        """Test feasibility check raising ValueError on degenerate matrix."""
        alloc = np.array([[10, 0], [0, 0]])  # 1 allocation, need m+n-1 = 3
        cost = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            feasibility_cost(alloc, cost)


if __name__ == "__main__":
    unittest.main()
