"""Unit tests for lscm_claude.py — Least Unit Cost Matrix Allocation Method."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from lscm_claude import (
    parse_cost_matrix,
    RawInput,
    ComputationalData,
    get_user_input,
    assertions,
    tie_breaking,
    allocation,
    feasibility_cost,
    main,
)


# =========================================================================
# parse_cost_matrix
# =========================================================================
class TestParseCostMatrix(unittest.TestCase):
    """Tests for the parse_cost_matrix helper."""

    # --- valid inputs ---
    def test_nested_lists(self) -> None:
        result = parse_cost_matrix("[[1, 2], [3, 4]]")
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_nested_tuples(self) -> None:
        result = parse_cost_matrix("((1, 2), (3, 4))")
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_mixed_nesting(self) -> None:
        result = parse_cost_matrix("[(1, 2), [3, 4]]")
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_extra_whitespace(self) -> None:
        result = parse_cost_matrix("  [[ 1 , 2 ], [ 3 , 4 ]]  ")
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_single_row_nested(self) -> None:
        result = parse_cost_matrix("[[5, 6, 7]]")
        self.assertEqual(result, [[5, 6, 7]])

    def test_bare_row_tuple(self) -> None:
        """A bare tuple like (4, 5, 6) should become [[4, 5, 6]]."""
        result = parse_cost_matrix("(4, 5, 6)")
        self.assertEqual(result, [[4, 5, 6]])

    def test_bare_row_list(self) -> None:
        """A bare list like [4, 5, 6] should become [[4, 5, 6]]."""
        result = parse_cost_matrix("[4, 5, 6]")
        self.assertEqual(result, [[4, 5, 6]])

    def test_single_element_matrix(self) -> None:
        result = parse_cost_matrix("[[10]]")
        self.assertEqual(result, [[10]])

    # --- invalid inputs ---
    def test_float_element_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_cost_matrix("[[2.5, 3], [5, 1]]")
        self.assertIn("float", str(ctx.exception))

    def test_string_element_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_cost_matrix("[['a', 'b'], [1, 2]]")

    def test_non_sequence_input(self) -> None:
        with self.assertRaises(ValueError):
            parse_cost_matrix("42")

    def test_row_not_sequence(self) -> None:
        with self.assertRaises(ValueError):
            parse_cost_matrix("[[1, 2], 3]")

    def test_empty_string(self) -> None:
        with self.assertRaises((ValueError, SyntaxError)):
            parse_cost_matrix("")


# =========================================================================
# Data structures
# =========================================================================
class TestDataStructures(unittest.TestCase):
    """Tests for RawInput and ComputationalData NamedTuples."""

    def test_raw_input_creation(self) -> None:
        raw = RawInput(cost=[[1, 2], [3, 4]], supply=[10, 20], demand=[15, 15])
        self.assertEqual(raw.cost, [[1, 2], [3, 4]])
        self.assertEqual(raw.supply, [10, 20])
        self.assertEqual(raw.demand, [15, 15])

    def test_computational_data_creation(self) -> None:
        data = ComputationalData(
            cost_array=np.array([[1, 2], [3, 4]]),
            supply_array=np.array([10, 20]),
            demand_array=np.array([15, 15]),
        )
        np.testing.assert_array_equal(data.cost_array, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(data.supply_array, [10, 20])
        np.testing.assert_array_equal(data.demand_array, [15, 15])


# =========================================================================
# get_user_input
# =========================================================================
class TestGetUserInput(unittest.TestCase):
    """Tests for get_user_input (with mocked input)."""

    @patch("builtins.input", side_effect=[
        "[[2, 3], [5, 1]]",    # cost matrix
        "[20, 30]",             # supply
        "[25, 25]",             # demand
    ])
    def test_valid_input(self, _mock_input) -> None:
        raw = get_user_input()
        self.assertIsInstance(raw, RawInput)
        self.assertEqual(raw.cost, [[2, 3], [5, 1]])
        self.assertEqual(raw.supply, [20, 30])
        self.assertEqual(raw.demand, [25, 25])

    @patch("builtins.input", side_effect=[
        "((1, 2), (3, 4))",    # tuples
        "(10, 20)",            # tuple supply
        "(15, 15)",            # tuple demand
    ])
    def test_tuple_input(self, _mock_input) -> None:
        raw = get_user_input()
        self.assertEqual(raw.cost, [[1, 2], [3, 4]])
        self.assertEqual(raw.supply, [10, 20])
        self.assertEqual(raw.demand, [15, 15])

    @patch("builtins.input", side_effect=["not valid python"])
    def test_invalid_cost_matrix(self, _mock_input) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_user_input()
        self.assertIn("cost matrix", str(ctx.exception).lower())

    @patch("builtins.input", side_effect=[
        "[[1, 2], [3, 4]]",
        "not_a_list",           # bad supply
    ])
    def test_invalid_supply_parse(self, _mock_input) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_user_input()
        self.assertIn("supply", str(ctx.exception).lower())

    @patch("builtins.input", side_effect=[
        "[[1, 2], [3, 4]]",
        "[10, 20]",
        "not_a_list",           # bad demand
    ])
    def test_invalid_demand_parse(self, _mock_input) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_user_input()
        self.assertIn("demand", str(ctx.exception).lower())

    @patch("builtins.input", side_effect=[
        "[[1, 2], [3, 4]]",
        "42",                   # not a list/tuple
    ])
    def test_supply_not_sequence(self, _mock_input) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_user_input()
        self.assertIn("list or tuple", str(ctx.exception).lower())

    @patch("builtins.input", side_effect=[
        "[[1, 2], [3, 4]]",
        "[10, 20]",
        "42",                   # not a list/tuple
    ])
    def test_demand_not_sequence(self, _mock_input) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_user_input()
        self.assertIn("list or tuple", str(ctx.exception).lower())

    @patch("builtins.input", side_effect=[
        "[[1, 2], [3, 4]]",
        "[10, 2.5]",            # float in supply
    ])
    def test_supply_float_rejected(self, _mock_input) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_user_input()
        self.assertIn("integer", str(ctx.exception).lower())

    @patch("builtins.input", side_effect=[
        "[[1, 2], [3, 4]]",
        "[10, 20]",
        "[15, 1.5]",            # float in demand
    ])
    def test_demand_float_rejected(self, _mock_input) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_user_input()
        self.assertIn("integer", str(ctx.exception).lower())


# =========================================================================
# assertions
# =========================================================================
class TestAssertions(unittest.TestCase):
    """Tests for the assertions validation function."""

    def test_valid_input_passes(self) -> None:
        raw = RawInput(
            cost=[[2, 3, 4], [5, 1, 6]],
            supply=[30, 20],
            demand=[15, 20, 15],
        )
        assertions(raw)  # should not raise

    def test_ragged_matrix(self) -> None:
        raw = RawInput(
            cost=[[1, 2, 3], [4, 5]],
            supply=[10, 10],
            demand=[5, 5, 10],
        )
        with self.assertRaises(ValueError) as ctx:
            assertions(raw)
        self.assertIn("ragged", str(ctx.exception).lower())

    def test_supply_length_mismatch(self) -> None:
        raw = RawInput(
            cost=[[1, 2], [3, 4]],
            supply=[10, 20, 30],     # 3 elements, need 2
            demand=[15, 15],
        )
        with self.assertRaises(ValueError) as ctx:
            assertions(raw)
        self.assertIn("supply length", str(ctx.exception).lower())

    def test_demand_length_mismatch(self) -> None:
        raw = RawInput(
            cost=[[1, 2], [3, 4]],
            supply=[10, 20],
            demand=[15, 10, 5],      # 3 elements, need 2
        )
        with self.assertRaises(ValueError) as ctx:
            assertions(raw)
        self.assertIn("demand length", str(ctx.exception).lower())

    def test_unbalanced_problem(self) -> None:
        raw = RawInput(
            cost=[[1, 2], [3, 4]],
            supply=[20, 30],         # total = 50
            demand=[25, 30],         # total = 55
        )
        with self.assertRaises(ValueError) as ctx:
            assertions(raw)
        self.assertIn("not balanced", str(ctx.exception).lower())


# =========================================================================
# tie_breaking
# =========================================================================
class TestTieBreaking(unittest.TestCase):
    """Tests for the tie_breaking function."""

    def test_rule1_unique_max_amount(self) -> None:
        """Rule 1: single cell with maximum allocatable quantity wins."""
        candidates = [(0, 0), (1, 1)]
        supply = np.array([50, 10])
        demand = np.array([20, 30])
        # amounts: min(50,20)=20 vs min(10,30)=10 → (0,0) wins
        result = tie_breaking(candidates, supply, demand)
        self.assertEqual(result, (0, 0))

    def test_rule2_supply_geq_demand(self) -> None:
        """Rule 2: equal max amounts, prefer supply >= demand."""
        candidates = [(0, 0), (1, 1)]
        supply = np.array([20, 30])
        demand = np.array([20, 20])
        # amounts: min(20,20)=20 vs min(30,20)=20 → tied
        # supply >= demand: (0,0) 20>=20 yes, (1,1) 30>=20 yes
        # Both qualify — first in index order: (0,0)
        result = tie_breaking(candidates, supply, demand)
        self.assertEqual(result, (0, 0))

    def test_rule2_selects_supply_geq_demand_cell(self) -> None:
        """Rule 2: among tied max-amount cells, pick where supply >= demand."""
        candidates = [(0, 0), (1, 1)]
        supply = np.array([10, 25])
        demand = np.array([10, 10])
        # amounts: min(10,10)=10 vs min(25,10)=10 → tied
        # supply >= demand: (0,0) 10>=10 yes, (1,1) 25>=10 yes
        # Both qualify — first in index order: (0,0)
        result = tie_breaking(candidates, supply, demand)
        self.assertEqual(result, (0, 0))

    def test_rule2_only_one_qualifies(self) -> None:
        """Rule 2: only one cell has supply >= demand."""
        candidates = [(0, 1), (1, 0)]
        supply = np.array([5, 20])
        demand = np.array([10, 10])
        # amounts: min(5,10)=5 vs min(20,10)=10 → (1,0) wins by rule 1
        result = tie_breaking(candidates, supply, demand)
        self.assertEqual(result, (1, 0))

    def test_returns_none_when_no_rule_resolves(self) -> None:
        """Neither rule resolves: all tied amounts, no supply >= demand."""
        candidates = [(0, 0), (1, 1)]
        supply = np.array([5, 5])
        demand = np.array([10, 10])
        # amounts: min(5,10)=5 vs min(5,10)=5 → tied
        # supply >= demand: 5>=10 no, 5>=10 no → none qualify
        result = tie_breaking(candidates, supply, demand)
        self.assertIsNone(result)


# =========================================================================
# allocation
# =========================================================================
class TestAllocation(unittest.TestCase):
    """Tests for the allocation (orchestrator) function."""

    def _make_data(
        self,
        cost: list[list[int]],
        supply: list[int],
        demand: list[int],
    ) -> ComputationalData:
        return ComputationalData(
            cost_array=np.array(cost, dtype=int),
            supply_array=np.array(supply, dtype=int),
            demand_array=np.array(demand, dtype=int),
        )

    def test_3x3_standard(self) -> None:
        """Classic 3×3 textbook example."""
        data = self._make_data(
            cost=[[2, 7, 4], [5, 4, 6], [8, 1, 3]],
            supply=[120, 80, 50],
            demand=[90, 100, 60],
        )
        alloc = allocation(data)
        expected = np.array([
            [90,  0, 30],
            [ 0, 50, 30],
            [ 0, 50,  0],
        ])
        np.testing.assert_array_equal(alloc, expected)

    def test_3x4_problem(self) -> None:
        """3×4 problem from user test case."""
        data = self._make_data(
            cost=[[7, 6, 4, 3], [9, 5, 2, 6], [4, 8, 5, 3]],
            supply=[20, 30, 50],
            demand=[15, 37, 23, 25],
        )
        alloc = allocation(data)
        expected = np.array([
            [ 0, 20,  0,  0],
            [ 0,  7, 23,  0],
            [15, 10,  0, 25],
        ])
        np.testing.assert_array_equal(alloc, expected)

    def test_single_row(self) -> None:
        """1×3 single-source problem."""
        data = self._make_data(
            cost=[[4, 5, 6]],
            supply=[100],
            demand=[30, 40, 30],
        )
        alloc = allocation(data)
        expected = np.array([[30, 40, 30]])
        np.testing.assert_array_equal(alloc, expected)

    def test_2x2_simple(self) -> None:
        """Minimal 2×2 balanced problem."""
        data = self._make_data(
            cost=[[1, 5], [3, 2]],
            supply=[30, 20],
            demand=[25, 25],
        )
        alloc = allocation(data)
        # min cost = 1 at (0,0): allocate min(30,25)=25 → supply[0]=5
        # min cost = 2 at (1,1): allocate min(20,25)=20 → demand[1]=5
        # min cost = 3 at (1,0): blocked (supply[1]=0)
        # min cost = 5 at (0,1): allocate min(5,5)=5
        expected = np.array([
            [25, 5],
            [ 0, 20],
        ])
        np.testing.assert_array_equal(alloc, expected)

    def test_supply_equals_demand_balanced(self) -> None:
        """Verify supply and demand are fully consumed."""
        data = self._make_data(
            cost=[[2, 7, 4], [5, 4, 6], [8, 1, 3]],
            supply=[120, 80, 50],
            demand=[90, 100, 60],
        )
        alloc = allocation(data)
        # Row sums should match supply
        np.testing.assert_array_equal(alloc.sum(axis=1), [120, 80, 50])
        # Column sums should match demand
        np.testing.assert_array_equal(alloc.sum(axis=0), [90, 100, 60])


# =========================================================================
# feasibility_cost
# =========================================================================
class TestFeasibilityCost(unittest.TestCase):
    """Tests for the feasibility_cost function."""

    def test_feasible_3x3(self) -> None:
        alloc = np.array([
            [90,  0, 30],
            [ 0, 50, 30],
            [ 0, 50,  0],
        ])
        cost = np.array([
            [2, 7, 4],
            [5, 4, 6],
            [8, 1, 3],
        ])
        is_feasible, total_cost = feasibility_cost(alloc, cost)
        self.assertTrue(is_feasible)
        self.assertEqual(total_cost, 730)

    def test_feasible_3x4(self) -> None:
        alloc = np.array([
            [ 0, 20,  0,  0],
            [ 0,  7, 23,  0],
            [15, 10,  0, 25],
        ])
        cost = np.array([
            [7, 6, 4, 3],
            [9, 5, 2, 6],
            [4, 8, 5, 3],
        ])
        is_feasible, total_cost = feasibility_cost(alloc, cost)
        self.assertTrue(is_feasible)
        self.assertEqual(total_cost, 416)

    def test_degenerate_raises(self) -> None:
        """A solution with too few allocations must raise ValueError."""
        # 3×3 needs 5 allocations; this one has only 3
        alloc = np.array([
            [90,  0,  0],
            [ 0, 80,  0],
            [ 0,  0, 50],
        ])
        cost = np.array([
            [2, 7, 4],
            [5, 4, 6],
            [8, 1, 3],
        ])
        with self.assertRaises(ValueError) as ctx:
            feasibility_cost(alloc, cost)
        self.assertIn("degenerate", str(ctx.exception).lower())


# =========================================================================
# main (integration)
# =========================================================================
class TestMain(unittest.TestCase):
    """Integration tests for the main entry point."""

    @patch("builtins.input", side_effect=[
        "[[2, 7, 4], [5, 4, 6], [8, 1, 3]]",
        "[120, 80, 50]",
        "[90, 100, 60]",
    ])
    def test_main_runs_without_error(self, _mock_input) -> None:
        """main() should complete without raising on valid input."""
        main()  # should not raise

    @patch("builtins.input", side_effect=["garbage"])
    def test_main_handles_bad_cost(self, _mock_input) -> None:
        """main() should handle invalid cost gracefully (no traceback)."""
        main()  # should not raise — catches ValueError internally

    @patch("builtins.input", side_effect=[
        "[[1, 2], [3, 4]]",
        "[20, 30]",
        "[25, 30]",            # unbalanced: 50 != 55
    ])
    def test_main_handles_unbalanced(self, _mock_input) -> None:
        """main() should handle unbalanced problem gracefully."""
        main()  # should not raise — catches ValueError internally


if __name__ == "__main__":
    unittest.main()
