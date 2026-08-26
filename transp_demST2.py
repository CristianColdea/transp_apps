"""
Script to compute short term transport demand.
Six methods are coded: gravitational model (for reference),
Furness, Fratar, Detroit, average growth factor and a weighted-demand method.

Refactored for better architecture:
  - Matrix utility helpers extracted to avoid code duplication.
  - All print() calls replaced with structured logging.
  - Execution wrapped in main() / ``if __name__ == "__main__"`` guard.
  - Hard-coded matrix size removed (derived from input dimensions).
  - Several correctness bugs fixed (shallow copies, flag overwrites,
    variable shadowing, uninitialised variables, exit() → ValueError).
"""

from __future__ import annotations

import logging
from collections.abc import Sized

logging.basicConfig(
    level=logging.DEBUG,  # switch to logging.INFO to silence the .debug() calls
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Type alias ──────────────────────────────────────────────────────────────
# A square matrix of numbers.  Travel counts are logically integers, but they
# get mixed into float division throughout this script (friction factors,
# coefficients, etc.), so everything numeric is typed as float.
Matrix = list[list[float]]

# Safety limit — methods like Detroit may take tens of thousands of iterations
# with certain inputs.  This prevents accidental infinite loops.
MAX_ITERATIONS: int = 100_000


# ── Matrix utility helpers ──────────────────────────────────────────────────

def row_sums(mat: Matrix) -> list[float]:
    """Return the sum of each row."""
    return [sum(row) for row in mat]


def col_sums(mat: Matrix) -> list[float]:
    """Return the sum of each column (transpose + row sums)."""
    return [sum(col) for col in zip(*mat)]


def transpose(mat: Matrix) -> Matrix:
    """Transpose a matrix (does not require it to be square)."""
    return [list(col) for col in zip(*mat)]


def flatten_to_matrix(flat: list[float], n: int) -> Matrix:
    """Reshape a flat list into an n-column matrix."""
    return [flat[i:i + n] for i in range(0, len(flat), n)]


def round_matrix(mat: Matrix) -> Matrix:
    """Round every element of *mat* and return as a new matrix."""
    return [[round(val) for val in row] for row in mat]


def deep_copy_matrix(mat: Matrix) -> Matrix:
    """Return a deep (row-level) copy so mutations don't corrupt the original."""
    return [row[:] for row in mat]


def _validate_same_length(*arrays: Sized, msg: str = "") -> None:
    """Raise ValueError if the arrays differ in length."""
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        detail = msg or f"Length mismatch: {lengths}"
        logger.error(detail)
        raise ValueError(detail)


# ── Initial data ────────────────────────────────────────────────────────────
# (module-level constants; kept here so ``main()`` can reference them easily)

# number of travels: produced on rows, attracted on columns
TRAVS: Matrix = [
    [40, 110, 150],
    [50,  20,  30],
    [110, 30,  10],
]

# matching friction factors
FFS: Matrix = [
    [0.753, 1.597, 0.753],
    [0.987, 0.753, 0.765],
    [1.597, 0.765, 0.753],
]

# neutral calibration coefficients
K_IJ0: Matrix = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]

# auto travels cost
TCA: Matrix = [
    [0.5, 1,   1.4],
    [1.2, 0.8, 1.2],
    [1.7, 1.5, 0.7],
]

# transit travels cost
TCT: Matrix = [
    [1,   1.5, 2],
    [1.8, 1.2, 1.9],
    [1.7, 1.5, 0.7],
]

# auto travels duration
TDA: Matrix = [
    [3,  12, 7],
    [13,  3, 19],
    [9,  16, 4],
]

# transit travels duration
TDT: Matrix = [
    [15,  5, 12],
    [15,  6, 26],
    [20, 21,  8],
]

# future friction factors
FFS_F: Matrix = [
    [0.753, 0.987, 1.597],
    [0.987, 0.753, 0.765],
    [1.597, 0.765, 0.753],
]

# future produced travels
P_IS: list[float] = [750, 580, 480]

# future attracted travels
A_JS: list[float] = [722, 786, 302]


# ── Gravitational-model class ──────────────────────────────────────────────

class GravitMod:
    """
    Namespace for transport-demand estimation methods.

    All methods are static — the class serves as a logical grouping rather
    than an object with mutable state.
    """

    @staticmethod
    def gravmod_init(travs: Matrix, ffs: Matrix, k_ijs: Matrix) -> Matrix:
        """
        Compute gravitational-model values to determine calibration factors.

        Parameters
        ----------
        travs : Matrix
            Observed (historical) travel matrix.
        ffs : Matrix
            Friction-factor matrix (same shape as *travs*).
        k_ijs : Matrix
            Calibration-coefficient matrix (same shape as *travs*).

        Returns
        -------
        Matrix
            Rounded computed-travel matrix.
        """
        _validate_same_length(travs, ffs, k_ijs,
                              msg="travs, ffs and k_ijs must have the same dimensions.")

        n = len(travs)
        s_Aj = col_sums(travs)
        s_Pi = row_sums(travs)

        # compute travels with gravitational model
        gvals_init: list[float] = []
        for i in range(n):
            pdsum = sum(aj * ff for aj, ff in zip(s_Aj, ffs[i]))
            for k1 in range(n):
                gvals_init.append(
                    s_Pi[i] * ffs[i][k1] * s_Aj[k1] * k_ijs[i][k1] / pdsum
                )

        gvals_init_r: list[float] = [float(round(v)) for v in gvals_init]
        gvals_init_m = flatten_to_matrix(gvals_init_r, n)

        return gvals_init_m

    @staticmethod
    def comp(s_ih: list[float], s_ic: list[float], tlr: float = 0.05) -> bool:
        """
        Check whether two value-lists are within relative *tlr* tolerance.

        Returns ``True`` if every pair is within tolerance, ``False`` otherwise.
        """
        for ih, ic in zip(s_ih, s_ic):
            if abs(ih - ic) / ih >= tlr:
                return False
        return True

    @staticmethod
    def iter_adj_in(travs: Matrix, travsc: Matrix, tlr: float = 0.01) -> Matrix:
        """
        Iteratively adjust travels computed with the gravitational model.

        Parameters
        ----------
        travs : Matrix
            Observed (historical) travel matrix.
        travsc : Matrix
            Initially computed travel matrix (will be adjusted in-place copy).
        tlr : float
            Convergence tolerance.

        Returns
        -------
        Matrix
            Adjusted, rounded travel matrix.
        """
        logger.info("Enter iter_adj_in method.")

        _validate_same_length(travs, travsc,
                              msg="travs and travsc must have the same dimensions.")

        n = len(travs)
        travsc = deep_copy_matrix(travsc)

        s_Pih = row_sums(travs)
        s_Ajh = col_sums(travs)

        is_converged = False
        i = 0  # produced passes counter
        j = 0  # attracted passes counter

        while not is_converged:
            # ── Produced adjustment ──
            s_Pic = row_sums(travsc)

            if not GravitMod.comp(s_Pih, s_Pic):
                ccsi = [round(ph / pc, 3) for ph, pc in zip(s_Pih, s_Pic)]
                for x in range(n):
                    travsc[x] = [ccsi[x] * val for val in travsc[x]]
                i += 1

            # ── Attracted adjustment ──
            travsc_t = transpose(travsc)
            s_Ajc = [sum(col) for col in travsc_t]

            if not GravitMod.comp(s_Ajh, s_Ajc):
                ccsj = [round(ah / ac, 3) for ah, ac in zip(s_Ajh, s_Ajc)]
                for x in range(n):
                    travsc_t[x] = [ccsj[x] * val for val in travsc_t[x]]
                j += 1

            travsc = transpose(travsc_t)

            # ── Convergence check (both produced AND attracted) ──
            s_Ajc = col_sums(travsc)
            s_Pic = row_sums(travsc)

            is_converged = (GravitMod.comp(s_Ajh, s_Ajc)
                            and GravitMod.comp(s_Pih, s_Pic))

        travscrm = round_matrix(travsc)

        logger.info("Final rounded matrix: %s", travscrm)
        logger.info("Historical travels matrix: %s", travs)
        logger.debug("Produced passes i = %d", i)
        logger.debug("Attracted passes j = %d", j)
        logger.info("Exit iter_adj_in method.")

        return travscrm

    @staticmethod
    def iter_adj_wgt(travs: Matrix, travsc: Matrix, tlr: float = 0.01) -> Matrix:
        """
        Iteratively *weight*-adjust travels computed with the gravitational
        model.  Adjustments are proportional to each zone's share of total
        produced / attracted travels.

        Parameters
        ----------
        travs : Matrix
            Observed (historical) travel matrix.
        travsc : Matrix
            Initially computed travel matrix.
        tlr : float
            Convergence tolerance.

        Returns
        -------
        Matrix
            Adjusted, rounded travel matrix.
        """
        logger.info("Enter iter_adj_wgt method.")

        _validate_same_length(travs, travsc,
                              msg="travs and travsc must have the same dimensions.")

        n = len(travs)
        travsc = deep_copy_matrix(travsc)

        s_Pih = row_sums(travs)
        s_Ajh = col_sums(travs)
        travs_t = transpose(travs)

        # produced-share coefficients (weight of each cell within its row)
        c_Pi: Matrix = [[trav / P for trav in row]
                        for row, P in zip(travs, s_Pih)]

        # attracted-share coefficients (weight of each cell within its column)
        c_Aj: Matrix = [[trav / A for trav in col]
                        for col, A in zip(travs_t, s_Ajh)]

        is_converged = False
        i = 0
        j = 0

        while not is_converged:
            # ── Produced adjustment ──
            s_Pic = row_sums(travsc)

            if not GravitMod.comp(s_Pih, s_Pic):
                delta_P = [ph - pc for ph, pc in zip(s_Pih, s_Pic)]
                remind_P_flat = [c * d for cP, d in zip(c_Pi, delta_P)
                                 for c in cP]
                remind_P = flatten_to_matrix(remind_P_flat, n)
                travsP_flat = [rem + t
                               for remP, trav in zip(remind_P, travsc)
                               for rem, t in zip(remP, trav)]
                travsc = flatten_to_matrix(travsP_flat, n)
                i += 1

            # ── Attracted adjustment ──
            travsc_t = transpose(travsc)
            s_Ajc = [sum(col) for col in travsc_t]

            if not GravitMod.comp(s_Ajh, s_Ajc):
                delta_A = [ah - ac for ah, ac in zip(s_Ajh, s_Ajc)]
                remind_A_flat = [c * d for cA, d in zip(c_Aj, delta_A)
                                 for c in cA]
                remind_A = flatten_to_matrix(remind_A_flat, n)
                travsA_flat = [rem + t
                               for remA, trav in zip(remind_A, travsc_t)
                               for rem, t in zip(remA, trav)]
                travsc_t2 = flatten_to_matrix(travsA_flat, n)
                travsc = transpose(travsc_t2)
                j += 1

            # ── Convergence check ──
            s_Ajc = col_sums(travsc)
            s_Pic = row_sums(travsc)

            is_converged = (GravitMod.comp(s_Ajh, s_Ajc)
                            and GravitMod.comp(s_Pih, s_Pic))

        travscrm = round_matrix(travsc)

        logger.info("Final rounded matrix: %s", travscrm)
        logger.info("Historical travels matrix: %s", travs)
        logger.debug("Produced passes i = %d", i)
        logger.debug("Attracted passes j = %d", j)
        logger.info("Exit iter_adj_wgt method.")

        return travscrm

    @staticmethod
    def gravmod_fin(ffs: Matrix, k_ijs: Matrix,
                    P_is: list[float], A_js: list[float]) -> Matrix:
        """
        Compute future travels using the gravitational model.

        Parameters
        ----------
        ffs : Matrix
            Future friction-factor matrix.
        k_ijs : Matrix
            Calibration-coefficient matrix.
        P_is : list[float]
            Future produced-travel totals per zone.
        A_js : list[float]
            Future attracted-travel totals per zone.

        Returns
        -------
        Matrix
            Rounded future-travel matrix.
        """
        _validate_same_length(k_ijs, ffs, P_is, A_js,
                              msg="k_ijs, ffs, P_is and A_js must have "
                                  "the same number of zones.")

        n = len(k_ijs)
        gvals_fin: list[float] = []

        for i in range(n):
            pdsum = sum(aj * ff for aj, ff in zip(A_js, ffs[i]))
            for k1 in range(n):
                gvals_fin.append(
                    P_is[i] * ffs[i][k1] * A_js[k1] * k_ijs[i][k1] / pdsum
                )

        gvals_fin_r: list[float] = [float(round(v)) for v in gvals_fin]
        gvals_fin_m = flatten_to_matrix(gvals_fin_r, n)

        logger.info("Matrix of future rounded numbers: %s", gvals_fin_m)

        return gvals_fin_m

    @staticmethod
    def ccoeffs(gvalsradj: Matrix, travs: Matrix) -> Matrix:
        """
        Compute calibration coefficients for the gravitational model.

        Parameters
        ----------
        gvalsradj : Matrix
            Adjusted, rounded computed-travel matrix.
        travs : Matrix
            Observed (historical) travel matrix.

        Returns
        -------
        Matrix
            Matrix of calibration coefficients.
        """
        n = len(travs)
        ccoeffs_flat = [round(t_h / t_c, 2)
                        for row_h, row_c in zip(travs, gvalsradj)
                        for t_h, t_c in zip(row_h, row_c)]
        return flatten_to_matrix(ccoeffs_flat, n)

    @staticmethod
    def furness(travs: Matrix, P_is: list[float], A_js: list[float],
                tlr: float = 0.01) -> Matrix:
        """
        Iteratively compute future travel distribution using the Furness method.

        Parameters
        ----------
        travs : Matrix
            Observed (historical) or seed travel matrix.
        P_is : list[float]
            Future produced-travel totals per zone.
        A_js : list[float]
            Future attracted-travel totals per zone.
        tlr : float
            Convergence tolerance.

        Returns
        -------
        Matrix
            Adjusted, rounded travel matrix.
        """
        logger.info("Enter Furness method.")

        _validate_same_length(travs, P_is, A_js,
                              msg="travs, P_is and A_js must have the same "
                                  "number of zones.")

        n = len(travs)
        travsc = deep_copy_matrix(travs)

        is_converged = False
        i = 0  # produced passes
        j = 0  # attracted passes

        while not is_converged:
            # ── Produced adjustment ──
            s_Pic = row_sums(travsc)

            if not GravitMod.comp(P_is, s_Pic):
                ccs = [round(ps / pc, 3) for ps, pc in zip(P_is, s_Pic)]
                for x in range(n):
                    travsc[x] = [ccs[x] * val for val in travsc[x]]
                i += 1

            # ── Attracted adjustment ──
            travsc_t = transpose(travsc)
            s_Ajc = [sum(col) for col in travsc_t]

            if not GravitMod.comp(A_js, s_Ajc):
                ccs_a = [round(ats / ac, 3) for ats, ac in zip(A_js, s_Ajc)]
                for x in range(n):
                    travsc_t[x] = [ccs_a[x] * val for val in travsc_t[x]]
                j += 1

            travsc = transpose(travsc_t)

            # ── Convergence check ──
            s_Ajc = col_sums(travsc)
            s_Pic = row_sums(travsc)

            is_converged = (GravitMod.comp(A_js, s_Ajc)
                            and GravitMod.comp(P_is, s_Pic))

        travscrm = round_matrix(travsc)

        logger.info("Final rounded matrix, Furness: %s", travscrm)
        logger.info("Historical travels matrix: %s", travs)
        logger.debug("Produced passes i = %d", i)
        logger.debug("Attracted passes j = %d", j)
        logger.info("Exit Furness method.")

        return travscrm

    @staticmethod
    def fratar(travs: Matrix, P_is: list[float], A_js: list[float],
               tlr: float = 0.01, max_passes: int = 1) -> Matrix:
        """
        Iteratively compute future travel distribution using the Fratar method.

        Parameters
        ----------
        travs : Matrix
            Observed (historical) travel matrix.
        P_is : list[float]
            Future produced-travel totals per zone.
        A_js : list[float]
            Future attracted-travel totals per zone.
        tlr : float
            Convergence tolerance.
        max_passes : int
            Maximum number of Fratar computational passes.  The classic
            Fratar method is typically applied as a single-pass estimator
            (default ``1``); set higher to iterate until convergence.

        Returns
        -------
        Matrix
            Adjusted, rounded travel matrix.
        """
        logger.info("Enter Fratar method.")

        _validate_same_length(travs, P_is, A_js,
                              msg="travs, P_is and A_js must have the same "
                                  "number of zones.")

        n = len(travs)
        travsc = deep_copy_matrix(travs)

        is_converged = False
        p = 0  # passes counter

        while not is_converged:
            s_Pic = row_sums(travsc)
            travsc_t = transpose(travsc)
            s_Ajc = col_sums(travsc)

            is_cmp_flgA = GravitMod.comp(A_js, s_Ajc)
            is_cmp_flgP = GravitMod.comp(P_is, s_Pic)

            if (is_cmp_flgA and is_cmp_flgP) or p >= max_passes:
                is_converged = True
                break

            ccsi = [round(ps / pc, 3) for ps, pc in zip(P_is, s_Pic)]
            ccsj = [round(ats / ac, 3) for ats, ac in zip(A_js, s_Ajc)]

            travs_ij: list[float] = []
            travs_ji: list[float] = []

            for r in range(n):
                asum = sum(travsc[r][t] * ccsj[t] for t in range(n))
                for t in range(n):
                    travs_ij.append(travsc[r][t] * P_is[r] * ccsj[t] / asum)

                psum = sum(travsc_t[r][tt] * ccsi[tt] for tt in range(n))
                for t in range(n):
                    travs_ji.append(travsc[r][t] * A_js[t] * ccsi[r] / psum)

            travsc0 = [(ij + ji) / 2 for ij, ji in zip(travs_ij, travs_ji)]
            travsc = flatten_to_matrix(travsc0, n)

            p += 1

        travscrm = round_matrix(travsc)

        logger.info("Historical travels matrix: %s", travs)
        logger.debug("Passes p = %d", p)
        logger.info("Exit Fratar method.")

        return travscrm

    @staticmethod
    def average_gf(travs: Matrix, P_is: list[float], A_js: list[float],
                   tlr: float = 0.01) -> Matrix:
        """
        Iteratively compute future travel distribution using the average
        growth-factor method.

        Parameters
        ----------
        travs : Matrix
            Observed (historical) travel matrix.
        P_is : list[float]
            Future produced-travel totals per zone.
        A_js : list[float]
            Future attracted-travel totals per zone.
        tlr : float
            Convergence tolerance.

        Returns
        -------
        Matrix
            Adjusted, rounded travel matrix.
        """
        logger.info("Enter average_gf method.")

        _validate_same_length(travs, P_is, A_js,
                              msg="travs, P_is and A_js must have the same "
                                  "number of zones.")

        n = len(travs)
        travsc = deep_copy_matrix(travs)

        is_converged = False
        p = 0

        while not is_converged:
            s_Pic = row_sums(travsc)
            is_cmp_flgP = GravitMod.comp(P_is, s_Pic)

            if not is_cmp_flgP:
                ccsi = [round(ps / pc, 3) for ps, pc in zip(P_is, s_Pic)]

            s_Ajc = col_sums(travsc)
            is_cmp_flgA = GravitMod.comp(A_js, s_Ajc)

            if not is_cmp_flgA:
                ccsj = [round(ats / ac, 3) for ats, ac in zip(A_js, s_Ajc)]

            # update the travel matrix
            if not is_cmp_flgP or not is_cmp_flgA:
                travsc_interm: list[float] = []
                for x in range(n):
                    for t in range(n):
                        travsc_interm.append(
                            travsc[x][t] * (ccsi[x] + ccsj[t]) / 2
                        )
                travsc = flatten_to_matrix(travsc_interm, n)
                p += 1

            if is_cmp_flgP and is_cmp_flgA:
                is_converged = True

        travscrm = round_matrix(travsc)

        logger.info("Final rounded matrix, average_gf: %s", travscrm)
        logger.info("Historical travels matrix: %s", travs)
        logger.debug("Passes p = %d", p)
        logger.info("Exit average_gf method.")

        return travscrm

    @staticmethod
    def detroit(travs: Matrix, P_is: list[float], A_js: list[float],
                tlr: float = 0.01) -> Matrix:
        """
        Iteratively compute future travel distribution using the Detroit method.

        Parameters
        ----------
        travs : Matrix
            Observed (historical) travel matrix.
        P_is : list[float]
            Future produced-travel totals per zone.
        A_js : list[float]
            Future attracted-travel totals per zone.
        tlr : float
            Convergence tolerance.

        Returns
        -------
        Matrix
            Adjusted, rounded travel matrix.
        """
        logger.info("Enter Detroit method.")

        _validate_same_length(travs, P_is, A_js,
                              msg="travs, P_is and A_js must have the same "
                                  "number of zones.")

        n = len(travs)
        travsc = deep_copy_matrix(travs)

        is_converged = False
        p = 0

        while not is_converged:
            if p >= MAX_ITERATIONS:
                logger.warning("Detroit: hit MAX_ITERATIONS (%d) — stopping.",
                               MAX_ITERATIONS)
                break

            s_Pic = row_sums(travsc)
            is_cmp_flgP = GravitMod.comp(P_is, s_Pic)

            if not is_cmp_flgP:
                ccsi = [round(ps / pc, 3) for ps, pc in zip(P_is, s_Pic)]

            s_Ajc = col_sums(travsc)
            is_cmp_flgA = GravitMod.comp(A_js, s_Ajc)

            if not is_cmp_flgA:
                ccsj = [round(ats / ac, 3) for ats, ac in zip(A_js, s_Ajc)]

            # update the travel matrix
            if not is_cmp_flgP or not is_cmp_flgA:
                global_ratio = sum(P_is) / sum(s_Pic)
                travsc_interm: list[float] = []
                for x in range(n):
                    for t in range(n):
                        travsc_interm.append(
                            travsc[x][t] * (ccsi[x] * ccsj[t]) / global_ratio
                        )
                travsc = flatten_to_matrix(travsc_interm, n)
                p += 1

            if is_cmp_flgP and is_cmp_flgA:
                is_converged = True

        travscrm = round_matrix(travsc)

        logger.debug("Final rounded matrix, Detroit: %s", travscrm)
        logger.debug("Historical travels matrix: %s", travs)
        logger.debug("Passes p = %d", p)
        logger.info("Exit Detroit method.")

        return travscrm

    @staticmethod
    def iter_wgt_dmd(travs: Matrix, P_is: list[float], A_js: list[float],
                     tlr: float = 0.02) -> Matrix:
        """
        Iteratively compute future travel distribution using weighted
        coefficients.  Adjustments are proportional to each zone's share
        of total produced / attracted travels.

        Parameters
        ----------
        travs : Matrix
            Observed (historical) travel matrix.
        P_is : list[float]
            Future produced-travel totals per zone.
        A_js : list[float]
            Future attracted-travel totals per zone.
        tlr : float
            Convergence tolerance.

        Returns
        -------
        Matrix
            Adjusted, rounded travel matrix.
        """
        logger.info("Enter iter_wgt_dmd method.")

        # check with the future produced
        if len(travs) != len(P_is):
            msg = ("The travels matrix doesn't match with the future "
                   "produced! Please fix it.")
            logger.error(msg)
            raise ValueError(msg)

        # check with the future attracted
        if len(travs[0]) != len(A_js):
            msg = ("The travels matrix doesn't match with the future "
                   "attracted! Please fix it.")
            logger.error(msg)
            raise ValueError(msg)

        n = len(travs)
        travsc = deep_copy_matrix(travs)

        s_Pih = row_sums(travs)
        s_Ajh = col_sums(travs)
        travs_t = transpose(travs)

        # produced-share coefficients
        c_Pi: Matrix = [[trav / P for trav in row]
                        for row, P in zip(travs, s_Pih)]

        # attracted-share coefficients
        c_Aj: Matrix = [[trav / A for trav in col]
                        for col, A in zip(travs_t, s_Ajh)]

        is_converged = False
        i = 0
        j = 0

        while not is_converged:
            # ── Produced adjustment ──
            s_Pic = row_sums(travsc)

            if not GravitMod.comp(P_is, s_Pic):
                delta_P = [pis - pic for pis, pic in zip(P_is, s_Pic)]
                remind_P_flat = [c * d for cP, d in zip(c_Pi, delta_P)
                                 for c in cP]
                remind_P = flatten_to_matrix(remind_P_flat, n)
                travsP_flat = [rem + t
                               for remP, trav in zip(remind_P, travsc)
                               for rem, t in zip(remP, trav)]
                travsc = flatten_to_matrix(travsP_flat, n)
                i += 1

            s_Pic = row_sums(travsc)

            # ── Attracted adjustment ──
            travsc_t = transpose(travsc)
            s_Ajc = [sum(col) for col in travsc_t]

            if not GravitMod.comp(A_js, s_Ajc):
                delta_A = [ajs - ajc for ajs, ajc in zip(A_js, s_Ajc)]
                remind_A_flat = [c * d for cA, d in zip(c_Aj, delta_A)
                                 for c in cA]
                remind_A = flatten_to_matrix(remind_A_flat, n)
                travsA_flat = [rem + t
                               for remA, trav in zip(remind_A, travsc_t)
                               for rem, t in zip(remA, trav)]
                # travsc_t stays column-oriented (adjusted columns)
                travsc_t = flatten_to_matrix(travsA_flat, n)
                # travsc becomes the row-oriented form
                travsc = transpose(travsc_t)
                j += 1

            # ── Convergence check ──
            # travsc_t[j] = column j  →  column sums = attracted
            s_Ajc = [sum(col) for col in travsc_t]
            s_Pic = row_sums(travsc)

            is_converged = (GravitMod.comp(A_js, s_Ajc)
                            and GravitMod.comp(P_is, s_Pic))

        travscrm = round_matrix(travsc)

        logger.debug("Produced passes i = %d", i)
        logger.debug("Attracted passes j = %d", j)
        logger.info("Final rounded matrix, weighted: %s", travscrm)
        logger.info("Historical travels matrix: %s", travs)
        logger.info("Exit iter_wgt_dmd method.")

        return travscrm


# ── Main entry point ────────────────────────────────────────────────────────

def main() -> None:
    """Run all transport-demand estimation methods and print results."""

    gvalsr = GravitMod.gravmod_init(TRAVS, FFS, K_IJ0)

    n = len(TRAVS)
    gvalsr_m = flatten_to_matrix(
        [v for row in gvalsr for v in row], n
    )
    logger.info("****")
    logger.info("gvalsr matrix: %s\n", gvalsr_m)

    gvalsadjA = GravitMod.iter_adj_in(TRAVS, gvalsr)

    ccoeffsA = GravitMod.ccoeffs(gvalsadjA, TRAVS)

    gvalsr_finAm = GravitMod.gravmod_fin(FFS_F, ccoeffsA, P_IS, A_JS)
    gvalsr_finAf = GravitMod.furness(gvalsr_finAm, P_IS, A_JS)

    logger.info("")
    logger.info("****")
    logger.info("Future demand estimation via gravitational model A final: %s",
                gvalsr_finAf)

    travsc_furn = GravitMod.furness(TRAVS, P_IS, A_JS)
    logger.info("")
    logger.info("****")
    logger.info("Travels with Furness method: %s\n", travsc_furn)

    travs_frat = GravitMod.fratar(TRAVS, P_IS, A_JS)
    logger.info("")
    logger.info("****")
    logger.info("Raw Fratar method result: %s", travs_frat)
    logger.info("Matrix of travels obtained with Fratar (Furness corrected): %s\n",
                GravitMod.furness(travs_frat, P_IS, A_JS))

    travsc_avgf = GravitMod.average_gf(TRAVS, P_IS, A_JS)
    logger.info("****")
    logger.info("Matrix of travels obtained with average growth factor: %s\n",
                travsc_avgf)

    travsc_detr = GravitMod.detroit(TRAVS, P_IS, A_JS)
    logger.info("****")
    logger.info("Matrix of travels obtained with Detroit method: %s\n",
                travsc_detr)

    travsc_wgtd_prod = GravitMod.iter_wgt_dmd(TRAVS, P_IS, A_JS)
    logger.info("****")
    logger.info("Matrix of travels (produced) obtained with weighted "
                "coefficients: %s\n", travsc_wgtd_prod)

    # check weighted estimation starting with attracted (transposed input)
    travs_t = transpose(TRAVS)
    logger.info("Transpose travel matrix: %s", travs_t)
    logger.info("Travel matrix: %s\n", TRAVS)

    travsc_wgtd_at = GravitMod.iter_wgt_dmd(travs_t, A_JS, P_IS)

    travsc_wgtd_atL: Matrix = transpose(travsc_wgtd_at)
    logger.info("Matrix of travels (attracted) obtained with weighted "
                "coefficients: %s\n", travsc_wgtd_atL)

    logger.info("****")
    travsc_wgtd_aver: list[float] = []
    for r1, r2 in zip(travsc_wgtd_prod, travsc_wgtd_atL):
        for t1, t2 in zip(r1, r2):
            travsc_wgtd_aver.append(round((t1 + t2) / 2, 0))
    travsc_wgtd_aver_m = flatten_to_matrix(travsc_wgtd_aver, n)
    logger.info("Matrix of travels (averaged) obtained with weighted "
                "coefficients: %s", travsc_wgtd_aver_m)


if __name__ == "__main__":
    main()
