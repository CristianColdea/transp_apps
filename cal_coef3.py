"""
Script to compute the individual traveler utilities 
and the travels weights using LOGIT model.
The weights are to be applied on the gravitational model
table of travels.

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
from math import e


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


def comp(s_ih: list[float], s_ic: list[float], tlr: float = 0.05) -> bool:
    """
    Check whether two value-lists are within relative *tlr* tolerance.

    Returns ``True`` if every pair is within tolerance, ``False`` otherwise.
    """
    for ih, ic in zip(s_ih, s_ic):
        if abs(ih - ic) / ih >= tlr:
            return False
    return True

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

# the adjusted future travels, gravitational model
TRAVS_ADJ_FIN: Matrix = [[114, 375, 244],
                         [298, 240, 50],
                         [310, 171, 8]]

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

            if not comp(s_Pih, s_Pic):
                ccsi = [round(ph / pc, 3) for ph, pc in zip(s_Pih, s_Pic)]
                for x in range(n):
                    travsc[x] = [ccsi[x] * val for val in travsc[x]]
                i += 1

            # ── Attracted adjustment ──
            travsc_t = transpose(travsc)
            s_Ajc = [sum(col) for col in travsc_t]

            if not comp(s_Ajh, s_Ajc):
                ccsj = [round(ah / ac, 3) for ah, ac in zip(s_Ajh, s_Ajc)]
                for x in range(n):
                    travsc_t[x] = [ccsj[x] * val for val in travsc_t[x]]
                j += 1

            travsc = transpose(travsc_t)

            # ── Convergence check (both produced AND attracted) ──
            s_Ajc = col_sums(travsc)
            s_Pic = row_sums(travsc)

            is_converged = (comp(s_Ajh, s_Ajc)
                            and comp(s_Pih, s_Pic))

        travscrm = round_matrix(travsc)

        logger.info("Final rounded matrix: %s", travscrm)
        logger.info("Historical travels matrix: %s", travs)
        logger.debug("Produced passes i = %d", i)
        logger.debug("Attracted passes j = %d", j)
        logger.info("Exit iter_adj_in method.")

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



# ── Modal option functions section ───────────────────────────────────────────

# function for modal option
def modopt(tca: Matrix, tct: Matrix, tda: Matrix,
           tdt: Matrix) -> tuple[list[float], list[float]]:
    """
    Function to compute modal option, i.e. auto and transit.

    Parameters
    ----------
    tca : Matrix
        Matrix of auto travel costs.
    tct : Matrix
        Matrix of transit travel costs.
    tda : Matrix
        Matrix of auto travel durations.
    tdt : Matrix
        Matrix of transit travel durations.

    Returns
    -------
        The travel utilities of auto and transit travels for each zone to zone.
    """

    # compute utilities for auto and transit modes
    u_a: list[float] = []    # store auto utility results
    u_t: list[float] = []    # store transit utility results

    for i in range(len(tca)):
        for ca, da in zip(tca[i], tda[i]):
            u_a.append(round(2.5 - 0.5 * ca - 0.01 * da, 2))
        for ct, dt in zip(tct[i], tdt[i]):
            u_t.append(round(-0.4 * ct - 0.012 * dt, 2))

    logger.debug("Auto utilities, %s", u_a)
    logger.debug("Transit utilities, %s", u_t)

    return (u_a, u_t)

# function to compute auto and transit proportions, from to each zone
def logit(u_a: list[float],
          u_t: list[float]) -> tuple[list[float], list[float]]:
    """
    Function to compute travels proportions for each zone.

    Parameters
    ----------
    u_a : list[float]
        List of auto travel utilities.
    u_t : List[float]
        List of transit travel utilities.

    Returns
    -------
        Auto and transit proportions for each zone to zone combination.
    """

    w_a: list[float] = []    # store auto weights
    w_t: list[float] = []    # store transit weights

    for ua_i, ut_i in zip(u_a, u_t):
        w_i = e**ua_i / (e**ua_i + e**ut_i)
        w_i = round(w_i, 2)
        w_a.append(w_i)
        w_t.append(round(1-w_i, 2))

    logger.debug("Auto travels weights, %s", w_a)
    logger.debug("Trasit travels weights, %s", w_t)

    return (w_a, w_t)


# ── Main entry point ────────────────────────────────────────────────────────

def main() -> None:
    """Run all transport-demand estimation methods and print results."""

    gvalsr = GravitMod.gravmod_init(TRAVS, FFS, K_IJ0)

    gvals_adj = GravitMod.iter_adj_in(TRAVS, gvalsr)

    ccoeffs = GravitMod.ccoeffs(gvals_adj, TRAVS)

    logger.info("Adjusted matrix, %s", gvals_adj)
    logger.info("Calibration coefficients matrix, %s", ccoeffs)

    n = len(TRAVS)
    ccoeffs_m = flatten_to_matrix(
            [v for row in ccoeffs for v in row], n
    )

    logger.debug("Calibration coefficients, %s", ccoeffs_m)

    gvalsr_fin = GravitMod.gravmod_fin(FFS_F, ccoeffs, P_IS, A_JS)

    u_a, u_t = modopt(TCA, TCT, TDA, TDT)

    logger.info("Auto travels utilities, %s", u_a)
    logger.info("Transit travels utilities, %s", u_t)

    w_a, w_t = logit(u_a, u_t)

    logger.info("Auto travels proportions, %s", w_a)
    logger.info("Transit travels proportions, %s", w_t)

    # applying weights to travels
    travs_adj_fin_flatten: list[float] = []
    for row in gvalsr_fin:
        for travel in row:
            travs_adj_fin_flatten.append(travel)

    logger.debug("Flatten final travels, %s", travs_adj_fin_flatten)

    travels_auto: list[float] = []
    for weight, travel in zip(w_a, travs_adj_fin_flatten):
        travels_auto.append(round(weight * travel, 0))

    logger.info("Auto travels, %s", travels_auto)
    travels_transit: list[float] = []
    for total, auto in zip(travs_adj_fin_flatten, travels_auto):
        travels_transit.append(total - auto)

    logger.info("Transit travels, %s", travels_transit)


if __name__ == "__main__":
    main()
