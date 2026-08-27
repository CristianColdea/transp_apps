"""
Script to compute the individual traveler utilities 
and the travels weights using LOGIT model.
The weights are to be applied on the gravitational model
table of travels.
"""

from __future__ import annotations

import logging
from math import e

logging.basicConfig(
    level=logging.DEBUG,  # switch to logging.INFO to silence the .debug() calls
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

# a square matrix of numbers, e.g. rows/columns indexed by zone.
# Travel counts are logically integers, but they get mixed into float
# division throughout this script (friction factors, coefficients, etc.),
# so everything numeric is typed as float here. list[int] would look more
# accurate for "number of travels", but Python's generic containers are
# invariant, so a list[list[int]] can't be passed where list[list[float]]
# is expected -- it's simpler and equally correct to use float everywhere.
# And took some time to understand this subtlety ...
Matrix = list[list[float]]

# number of travels as a matrix with produced travels on lines
# and attracted travels on columns

TRAVS: Matrix = [[40, 110, 150],
                 [50, 20, 30],
                 [110, 30, 10]]

# the matching friction factors, same arrangement

FFS: Matrix = [[0.753, 1.597, 0.753],
               [0.987, 0.753, 0.765],
               [1.597, 0.765, 0.753]]

# neutral calibration coefficients

K_IJ0: Matrix = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]

# auto travels cost

TCA: Matrix = [[0.5, 1, 1.4],
               [1.2, 0.8, 1.2],
               [1.7, 1.5, 0.7]]

# transit travels cost

TCT: Matrix = [[1, 1.5, 2],
               [1.8, 1.2, 1.9],
               [1.7, 1.5, 0.7]]

# auto travels duration

TDA: Matrix = [[3, 12, 7],
               [13, 3, 19],
               [9, 16, 4]]

# transit travels duration

TDT: Matrix = [[15, 5, 12],
               [15, 6, 26],
               [20, 21, 8]]

# the future friction factors

FFS_F: Matrix = [[0.753, 0.987, 1.597],
                  [0.987, 0.753, 0.765],
                  [1.597, 0.765, 0.753]]

# the future produced travels

P_IS: list[float] = [750, 580, 480]

# the future attracted travels

A_JS: list[float] = [722, 786, 302]

class GravitMod:
    def __init__(self, travs: Matrix, ffs: Matrix, k_ijs: Matrix,
                 P_is: list[float], A_js: list[float]) -> None:
        self.travs = travs
        self.ffs = ffs
        self.k_ijs = k_ijs
        self.P_is = P_is
        self.A_js = A_js

    @staticmethod
    def gravmod_init(travs: Matrix, ffs: Matrix,
                      k_ijs: Matrix) -> Matrix:
        """
        Method to compute gravitational model values in order to determine the
        calibration factors.
        Takes as input the travels, friction factors and calibration
        coefficients matrices.
        Returns a matrix with the computed travels.
        """
    
        # check if the matrices have the same shape
        if(len(travs) != len(ffs) or (len(travs) != len(k_ijs))):
            logger.error("The matrices doesn't match. Please fix it.")
            exit()
    
        # transpose de matrices
        travs_tt = list(zip(*travs))
        ffs_tt = list(zip(*ffs))
        travs_t = [list(sublist) for sublist in travs_tt]
        ffs_t = [list(sublist) for sublist in ffs_tt]
            
        # get attracted travels sums (cycling on transposes)
        s_Aj: list[float] = []   # store the attracted sums

        for col in travs_tt:
            s_Aj.append(sum(col))

        # get produced travels sums
        s_Pi: list[float] = []
        for row in travs:
            s_Pi.append(sum(row))
    
        # compute travels with gravitational model
        gvals_init: list[float] = []    # to store computed values
        for i in range(len(travs)):
            pdsum = 0.0
            for j1, j2 in zip(s_Aj, ffs[i]):
                pdsum = pdsum + j1 * j2
                # print(pdsum)
                # print(ffs[i])
            for k1 in range(len(ffs[i])):
                gvals_init.append((s_Pi[i] * ffs[i][k1] * s_Aj[k1] * k_ijs[i][k1] /
                                   pdsum))

        #print("Initial travels obtained with gravitational model, ", gvals_init)

        # check raw produced travels
        gvals_init_m0 = [gvals_init[i:i + 3] for i in range(0, len(gvals_init), 3)]
        vals_r = []
        for r in gvals_init_m0:
            r_r = [round(el, 2) for el in r]
            vals_r.append(r_r)
        logger.debug("Initial travels matrix (trimed) is, %s", vals_r)

    
        # for p1, p2 in zip(travs, gvals_init_m0):
            #print(round(sum(p1)) == round(sum(p2)))
            #print(sum(p2))

        # round the number of travels
        gvals_init_r: list[float] = []
        for val in gvals_init:
            gvals_init_r.append(round(val))
    
        # group flatten list 'gvals_init_r' as a matrix
        gvals_init_m = [gvals_init_r[i:i + 3] for i in range(0,
                         len(gvals_init_r), 3)]
        # print(gvals_init_m)
        logger.debug("Matrix of rounded numbers, %s", gvals_init_m)

     
        # check attracted travels sum
        # transpose the matrix first
        gvals_init_m_tt = list(zip(*gvals_init_m))
    
        return gvals_init_m

    @staticmethod
    def iter_adj_in(travs: Matrix, travsc: Matrix,
                     tlr: float = 0.01) -> Matrix:

        """
        Method to iteratively adjust travels computed with gravitational model.
        Takes as input the observed (historical) travels,the computed
        ones, in the form of matrices and the precision (tolerance) of
        adjustment.
        Returns a matrix with adjusted travels.
        """

        logger.debug("")
        logger.debug("Enter iter_adj_in method.")
    
        # check if the matrices have the same shape
        if(len(travs) != len(travsc)):
            logger.error("The matrices doesn't match. Please fix it.")
            exit()
        
        # function to compare the produced, respectively attracted travels
        # within a certain tolerance
        def comp(s_ih: list[float], s_ic: list[float], tlr: float) -> bool:
            """
            Function within method to compare two values, within tolerance.
            Takes as inputs the lists of to be compared values
            and the precision/tolerance.
            Returns True of False.
            """
            
            # set a flag
            flag = True

            for ih, ic in zip(s_ih, s_ic):
                if(abs(ih - ic) / ih >= tlr): 
                    flag = False
                    break

            return flag
        
        # get produced travels sums on observed travels
        s_Pih: list[float] = []
        for row in travs:
            s_Pih.append(sum(row))

        # get attracted travels sums on observed travels (cycling on transposes)
        s_Ajh: list[float] = []   # store the attracted sums

        travs_tt = list(zip(*travs))    # transpose the matrix hist travs

        for col in travs_tt:
            s_Ajh.append(sum(col))

        cmp_flg = False  # comparison flag to govern the following cycle
        i = 0   # produced passes counter
        j = 0   # attracted passes counter

        # rounded, flattened values from the last pass; declared here so it
        # has a value even if the while loop below never runs
        travscr: list[float] = []

        while(cmp_flg == False):
                       
            # get produced travels sums on computed travels
            s_Pic: list[float] = []
            for row in travsc:
                s_Pic.append(sum(row))
            
                        
            cmp_flg = comp(s_Pih, s_Pic, tlr)
            
            if (comp(s_Pih, s_Pic, tlr) == False):
                ccsi: list[float] = []   # list to store produced travels coefficients
                for ph, pc in zip(s_Pih, s_Pic):
                    ccsi.append(round(ph/pc, 3))

                logger.debug("travsc, %s", travsc)
                logger.debug("coefficients on produced travels, %s", ccsi)

                for x in range(len(travsc)):
                    travsc[x] = [ccsi[x]*val for val in travsc[x]]
            
                i += 1

            # *********
            # working on attracted travels

            # transpose de matrices (as a list of lists, not tuples, since
            # the rows get reassigned in place further down)
            travsc_tt: Matrix = [list(col) for col in zip(*travsc)]

            # travs_t = [list(sublist) for sublist in travs_tt]
            # travsc_t = [list(sublist) for sublist in travsc_tt]

                                
            # get attracted travels sums on computed travels (cycling on transposes)
            s_Ajc: list[float] = []   # store the attracted sums

            for ccol in travsc_tt:
                s_Ajc.append(sum(ccol))
            
                        
            if (comp(s_Ajh, s_Ajc, tlr) == False):
                ccsj: list[float] = []   # list to store attracted travels coefficients
                for ah, ac in zip(s_Ajh, s_Ajc):
                    ccsj.append(round(ah/ac, 3))

                logger.debug("travsc, %s", travsc)
                logger.debug("coefficients on attracted travels, %s", ccsj)

                for x in range(len(travsc_tt)):
                    travsc_tt[x] = [ccsj[x]*val for val in travsc_tt[x]]
            
                j += 1

               
            travsc = [list(row) for row in zip(*travsc_tt)]
            
            # update the attracted sums
                
            # get attracted travels sums on new computed travels (cycling on transposes)
            s_Ajc.clear()   # clear the computed attracted sums

            for ccol in travsc_tt:
                s_Ajc.append(sum(ccol))

            # update the produced sums
            s_Pic.clear()

            for row in travsc:
                s_Pic.append(sum(row))

            
            cmp_flg = comp(s_Ajh, s_Ajc, tlr)
                        
            cmp_flg = comp(s_Pih, s_Pic, tlr)
            
            travscr = []     # list to store rounded values, flatten form
            for row in travsc:
                for val in row:
                    travscr.append(round(val))

        travscrm = [travscr[i:i + 3] for i in range(0, len(travscr), 3)]
            
        logger.info("Final rounded matrix, %s", travscrm)
        logger.debug("Historical travels matrix, %s", travs)
        logger.debug("i is, %s", i)
        logger.debug("j is, %s", j)
        logger.debug("Exit iter_adj_in method.")
        
        return travscrm

    # method to compute gravitational model travels projected into the future
    @staticmethod
    def gravmod_fin(ffs: Matrix, k_ijs: Matrix, P_is: list[float],
                     A_js: list[float]) -> list[float]:
        """
        Method to compute future travels using gravitational model.
        Takes as inputs the future friction factors matrix, the previously
        computed calibration coefficients matrix, the matrix of produced
        travels and the matrix of attracted travels.
        Returns a matrix with future travels determined with gravitational
        model.
        """
        
        logger.debug("")
        logger.debug("Enter the gravitmod final method.")
        # check if the matrices have the same shape
        if(len(k_ijs) != len(ffs) or (len(k_ijs) != len(P_is)) or \
           (len(k_ijs) != len(A_js))):
            logger.error("The matrices doesn't match. Please fix it.")
            exit()
    
        # compute travels with gravitational model
        gvals_fin: list[float] = []    # to store computed values
        for i in range(len(k_ijs)):
            pdsum = 0.0
            for j1, j2 in zip(A_js, ffs[i]):
                pdsum = pdsum + j1 * j2
                #print("pdsum fin, ", pdsum)
                #print("ffs[i] fin, ", ffs[i])
                # print("A_j, ", j1)
            for k1 in range(len(ffs[i])):
                gvals_fin.append((P_is[i] * ffs[i][k1] * A_js[k1] * k_ijs[i][k1] / pdsum))

        logger.debug("Future travels obtained with gravitational model, %s", gvals_fin)

        # check raw produced travels
        gvals_fin_m0 = [gvals_fin[i:i + 3] for i in range(0, len(gvals_fin), 3)]
        # print(gvals_fin_m0)

    
        # round the number of travels
        gvals_fin_r: list[float] = []
        for val in gvals_fin:
            gvals_fin_r.append(round(val))
    
        logger.debug("Rounded number of future travels, %s", gvals_fin_r)
        logger.info("Total travels sum, %s", sum(gvals_fin_r))

        # group flatten list 'gvals_fin_r' as a matrix
        gvals_fin_m = [gvals_fin_r[i:i + 3] for i in range(0,
                         len(gvals_fin_r), 3)]

        logger.debug("Matrix of future rounded numbers, %s", gvals_fin_m)

        return gvals_fin_r

        
    @staticmethod
    def ccoeffs(gvalsradj: Matrix, travs: Matrix) -> Matrix:
        # compute calibration coefficients
        ccoeffs: list[float] = []
        for row_h, row_c in zip(travs, gvalsradj):
            for t_h, t_c in zip(row_h, row_c):
                ccoeffs.append(round(t_h / t_c, 2))
        ccoeffs_m = [ccoeffs[i:i + 3] for i in range(0, len(ccoeffs), 3)]

        return ccoeffs_m

# function for modal option
def modopt(tca: Matrix, tct: Matrix, tda: Matrix,
           tdt: Matrix) -> tuple[list[float], list[float]]:
    """
    Function to compute modal option, i.e. auto and transit.
    Takes as inputs the matrices of travels cost, auto and transit, and
    duration, respectively.
    Returns the weights of auto and transit travels for each zone to zone
    combination.
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

# function to compute auto and transit weight, from to each zone
def logit(u_a: list[float],
          u_t: list[float]) -> tuple[list[float], list[float]]:
    """
    Function to compute travels weights for each zone.
    Takes as inputs the utilities lists, auto and transit.
    Returns auto and transit weights for each zone to zone combination.
    """
    w_a: list[float] = []    # store auto weights
    w_t: list[float] = []    # store transit weights
    for ua_i, ut_i in zip(u_a, u_t):
        w_i = e**ua_i / (e**ua_i + e**ut_i)
        # print(w_i)
        w_i = round(w_i, 2)
        # print(w_i)
        w_a.append(w_i)
        w_t.append(round(1-w_i, 2))

    logger.debug("Auto travels weights, %s", w_a)
    logger.debug("Trasit travels weights, %s", w_t)

    return (w_a, w_t)

gvalsr = GravitMod.gravmod_init(travs, ffs, k_ij0)
# print("gvalsr is, ", gvalsr)

# gvalsr_m = [gvalsr[i:i + 3] for i in range(0, len(gvalsr), 3)]

gvalsadjA = GravitMod.iter_adj_in(travs, gvalsr)
# gvalsadjB = GravitMod.iter_adj_wgt(travs, gvalsr)

ccoeffsA = GravitMod.ccoeffs(gvalsadjA, travs)
# ccoeffsB = GravitMod.ccoeffs(gvalsadjB, travs)

# travsc_wgtd = GravitMod.iter_wgt_dmd(travs, P_is, A_js)
# print()
# print("Matrix of travels obtained with weighted coefficients is, ",
#       travsc_wgtd)

logger.info("Adjusted matrix A, %s", gvalsadjA)
logger.info("Calibration coefficients matrix A, %s", ccoeffsA)

# print("ccoeffsA, ", ccoeffsA)

# ccoeffs_it = GravitMod.ccoeffs(gvalsradj_it, travs)

# print("ccoeffs_it, ", ccoeffs_it)

ccoeffs_m = [ccoeffsA[i:i + 3] for i in range(0, len(ccoeffsA), 3)]
logger.debug("Calibration coefficients, %s", ccoeffs_m)

gvalsr_fin = GravitMod.gravmod_fin(ffs_f, ccoeffsA, P_is, A_js)

# print("Future number of rounded travels, ", gvalsr_fin)

u_a, u_t = modopt(tca, tct, tda, tdt)

w_a, w_t = logit(u_a, u_t)

# The adjusted travels, gravitational model
travs_adj_fin: Matrix = [[114, 375, 244],
                         [298, 240, 50],
                         [310, 171, 8]]

# applying weights to travels
travs_adj_fin_flatten: list[float] = []
for row in travs_adj_fin:
    for travel in row:
        travs_adj_fin_flatten.append(travel)

logger.debug("flatten final travels, %s", travs_adj_fin_flatten)

travels_auto: list[float] = []
for weight, travel in zip(w_a, travs_adj_fin_flatten):
    travels_auto.append(round(weight * travel, 0))

logger.info("Auto travels, %s", travels_auto)
travels_transit: list[float] = []
for total, auto in zip(travs_adj_fin_flatten, travels_auto):
    travels_transit.append(total - auto)

logger.info("Transit travels, %s", travels_transit)

