"""
Script to compute short term transport demand.
Six methods will be coded: gravitational model (for reference),
Furness, Fratar, Detroit, average growth factor and a new one.
This script is going to be rewriten towards a better
and cleaner code, Method decomposition to be heavily used.
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

"""
INITIAL DATA SECTION
"""

# number of travels as a matrix with produced travels on lines
# and attracted travels on columns

travs: Matrix = [[40, 110, 150],
                 [50, 20, 30],
                 [110, 30, 10]]

# the matching friction factors, same arrangement

ffs: Matrix = [[0.753, 1.597, 0.753],
               [0.987, 0.753, 0.765],
               [1.597, 0.765, 0.753]]

# neutral calibration coefficients

k_ij0: Matrix = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]

# auto travels cost

tca = [[0.5, 1, 1.4],
       [1.2, 0.8, 1.2],
       [1.7, 1.5, 0.7]]

# transit travels cost

tct: Matrix = [[1, 1.5, 2],
               [1.8, 1.2, 1.9],
               [1.7, 1.5, 0.7]]

# auto travels duration

tda = [[3, 12, 7],
       [13, 3, 19],
       [9, 16, 4]]

# transit travels duration

tdt: Matrix = [[15, 5, 12],
               [15, 6, 26],
               [20, 21, 8]]

# the future friction factors

ffs_f: Matrix = [[0.753, 0.987, 1.597],
                 [0.987, 0.753, 0.765],
                 [1.597, 0.765, 0.753]]

# the future produced travels

P_is: list[float] = [750, 580, 480]

# the future attracted travels

A_js: list[float] = [722, 786, 302]



"""
GRAVITATIONAL MODEL CLASS
"""


class GravitMod:
    def __init__(self, travs: Matrix, ffs: Matrix, 
                 k_ijs: Matrix, P_is: list[float], A_js: list[float]):
        self.travs = travs
        self.ffs = ffs
        self.k_ijs = k_ijs
        self.P_is = P_is
        self.A_js = A_js


    @staticmethod
    def transp_mat(mat: Matrix):
        """
        Method to transpose a squared matrix.
        Takes as input the matrix, returns the transpose, same shape.
        """
        # check for same number of rows and columns (square matrix)
        if(len(mat) != len(mat[0])):
            print("The matrix is not squared. Please provide a squared matrix.")
            exit()

        # transpose the matrix
        transp = list(zip(*mat))

        return [list(sublist) for sublist in transp]


    @staticmethod
    def gravmod_init(travs: Matrix,ffs: Matrix, k_ijs: Matrix):
        """
        Method to compute gravitational model values in order to determine the
        calibration factors.
        Takes as input the travels, friction factors and neutral calibration
        coefficients matrices.
        Returns a matrix with the computed travels.
        """
    
        # check if the matrices have the same shape
        if(len(travs) != len(ffs) or (len(travs) != len(k_ijs))):
            print("The matrices doesn't match. Please fix it.")
            exit()
    
        # transpose de matrices
        travs_tt = list(zip(*travs))
        ffs_tt = list(zip(*ffs))
        travs_t = [list(sublist) for sublist in travs_tt]
        ffs_t = [list(sublist) for sublist in ffs_tt]
            
        # get attracted travels sums (cycling on transposes)
        s_Aj = []   # store the attracted sums

        for item in travs_tt:
            s_Aj.append(sum(item))

        # get produced travels sums
        s_Pi = []
        for item in travs:
            s_Pi.append(sum(item))
    
        # compute travels with gravitational model
        gvals_init = []    # to store computed values
        for i in range(len(travs)):
            pdsum = 0
            for j1, j2 in zip(s_Aj, ffs[i]):
                pdsum = pdsum + j1 * j2
                # print(pdsum)
                # print(ffs[i])
            for k1 in range(len(ffs[i])):
                gvals_init.append((s_Pi[i] * ffs[i][k1] * s_Aj[k1] * k_ijs[i][k1] /
                                   pdsum))

        # check raw produced travels
        gvals_init_m0 = [gvals_init[i:i + 3] for i in range(0, len(gvals_init), 3)]
        #print("Initial travels matrix is ,", gvals_init_m0)

    
        # for p1, p2 in zip(travs, gvals_init_m0):
            #print(round(sum(p1)) == round(sum(p2)))
            #print(sum(p2))

        # round the number of travels
        gvals_init_r = []
        for item in gvals_init:
            gvals_init_r.append(round(item))
    
        #print("Rounded number of initial travels, ", gvals_init_r)

        # group flatten list 'gvals_init_r' as a matrix
        gvals_init_m = [gvals_init_r[i:i + 3] for i in range(0,
                         len(gvals_init_r), 3)]
        # print(gvals_init_m)
        # print("Matrix of rounded numbers, ", gvals_init_m)

        # check produced travels sum
        # for p1, p2 in zip(travs, gvals_init_m):
            # print(sum(p1) == sum(p2))

        # check attracted travels sum
        # transpose the matrix first
        gvals_init_m_tt = list(zip(*gvals_init_m))
        # print(gvals_init_m_tt, travs_tt)
        # for a1, a2 in zip(gvals_init_m_tt, travs_tt):
            # print(sum(a1) == sum(a2))
            # print(sum(a1))
            # print(sum(a2))

        return gvals_init_m

    @staticmethod
    def comp(s_ih: list[float], s_ic: list[float], tlr=0.05):
        """
        Method to compare two values, within tolerance.
        Takes as inputs the lists of to be compared values
        and the precision/tolerance.
        Returns True of False.
        """
            
        # set a flag
        is_within = True

        for ih, ic in zip(s_ih, s_ic):
            if(abs(ih - ic) / ih >= tlr): 
                is_within = False
                break

        return is_within


    @staticmethod
    def iter_adj_in(travs: Matrix, travsc: Matrix, tlr=0.01):
        """
        Method to iteratively adjust travels computed with gravitational model.
        Takes as input the observed (historical) travels,the computed
        ones, in the form of matrices and the precision (tolerance) of
        adjustment.
        Returns a matrix with adjusted travels.
        """

        # print()
        print("Enter iter_adj_in method.")
    
        # check if the matrices have the same shape
        if(len(travs) != len(travsc)):
            print("The matrices doesn't match. Please fix it.")
            exit()
        
                        
        # get produced travels sums on observed travels
        s_Pih = []
        for item in travs:
            s_Pih.append(sum(item))

        # get attracted travels sums on observed travels (cycling on transposes)
        s_Ajh = []   # store the attracted sums

        travs_tt = list(zip(*travs))    # transpose the matrix hist travs

        for item in travs_tt:
            s_Ajh.append(sum(item))

        is_cmp_flg = False  # comparison flag to govern the following cycle
        i = 0   # produced passes counter
        j = 0   # attracted passes counter

        while(is_cmp_flg == False):
                       
            # get produced travels sums on computed travels
            s_Pic = []
            for item in travsc:
                s_Pic.append(sum(item))
                        
            if (GravitMod.comp(s_Pih, s_Pic) == False):
                ccsi = []   # list to store produced travels coefficients
                for ph, pc in zip(s_Pih, s_Pic):
                    ccsi.append(round(ph/pc, 3))

                for x in range(len(travsc)):
                    travsc[x] = [ccsi[x]*item for item in travsc[x]]
            
                i += 1

            
            # *********
            # working on attracted travels

            # transpose de matrices
            travsc_tt = list(zip(*travsc))

            s_Ajc = []   # store the attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))
                        
            if (GravitMod.comp(s_Ajh, s_Ajc) == False):
                ccsj = []   # list to store attracted travels coefficients
                for ah, ac in zip(s_Ajh, s_Ajc):
                    ccsj.append(round(ah/ac, 3))

                for x in range(len(travsc_tt)):
                    travsc_tt[x] = [ccsj[x]*item for item in travsc_tt[x]]
            
                j += 1

            travsc_0 = list(zip(*travsc_tt))
            travsc = [list(sublist) for sublist in travsc_0]
            
            # update the attracted sums
                
            # get attracted travels sums on new computed travels (cycling on transposes)
            s_Ajc.clear()   # clear the computed attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))

            # update the produced sums
            s_Pic.clear()

            for item  in travsc:
                s_Pic.append(sum(item))

            is_cmp_flg = GravitMod.comp(s_Ajh, s_Ajc)
                        
            is_cmp_flg = GravitMod.comp(s_Pih, s_Pic)
            
            travscr = []     # list to store rounded values, flatten form
            for item in travsc:
                for item in item:
                    travscr.append(round(item))

        travscrm = [travscr[i:i + 3] for i in range(0, len(travscr), 3)]
            
        #print()
        print("Final rounded matrix, ", travscrm)
        print("Historical travels matrix, ", travs)
        print("i is, ", i)
        print("j is, ", j)
        print("Exit iter_adj_in method.")
        
        return travscrm

    @staticmethod
    def iter_adj_wgt(travs, travsc, tlr=0.01):

        """
        Method to iteratively weight adjust travels computed with 
        gravitational model. The main point is to operate adjustments
        according to the weight of each produced and attracted travels.
        Takes as input the observed (historical) travels,the computed
        ones, in the form of matrices and the precision (tolerance) of
        adjustment.
        Returns a matrix with adjusted travels.
        """

        # print()
        print("Enter iter_adj_wgt method")

        # check if the matrices have the same shape
        if(len(travs) != len(travsc)):
            print("The matrices doesn't match. Please fix it.")
            exit()
        
        # get produced travels sums on observed travels
        s_Pih = []
        for item in travs:
            s_Pih.append(sum(item))

        # get attracted travels sums on observed travels (cycling on transposes)
        s_Ajh = []   # store the attracted sums

        travs_tt = list(zip(*travs))    # transpose hist travels matrix

        for item in travs_tt:
            s_Ajh.append(sum(item))

        # generate the produced coefficients matrix (produced perspective)
        c_Pi0 = []
        for t, P in zip(travs, s_Pih):
            for trav in t:
                c_Pi0.append(trav/P)
        
        c_Pi = [c_Pi0[i:i + 3] for i in range(0, len(c_Pi0), 3)]

        # generate the attracted coefficients matrix (attracted perspective)
        c_Aj0= []
        for t, A in zip(travs_tt, s_Ajh):
            for trav in t:
                c_Aj0.append(trav/A)

        c_Aj = [c_Aj0[j:j + 3] for j in range(0, len(c_Aj0), 3)]

        is_cmp_flg = False  # comparison flag to govern the following cycle
        i = 0   # produced passes counter
        j = 0   # attracted passes counter

        while(is_cmp_flg == False):
                       
            # get produced travels sums on computed travels
            s_Pic = []
            for item in travsc:
                s_Pic.append(sum(item))
            
            if (GravitMod.comp(s_Pih, s_Pic) == False):
                delta_P = []    #list to store the deltas of produced travels
                for Pih, Pic in zip(s_Pih, s_Pic):
                    delta_P.append(Pih - Pic)
                remind_P = []    #matrix of additions to travels, produced
                for cP, delta_i in zip(c_Pi, delta_P):
                    for c in cP:
                        remind_P.append(c*delta_i)   #append weighted delta
                remind_P = [remind_P[i:i + 3] for i in range(0, len(remind_P), 3)]
                travsP = []   # list to store adjusted travels matrix, produced
                for remP, trav in zip(remind_P, travsc):
                    for rem, t in zip(remP, trav):
                        travsP.append(rem+t)

                travsc = [travsP[i:i + 3] for i in range(0, len(travsP), 3)]

                i += 1
            
            # working on attracted travels

            # transpose de matrix
            travsc_tt = list(zip(*travsc))

            # get attracted travels sums on computed travels (cycling on transposes)
            s_Ajc = []   # store the attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))
                        
            if (GravitMod.comp(s_Ajh, s_Ajc) == False):
                delta_A = []    #list to store the deltas of attracted travels
                for Ajh, Ajc in zip(s_Ajh, s_Ajc):
                    delta_A.append(Ajh - Ajc)
                remind_A = []    #matrix of additions to travels, atrracted
                for cA, delta_j in zip(c_Aj, delta_A):
                    for c in cA:
                        remind_A.append(c*delta_j)   #append weighted delta
                remind_A = [remind_A[j:j + 3] for j in range(0, len(remind_A), 3)]
                travsA = []   # list to store adjusted travels matrix, produced
                for remA, trav in zip(remind_A, travsc_tt):
                    for rem, t in zip(remA, trav):
                        travsA.append(rem+t)

                # getting the flatten list into a matrix
                travsc_t = [travsA[i:i + 3] for i in range(0, len(travsA), 3)]
                # transposing 'the transposed'
                travsc_tt = list(zip(*travsc_t))
                # obtain the matrix after iteration
                travsc = [list(sublist) for sublist in travsc_tt]
                
                j += 1

            # update the attracted sums
                
            # get attracted travels sums on new computed travels (cycling on transposes)
            s_Ajc.clear()   # clear the computed attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))

            # update the produced sums
            s_Pic.clear()

            for item  in travsc:
                s_Pic.append(sum(item))

            # print()
            # print("s_Pic, ", s_Pic)

            is_cmp_flgA = GravitMod.comp(s_Ajh, s_Ajc)
            
            is_cmp_flgP = GravitMod.comp(s_Pih, s_Pic)

            if(is_cmp_flgA == True and is_cmp_flgP == True):
                is_cmp_flg = True
            else:
                is_cmp_flg = False

            travscr = []     # list to store rounded values, flatten form
            for item in travsc:
                for item in item:
                    travscr.append(round(item))

        travscrm = [travscr[i:i + 3] for i in range(0, len(travscr), 3)]
            
        # print()
        print("Final rounded matrix, ", travscrm)
        print("Historical travels matrix, ", travs)
        print("i is, ", i)
        print("j is, ", j)
        print("Exit iter_adj_wgt method.")
        
        return travscrm

    # method to compute gravitational model travels projected into the future
    @staticmethod
    def gravmod_fin(ffs, k_ijs, P_is, A_js):
        """
        Method to compute future travels using gravitational model.
        Takes as inputs the future friction factors matrix, the previously
        computed calibration coefficients matrix, the matrix of produced
        travels and the matrix of attracted travels.
        Returns a matrix with future travels determined with gravitational
        model.
        """
        
        # check if the matrices have the same shape
        if(len(k_ijs) != len(ffs) or (len(k_ijs) != len(P_is)) or \
           (len(k_ijs) != len(A_js))):
            print("The matrices doesn't match. Please fix it.")
            exit()
    
        # compute travels with gravitational model
        gvals_fin = []    # to store computed values
        for i in range(len(k_ijs)):
            pdsum = 0
            for j1, j2 in zip(A_js, ffs[i]):
                pdsum = pdsum + j1 * j2
            for k1 in range(len(ffs[i])):
                gvals_fin.append((P_is[i] * ffs[i][k1] * A_js[k1] * k_ijs[i][k1] / pdsum))

        # check raw produced travels
        gvals_fin_m0 = [gvals_fin[i:i + 3] for i in range(0, len(gvals_fin), 3)]
        # print(gvals_fin_m0)

    
        # round the number of travels
        gvals_fin_r = []
        for item in gvals_fin:
            gvals_fin_r.append(round(item))
    
        # group flatten list 'gvals_fin_r' as a matrix
        gvals_fin_m = [gvals_fin_r[i:i + 3] for i in range(0,
                         len(gvals_fin_r), 3)]

        print("Matrix of future rounded numbers, ", gvals_fin_m)

        return gvals_fin_r

    @staticmethod    
    def ccoeffs(gvalsradj, travs):
        """
        Method to compute calibration coefficients for gravitational model.
        Takes as inputs the adjusted and rounded travels matrix initially
        computed with neutral calibration coefficients and original
        travels (historical) matrix.
        Returns the matrix of coefficients.
        """
        ccoeffs = []
        for row_h, row_c in zip(travs, gvalsradj):
            for t_h, t_c in zip(row_h, row_c):
                ccoeffs.append(round(t_h / t_c, 2))
        ccoeffs_m = [ccoeffs[i:i + 3] for i in range(0, len(ccoeffs), 3)]

        return ccoeffs_m


    @staticmethod
    def furness(travs, P_is, A_js, tlr=0.01):

        """
        Method to iteratively compute the future travels distribution using
        Furness method.
        Takes as input the observed (historical) travels in the form of squared
        matrix,the future produced and attracted ones, respectively, in the
        form of one-line matrices (one for produced and one for attracted)
        and the precision (tolerance) as indicated in technical literature.
        Returns a matrix with adjusted travels.
        """

        # print()
        print("Enter Furness method.")

        # check if the matrices have the same shape
        if(len(travs) != len(P_is) or len(travs) != len(A_js)):
            print("The matrices doesn't match. Please fix it.")
            exit()
           
        is_cmp_flg = False  # comparison flag to govern the following cycle
        i = 0   # produced passes counter
        j = 0   # attracted passes counter
        
        # allocate initial computed travels matrix
        travsc = []

        for row in travs:
            travsc.append(row)

        while(is_cmp_flg == False):
                       
            # get produced travels sums on computed travels
            s_Pic = []
            for item in travsc:
                s_Pic.append(sum(item))

            # print()
            # print("P_is, ", P_is)
            # print("s_Pic, ", s_Pic)
            
            is_cmp_flg = GravitMod.comp(P_is, s_Pic)
            # print(is_cmp_flg)

            if (is_cmp_flg == False):
                ccs = []   # list to store produced travels coefficients
                for ps, pc in zip(P_is, s_Pic):
                    ccs.append(round(ps/pc, 3))

                # print("coefficients on produced travels, ", ccs)

                for x in range(len(travsc)):
                    travsc[x] = [ccs[x]*item for item in travsc[x]]
            
                i += 1

                # print()
                # print("travsc after pass  i = ", i, "is ", travsc)
            
            
            # *********
            # working on attracted travels

            # clear the coefficients vector
            ccs.clear()

            # transpose de matrices
            travsc_tt = list(zip(*travsc))

            # print()
            # print("travs_t, ", travs_t)
            # print("travsc_t ", travsc_t)
    
                    
            # get attracted travels sums on computed travels (cycling on transposes)
            s_Ajc = []   # store the attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))
            
            is_cmp_flg = GravitMod.comp(A_js, s_Ajc)
            
            if (is_cmp_flg == False):
                for ats, ac in zip(A_js, s_Ajc):
                    ccs.append(round(ats/ac, 3))

                for x in range(len(travsc_tt)):
                    travsc_tt[x] = [ccs[x]*item for item in travsc_tt[x]]
            
                j += 1

            travsc_0 = list(zip(*travsc_tt))
            travsc = [list(sublist) for sublist in travsc_0]

            #print()
            #print("travsc,  ", travsc)
            
            # update the attracted sums
                
            # get attracted travels sums on new computed travels (cycling on transposes)
            s_Ajc.clear()   # clear the computed attracted sums
            ccs.clear()    # clear the coefficients vector

            for item in travsc_tt:
                s_Ajc.append(sum(item))

            # update the produced sums
            s_Pic.clear()

            for item  in travsc:
                s_Pic.append(sum(item))

            is_cmp_flgA = GravitMod.comp(A_js, s_Ajc)
            
            is_cmp_flgP = GravitMod.comp(P_is, s_Pic)

            if(is_cmp_flgA == True and is_cmp_flgP == True):
                is_cmp_flg = True
            else:
                is_cmp_flg = False

        
            travscr = []     # list to store rounded values, flatten form
            for item in travsc:
                for item in item:
                    travscr.append(round(item))

        travscrm = [travscr[i:i + 3] for i in range(0, len(travscr), 3)]
            
        print("Final rounded matrix, Furness, ", travscrm)
        print("Historical travels matrix, ", travs)
        print("i is, ", i)
        print("j is, ", j)
        print("Exit Furness method.")
        
        return travscrm


    @staticmethod
    def fratar(travs, P_is, A_js, tlr=0.01):
        """
        Method to iteratively compute the future travels distribution using
        Fratar method.
        Takes as input the observed (historical) travels in the form of squared
        matrix,the future produced and attracted ones, respectively, in the
        form of one-line matrices (one for produced and one for attracted)
        and the precision (tolerance).
        Returns a matrix with adjusted travels.
        """

        # print()
        print("Enter Fratar method.")

        # check if the matrices have the same shape
        if(len(travs) != len(P_is) or len(travs) != len(A_js)):
            print("The matrices doesn't match. Please fix it.")
            exit()
                 
        is_cmp_flg = False  # comparison flag to govern the following cycle
        
        p = 0   # passes counter
                
        # allocate initial computed travels matrix
        travsc = []

        for row in travs:
            travsc.append(row)

        # print("Current matrix is, ", travs)

        while(is_cmp_flg == False):
                       
            # get produced travels sums on current travels
            s_Pic = []
            for item in travsc:
                s_Pic.append(sum(item))
            
            # working on attracted travels

            # clear the coefficients vector
           
            # transpose de matrix
            travsc_tt = list(zip(*travsc))
                 
            # get attracted travels sums (cycling on transposes)
            s_Ajc = []   # store the attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))
            
            is_cmp_flgA = GravitMod.comp(A_js, s_Ajc)
            
            is_cmp_flgP = GravitMod.comp(P_is, s_Pic)

            if(is_cmp_flgA == True and is_cmp_flgP == True):
                is_cmp_flg = True
            else:
                is_cmp_flg = False

            if p == 1:
               is_cmp_flg = True

            if (is_cmp_flg == False):
                ccsi = []   # list to store produced travels growth factors
                for ps, pc in zip(P_is, s_Pic):
                    ccsi.append(round(ps/pc, 3))

                ccsj = []    # list to store attracted travels growth factors
                for ats, ac in zip(A_js, s_Ajc):
                    ccsj.append(round(ats/ac, 3))
                
                # matrix to store t_ij travels
                travs_ij = []

                # matrix to store t_ji travels
                travs_ji = []

                for r in range(len(travsc)):
                    asum = 0
                    psum = 0

                    # computing the sum of product between attracted and ccsj
                    for t in range(len(travsc[r])):
                        asum = asum + travsc[r][t] * ccsj[t]
                        
                    # computing the t_ij travels with Fratar formula
                    for t in range(len(travsc[r])):
                        t_ij = travsc[r][t] * P_is[r] * ccsj[t] / asum
                        travs_ij.append(t_ij)

                    # computing the sum of product between produced and ccsi
                    for tt in range(len(travsc_tt[r])):
                        psum = psum + travsc_tt[r][tt] * ccsi[tt]
                        # print("psum is, ", psum)
                    
                    # computing the t_ji travels with Fratar formula 
                    for tt in range(len(travsc[r])):
                        t_ji = travsc[r][tt] * A_js[tt] * ccsi[r] / psum
                        travs_ji.append(t_ji)

            travsc0 = []

            for ij, ji in zip(travs_ij, travs_ji):
                travsc0.append((ij+ji)/2)
        
            travsc.clear()
            travsc = [travsc0[x:x + 3] for x in range(0, len(travsc0), 3)]

                
            p += 1

        travscr = []     # list to store rounded values, flatten form
        for item in travsc:
            for item in item:
                travscr.append(round(item))

        travscrm = [travscr[i:i + 3] for i in range(0, len(travscr), 3)]

        print("Historical travels matrix, ", travs)
        print("p is, ", p)
        print("Exit Fratar method.")
        
        return travscrm

    @staticmethod
    def average_gf(travs, P_is, A_js, tlr=0.01):

        """
        Method to iteratively compute the future travels distribution using
        average growth method.
        Takes as input the observed (historical) travels in the form of squared
        matrix,the future produced and attracted ones, respectively, in the
        form of one-line matrices (one for produced and one for attracted)
        and the precision (tolerance).
        Returns a matrix with adjusted travels.
        """

        # print()
        print("Enter average_gf method.")

        # check if the matrices have the same shape
        if(len(travs) != len(P_is) or len(travs) != len(A_js)):
            print("The matrices doesn't match. Please fix it.")
            exit()
                 
        is_cmp_flg = False  # comparison flag to govern the following cycle
        p = 0   # passes counter
               
        # allocate initial computed travels matrix
        travsc = []

        for row in travs:
            travsc.append(row)
        
        while(is_cmp_flg == False):                       
            # get produced travels sums on computed travels
            s_Pic = []
            for item in travsc:
                s_Pic.append(sum(item))

            is_cmp_flgP = GravitMod.comp(P_is, s_Pic)

            if (is_cmp_flgP == False):
                ccsi = []   # list to store produced travels coefficients
                for ps, pc in zip(P_is, s_Pic):
                    ccsi.append(round(ps/pc, 3))

            # *********
            # working on attracted sums

            # transpose de matrix
            travsc_tt = list(zip(*travsc))
                    
            # get attracted travels sums on computed travels (cycling on transposes)
            s_Ajc = []   # store the attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))
            
            is_cmp_flgA = GravitMod.comp(A_js, s_Ajc)
            
            if (is_cmp_flgA == False):
                ccsj = []
                for ats, ac in zip(A_js, s_Ajc):
                    ccsj.append(round(ats/ac, 3))

            # updating the travels matrix
                       
            if(is_cmp_flgP == False or is_cmp_flgA == False):
                travsc_interm = []    #store the modified values
                for x in range(len(travsc)):
                    for t in range(len(travsc[x])):
                        travsc_interm.append(travsc[x][t] * (ccsi[x] + ccsj[t])/2)
                p += 1

                
            travsc = [travsc_interm[i:i + 3]
                      for i in range(0, len(travsc_interm), 3)]

            # recreate the current travels matrix
            if(is_cmp_flgP == True and is_cmp_flgA == True):
                is_cmp_flg = True
                  
        travscr = []     # list to store rounded values, flatten form
        for item in travsc:
            for item in item:
                travscr.append(round(item))

        travscrm = [travscr[i:i + 3] for i in range(0, len(travscr), 3)]
            
        #print()
        print("Final rounded matrix,i average_gf ", travscrm)
        print("Historical travels matrix, ", travs)
        print("p is, ", p)
        # compute the final sums of produced
        s_Pic = []
        for item in travscrm:
            s_Pic.append(sum(item))
            
        # working on attracted sums

        # transpose de matrix
        travsc_tt = list(zip(*travscrm))

        # get attracted travels sums on computed travels (cycling on transposes)
        s_Ajc = []   # store the attracted sums

        for item in travsc_tt:
            s_Ajc.append(sum(item))
            
        logger.info("Exit average_gf method.")
        
        return travscrm

    @staticmethod
    def detroit(travs, P_is, A_js, tlr=0.01):

        """
        Method to iteratively compute the future travels distribution using
        Detroit method.
        Takes as input the observed (historical) travels in the form of squared
        matrix,the future produced and attracted ones, respectively, in the
        form of one-line matrices (one for produced and one for attracted)
        and the precision (tolerance).
        Returns a matrix with adjusted travels.
        """

        print("Enter Detroit method.")

        # check if the matrices have the same shape
        if(len(travs) != len(P_is) or len(travs) != len(A_js)):
            print("The matrices doesn't match. Please fix it.")
            exit()
        
        is_cmp_flg = False  # comparison flag to govern the following cycle
        p = 0   # passes counter
               
        # allocate initial computed travels matrix
        travsc = []

        for row in travs:
            travsc.append(row)
                
        while(is_cmp_flg == False):
            # get produced travels sums on computed travels
            s_Pic = []
            for item in travsc:
                s_Pic.append(sum(item))
            
            is_cmp_flgP = GravitMod.comp(P_is, s_Pic)
            
            if (is_cmp_flgP == False):
                ccsi = []   # list to store produced travels coefficients
                for ps, pc in zip(P_is, s_Pic):
                    ccsi.append(round(ps/pc, 3))
                            
            # *********
            # working on attracted sums

            # transpose de matrix
            travsc_tt = list(zip(*travsc))
                    
            # get attracted travels sums on computed travels (cycling on transposes)
            s_Ajc = []   # store the attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))
            
            is_cmp_flgA = GravitMod.comp(A_js, s_Ajc)
                        
            if (is_cmp_flgA == False):
                ccsj = []
                for ats, ac in zip(A_js, s_Ajc):
                    ccsj.append(round(ats/ac, 3))

            # updating the travels matrix
                       
            if(is_cmp_flgP == False or is_cmp_flgA == False):
                travsc_interm = []    #store the modified values
                for x in range(len(travsc)):
                    for t in range(len(travsc[x])):
                        travsc_interm.append(travsc[x][t] * (ccsi[x] * ccsj[t])/
                                             (sum(P_is)/sum(s_Pic)))
                p += 1

            travsc = [travsc_interm[i:i + 3]
                      for i in range(0, len(travsc_interm), 3)]

            # recreate the current travels matrix
            if(is_cmp_flgP == True and is_cmp_flgA == True):
                is_cmp_flg = True
                                                  
        travscr = []     # list to store rounded values, flatten form
        for item in travsc:
            for item in item:
                travscr.append(round(item))

        travscrm = [travscr[i:i + 3] for i in range(0, len(travscr), 3)]
            
        logger.debug("Final rounded matrix, Detroit, %s", travscrm)
        logger.debug("Historical travels matrix, %s", travs)
        logger.debug("p is, %s", p)
        
        # compute the final sums of produced
        s_Pic = []
        for item in travscrm:
            s_Pic.append(sum(item))
        # print()
        # print("s_Pic, ", s_Pic)
        # print("P_is, ", P_is)

        # working on attracted sums

        # transpose de matrix
        travsc_tt = list(zip(*travscrm))

        # get attracted travels sums on computed travels (cycling on transposes)
        s_Ajc = []   # store the attracted sums

        for item in travsc_tt:
            s_Ajc.append(sum(item))
        # print()
        # print("s_Ajc, ", s_Ajc)
        # print("A_js, ", A_js)

        print("Exit Detroit method.")
        
        return travscrm

    @staticmethod
    def iter_wgt_dmd(travs, P_is, A_js, tlr=0.02):

        """
        Method to iteratively weight compute the future travels distribution.
        The main point is to create the travel distribution according to the
        weight of each produced and attracted travels.
        Takes as input the observed (historical) travels in the form of squared
        matrix,the future produced and attracted ones, respectively, in the
        form of one-line matrices (one for produced and one for attracted)
        and the precision (tolerance).
        Returns a matrix with adjusted travels.
        """

        # print()
        print("Enter iter_wgt_dmd method")

        # print()
        # print("Historical travels matrix travs is, ", travs)
        # print()

        # check if the matrices have the correct shape
        # check with the future produced
        if(len(travs) != len(P_is)):
            print("The travels matrix doesn't match with the future\
                  produced! Please fix it.")
            exit()

        # check with the future attracted
        if(len(travs[0]) != len(A_js)):
            print("The travesl matrix doesn't match with the future\
                  attracted! Please fix it.")

            exit()
        
        # get produced travels sums on observed travels
        s_Pih = []
        for item in travs:
            s_Pih.append(sum(item))

        # get attracted travels sums on observed travels (cycling on transposes)
        s_Ajh = []   # store the attracted sums

        travs_tt = list(zip(*travs))

        for item in travs_tt:
            s_Ajh.append(sum(item))

        # generate the produced coefficients matrix (produced perspective)
        c_Pi0 = []
        for t, P in zip(travs, s_Pih):
            for trav in t:
                c_Pi0.append(trav/P)
        
        c_Pi = [c_Pi0[i:i + 3] for i in range(0, len(c_Pi0), 3)]

        # generate the attracted coefficients matrix (attracted perspective)
        c_Aj0= []
        for t, A in zip(travs_tt, s_Ajh):
            for trav in t:
                c_Aj0.append(trav/A)

        c_Aj = [c_Aj0[j:j + 3] for j in range(0, len(c_Aj0), 3)]

       # print("c_Pi matrix, ", c_Pi)
       # print("c_Aj matrix, ", c_Aj)

        is_cmp_flg = False  # comparison flag to govern the following cycle
        i = 0   # produced passes counter
        j = 0   # attracted passes counter
        
        # allocate initial computed travels matrix
        travsc = []
        for row in travs:
            travsc.append(row)

        while(is_cmp_flg == False):
                       
            # get produced travels sums on computed travels
            s_Pic = []
            for item in travsc:
                s_Pic.append(sum(item))

            # print()
            # print("P_is, ", P_is)
            # print("s_Pic, ", s_Pic)
            
            is_cmp_flg = GravitMod.comp(P_is, s_Pic)
            # print(is_cmp_flg)
            if (is_cmp_flg == False):
                delta_P = []    #list to store the deltas of produced travels
                for Pis, Pic in zip(P_is, s_Pic):
                    delta_P.append(Pis - Pic)
                remind_P = []    #matrix of additions to travels, produced
                for cP, delta_i in zip(c_Pi, delta_P):
                    for c in cP:
                        remind_P.append(c*delta_i)   #append weighted delta
                remind_P = [remind_P[i:i + 3] for i in range(0, len(remind_P), 3)]
                # print("remind_P, ", remind_P)
                travsP = []   # list to store adjusted travels matrix, produced
                for remP, trav in zip(remind_P, travsc):
                    for rem, t in zip(remP, trav):
                        travsP.append(rem+t)

                #travsc.clear()

                travsc = [travsP[i:i + 3] for i in range(0, len(travsP), 3)]

                # print()
                # print("travs, ", travs)
                # print("travsc, ", travsc)
                
                i += 1

                # print()
                # print("travsc after pass  i = ", i, "is ", travsc)
            
            s_Pic.clear()

            for item in travsc:
                s_Pic.append(sum(item))

            # print()
            # print("s_Pic, ", s_Pic)
            
            """
            working on attracted travels
            """

            # transpose de matrix
            travsc_tt = list(zip(*travsc))

            travsc_t = [list(sublist) for sublist in travsc_tt]

            # get attracted travels sums on computed travels (cycling on transposes)
            s_Ajc = []   # store the attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))
            
            # print()
            # print("A_js, ", A_js)
            # print("s_Ajc, ", s_Ajc)

            if (GravitMod.comp(A_js, s_Ajc) == False):
                delta_A = []    #list to store the deltas of attracted travels
                for Ajs, Ajc in zip(A_js, s_Ajc):
                    delta_A.append(Ajs - Ajc)
                # print()
                # print("delta_A, ", delta_A)
                # print()

                remind_A = []    #matrix of additions to travels, atrracted
                for cA, delta_j in zip(c_Aj, delta_A):
                    for c in cA:
                        remind_A.append(c*delta_j)   #append weighted delta
                remind_A = [remind_A[j:j + 3] for j in range(0, len(remind_A), 3)]
                # print("remind_A, ", remind_A)
                # print("travsc_tt, ", travsc_tt)
                travsA = []   # list to store adjusted travels matrix, produced
                for remA, trav in zip(remind_A, travsc_tt):
                    #print("remA, ", remA)
                    #print("trav, ", trav)
                    for rem, t in zip(remA, trav):
                        travsA.append(rem+t)

                #travsc.clear()
                #travsc_t.clear()

                travsc_t = [travsA[i:i + 3] for i in range(0, len(travsA), 3)]
                travsc_tt = list(zip(*travsc_t))
                travsc = [list(sublist) for sublist in travsc_tt]
                # print("travsA, ", travsA)
                # print("travsc_t, ", travsc_t)
                # print("travsc_tt, ", travsc_tt)

                # print()
                # print("travs, ", travs)
                # print("travsc, ", travsc)
                
                j += 1

                # print()
                # print("travsc_tt after pass j = ", j, "is ", travsc_tt)             
                
            # update the attracted sums
                
            # get attracted travels sums on new computed travels (cycling on transposes)
            s_Ajc.clear()   # clear the computed attracted sums

            for item in travsc_t:
                s_Ajc.append(sum(item))

            # print()
            # print("s_Ajc, ", s_Ajc)

            # update the produced sums
            s_Pic.clear()

            for item  in travsc:
                s_Pic.append(sum(item))

            # print()
            # print("s_Pic, ", s_Pic)

            is_cmp_flgA = GravitMod.comp(A_js, s_Ajc)
            # print("Flag on attracted, ", is_cmp_flgA)
            
            is_cmp_flgP = GravitMod.comp(P_is, s_Pic)
            # print("Flag on produced, ", is_cmp_flgP)

            if(is_cmp_flgA == True and is_cmp_flgP == True):
                is_cmp_flg = True
            else:
                is_cmp_flg = False

            #if(j == 3):   #block the loop at 3 iterations
            #    is_cmp_flg = True

            travscr = []     # list to store rounded values, flatten form
            for item in travsc:
                for item in item:
                    travscr.append(round(item))

        #print()
        #print("Final rounded and flatten, ", travscr)
        travscrm = [travscr[i:i + 3] for i in range(0, len(travscr), 3)]
            
        # print()
        print("i is, ",i)
        print("j is, ", j)
        print("Final rounded matrix, weighted, ", travscrm)
        print("Historical travels matrix, ", travs)
        print("Exit iter_adj_wgt method.")
        
        return travscrm
        


"""
Enter the methods call section here
"""

gvalsr = GravitMod.gravmod_init(travs, ffs, k_ij0)
# print("gvalsr is, ", gvalsr)

gvalsr_m = [gvalsr[i:i + 3] for i in range(0, len(gvalsr), 3)]
print("****")
print("gvalsr matrix, ", gvalsr_m, '\n')

gvalsadjA = GravitMod.iter_adj_in(travs, gvalsr)
# gvalsadjB = GravitMod.iter_adj_wgt(travs, gvalsr)

# print()
# print("****")
# print("gvalsadj iterative, ", gvalsadjA)
# print("gvalsadj weighted, ", gvalsadjB, '\n')

ccoeffsA = GravitMod.ccoeffs(gvalsadjA, travs)
# ccoeffsB = GravitMod.ccoeffs(gvalsadjB, travs)
ccoeffs = [[0.47,0.99,1.45], [1.27,1.06,0.72], [1.47,0.98,0.23]]

# print("****")
# print("Calibration coefficients iterative, ", ccoeffsA)
# print("Calibration coefficients weighted, ", ccoeffsB, '\n')

gvalsr_finA = GravitMod.gravmod_fin(ffs_f, ccoeffsA, P_is, A_js)
# gvalsr_finB = GravitMod.gravmod_fin(ffs_f, ccoeffsB, P_is, A_js)
gvalsr_finAm = [gvalsr_finA[i:i + 3] for i in range(0, len(gvalsr_finA), 3)]
# gvalsr_finBm = [gvalsr_finB[i:i + 3] for i in range(0, len(gvalsr_finB), 3)]
gvalsr_finAf = GravitMod.furness(gvalsr_finAm, P_is, A_js)
# gvalsr_finBf = GravitMod.furness(gvalsr_finBm, P_is, A_js)

print()
print("****")
# print("Future demand estimation via gravitational model A, ", gvalsr_finA)
# print("Future demand estimation via gravitational model B, ", gvalsr_finB)
print("Future demand estimation via gravitational model A final, ",
      gvalsr_finAf)
# print("Future demand estimation via gravitational model B final, ",
#       gvalsr_finBf, '\n')

travsc_furn = GravitMod.furness(travs, P_is, A_js)
print()
print("****")
print("Travels with Furness method, ", travsc_furn, '\n')

travs_frat = GravitMod.fratar(travs, P_is, A_js)
print()
print("****")
print("Raw Fratar method result, ", travs_frat)
print("Matrix of travels obtained with Fratar (Furness corrected), ",
      GravitMod.furness(travs_frat, P_is, A_js), '\n')

travsc_avgf = GravitMod.average_gf(travs, P_is, A_js)
print("****")
print("Matrix of travels obtained with average growth factor, ", travsc_avgf,
      '\n')

travsc_detr = GravitMod.detroit(travs, P_is, A_js)
print("****")
print("Matrix of travels obtained with_Detroit method, ", travsc_detr, '\n')

travsc_wgtd_prod = GravitMod.iter_wgt_dmd(travs, P_is, A_js)

print("****")
print("Matrix of travels (produced) obtained with weighted coefficients is, ",
      travsc_wgtd_prod, '\n')

# check weighted estimation starting with attracted (not produced, like
# previous call

# transpose the travels matrix
travs_tt = list(zip(*travs))
print("Transpose travel matrix, ", travs_tt)
print("Travel matrix, ", travs, '\n')

travsc_wgtd_at = GravitMod.iter_wgt_dmd(travs_tt,A_js, P_is)

# print("Matrix of travels (attracted) obtained with weighted coefficients is, ",
#      list(zip(*travsc_wgtd_at)))

travsc_wgtd_atL = []   #matrix as list of lists
for row in list(zip(*travsc_wgtd_at)):
    travsc_wgtd_atL.append(list(row))
print("Matrix of travels (attracted) obtained with weighted coefficients is, " , travsc_wgtd_atL, '\n')

print("****")
travsc_wgtd_aver = []
for r1, r2 in zip(travsc_wgtd_prod, travsc_wgtd_atL):
    for t1, t2 in zip(r1, r2):
        travsc_wgtd_aver.append(round((t1+t2)/2, 0))
travsc_wgtd_aver_m = [travsc_wgtd_aver[i:i + 3] for i in range(0,
                                                               len(travsc_wgtd_aver), 3)]
print("Matrix of travels (averaged) obtained with weighted coefficients is, ",
      travsc_wgtd_aver_m)
