"""
Script to compute the individual traveler utilities 
and the travels weights using LOGIT model.
The weights are to be applied on the gravitational model
table of travels.
"""

# number of travels as a matrix with produced travels on lines
# and attracted travels on columns

travs = [[40, 110, 150],
         [50, 20, 30],
         [110, 30, 10]]

# the matching friction factors, same arrangement

ffs = [[0.753, 1.597, 0.753],
       [0.987, 0.753, 0.765],
       [1.597, 0.765, 0.753]]

# neutral calibration coefficients

k_ij0 = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]

# auto travels cost

tca = [[0.5, 1, 1.4],
       [1.2, 0.8, 1.2],
       [1.7, 1.5, 0.7]]

# transit travels cost

tct = [[1, 1.5, 2],
       [1.8, 1.2, 1.9],
       [1.7, 1.5, 0.7]]

# auto travels duration

tda = [[3, 12, 7],
       [13, 3, 19],
       [9, 16, 4]]

# transit travels duration

tdt = [[15, 5, 12],
       [15, 6, 26],
       [20, 21, 8]]

# the future friction factors

ffs_f = [[0.753, 0.987, 1.597],
         [0.987, 0.753, 0.765],
         [1.597, 0.765, 0.753]]

# the future produced travels

P_is = [750, 580, 480]

# the future attracted travels

A_js = [722, 786, 302]

class GravitMod:
    def __init__(self, travs, ffs, k_ijs, P_is, A_js):
        self.travs = travs
        self.ffs = ffs
        self.k_ijs = k_ijs
        self.P_is = P_is
        self.A_js = A_js

    def gravmod_init(travs,ffs, k_ijs):
        """
        Method to compute gravitational model values in order to determine the
        calibration factors.
        Takes as input the travels, friction factors and calibration
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
        # print("travs_tt, ", travs_tt)
        # print("travs_t, ", travs_t)
        # print(ffs_t)
    
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

        #print("Initial travels obtained with gravitational model, ", gvals_init)

        # check raw produced travels
        gvals_init_m0 = [gvals_init[i:i + 3] for i in range(0, len(gvals_init), 3)]
        vals_r = []
        for r in gvals_init_m0:
            r_r = [round(el, 2) for el in r]
            vals_r.append(r_r)
        print("Initial travels matrix (trimed) is ,", vals_r)

    
        # for p1, p2 in zip(travs, gvals_init_m0):
            #print(round(sum(p1)) == round(sum(p2)))
            #print(sum(p2))

        # round the number of travels
        gvals_init_r = []
        for item in gvals_init:
            gvals_init_r.append(round(item))
    
        # print("Rounded number of initial travels, ", gvals_init_r)

        # group flatten list 'gvals_init_r' as a matrix
        gvals_init_m = [gvals_init_r[i:i + 3] for i in range(0,
                         len(gvals_init_r), 3)]
        # print(gvals_init_m)
        print("Matrix of rounded numbers, ", gvals_init_m)

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

    def iter_adj_in(travs, travsc, tlr=0.01):

        """
        Method to iteratively adjust travels computed with gravitational model.
        Takes as input the observed (historical) travels,the computed
        ones, in the form of matrices and the precision (tolerance) of
        adjustment.
        Returns a matrix with adjusted travels.
        """

        print()
        print("Enter iter_adj_in method.")
    
        # check if the matrices have the same shape
        if(len(travs) != len(travsc)):
            print("The matrices doesn't match. Please fix it.")
            exit()
        
        # function to compare the produced, respectively attracted travels
        # within a certain tolerance
        def comp(s_ih, s_ic, tlr):
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
        s_Pih = []
        for item in travs:
            s_Pih.append(sum(item))

        # get attracted travels sums on observed travels (cycling on transposes)
        s_Ajh = []   # store the attracted sums

        travs_tt = list(zip(*travs))    # transpose the matrix hist travs

        for item in travs_tt:
            s_Ajh.append(sum(item))

        cmp_flg = False  # comparison flag to govern the following cycle
        i = 0   # produced passes counter
        j = 0   # attracted passes counter

        while(cmp_flg == False):
                       
            # get produced travels sums on computed travels
            s_Pic = []
            for item in travsc:
                s_Pic.append(sum(item))
            
            #print()
            #print("Enter iter_adj_in method.")
            #print("s_Pih, ", s_Pih)
            #print("s_Pic, ", s_Pic)
            
            cmp_flg = comp(s_Pih, s_Pic, tlr)
            #print(cmp_flg)
            if (comp(s_Pih, s_Pic, tlr) == False):
                ccsi = []   # list to store produced travels coefficients
                for ph, pc in zip(s_Pih, s_Pic):
                    ccsi.append(round(ph/pc, 3))

                #print()
                #print("travs, ", travs)
                print("travsc, ", travsc)
                print("coefficients on produced travels, ", ccsi)

                for x in range(len(travsc)):
                    travsc[x] = [ccsi[x]*item for item in travsc[x]]
            
                i += 1

                #print()
                #print("travsc after pass  i = ", i, "is ", travsc)
            
            
            # *********
            # working on attracted travels

            # transpose de matrices
            travsc_tt = list(zip(*travsc))

            # travs_t = [list(sublist) for sublist in travs_tt]
            # travsc_t = [list(sublist) for sublist in travsc_tt]

            # print()
            # print("travs_t, ", travs_t)
            # print("travsc_t ", travsc_t)
    
                    
            # get attracted travels sums on computed travels (cycling on transposes)
            s_Ajc = []   # store the attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))
            
            # cmp_flg = comp(s_Ajh, s_Ajc, tlr)
            # print(cmp_flg)
            
            #print()
            #print("s_Ajh, ", s_Ajh)
            #print("s_Ajc, ", s_Ajc)
            
            if (comp(s_Ajh, s_Ajc, tlr) == False):
                ccsj = []   # list to store attracted travels coefficients
                for ah, ac in zip(s_Ajh, s_Ajc):
                    ccsj.append(round(ah/ac, 3))

                #print()
                #print("travs, ", travs)
                print("travsc, ", travsc)
                print("coefficients on attracted travels, ", ccsj)

                for x in range(len(travsc_tt)):
                    travsc_tt[x] = [ccsj[x]*item for item in travsc_tt[x]]
            
                j += 1

               #print()
               #print("travsc_tt after pass j = ", j, "is ", travsc_tt)
                
                

            travsc_0 = list(zip(*travsc_tt))
            travsc = [list(sublist) for sublist in travsc_0]

            #print()
            #print("travsc,  ", travsc)
            
            # update the attracted sums
                
            # get attracted travels sums on new computed travels (cycling on transposes)
            s_Ajc.clear()   # clear the computed attracted sums

            for item in travsc_tt:
                s_Ajc.append(sum(item))

            #print()
            #print("s_Ajc, ", s_Ajc)

            # update the produced sums
            s_Pic.clear()

            for item  in travsc:
                s_Pic.append(sum(item))

            #print()
            #print("s_Pic, ", s_Pic)

            cmp_flg = comp(s_Ajh, s_Ajc, tlr)
            #print("Flag on attracted, ", comp(s_Ajh, s_Ajc, tlr))
            
            cmp_flg = comp(s_Pih, s_Pic, tlr)
            #print("Flag on produced, ", cmp_flg)

            travscr = []     # list to store rounded values, flatten form
            for item in travsc:
                for item in item:
                    travscr.append(round(item))

        #print()
        #print("Final rounded and flatten, ", travscr)
        travscrm = [travscr[i:i + 3] for i in range(0, len(travscr), 3)]
            
        #print()
        print("Final rounded matrix, ", travscrm)
        print("Historical travels matrix, ", travs)
        print("i is, ", i)
        print("j is, ", j)
        print("Exit iter_adj_in method.")
        
        return travscrm

    # method to compute gravitational model travels projected into the future
    def gravmod_fin(ffs, k_ijs, P_is, A_js):
        """
        Method to compute future travels using gravitational model.
        Takes as inputs the future friction factors matrix, the previously
        computed calibration coefficients matrix, the matrix of produced
        travels and the matrix of attracted travels.
        Returns a matrix with future travels determined with gravitational
        model.
        """
        
        print()
        print("Enter the gravitmod final method.")
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
                #print("pdsum fin, ", pdsum)
                #print("ffs[i] fin, ", ffs[i])
                # print("A_j, ", j1)
            for k1 in range(len(ffs[i])):
                gvals_fin.append((P_is[i] * ffs[i][k1] * A_js[k1] * k_ijs[i][k1] / pdsum))

        print("Future travels obtained with gravitational model, ", gvals_fin)

        # check raw produced travels
        gvals_fin_m0 = [gvals_fin[i:i + 3] for i in range(0, len(gvals_fin), 3)]
        # print(gvals_fin_m0)

    
        # round the number of travels
        gvals_fin_r = []
        for item in gvals_fin:
            gvals_fin_r.append(round(item))
    
        print("Rounded number of future travels, ", gvals_fin_r)
        print("Total travels sum, ", sum(gvals_fin_r))

        # group flatten list 'gvals_fin_r' as a matrix
        gvals_fin_m = [gvals_fin_r[i:i + 3] for i in range(0,
                         len(gvals_fin_r), 3)]

        print("Matrix of future rounded numbers, ", gvals_fin_m)

        return gvals_fin_r

        
    def ccoeffs(gvalsradj, travs):
        # compute calibration coefficients
        ccoeffs = []
        for row_h, row_c in zip(travs, gvalsradj):
            for t_h, t_c in zip(row_h, row_c):
                ccoeffs.append(round(t_h / t_c, 2))
        ccoeffs_m = [ccoeffs[i:i + 3] for i in range(0, len(ccoeffs), 3)]

        return ccoeffs_m

# function for modal option
def modopt(tca, tct, tda, tdt):
    """
    Function to compute modal option, i.e. auto and transit.
    Takes as inputs the matrices of travels cost, auto and transit, and
    duration, respectively.
    Returns the weights of auto and transit travels for each zone to zone
    combination.
    """
    # compute utilities for auto and transit modes
    u_a = []    # store auto utility results
    u_t = []    # store transit utility results

    for i in range(len(tca)):
        for ca, da in zip(tca[i], tda[i]):
            u_a.append(round(2.5 - 0.5 * ca - 0.01 * da, 2))
        for ct, dt in zip(tct[i], tdt[i]):
            u_t.append(round(-0.4 * ct - 0.012 * dt, 2))

    print("Auto utilities, ", u_a)
    print("Transit utilities, ", u_t)

    return (u_a, u_t)

# function to compute auto and transit weight, from to each zone
def logit(u_a, u_t):
    """
    Function to compute travels weights for each zone.
    Takes as inputs the utilities lists, auto and transit.
    Returns auto and transit weights for each zone to zone combination.
    """
    # import to get Euler number
    from math import e
    # print("e, ", e)

    w_a = []    # store auto weights
    w_t = []    # store transit weights
    for u_a, u_t in zip(u_a, u_t):
        w_i = e**u_a / (e**u_a + e**u_t)
        # print(w_i)
        w_i = round(w_i, 2)
        # print(w_i)
        w_a.append(w_i)
        w_t.append(round(1-w_i, 2))

    print("Auto travels weights, ", w_a)
    print("Trasit travels weights, ", w_t)

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

print("Adjusted matrix A, ", gvalsadjA)
#print("Adjusted matrix B, ", gvalsadjB)
print("Calibration coefficients matrix A, ", ccoeffsA)
#print("Calibration coefficients matrix B, ", ccoeffsB)

# print("ccoeffsA, ", ccoeffsA)

# ccoeffs_it = GravitMod.ccoeffs(gvalsradj_it, travs)

# print("ccoeffs_it, ", ccoeffs_it)

ccoeffs_m = [ccoeffsA[i:i + 3] for i in range(0, len(ccoeffsA), 3)]
print("Calibration coefficients, ", ccoeffs_m)

gvalsr_fin = GravitMod.gravmod_fin(ffs_f, ccoeffsA, P_is, A_js)

# print("Future number of rounded travels, ", gvalsr_fin)

u_a, u_t = modopt(tca, tct, tda, tdt)

w_a, w_t = logit(u_a, u_t)

# The adjusted travels, gravitational model
travs_adj_fin = [[114, 375, 244],
                  [298, 240, 50],
                  [310, 171, 8]]

# applying weights to travels
travs_adj_fin_flatten = []
for row in travs_adj_fin:
    for travel in row:
        travs_adj_fin_flatten.append(travel)

print("flatten final travels, ", travs_adj_fin_flatten)

travels_auto = []
for weight, travel in zip(w_a, travs_adj_fin_flatten):
    travels_auto.append(round(weight * travel, 0))

print("Auto travels, ", travels_auto)
travels_transit = []
for travels, travels_auto in zip(travs_adj_fin_flatten, travels_auto):
    travels_transit.append(travels - travels_auto)

print("Transit travels, ", travels_transit)

