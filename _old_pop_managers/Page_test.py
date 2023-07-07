# PageTest v1.0.1
# http://jwcarr.github.io/PageTest/
#
# Copyright (c) 2013-2015 Jon W. Carr
# Licensed under the terms of the MIT License
import numpy as np
from scipy import stats

#   Run Page's test and return l, m, n, p, where l = Page's L statistic,
#   m = number of replications, n = number of treatments, and p = p-value.

def test(matrix, ascending=False, use_critical_values=False):
    """
    Takes a matrix, with treatments along the columns and replications along the
    rows, and returns Page's (1963) L statistic, along with its p-value.

    # https://github.com/jwcarr/PageTest/blob/master/Page.py

    Parameters
    ----------
    matrix : list
      Data matrix (formated as a list of lists) with treatments along the
      columns and replications along the rows.
    ascending : bool, optional
      Set to True if hypothesizing an ascending trend, False if hypothesizing
      a descending trend (default: False).
    use_critical_values : bool, optional
      Set to True to use the critical values from Page (1963) rather than
      compute an exact p-vaue (default: False).

    Returns
    -------
    L : float
      Page's L statistic
    m : int
      Number of replications
    n : int
      Number of treatments
    p : float or str
      P-value
    """
    validate_input(matrix, ascending, use_critical_values)
    if ascending == True:
        matrix = reverse_matrix(matrix)
    m = len(matrix)
    n = len(matrix[0])
    l = page_l(matrix, m, n, report=1)
    p = page_p(l, m, n, matrix, use_critical_values)
    return l, m, n, p

#   Calculate Page's L statistic.

def page_l(matrix, m, n, report=0):
    rank_matrix = []
    for i in range(0, m):
        rank = stats.rankdata(matrix[i])
        rank_list = []
        for j in range(0, n):
            rank_list.append(rank[n-j-1])
        rank_matrix.append(rank_list)

    ranks = []
    for i in range(0, n):
        total = sum([row[i] for row in rank_matrix])
        total *= i + 1
        ranks.append(total)

    if report == 1:
        print("rank_matrix:\n", np.array(rank_matrix))
        print("ranks:", ranks)
        rm = np.array(rank_matrix)
        print("sum down:", np.sum(rm,axis=0))
        print("sum accros:", np.sum(rm,axis=1))


    return sum(ranks)

#   Calculate a p-value for L using the appropriate method.

def page_p(l, m, n, matrix, use_critical_values):
    if use_critical_values == True:
        try:
            return page_critical_p(l, m, n)
        except IndexError:
            print('Large data matrix, so calculating exact p-value instead')
    return page_exact_p(l, m, n, matrix)

#   For small m and n, the exact p-value won't always agree with the critical
#   values given in Page (1963, p. 220). If you prefer, you can use Page's critical
#   values instead. This function looks up the critical values for m and n, and
#   finds the significance level for L.

def page_critical_p(l, m, n):
    values = critical_values[n-3][m-2]
    significance_levels = ['< 0.001', '< 0.01', '< 0.05']
    for i in range(0, 3):
        if l >= values[i] and values[i] != None:
            return significance_levels[i]
    return 'n.s.'

#   Calculate the exact p-value using Eqation 4 in Page (1963)

def page_exact_p(l, m, n, matrix):

    # # Calc p
    # chi_squared = ((12.0*l-3.0*m*n*(n+1.0)**2.0)**2.0)/(m*n**2.0*(n**2.0-1.0)*(n+1.0))
    chi_squared = calc_chi_squared(l,m,n)
    p_two_tailed = 1 - stats.chi2.cdf(chi_squared, 1)

    # # Calcualte L for the opposite trend
    alt_l = page_l(reverse_matrix(matrix), m, n)
    alt_chi_squared = calc_chi_squared(alt_l,m,n)
    alt_p_two_tailed = 1 - stats.chi2.cdf(alt_chi_squared, 1)
    print(chi_squared, alt_chi_squared)
    # If L for the opposite trend > L for the hypothesized trend, then the trend
    # can't be significant ...
    if alt_l > l:
        # ... so return 'n.s.', otherwise the exact p-value could be misleading
        # if the opposite trend happens to be significant
        print(" > L from the Opposite Trend L{B-A}=%.2f > chosen trend L{A-B}=%.2f" % (alt_l,l))
        print(" > p{A-B}=%.10f, p{B-A}=%.10f" % (p_two_tailed / 2.0, alt_p_two_tailed/2))
        return 'n.s.'

    # Return one-tailed p-value, since this is a one-tailed test
    return p_two_tailed / 2.0


def calc_chi_squared(l, m, n):

    chi_squared = ((12.0*l-3.0*m*n*(n+1.0)**2.0)**2.0)/(m*n**2.0*(n**2.0-1.0)*(n+1.0))

    k = n
    N = m
    paper_chi = (12*(l-0.5) - 3*N*k*(k+1)**2) / (k*(k+1)*(N*(k-1))**0.5)
    paper_chi_squared = paper_chi**2

    # print("f chi_squared:", chi_squared)
    # print("paper_chi_squared:", paper_chi_squared)

    return paper_chi_squared



def _l_p_asymptotic(L, m, n):
    '''Calculate the p-value of Page's L from the asymptotic distribution'''
    # Using [1] as a reference, the asymptotic p-value would be calculated as:
    # chi_L = (12*L - 3*m*n*(n+1)**2)**2/(m*n**2*(n**2-1)*(n+1))
    # p = chi2.sf(chi_L, df=1, loc=0, scale=1)/2
    # but this is insentive to the direction of the hypothesized ranking

    # See [2] page 151
    E0 = (m*n*(n+1)**2)/4
    V0 = (m*n**2*(n+1)*(n**2-1))/144
    Lambda = (L-E0)/np.sqrt(V0)
    # This is a one-sided "greater" test - calculate the probability that the
    # L statistic under H0 would be greater than the observed L statistic
    p = norm.sf(Lambda)
    return p



def reverse_matrix(matrix):
    """ Reverses the columns of a matrix """
    return [[row[i] for i in reversed(range(len(matrix[0])))] for row in matrix]

#   Validates the input arguments to catch common problems

def validate_input(matrix, ascending, use_critical_values):
    if type(matrix) != list:
        raise TypeError('Matrix should be represented as Python lists')
    for row_type in [type(row) for row in matrix]:
        if row_type != list:
            raise TypeError('Rows of the matrix should be represented as Python lists')
    if len(set([len(row) for row in matrix])) != 1:
        raise ValueError('Rows in matrix should have same length')
    if len(matrix) < 2:
        raise ValueError('Page\'s test requires at least 2 replications')
    if len(matrix[0]) < 3:
        raise ValueError('Page\'s test requires at least 3 treatments')
    if type(ascending) != bool:
        raise TypeError('The ascending argument should be set to True or False')
    if type(use_critical_values) != bool:
        raise TypeError('The use_critical_values argument should be set to True or False')

critical_values = [[[None, None, 28], [None, 42, 41], [56, 55, 54], [70, 68, 66], [83, 81, 79], [96, 93, 91], [109, 106, 104], [121, 119, 116], [134, 131, 128], [147, 144, 141], [160, 156, 153], [172, 169, 165], [185, 181, 178], [197, 194, 190], [210, 206, 202], [223, 218, 215], [235, 231, 227], [248, 243, 239], [260, 256, 251]], [[None, 60, 58], [89, 87, 84], [117, 114, 111], [145, 141, 137], [172, 167, 163], [198, 193, 189], [225, 220, 214], [252, 246, 240], [278, 272, 266], [305, 298, 292], [331, 324, 317]], [[109, 106, 103], [160, 155, 150], [210, 204, 197], [259, 251, 244], [307, 299, 291], [355, 346, 338], [403, 393, 384], [451, 441, 431], [499, 487, 477], [546, 534, 523], [593, 581, 570]], [[178, 173, 166], [260, 252, 244], [341, 331, 321], [420, 409, 397], [499, 486, 474], [577, 563, 550], [655, 640, 625], [733, 717, 701], [811, 793, 777], [888, 869, 852], [965, 946, 928]], [[269, 261, 252], [394, 382, 370], [516, 501, 487], [637, 620, 603], [757, 737, 719], [876, 855, 835], [994, 972, 950], [1113, 1088, 1065], [1230, 1205, 1180], [1348, 1321, 1295], [1465, 1437, 1410]], [[388, 376, 362], [567, 549, 532], [743, 722, 701], [917, 893, 869], [1090, 1063, 1037], [1262, 1232, 1204], [1433, 1401, 1371], [1603, 1569, 1537], [1773, 1736, 1703], [1943, 1905, 1868], [2112, 2072, 2035]]]
