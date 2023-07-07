# Importing the required libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple

from tqdm import tqdm
from scipy.stats import norm, chi2
from scipy.stats import t as t_dist
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold

from mlxtend.evaluate import mcnemar_table, mcnemar

from mod_methods.Page_test import test as ptest
# from scipy.stats import page_trend_test as p_test

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


# # pass in: y_test, y_algo1, y_algo2
def mcnemar_test(y_true, y_1, y_2):
    b = sum(np.logical_and((y_1 != y_true),(y_2 == y_true)))
    c = sum(np.logical_and((y_1 == y_true),(y_2 != y_true)))

    c_ = (np.abs(b - c) - 1)**2 / (b + c)
    p_value = chi2.sf(c_, 1)

    return c_, p_value



def mcnemar_test_2(y_true, y_1, y_2):

    tb = mcnemar_table(y_target=y_true,
                        y_model1=y_1,
                        y_model2=y_2)

    chi, p = mcnemar(ary=tb, corrected=True)
    print(">>", chi, p)
    return chi, p



def page_test(matrix):


    l, m, n, p = ptest(matrix)

    return l, m, n, p


def page_cut_subtract(x1,y1,x2,y2, num_points=4, max_x=None, invert=0):
    """
    Takes the two evolutionary algorithm curves and creates two new
    interpolated arrays with a unified x axis value.
    These are the "c cut points" used to subtract the results.
    and formulate the A-B trend.

    >>> Why should the number of cuts be ~0.5*num rows????????


    """

    # # Ensure the interpolation gives values close to what we are looking for
    l = max(len(x1), len(x2))
    nps = (int(l/num_points)+1)*num_points*10
    # print("l:", l, ", nps:", nps)

    # # Make sure the max value is the same between al algo piece wise combos
    if max_x is None:
        max_x = max([max(x2), max(x1)])
        max_x = np.around(max_x, decimals=-5)


    # # Interpolate to a larger number of points
    new_x = np.linspace(0, max_x, num=nps)
    new_y1 = np.interp(new_x, x1, y1)
    new_y2 = np.interp(new_x, x2, y2)

    # # Cut down to the newer number of points
    cuts = []
    real_c_locs = []
    c_locs = np.linspace(0, max_x, num=num_points)
    for c in c_locs:
        idx, val = find_nearest(new_x, c)

        if invert == 0:
            cuts.append(new_y1[idx]-new_y2[idx])
        elif invert == 1:
            cuts.append(new_y2[idx]-new_y1[idx])
        else:
            raise ValueError("Invalif invert argument")

        real_c_locs.append(val)
    #print("ideal c_locs:", c_locs)
    #print("real c_locs:", real_c_locs)
    #print("cuts:", cuts)

    """
    fig = plt.figure()
    plt.plot(x1,y1, 'r', label='OG algo 1', alpha=0.5)
    plt.plot(x2,y2, 'b', label='OG algo 2', alpha=0.5)
    plt.plot(new_x, new_y1, '--', color='r', label='interp algo 1', alpha=0.8)
    plt.plot(new_x, new_y2, '--', color='b', label='interp algo 2', alpha=0.8)

    plt.plot(c_locs, np.ones(len(cuts)), 'x', color='k', label='cuts')
    plt.legend()
    fig.savefig("temp.png", dpi=200)
    exit()
    #"""

    if invert == 0:
        y = new_y1-new_y2
    elif invert == 1:
        y = new_y2-new_y1
    else:
        raise ValueError("Invalif invert argument")

    rel_x = (new_x/np.max(new_x))*(num_points-1)


    DataGroup = namedtuple('PageCuts', ['x', 'y', 'cuts_x', 'cuts_y', 'cx'])
    return_data = DataGroup(new_x, y, c_locs, cuts, rel_x)


    return return_data
