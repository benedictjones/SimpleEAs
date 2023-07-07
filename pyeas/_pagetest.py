import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from typing import Any
from typing import cast, Union
from typing import Optional  # telling the type checker that either an object of the specific type is required, or None is required
import time

from collections import namedtuple
from scipy.stats import page_trend_test as ptest # "results\n L=%.2f, p-value=%.5f, method=%s" % (r.statistic, r.pvalue, r.method)

class PageTest:
    """Allows Page Trend Test to be performed to compare convegence of two EA's fitnesses.
        - This allows algorithms with different x-axis values to be compared via projection 
        onto a unified x-axis.
        - Algorithm results are added problem by problem. 

    Args:

        num_cuts:
            The number of cuts. Typically half the number of rows/problems considered.
        
        max_x:
            Select the maximum x-axis value to go up to.
        
        invert:
            Binary integer value. Rather than conidering algorith results A-B, consider B-A.

    """

    

    def __init__(
                self,
                num_cuts: int,
                max_x: Optional[float] = None,
                invert: int = 0,
                problem_labels = Optional[list] = None,
                ):
        
        # # Set number of cuts to half the number of problems
        self.num_cuts = num_cuts

        assert invert == 0 or invert == 1, "Invert must be binary integer"
        self.invert = invert
        self.max_x = max_x
        self.problem_labels = problem_labels

        self.results_so_far = []

        return


    def model_data(self):
        """Perform Page Tren Test on the problems added so far"""
        y = [c.y for c in self.results_so_far]
        x = [c.cx for c in self.results_so_far]
        # cx = [c.cuts_x for c in self.results_so_far]
        cy = [c.cuts_y for c in self.results_so_far]

        ymin = 1000
        ymax = -1000
        for f, arr in enumerate(y):
            if np.min(arr) < ymin:
                ymin = np.min(arr)
            if np.max(arr) > ymax:
                ymax = np.max(arr)

        fig, ax = plt.subplots()

        for f, arr in enumerate(y):
            
            if self.problem_labels is None:
                ax.plot(x[f],y[f], label='f%d' % (f), alpha=1)
            else:
                ax.plot(x[f],y[f], label='f%d (%s)' % (f, self.problem_labels[f]), alpha=1)


            ax.plot(cy[f],'x', color='k', alpha=0.75, markersize=5)

            for xx in range(len(cy)-1):
                ax.plot([xx,xx],[ymin,ymax],color='k', alpha=0.75, markersize=5)  # plot line

        ax.set_xlabel('cuts')
        ax.set_ylabel('fitness{A-B}')


        handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='k', marker='x', markersize=5, label='cuts', linestyle=''))

        ax.legend(handles=handles)
        fig.savefig("%s/Page__%s__Plots.pdf" % (exp_dir, labs[i]), dpi=300)
        plt.close()

        return

    def plot_data(self):
        """Perform Page Tren Test on the problems added so far"""
        matrix = [c.cuts_y for c in self.results_so_far]
        r = ptest(matrix)
        print("Page Test (%d problems, %d cuts) results:\n L=%.2f, p-value=%.5f, method=%s" % (len(matrix), self.num_cuts, r.statistic, r.pvalue, r.method))

        return


    def add_problem_interp(self, x1,y1,x2,y2):
        """ Adds the results from two algorithms.
        - Takes the two evolutionary algorithm curves and creates two new
        interpolated arrays with a unified x axis value.
        - These are the "c cut points" used to subtract the results.
        and formulate the A-B trend.
        """

        # # Ensure the interpolation gives values close to what we are looking for
        l = max(len(x1), len(x2))
        nps = (int(l/self.num_cuts)+1)*self.num_cuts*10
        # print("l:", l, ", nps:", nps)

        # # Make sure the max value is the same between al algo piece wise combos
        if self.max_x is None:
            max_x = max([max(x2), max(x1)])
            max_x = np.around(max_x, decimals=-5)
        else:
            max_x = self.max_x

        # # Interpolate to a larger number of points
        new_x = np.linspace(0, max_x, num=nps)
        new_y1 = np.interp(new_x, x1, y1)
        new_y2 = np.interp(new_x, x2, y2)

        # # Cut down to the newer number of points
        cuts = []
        real_c_locs = []
        c_locs = np.linspace(0, max_x, num=self.num_cuts)
        for c in c_locs:
            idx, val = self.find_nearest(new_x, c)
            if self.invert == 0:
                cuts.append(new_y1[idx]-new_y2[idx])
            elif self.invert == 1:
                cuts.append(new_y2[idx]-new_y1[idx])
            real_c_locs.append(val)

        # # Apply inversion to y values
        if self.invert == 0:
            y = new_y1-new_y2
        elif self.invert == 1:
            y = new_y2-new_y1

        # # 
        rel_x = (new_x/np.max(new_x))*(self.num_cuts-1)


        DataGroup = namedtuple('PageCuts', ['x', 'y', 'cuts_x', 'cuts_y', 'cx'])
        results = DataGroup(new_x, y, c_locs, cuts, rel_x)

        self.results_so_far.append(results)

        return 
    

    def add_problem(self, x, y1, y2):
        """ Adds the results from two algorithms.
        - Takes the two evolutionary algorithm curves.
        - These are the "c cut points" used to subtract the results
        and formulate the A-B trend.
        """

        # # Make sure the max value is the same between al algo piece wise combos
        if self.max_x is None:
            max_x = max(x)
            max_x = np.around(max_x, decimals=-5)
        else:
            max_x = self.max_x

        # # Cut down to the newer number of points
        cuts = []
        real_c_locs = []
        c_locs = np.linspace(0, max_x, num=self.num_cuts)
        for c in c_locs:
            idx, val = self.find_nearest(c, c)
            if self.invert == 0:
                cuts.append(y1[idx]-y2[idx])
            elif self.invert == 1:
                cuts.append(y2[idx]-y1[idx])
            real_c_locs.append(val)

        # # Apply inversion to y values
        if self.invert == 0:
            y = y1-y2
        elif self.invert == 1:
            y = y2-y1

        # # 
        rel_x = (x/np.max(x))*(self.num_cuts-1)


        DataGroup = namedtuple('PageCuts', ['x', 'y', 'cuts_x', 'cuts_y', 'cx'])
        results = DataGroup(x, y, c_locs, cuts, rel_x)

        self.results_so_far.append(results)

        return 

    def find_nearest(array, location):
        """Find the index & value of the point in an array clossest to the desired cut location """
        array = np.asarray(array)
        idx = (np.abs(array - location)).argmin()
        return idx, array[idx]                      