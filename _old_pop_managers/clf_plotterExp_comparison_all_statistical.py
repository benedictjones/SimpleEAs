from mod_load.FetchDataObj import FetchDataObj
from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap
import matplotlib.patches as mpatches
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines

from tqdm import tqdm
import os
from datetime import datetime
import h5py
import yaml
import scipy

from mod_methods.stats_test import mcnemar_test, page_cut_subtract, page_test
from mod_plotters.clf_loader import load_exp_data
from scipy.stats import page_trend_test as ptest

def plotterExpAlgoCompaStats(exp_dir, plot_prms='all', plt_std=1, formal=0):

    NESprm, NESres = load_exp_data(exp_dir, 'NES')
    DEprm, DEres = load_exp_data(exp_dir, 'DE')
    CMAESprm, CMAESres = load_exp_data(exp_dir, 'CMAES')

    if NESprm is not None:
        prm = NESprm
    elif DEprm is not None:
        prm = DEprm
    elif CMAESprm is not None:
        prm = CMAESprm


    matplotlib .rcParams['font.family'] = 'Arial'  # 'serif'
    matplotlib .rcParams['font.size'] = 8  # tixks and title
    matplotlib .rcParams['figure.titlesize'] = 'medium'
    matplotlib .rcParams['axes.labelsize'] = 10  # axis labels
    matplotlib .rcParams['axes.linewidth'] = 1  # box edge
    #matplotlib .rcParams['mathtext.fontset'] = 'Arial'  # 'cm'
    matplotlib.rc('pdf', fonttype=42)  # embeds the font, so can import to inkscape
    matplotlib .rcParams["legend.labelspacing"] = 0.25

    matplotlib .rcParams['lines.linewidth'] = 0.85
    matplotlib .rcParams['lines.markersize'] = 3.5
    matplotlib .rcParams['lines.markeredgewidth'] = 0.5

    if formal == 1:
        fig_size = [2.1,2.]  # [3.5,2.7]  , [2.8,2.3]
        fig_size = [2.8,2.3]  # [3.5,2.7]  , [2.8,2.3]
    else:
        fig_size = [4.6, 4]
    matplotlib .rcParams["figure.figsize"] = fig_size
    #matplotlib .rcParams["figure.autolayout"] = True


    ''' ######################################################
    Plot algos against each other on individual graphs per param
    '''
    Ttext = []
    Ttext.append('')
    Ttext.append("      {a:^10}{b:^10}{c:^10}{d:^10}{e:^10}{f:^10}{g:^10}{h:^10}{i:^10}{j:^10}".format(a='mean fit', b='std fit', c='best fit', d='mean accr', e='accr*', f='c accr*', g='bs', h='NP', i='pop frac', j='HL size'))


    Stext = []
    stats_test = 'acc'  # 'acc'  , 'loss'
    page_stats_test = 'loss'
    Stext.append('Using %s values for stats' % stats_test)
    Stext.append("{a:^10}{b:^10}{c:^10}{d:^10}{e:^10}".format(a='Algo1', b='Algo2', c='T-test', d='Rank Test', e='McNemar'))

    pageMatrix_DE_OAIES = []  # res['test']['mean']['loss']
    pageMatrix_DE_CMAES = []  # res['test']['mean']['loss']
    pageMatrix_OAIES_CMAES = []  # res['test']['mean']['loss']

    print("start...")
    for p, param in enumerate(prm['exp']['sweep']):
        if param in plot_prms or 'all' in plot_prms:

            Ttext.append("\n >>%s<<" % param)
            Stext.append("\n >>%s<<" % param)

            print("\n >> %s <<" % (param))

            fig, ax = plt.subplots(1, sharex=True)

            if formal == 0:
                fig.suptitle('SE of %s Results' % (param))

            y = []
            x = []
            final_fits = DEres['test']['formatted']['loss'][p][:,-1]
            print("final_fits:", final_fits)
            for nc in range(1, len(final_fits)):
                g = final_fits[:nc+1]
                gg = scipy.stats.sem(g)
                # print(nc, g, gg)
                y.append(gg)
                x.append(nc+1)
            ax.plot(x, y, color='g', label="DE")
            Ttext.append("DE    {a:^10}{b:^10}{c:^10}{d:^10}{e:^10}{f:^10}{g:^10}{h:^10}{i:^10}{j:^10}".format(a='%.4f' % y[-1],
                                                                                                  b='%.4f' % DEres['test']['std']['loss'][p][-1],
                                                                                                  c='%.4f' % DEres['test']['best']['loss'][p],  # best fit
                                                                                                  d='%.4f' % DEres['test']['mean']['acc'][p][-1],  # mean accr
                                                                                                  e='%.4f' % DEres['test']['best']['acc'][p], # best accr
                                                                                                  f='%.4f' % DEres['test']['best']['acc_corresponding'][p], # coressponding best accr
                                                                                                  g='%.4f' % DEres['prms'][p]['algo']['batch_size'],  # bs
                                                                                                  h='%.4f' % DEres['prms'][p]['algo']['popsize'],  # NP
                                                                                                  i='%.4f' % DEres['prms'][p]['algo']['popsize_frac'], # pop frac
                                                                                                  j='%.4f' % DEres['prms'][p]['network']['hiddenSize']))  # HL size


            y = []
            x = []
            final_fits = NESres['test']['formatted']['loss'][p][:,-1]
            for nc in range(1, len(final_fits)):
                y.append(scipy.stats.sem(final_fits[:nc+1]))
                x.append(nc+1)
            ax.plot(x, y, color='r', label="OAIES")
            Ttext.append("OAIES {a:^10}{b:^10}{c:^10}{d:^10}{e:^10}{f:^10}{g:^10}{h:^10}{i:^10}{j:^10}".format(a='%.4f' % y[-1],
                                                                                                  b='%.4f' % NESres['test']['std']['loss'][p][-1],
                                                                                                  c='%.4f' % NESres['test']['best']['loss'][p],  # best fit
                                                                                                  d='%.4f' % NESres['test']['mean']['acc'][p][-1],  # mean accr
                                                                                                  e='%.4f' % NESres['test']['best']['acc'][p], # best accr
                                                                                                  f='%.4f' % NESres['test']['best']['acc_corresponding'][p], # coressponding best accr
                                                                                                  g='%.4f' % NESres['prms'][p]['algo']['batch_size'],  # bs
                                                                                                  h='%.4f' % NESres['prms'][p]['algo']['popsize'],  # NP
                                                                                                  i='%.4f' % NESres['prms'][p]['algo']['popsize_frac'], # pop frac
                                                                                                  j='%.4f' % NESres['prms'][p]['network']['hiddenSize']))  # HL size


            y = []
            x = []
            final_fits = CMAESres['test']['formatted']['loss'][p][:,-1]
            for nc in range(1, len(final_fits)):
                y.append(scipy.stats.sem(final_fits[:nc+1]))
                x.append(nc+1)
            ax.plot(x, y, color='b', label="CMAES")
            Ttext.append("CMAES {a:^10}{b:^10}{c:^10}{d:^10}{e:^10}{f:^10}{g:^10}{h:^10}{i:^10}{j:^10}".format(a='%.4f' % y[-1],
                                                                                                  b='%.4f' % CMAESres['test']['std']['loss'][p][-1],
                                                                                                  c='%.4f' % CMAESres['test']['best']['loss'][p],  # best fit
                                                                                                  d='%.4f' % CMAESres['test']['mean']['acc'][p][-1],  # mean accr
                                                                                                  e='%.4f' % CMAESres['test']['best']['acc'][p], # best accr
                                                                                                  f='%.4f' % CMAESres['test']['best']['acc_corresponding'][p], # coressponding best accr
                                                                                                  g='%.4f' % CMAESres['prms'][p]['algo']['batch_size'],  # bs
                                                                                                  h='%.4f' % CMAESres['prms'][p]['algo']['popsize'],  # NP
                                                                                                  i='%.4f' % CMAESres['prms'][p]['algo']['popsize_frac'], # pop frac
                                                                                                  j='%.4f' % CMAESres['prms'][p]['network']['hiddenSize']))  # HL size


            if formal == 0:
                ax.set_xlabel('Repetitions')
                ax.set_ylabel('SE of Mean Test Fitness')

            if formal == 0:
                ax.legend(fontsize=6)

            if formal == 0:
                fig_path = "%s/cFIG_Algo_SE_comparison_%s.png" % (exp_dir, param)
                fig.savefig(fig_path, dpi=200)
            else:
                fig_path = "%s/cFIG_Algo_SE_comparison_%s.pdf" % (exp_dir, param)
                fig.savefig(fig_path, dpi=200)

            plt.close(fig)

            #

            #

            #

            # Stats tests
            a = DEres['test']['formatted'][stats_test][p][:,-1]
            b = NESres['test']['formatted'][stats_test][p][:,-1]
            t = scipy.stats.ttest_ind(a, b, equal_var=False)
            wr = scipy.stats.wilcoxon(a, b)
            return_data = page_cut_subtract(DEres['test']['formatted']['ncomps'][p], DEres['test']['mean'][page_stats_test][p], NESres['test']['formatted']['ncomps'][p], NESres['test']['mean'][page_stats_test][p])
            pageMatrix_DE_OAIES.append(return_data)
            extended_test = [DEres['details']['test_data'][p].y for cont in range(len(a))]
            extended_test = np.concatenate(extended_test)

            # McNemar’s Test
            # https://towardsdatascience.com/statistical-tests-for-comparing-classification-algorithms-ac1804e79bb7
            if 'acc' in stats_test:
                mcn, mcn_p = mcnemar_test(extended_test ,np.concatenate(DEres['test']['formatted']['final_pred'][p]), np.concatenate(NESres['test']['formatted']['final_pred'][p]))
            else:
                mcn_p = np.nan
            #print(t)
            #print(t.pvalue)
            #print(wr)
            #print(f"chi² statistic: {mcn}, p-value: {mcn_p}\n")
            #exit()
            Stext.append("{a:^10}&{b:^10}&{c:^10}&{d:^10}&{e:^10}".format(a='DE', b='OAIES', c='%.4f' % float(t.pvalue), d='%.4f' % float(wr.pvalue), e='%.4f' % mcn_p))

            #

            # Stats tests
            a = DEres['test']['formatted'][stats_test][p][:,-1]
            b = CMAESres['test']['formatted'][stats_test][p][:,-1]
            t = scipy.stats.ttest_ind(a, b, equal_var=False)
            wr = scipy.stats.wilcoxon(a, b)
            return_data = page_cut_subtract(DEres['test']['formatted']['ncomps'][p], DEres['test']['mean'][page_stats_test][p], CMAESres['test']['formatted']['ncomps'][p], CMAESres['test']['mean'][page_stats_test][p])
            pageMatrix_DE_CMAES.append(return_data)
            if 'acc' in stats_test:
                mcn, mcn_p = mcnemar_test(extended_test ,np.concatenate(DEres['test']['formatted']['final_pred'][p]), np.concatenate(CMAESres['test']['formatted']['final_pred'][p]))
            else:
                mcn_p = np.nan
            Stext.append("{a:^10}&{b:^10}&{c:^10}&{d:^10}&{e:^10}".format(a='DE', b='CMAES', c='%.4f' % float(t.pvalue), d='%.4f' % float(wr.pvalue), e='%.4f' % mcn_p))

            #

            # Stats tests
            a = NESres['test']['formatted'][stats_test][p][:,-1]
            b = CMAESres['test']['formatted'][stats_test][p][:,-1]
            t = scipy.stats.ttest_ind(a, b, equal_var=False)
            wr = scipy.stats.wilcoxon(a, b)
            return_data = page_cut_subtract(NESres['test']['formatted']['ncomps'][p], NESres['test']['mean'][page_stats_test][p], CMAESres['test']['formatted']['ncomps'][p], CMAESres['test']['mean'][page_stats_test][p])
            pageMatrix_OAIES_CMAES.append(return_data)
            if 'acc' in stats_test:
                mcn, mcn_p = mcnemar_test(extended_test ,np.concatenate(NESres['test']['formatted']['final_pred'][p]), np.concatenate(CMAESres['test']['formatted']['final_pred'][p]))
            else:
                mcn_p = np.nan
            Stext.append("{a:^10}&{b:^10}&{c:^10}&{d:^10}&{e:^10}".format(a='OAIES', b='CMAES', c='%.4f' % float(t.pvalue), d='%.4f' % float(wr.pvalue), e='%.4f' % mcn_p))

            #if 'iris' in param:
            #    exit()


    #"""
    print("\n\nAll Page Trend values...")

    titles = []

    # return_data --> x, y, cuts_x, cuts_y

    matrix = [c.cuts_y for c in pageMatrix_DE_OAIES]
    # l, m, n, p = page_test(matrix)  # my page test, doesn't give opposing confidence
    r = ptest(matrix)
    print("DE/OAIES page results:", r)
    titles.append("DE/OAIES page results\n L=%.2f, p-value=%.5f, method=%s" % (r.statistic, r.pvalue, r.method))



    matrix = [c.cuts_y for c in pageMatrix_DE_CMAES]
    #l, m, n, p = page_test(matrix)
    r = ptest(matrix)
    print("DE/CMAES page p-value:", r)
    titles.append("DE/CMAES page results\n L=%.2f, p-value=%.5f, method=%s" % (r.statistic, r.pvalue, r.method))
    #print(np.array(matrix))
    #exit()


    matrix = [c.cuts_y for c in pageMatrix_OAIES_CMAES]
    # l, m, n, p = page_test(matrix)
    r = ptest(matrix)
    print("OAIES/CMAES page p-value:", r)
    titles.append("OAIES/CMAES page results\n L=%.2f, p-value=%.5f, method=%s" % (r.statistic, r.pvalue, r.method))
    # """

    #print(">", np.around(np.matrix(matrix), 4))
    #exit()


    #"""
    labs = ['DE_OAIES', 'DE_CMAES', 'OAIES_CMAES']
    for i, Strut in enumerate([pageMatrix_DE_OAIES, pageMatrix_DE_CMAES, pageMatrix_OAIES_CMAES]):

        matrix = [c.cuts_y for c in Strut]

        fig = plt.figure()
        for f, arr in enumerate(matrix):
            plt.plot(arr, label='f%d (%s)' % (f, prm['exp']['sweep'][f]), alpha=0.75)
        plt.xlabel('cuts')
        plt.ylabel('fitness{A-B}')
        plt.title(titles[i])
        plt.legend()
        fig.savefig("%s/Page__%s__Cuts.png" % (exp_dir, labs[i]), dpi=300)
        plt.close()
    #exit()
    #"""




    #"""
    labs = ['DE_OAIES', 'DE_CMAES', 'OAIES_CMAES']
    for i, Strut in enumerate([pageMatrix_DE_OAIES, pageMatrix_DE_CMAES, pageMatrix_OAIES_CMAES]):

        y = [c.y for c in Strut]
        x = [c.cx for c in Strut]
        # cx = [c.cuts_x for c in Strut]
        cy = [c.cuts_y for c in Strut]

        ymin = 1000
        ymax = -1000
        for f, arr in enumerate(y):
            if np.min(arr) < ymin:
                ymin = np.min(arr)
            if np.max(arr) > ymax:
                ymax = np.max(arr)

        fig, ax = plt.subplots()

        for f, arr in enumerate(y):
            ax.plot(x[f],y[f], label='f%d (%s)' % (f, prm['exp']['sweep'][f]), alpha=1)
            ax.plot(cy[f],'x', color='k', alpha=0.75, markersize=5)
            for xx in range(len(cy)-1):
                ax.plot([xx,xx],[ymin,ymax],color='k', alpha=0.75, markersize=5)
        ax.set_xlabel('cuts')
        ax.set_ylabel('fitness{A-B}')

        if formal == 0:
            fig.suptitle(titles[i])

        handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='k', marker='x', markersize=5, label='cuts', linestyle=''))

        ax.legend(handles=handles)
        fig.savefig("%s/Page__%s__Plots.pdf" % (exp_dir, labs[i]), dpi=300)
        plt.close()
    #exit()
    #"""



    #

    Deets = "\n>> Results << \n"
    for Deets_Line in Ttext:
        Deets = '%s%s \n' % (Deets, Deets_Line)
    #print(Deets)
    #exit()

    #

    Deets = "\n>> Stats Results << \n"
    for Deets_Line in Stext:
        Deets = '%s%s \n' % (Deets, Deets_Line)
    #print(Deets)


    return






#

#

# fin
