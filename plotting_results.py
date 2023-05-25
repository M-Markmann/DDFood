
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot
import sklearn
import numpy as np
import statistics_and_plots as stats
import math
from math import exp
import scipy.optimize
from scipy.optimize import curve_fit
import os
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import pandas as pd
import glob
import scipy
import scipy.stats
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage import uniform_filter1d
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
#
from sklearn.metrics import r2_score


def get_wilcoxon_rank_and_make_fancy_graphics(df,significance_value):
    df = df
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how="outer")
    correlations = dfcols.transpose().join(dfcols, how="outer")
    correlations = correlations.to_numpy()
    pvalues = pvalues.to_numpy()
    df_numpy = df.to_numpy()
    for ix in range(np.size(df_numpy, 1)):
        for jx in range(np.size(df_numpy, 1)):
            if ix is not jx:
                sp = scipy.stats.wilcoxon(df_numpy[:, ix], df_numpy[:, jx])
                correlations[ix, jx] = sp[0]
                pvalues[ix, jx] = sp[1]  # Only store values below the diagonal
            else:
                correlations[ix, jx] = 1
                pvalues[ix, jx] = 1

    uncorrected_p_values = pd.DataFrame(pvalues,
                                        columns=['Exponential', 'Hyperbole', 'Hyperbole \n with time scaling \n Delay', 'Hyperbole \n with scaling \n Delay & Discounting',
     'Exponential \n with time scaling'], #df.columns.values.tolist(),
                                        index=['Exponential', 'Hyperbole', 'Hyperbole \n with time scaling \n Delay', 'Hyperbole \n with scaling \n Delay & Discounting',
     'Exponential \n with time scaling'], dtype="float")


    g = sns.clustermap(uncorrected_p_values, cmap=sns.dark_palette("#D2042D",reverse=True), vmin=0, vmax=0.1, row_cluster = False, col_cluster = False)
    #"rocket"
    #plt.xticks([0.5, 1.5])
    #plt.yticks([0.5, 1.5])
    #plt.title('Confusion matrix')
    #plt.xlabel('Actual label')
    #plt.ylabel('Predicted label');

    # Here labels on the y-axis are rotated
    for tick in g.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    as_numpy_uncorrected = uncorrected_p_values.to_numpy()
    counter = [0,1,2,3,4]
    # Here we add asterisks onto cells with signficant correlations
    for i, ix in enumerate(counter):#enumerate(g.dendrogram_row.reordered_ind):
        for j, jx in enumerate(counter):#enumerate(g.dendrogram_row.reordered_ind):
            if i != j:
                if as_numpy_uncorrected[ix, jx] <= significance_value and as_numpy_uncorrected [ix, jx]> significance_value/10:
                    text_value = "*"
                elif  as_numpy_uncorrected[ix, jx] <= significance_value/10  and as_numpy_uncorrected [ix, jx]> significance_value/1000:
                    text_value = "**"
                elif  as_numpy_uncorrected[ix, jx] <= significance_value/1000 and as_numpy_uncorrected [ix, jx]> significance_value/10000:
                    text_value = "***"
                elif  as_numpy_uncorrected[ix, jx] <= significance_value/10000:
                    text_value = "****"
                else:
                    text_value = ""
                text = g.ax_heatmap.text(
                    j + 0.5,
                    i + 0.5,
                    text_value,
                    ha="center",
                    va="center",
                    color="white",
                )
                text.set_fontsize(18)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=18)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=18)
    return g

#food_rsquared_heatmap = get_wilcoxon_rank_and_make_fancy_graphics(rsquared_data_unfiltered[['r2_exponential_food','r2_hyberbole_food','r2_mazur_food','r2_GM_food','r2_prelec_food']],significance_value=0.05)

def get_correlation_strength_and_make_fancy_graphics(df,significance_value,labels_to_be_correlated):
   # df = pd.DataFrame(df)
    df = df
    #len = df.columns()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how="outer")
    correlations = dfcols.transpose().join(dfcols, how="outer")
    correlations = correlations.to_numpy()

    pvalues = pvalues.to_numpy()
    df_numpy = df.to_numpy(dtype=float)
    for ix in range(np.size(df_numpy, 1)):
        for jx in range(np.size(df_numpy, 1)):
            if ix is not jx:
                corr_strength, corr_significancy = scipy.stats.spearmanr(df_numpy[:, ix], df_numpy[:, jx])
                correlations[ix, jx] = corr_strength
                pvalues[ix, jx] = corr_significancy  # Only store values below the diagonal
            else:
                correlations[ix, jx] = 1
                pvalues[ix, jx] = 1

    uncorrected_correlation_table = pd.DataFrame(correlations,
                                        columns=labels_to_be_correlated, #df.columns.values.tolist(),
                                        index=labels_to_be_correlated, dtype="float")




    g = sns.heatmap(uncorrected_correlation_table, cmap="rocket", vmin=0, vmax=0.05)

    #plt.xticks([0.5, 1.5])
    #plt.yticks([0.5, 1.5])
    #plt.title('Confusion matrix')
    #plt.xlabel('Actual label')
    #plt.ylabel('Predicted label');

    # Here labels on the y-axis are rotated
    for tick in g.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    as_numpy_uncorrected = uncorrected_correlation_table.to_numpy()

    # Here we add asterisks onto cells with signficant correlations
    for i, ix in enumerate(range(0,g.mask.shape[0]-1)):
        for j, jx in enumerate(range(0,g.mask.shape[0]-1)):
            if i != j:
                if pvalues[ix, jx] < significance_value and pvalues [ix, jx]> significance_value/10:
                    text_value = "*"
                elif  pvalues[ix, jx] < significance_value/100  and pvalues [ix, jx]> significance_value/1000:
                    text_value = "**"
                elif  pvalues[ix, jx] < significance_value/1000 and pvalues [ix, jx]> significance_value/10000:
                    text_value = "***"
                elif  pvalues[ix, jx] < significance_value/10000:
                    text_value = "****"
                else:
                    text_value = ""
                text = g.ax_heatmap.text(
                    j + 0.5,
                    i + 0.5,
                    text_value,
                    ha="center",
                    va="center",
                    color="white",
                )
                if (labels_to_be_correlated.__len__() > 6):
                    text.set_fontsize(12)
                else:
                    text.set_fontsize(12)

    return g