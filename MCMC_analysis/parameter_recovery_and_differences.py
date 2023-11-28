import matplotlib.pyplot
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import expit as inv_logit
import numpy as np
import pandas as pd
import scipy
import arviz
import pymc as pm
import pytensor
import matplotlib.pyplot as plt
import hpd
from hpd import hpd_grid



def Phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    # A.k.a the probit transform
    error = 0.001
    inv_logit_value = inv_logit(x)
    return (error + (1.0-2.0*error)*inv_logit_value)
    #return pm.math.sigmoid(x*10)

def cnv(x,g):
    return np.log(1+x*g)/g


def parameter_comparison(trace_food,trace_money):
    x = np.exp(trace_food.posterior['k_mean'][:, :, :].values).flatten() - np.exp(
        trace_money.posterior['k_mean'][:, :, :].values).flatten()
    # x = trace_food.posterior['k_mean'].values.flatten() - trace_money.posterior['k_mean'].values.flatten()
    hdi, trash, trash2, mean = hpd_grid(x, alpha=0.05, roundto=1)
    result1 = plt.hist(x, color='k', edgecolor='k', alpha=0.8, bins=14)
    mean_value = np.mean(x)
    box_width = result1[0][1] - result1[0][0]
    new_valueMin = np.interp(hdi[0][0], [x.min(), x.max()], [0, 1])  #
    new_valueMax = np.interp(hdi[0][1], [x.min(), x.max()], [0, 1])  # [result1[1][0],result1[1][-1]]
    plt.axhline(xmin=new_valueMin, xmax=new_valueMax - 0.01, linewidth=6, label='95%HDI', color='r')  #
    plt.axvline(x=mean_value, label='mean', color='b', ls='--')
    plt.legend()
    plt.title('Difference between estimates for k')

    return 0

def parameter_recovery(trace_model,dataframe):
    stats.pearsonr(np.exp(trace_model.posterior['k'].values.mean(axis=(0, 1))), np.exp(dataframe['k']))
    stats.pearsonr(trace_model.posterior['s'].values.mean(axis=(0, 1)), dataframe['s'])
    stats.pearsonr(trace_model.posterior['error'].values.mean(axis=(0, 1)), dataframe['error'])

    return 0


