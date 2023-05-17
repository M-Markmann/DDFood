
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot
import sklearn
import numpy as np
import statistics as stats
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

def hyberbolic_discounting(Delay,k):
    SV_delayed = 40/(1+k*Delay)
    return SV_delayed

def exponential_discounting(Delay, k):
    SV_delayed = 40*exp(-k*Delay)
    return SV_delayed

def green_meyerson_discounting(Delay, k, s):
    SV_delayed = 40/np.power((1+k*Delay),s)
    return SV_delayed

def mazur(Delay, k, s):
    SV_delayed = 40/(1+k*np.power(Delay,s))
    return SV_delayed

def prelec(Delay, k, s):
    SV_Delayed = 40 * exp(-(np.power(((k*Delay)),s)))
    return SV_Delayed


def single_parameter_softmax_hyperbol(Choosen_Reward_D,immediate_Offer,Delay,k, beta):
    if Choosen_Reward_D == 0:
        p_choosen = (exp(hyberbolic_discounting(Delay,k)/beta))/((exp(immediate_Offer/beta))+(exp(hyberbolic_discounting(Delay,k)/beta)))
    if Choosen_Reward_D == 1:
        p_choosen = (exp(immediate_Offer/beta))/((exp(immediate_Offer/beta))+(exp(hyberbolic_discounting(Delay,k)/beta)))

    return p_choosen

def single_parameter_softmax_exponential(Choosen_Reward_D,immediate_Offer,Delay,k, beta):
    if Choosen_Reward_D == 0:
        p_choosen = (exp(exponential_discounting(Delay,k)/beta))/((exp(immediate_Offer/beta))+(exp(exponential_discounting(Delay,k)/beta)))
    if Choosen_Reward_D == 1:
        p_choosen = (exp(immediate_Offer/beta))/((exp(immediate_Offer/beta))+(exp(exponential_discounting(Delay,k)/beta)))

    return p_choosen

def two_parameter_softmax_green(Choosen_Reward_D, immediate_Offer, Delay, k, beta, s):
    if Choosen_Reward_D == 0:
        p_choosen = (exp(green_meyerson_discounting(Delay,k, s)/beta))/((exp(immediate_Offer/beta))+(exp(green_meyerson_discounting(Delay,k, s)/beta)))
    if Choosen_Reward_D == 1:
        p_choosen = (exp(immediate_Offer/beta))/((exp(immediate_Offer/beta))+(exp(green_meyerson_discounting(Delay,k, s)/beta)))

    return p_choosen

def two_parameter_softmax_mazur(Choosen_Reward_D, immediate_Offer, Delay, k, beta, s):
    if Choosen_Reward_D == 0:
        p_choosen = (exp(mazur(Delay,k, s)/beta))/((exp(immediate_Offer/beta))+(exp(mazur(Delay,k, s)/beta)))
    if Choosen_Reward_D == 1:
        p_choosen = (exp(immediate_Offer/beta))/((exp(immediate_Offer/beta))+(exp(mazur(Delay,k, s)/beta)))

    return p_choosen


def two_parameter_softmax_prelec(Choosen_Reward_D, immediate_Offer, Delay, k, beta, s):
    if Choosen_Reward_D == 0:
        p_choosen = (exp(prelec(Delay,k, s)/beta))/((exp(immediate_Offer/beta))+(exp(prelec(Delay,k, s)/beta)))
    if Choosen_Reward_D == 1:
        p_choosen = (exp(immediate_Offer/beta))/((exp(immediate_Offer/beta))+(exp(prelec(Delay,k, s)/beta)))

    return p_choosen
