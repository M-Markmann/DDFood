# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:14:48 2020
Script for Elody for calculating k values from log files
@author: marius
"""

# This is a sample Python script.
import csv
from scipy.interpolate import interp1d

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
#import pyddm



def aicc_correction(parameter,sample_size):
    return (2 * pow(2, parameter) + 2 * parameter) / (sample_size - parameter - 1)


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



def indifference_point_calculator(numbers, start, end):
    indif = int(20)
    if numbers[0, 4] == 1:
        indif += -10
    elif numbers[0, 4] == 0:
        indif += 10
    if numbers[1, 4] == 1:
        indif += -5
    elif numbers[1, 4] == 0:
        indif += 5
    if numbers[2, 4] == 1:
        indif += -2.5
    elif numbers[2, 4] == 0:
        indif += 2.5
    if numbers[3, 4] == 1:
        indif += -1
    elif numbers[3, 4] == 0:
        indif += 1

    change_happened = any(numbers[:,4])
    return indif, change_happened
    # if numbers[start]
    # print(numbers)/media/data/elodiesdaten/BehavioralDATA
    # return


def hyperbole_function(x, k):
    return 40 / (1 + x * k)

def calculate_indiff_bayesian_prelec(delay,k,b,s,return_prop_function=False):
    current_best_result =40
    current_best_index = 0
    step_size = 0.01
    prop_list = []
    index_list = []
    for i in np.arange(0,40,step_size):
        result = two_parameter_softmax_prelec(1, i, delay, k, b, s)
        distance = abs(0.5-result)
        prop_list.append(result)
        index_list.append(i)
        #print(distance)
        if distance<current_best_result:
            current_best_result = distance
            current_best_index = i
    if return_prop_function:
        return prop_list, index_list
    else:
        return [current_best_result, current_best_index]


def calculate_indiff_bayesian_mazur(delay,k,b,s,return_prop_function=False):
    current_best_result =40
    current_best_index = 0
    step_size = 0.01
    prop_list = []
    index_list = []
    for i in np.arange(0, 40, step_size):
        result = two_parameter_softmax_prelec(1, i, delay, k, b, s)
        distance = abs(0.5 - result)
        prop_list.append(result)
        index_list.append(i)
        if distance<current_best_result:
            current_best_result = distance
            current_best_index = i

    if return_prop_function:
        return prop_list, index_list
    else:
        return [current_best_result, current_best_index]

def single_result(method):

    method = 'SLSQP'
    folder = 'C:/Users/mariu/Documents/Arbeit/Doktorarbeit/Abbildungen/'
    results_dir = os.path.dirname(folder)
    results_methods_dir = os.path.join(results_dir, method)
    if not os.path.exists(results_methods_dir):
        os.makedirs(results_methods_dir)
    data_path = 'C:/Users/mariu/Documents/Arbeit/DelayDiscountingFood/BehavioralDATA/'
    q = os.listdir(data_path)
    method = method  # was nelder mean
    maxiter = 100000
    all_data = []
    rsquared_parameter_estimates = []
    rsquared_data = []
    rsquared_data_unfiltered = []
    rsquared_parameters = []
    log_likelihood_food =[]
    log_likelihood_money = []
    counter = 0
    for file in q:
        counter = counter + 1

        file_name = data_path + file
        f = open(file_name, "r")
        participant_code = file[:7]
        # indifference_list = np.ndarray(shape=(2,2,6,5),dtype=int)
        indifference_liste = []
        Coin_Task = 0  # 0 = food
        Counter_Decisions = 0;
        Decision = 0  # later = 0 now = 1
        f1 = f.readlines()

        act_vol_time = 0
        pulse_counter = 0
        Offer_Food = []
        Offer_Food_now = []
        Offer_Food_later = []
        Offer_Coin = []
        Offer_Coin_now = []
        Offer_Coin_later = []
        indifference_liste_Coin_Now = []
        indifference_liste_Coin_Later = []
        indifference_liste_Food_Now = []
        indifference_liste_Food_Later = []
        error_trials_foos = []
        error_trials_coin = []
        Trial_duration_Food = []
        Trial_duration_Food_Now = []
        Trial_duration_Food_Later = []
        Trial_duration_Coin = []
        Trial_duration_Coin_now = []
        Trial_duration_Coin_Later = []
        end_offer_timing = 0
        feedback_trials_food = []
        feedback_trials_coin = []
        events_script = []
        start_point = 0
        startl_point = 0

        f1 = [x for x in f1 if not str('last trial') in x]
        f1 = [x for x in f1 if not str('start trial') in x]
        entry_counter = 0
        immediate_offer = 20
        immediate_offer_next = 20
        choice_counter = 1
        for x in f1:  # This analyses the logfile line by line, extracting all Decisions made,the Conditiona and the Time displayed
            split_string = x.split("\t");
            # print(split_string)
            if "Pulse" in split_string:
                act_vol_time = float(split_string[4]) / 10000 - startl_point
                pulse_counter += 1
                if pulse_counter == 1:
                    start_point = 0  # act_vol_time-2.5
                    startl_point = float(split_string[4]) / 10000
            elif "Response" in split_string and pulse_counter > 1:
                diftime_response = float(split_string[4]) / 10000 - act_vol_time
                ons_response = (pulse_counter * 2.5 + diftime_response - start_point)
                reaction_time = ons_response - end_offer_timing
            elif "coin.jpg" in str(split_string) and pulse_counter > 1:
                Coin_Task = 1
                diftime_offer = float(split_string[4]) / 10000 - act_vol_time
                ons_offer_coin = (pulse_counter * 2.5 + diftime_offer - start_point)
                end_offer_timing = ons_offer_coin + float(split_string[5]) / 10000

            elif "food" in str(split_string) and pulse_counter > 1 and not "foodchoice" in str(
                    split_string) and ".jpg" in str(split_string):
                Coin_Task = 0
                if "food1.jpg" in split_string:
                    food_chooser = 1
                elif "food2.jpg" in split_string:
                    food_chooser = 2
                elif "food3.jpg" in split_string:
                    food_chooser = 3
                elif "food4.jpg" in split_string:
                    food_chooser = 4
                diftime_offer = float(split_string[4]) / 10000 - act_vol_time
                ons_offer_food = (pulse_counter * 2.5 + diftime_offer - start_point)
                end_offer_timing = ons_offer_food + float(split_string[5]) / 10000
                # split_string = split_string[3].split("/t")
            elif "Picture" in split_string and ".jpg" not in str(split_string) and "foodchoice" not in str(
                    split_string) and pulse_counter > 1:
                # print(split_strin1g[3])
                # print(Coin_Task)
                # print(split_string)
                diftime_fdb = float(split_string[4]) / 10000 - act_vol_time
                relevant_data = split_string[3].split(",")[0]
                # print(split_string)
                ons_fdb = ((pulse_counter * 2.5) + diftime_fdb - start_point)

                split_string_timing = split_string[3].split(";")
                split_string_decision = split_string_timing[0].split(",")
                split_string_timing = split_string_timing[1]
                split_string_decision = split_string_decision[1]

                # print(split_string_timing[1])
                if "in 2 Tagen" in str(split_string_timing):
                    day = 2
                elif "in 2 Wochen" in str(split_string_timing):
                    day = 14
                elif "in 1 Monat" in str(split_string_timing):
                    day = 31
                elif "in 3 Monaten" in str(split_string_timing):
                    day = 91
                elif "in 6 Monaten" in str(split_string_timing):
                    day = 182.5
                elif "in 1 Jahr" in str(split_string_timing):
                    day = 365
                Counter_Decisions += 1

                if Counter_Decisions > 5:
                    Counter_Decisions = 1

                if split_string_decision == split_string_timing:
                    Decision = 0  # später
                else:
                    Decision = 1  # jetzt
                # print(Coin_Task,First_Round,Counter_Decisions,Decision)
                # print(Coin_Task)
                if choice_counter > 5:
                    choice_counter = 1

                if (choice_counter == 1 and Decision == 0):  # later
                    immediate_offer = 20
                    immediate_offer_next = immediate_offer + 10
                elif (choice_counter == 1 and Decision == 1):  # now
                    immediate_offer = 20
                    immediate_offer_next = immediate_offer - 10
                elif (choice_counter == 2 and Decision == 0):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer + 5
                elif (choice_counter == 2 and Decision == 1):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer - 5
                elif (choice_counter == 3 and Decision == 0):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer + 2.5
                elif (choice_counter == 3 and Decision == 1):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer - 2.5
                elif (choice_counter == 4 and Decision == 0):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer + 1
                elif (choice_counter == 4 and Decision == 1):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer - 1
                elif (choice_counter == 5):
                    immediate_offer = immediate_offer_next

                if Coin_Task == 0:
                    Offer_Food.append(ons_offer_food)  # was ons_offer_food
                    Trial_duration_Food.append(ons_response - ons_offer_food)
                    feedback_trials_food.append(ons_fdb)
                elif Coin_Task == 1:
                    Offer_Coin.append(ons_offer_coin)
                    Trial_duration_Coin.append(ons_response - ons_offer_coin)
                    feedback_trials_coin.append(ons_fdb)

                if Coin_Task == 0 and Decision == 0:
                    if reaction_time<8:
                        Trial_duration_Food_Later.append(ons_response - ons_offer_food)
                        Offer_Food_later.append(ons_response)
                    else:
                        error_trials_foos.append(ons_response)
                    indifference_liste_Food_Later.append(
                        (entry_counter, Coin_Task, day, Counter_Decisions, Decision, immediate_offer))
                elif Coin_Task == 0 and Decision == 1:
                    if reaction_time < 8:
                        Trial_duration_Food_Now.append(ons_response - ons_offer_food)
                        Offer_Food_now.append(ons_response)
                    else:
                        error_trials_foos.append(ons_response)
                    indifference_liste_Food_Now.append(
                        (entry_counter, Coin_Task, day, Counter_Decisions, Decision, immediate_offer))
                elif Coin_Task == 1 and Decision == 0:
                    if reaction_time < 8:
                        Trial_duration_Coin_Later.append(ons_response - ons_offer_coin)
                        Offer_Coin_later.append(ons_response)
                    else:
                        error_trials_coin.append(ons_response)
                    indifference_liste_Coin_Later.append(
                        (entry_counter, Coin_Task, day, Counter_Decisions, Decision, immediate_offer))
                elif Coin_Task == 1 and Decision == 1:
                    if reaction_time < 8:
                        Trial_duration_Coin_now.append(ons_response - ons_offer_coin)
                        Offer_Coin_now.append(ons_response)
                    else:
                        error_trials_coin.append(ons_response)
                    indifference_liste_Coin_Now.append(
                        (entry_counter, Coin_Task, day, Counter_Decisions, Decision, immediate_offer))

                choice_counter = choice_counter + 1
                entry_counter = entry_counter + 1
                if Coin_Task == 0:
                    duration = ons_response - ons_offer_food
                    trial_start = ons_offer_food
                    condition = "Food"

                else:
                    duration = ons_response -ons_offer_coin
                    trial_start = ons_offer_coin
                    condition = "Coin"

                if Decision== 0:
                    choice = "Later"
                else:
                    choice = "Now"
                events_script.append((trial_start,duration,day,condition,immediate_offer,choice))
                indifference_liste.append((entry_counter, Coin_Task, day, Counter_Decisions, Decision, immediate_offer))

        indifference_liste = np.array(indifference_liste)
        indifference_liste_with_all = indifference_liste
        #indifference_liste = indifference_liste[indifference_liste[:,2]< 180]
        indifference_liste_Coin_Later = np.array(indifference_liste_Coin_Later)
        indifference_liste_Coin_Now = np.array(indifference_liste_Coin_Now)
        indifference_liste_Food_Later = np.array(indifference_liste_Food_Later)
        indifference_liste_Food_Now = np.array(indifference_liste_Food_Now)
        Food = indifference_liste[indifference_liste[:, 1] < 0.5]
        Money = indifference_liste[indifference_liste[:, 1] > 0.5]

        indifference_liste = np.array(indifference_liste_with_all)
        list_of_indifference_points = []
        did_change_happened = []
        food_chooser = [food_chooser]
        for i in range(int(entry_counter / 5)):
            # print((i+1)*5)
            start = ((i + 1) * 5) - 5
            end = ((i + 1) * 5)
            # print(start,end)
            indifference_point, change_happened = indifference_point_calculator(indifference_liste_with_all[start:end][:][:][:][:], start, end)
            list_of_indifference_points.append(
                (start, indifference_liste_with_all[start, 1], indifference_liste_with_all[start, 2], indifference_point))
            did_change_happened.append(change_happened)
            # print(indifference_point)
        list_of_indifference_points = np.array(list_of_indifference_points)
        # list_of_indifference_points= list_of_indifference_points[list_of_indifference_points[:,1].argsort()]
        list_of_indifference_points = list_of_indifference_points[list_of_indifference_points[:, 2].argsort()]
        Food_indifference_list = list_of_indifference_points[
            list_of_indifference_points[:, 1] < 0.5]  # Create List for all Food Indifference Points
        Money_indifference_list = list_of_indifference_points[
            list_of_indifference_points[:, 1] > 0.5]  # Create List for all Money Decisions
        Food_Decision1 = []
        Food_Decision2 = []
        Money_Decision1 = []
        Money_Decision2 = [] #np.sum()
        Food_is_calculatable = np.array(np.multiply(did_change_happened[0:5],1) + np.multiply(did_change_happened[13:18],1))
        Food_is_calculatable = np.min(Food_is_calculatable) >= 1
        Coin_is_calculatable = np.array((np.multiply(did_change_happened[6:11],1))+ np.multiply(did_change_happened[19:24],1))
        Coin_is_calculatable = np.min(Coin_is_calculatable) >= 1
        for i in range(int(len(Food_indifference_list))):
            if i % 2 == 0:
                if Food[i, 0] > Food[i + 1, 0]:
                    Food_Decision1.append(Food_indifference_list[i + 1, 3])
                    Food_Decision2.append(Food_indifference_list[i, 3])
                else:
                    Food_Decision1.append(Food_indifference_list[i, 3])
                    Food_Decision2.append(Food_indifference_list[i + 1, 3])
        for i in range(int(len(Money_indifference_list))):
            if i % 2 == 0:
                if Money[i, 0] > Money[i + 1, 0]:
                    Money_Decision1.append(Money_indifference_list[i + 1, 3])
                    Money_Decision2.append(Money_indifference_list[i, 3])
                else:
                    Money_Decision1.append(Money_indifference_list[i, 3])
                    Money_Decision2.append(Money_indifference_list[i + 1, 3])

        # list_of_indifference_points = np.sort(list_of_indifference_points,axis=0)
        # indifference_point(Decisions_near)
        timings = [2, 14, 31, 91, 182.5, 365]
        indifference_attacher = []
        abs_value = []
        rel_value = []

        All_Money_Decisions = (np.array(Money_Decision1) + np.array(Money_Decision2)) / 2
        All_Food_Decisions = (np.array(Food_Decision1) + np.array(Food_Decision2)) / 2
        initial_guess_one_r2 = (0.00001)  # k
        initial_guess_tw0_r2 = [0.0055, 0.403]
        bounds_one_r2 = (0.00001, 1)  # k
        bounds_two_r2 = ((0.00001, 0.01), (1, 1.5))  # k,s
        bounds_two = [(0.1, 10), (0.00001, 1)]  # beta, k
        initial_guess_two = np.array([2, 0.055])  # beta, k
        bounds_three = [(0.1, 10), (0.00001, 1), (0.01,  1.5)]  # beta, k, s
        initial_guess_three = np.array([2, 0.055, 0.42])


        array = Food
        print(np.shape(array))

        def log_likelihood_single_parameter(params):
            beta, k = params
            log_likelihood = 0
            for fun_counter in range(np.size(array, axis=0)):
                Choosen_Reward_D = array[fun_counter, 4]
                immediate_offer = array[fun_counter, 5]
                Delay = array[fun_counter, 2]
                likeness = np.log(single_parameter_softmax_hyperbol(Choosen_Reward_D, immediate_offer, Delay, k, beta))
                log_likelihood = likeness + log_likelihood

            return 1 / log_likelihood

        DD_Food_Fit_hyperbol = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess_two,
                                                       method=method, bounds=bounds_two, options={'maxiter': maxiter})
        #   DD_Near_Fit_hyperbol = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess, method=method, options={'maxiter': maxiter})
        # ,bounds=bounds
        params = [DD_Food_Fit_hyperbol.x[0], DD_Food_Fit_hyperbol.x[1]]
        log_likelihood_food_hyperbol = log_likelihood_single_parameter(params)
        aic_food_hyperbol = -2 * 1 / log_likelihood_food_hyperbol + 2 * 2
        aic_food_hyperbol = aic_food_hyperbol + aicc_correction(2, np.size(array, axis=0))
        print("AIC_Food_Hyperbol")
        print(aic_food_hyperbol)

        # exponential -- Food
        def log_likelihood_single_parameter(params):
            beta, k = params
            log_likelihood = 0
            for fun_counter in range(np.size(array, axis=0)):
                Choosen_Reward_D = array[fun_counter, 4]
                immediate_offer = array[fun_counter, 5]
                Delay = array[fun_counter, 2]
                likeness = np.log(
                    single_parameter_softmax_exponential(Choosen_Reward_D, immediate_offer, Delay, k, beta))
                log_likelihood = likeness + log_likelihood

            return 1 / log_likelihood

        DD_Food_Fit_exponential = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess_two,
                                                          method=method, bounds=bounds_two,
                                                          options={'maxiter': maxiter})
        #   DD_Near_Fit_hyperbol = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess, method=method, options={'maxiter': maxiter})
        # , bounds=bounds
        params = [DD_Food_Fit_exponential.x[0], DD_Food_Fit_exponential.x[1]]
        log_likelihood_food_exponential = log_likelihood_single_parameter(params)
        aic_food_exponential = -2 * 1 / log_likelihood_food_exponential + 2 * 2
        aic_food_exponential = aic_food_exponential + aicc_correction(2, np.size(array, axis=0))
        print("AIC_Food_expmemtoaö")
        print(aic_food_exponential)

        def log_likelihood_single_parameter(params):
            beta, k, s = params
            log_likelihood = 0
            for fun_counter in range(np.size(array, axis=0)):
                Choosen_Reward_D = array[fun_counter, 4]
                immediate_offer = array[fun_counter, 5]
                Delay = array[fun_counter, 2]
                likeness = np.log(two_parameter_softmax_green(Choosen_Reward_D, immediate_offer, Delay, k, beta, s))
                log_likelihood = likeness + log_likelihood

            return 1 / log_likelihood

        DD_Food_Fit_meyer_green = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess_three,
                                                          bounds=bounds_three,
                                                          method=method, options={'maxiter': maxiter})
        # , bounds=bounds
        params = [DD_Food_Fit_meyer_green.x[0], DD_Food_Fit_meyer_green.x[1], DD_Food_Fit_meyer_green.x[2]]
        log_likelihood_Food_meyer_green = log_likelihood_single_parameter(params)
        aic_food_meyer_green = -2 * 1 / log_likelihood_Food_meyer_green + 2 * 3
        aic_food_meyer_green = aic_food_meyer_green + aicc_correction(3, np.size(array, axis=0))
        print("AIC Food meyer green")
        print(aic_food_meyer_green)

        # Mazur
        def log_likelihood_single_parameter(params):
            beta, k, s = params
            log_likelihood = 0
            for fun_counter in range(np.size(array, axis=0)):
                Choosen_Reward_D = array[fun_counter, 4]
                immediate_offer = array[fun_counter, 5]
                Delay = array[fun_counter, 2]
                likeness = np.log(two_parameter_softmax_mazur(Choosen_Reward_D, immediate_offer, Delay, k, beta, s))
                log_likelihood = likeness + log_likelihood

            return 1 / log_likelihood

        DD_Food_Fit_mazur = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess_three,
                                                    bounds=bounds_three,
                                                    method=method, options={'maxiter': maxiter})
        # , bounds=bounds
        params = [DD_Food_Fit_mazur.x[0], DD_Food_Fit_mazur.x[1], DD_Food_Fit_mazur.x[2]]
        log_likelihood_Food_mazur = log_likelihood_single_parameter(params)
        aic_food_mazur = -2 * 1 / log_likelihood_Food_mazur + 2 * 3 + aicc_correction(3, np.size(array, axis=0))
        print("AIC Food mazur")
        print(aic_food_mazur)

        # prelec
        def log_likelihood_single_parameter(params):
            beta, k, s = params
            log_likelihood = 0
            for fun_counter in range(np.size(array, axis=0)):
                Choosen_Reward_D = array[fun_counter, 4]
                immediate_offer = array[fun_counter, 5]
                Delay = array[fun_counter, 2]
                likeness = np.log(two_parameter_softmax_prelec(Choosen_Reward_D, immediate_offer, Delay, k, beta, s))
                log_likelihood = likeness + log_likelihood

            return 1 / log_likelihood

        DD_Food_Fit_prelec = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess_three,
                                                     bounds=bounds_three,
                                                     method=method, options={'maxiter': maxiter})
        # , bounds=bounds
        params = [DD_Food_Fit_prelec.x[0], DD_Food_Fit_prelec.x[1], DD_Food_Fit_prelec.x[2]]
        log_likelihood_Food_prelec = log_likelihood_single_parameter(params)
        aic_food_prelec = -2 * 1 / log_likelihood_Food_prelec + 2 * 3 + aicc_correction(3, np.size(array, axis=0))
        print("AIC Food prelec")
        print(aic_food_prelec)

        array = Money
        print(np.shape(array))

        def log_likelihood_single_parameter(params):
            beta, k = params
            log_likelihood = 0
            for fun_counter in range(np.size(array, axis=0)):
                Choosen_Reward_D = array[fun_counter, 4]
                immediate_offer = array[fun_counter, 5]
                Delay = array[fun_counter, 2]
                likeness = np.log(single_parameter_softmax_hyperbol(Choosen_Reward_D, immediate_offer, Delay, k, beta))
                log_likelihood = likeness + log_likelihood

            return 1 / log_likelihood

        DD_Money_Fit_hyperbol = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess_two,
                                                        bounds=bounds_two,
                                                        method=method, options={'maxiter': maxiter})
        # , bounds=bounds
        #   DD_Near_Fit_hyperbol = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess, method=method, options={'maxiter': maxiter})
        params = [DD_Money_Fit_hyperbol.x[0], DD_Money_Fit_hyperbol.x[1]]
        log_likelihood_money_hyperbol = log_likelihood_single_parameter(params)
        aic_money_hyperbol = -2 * 1 / log_likelihood_money_hyperbol + 2 * 2 + aicc_correction(2, np.size(array, axis=0))
        print("AIC_Money_Hyperbol")
        print(aic_money_hyperbol)

        # exponential -- Money
        def log_likelihood_single_parameter(params):
            beta, k = params
            log_likelihood = 0
            for fun_counter in range(np.size(array, axis=0)):
                Choosen_Reward_D = array[fun_counter, 4]
                immediate_offer = array[fun_counter, 5]
                Delay = array[fun_counter, 2]
                likeness = np.log(
                    single_parameter_softmax_exponential(Choosen_Reward_D, immediate_offer, Delay, k, beta))
                log_likelihood = likeness + log_likelihood

            return 1 / log_likelihood

        DD_Money_Fit_exponential = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess_two,
                                                           bounds=bounds_two,
                                                           method=method,
                                                           options={'maxiter': maxiter})

        # , bounds=bounds
        #   DD_Near_Fit_hyperbol = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess, method=method, options={'maxiter': maxiter})
        params = [DD_Money_Fit_exponential.x[0], DD_Money_Fit_exponential.x[1]]
        log_likelihood_money_exponential = log_likelihood_single_parameter(params)
        aic_money_exponential = -2 * 1 / log_likelihood_money_exponential + 2 * 2 + aicc_correction(2, np.size(array,
                                                                                                               axis=0))
        print("AIC_Money_exponential")
        print(aic_money_exponential)

        def log_likelihood_single_parameter(params):
            beta, k, s = params
            log_likelihood = 0
            for fun_counter in range(np.size(array, axis=0)):
                Choosen_Reward_D = array[fun_counter, 4]
                immediate_offer = array[fun_counter, 5]
                Delay = array[fun_counter, 2]
                likeness = np.log(two_parameter_softmax_green(Choosen_Reward_D, immediate_offer, Delay, k, beta, s))
                log_likelihood = likeness + log_likelihood

            return 1 / log_likelihood

        DD_Money_Fit_meyer_green = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess_three,
                                                           bounds=bounds_three,
                                                           method=method, options={'maxiter': maxiter})
        # ,bounds=bounds

        params = [DD_Money_Fit_meyer_green.x[0], DD_Money_Fit_meyer_green.x[1], DD_Money_Fit_meyer_green.x[2]]
        log_likelihood_Money_meyer_green = log_likelihood_single_parameter(params)
        aic_money_meyer_green = -2 * 1 / log_likelihood_Money_meyer_green + 2 * 3 + aicc_correction(3, np.size(array,
                                                                                                               axis=0))
        print("AIC Money meyer green")
        print(aic_money_meyer_green)

        # Mazur
        def log_likelihood_single_parameter(params):
            beta, k, s = params
            log_likelihood = 0
            for fun_counter in range(np.size(array, axis=0)):
                Choosen_Reward_D = array[fun_counter, 4]
                immediate_offer = array[fun_counter, 5]
                Delay = array[fun_counter, 2]
                likeness = np.log(two_parameter_softmax_mazur(Choosen_Reward_D, immediate_offer, Delay, k, beta, s))
                log_likelihood = likeness + log_likelihood

            return 1 / log_likelihood

        DD_Money_Fit_mazur = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess_three,
                                                     bounds=bounds_three,
                                                     method=method, options={'maxiter': maxiter})
        # , bounds=bounds
        params = [DD_Money_Fit_mazur.x[0], DD_Money_Fit_mazur.x[1], DD_Money_Fit_mazur.x[2]]
        log_likelihood_Money_mazur = log_likelihood_single_parameter(params)
        aic_money_mazur = -2 * 1 / log_likelihood_Money_mazur + 2 * 3 + aicc_correction(3, np.size(array, axis=0))
        print("AIC Money Mazur")
        print(aic_money_mazur)

        # prelec
        def log_likelihood_single_parameter(params):
            beta, k, s = params
            log_likelihood = 0
            for fun_counter in range(np.size(array, axis=0)):
                Choosen_Reward_D = array[fun_counter, 4]
                immediate_offer = array[fun_counter, 5]
                Delay = array[fun_counter, 2]
                likeness = np.log(two_parameter_softmax_prelec(Choosen_Reward_D, immediate_offer, Delay, k, beta, s))
                log_likelihood = likeness + log_likelihood

            return 1 / log_likelihood

        DD_Money_Fit_prelec = scipy.optimize.minimize(log_likelihood_single_parameter, x0=initial_guess_three,
                                                      bounds=bounds_three,
                                                      method=method, options={'maxiter': maxiter})
        # , bounds=bounds
        params = [DD_Money_Fit_prelec.x[0], DD_Money_Fit_prelec.x[1], DD_Money_Fit_prelec.x[2]]
        log_likelihood_Money_prelec = log_likelihood_single_parameter(params)
        aic_money_prelec = -2 * 1 / log_likelihood_Money_prelec + 2 * 3 + aicc_correction(3, np.size(array, axis=0))
        print("AIC Money prelec")
        print(aic_money_prelec)
        if not Trial_duration_Coin_Later:
            Trial_duration_Coin_Later.append(0)
        if not Trial_duration_Coin_now:
            Trial_duration_Coin_now.append(0)
        if not Trial_duration_Food_Later:
            Trial_duration_Food_Later.append(0)
        if not Trial_duration_Food_Now:
            Trial_duration_Food_Now.append(0)
        Coin_Later_RT = np.array(Trial_duration_Coin_Later).mean()
        Coin_Now_RT = np.array(Trial_duration_Coin_now).mean()
        Food_Later_RT = np.array(Trial_duration_Food_Later).mean()
        Food_Now_RT = np.array(Trial_duration_Food_Now).mean()

        def hyperbole_function_for_curve_fit(x, k):
            SV = []
            for i in x:
                SV.append(40 / (1 + i * k))
            return SV
        #All_Food_Decisions = All_Food_Decisions[0:4]
        #All_Money_Decisions = All_Money_Decisions[0:4]
        #timings = timings[0:5]
        k_hyperbole_money, covar = curve_fit(hyperbole_function_for_curve_fit, timings, All_Money_Decisions, p0=0.0055,
                                            maxfev=5000, bounds=bounds_one_r2)
        r2_hyberbole_money = sklearn.metrics.r2_score(All_Money_Decisions,
                                                      hyperbole_function_for_curve_fit(timings, k_hyperbole_money),
                                                      sample_weight=None)

        k_hyperbole_food, covar = curve_fit(hyperbole_function_for_curve_fit, timings, All_Food_Decisions, p0=0.0055,
                                            maxfev=5000, bounds=bounds_one_r2)
        r2_hyberbole_food = sklearn.metrics.r2_score(All_Food_Decisions,
                                                     hyperbole_function_for_curve_fit(timings, k_hyperbole_food),
                                                     sample_weight=None)

        def exponential_discounting_for_curve_fitting(x, k):
            SV = []
            for i in x:
                SV.append(40 * exp(-k * i))
            return SV

        k_exponential_money, covar = curve_fit(exponential_discounting_for_curve_fitting, timings, All_Money_Decisions,
                                               p0=0.0055, maxfev=5000, bounds=bounds_one_r2)
        r2_exponential_money = sklearn.metrics.r2_score(All_Money_Decisions,
                                                        exponential_discounting_for_curve_fitting(timings,
                                                                                                  k_exponential_money),
                                                        sample_weight=None)

        k_exponential_food, covar = curve_fit(exponential_discounting_for_curve_fitting, timings, All_Food_Decisions,
                                              p0=0.0055, maxfev=5000, bounds=bounds_one_r2)
        r2_exponential_food = sklearn.metrics.r2_score(All_Food_Decisions,
                                                       exponential_discounting_for_curve_fitting(timings,
                                                                                                 k_exponential_food),
                                                       sample_weight=None)

        timings = np.array(timings)
       #timings = list(timings)

        # All_Money_Decisions = list(All_Money_Decisions)

        def prelec_for_curve_fit(x, k, s):
            SV = []
            for i in x:
                SV.append(40 * exp(-1 * (np.power(((k * i)), s))))
            return SV

        k_prelec_food, covar = scipy.optimize.curve_fit(prelec_for_curve_fit, timings, All_Food_Decisions,
                                                        p0=initial_guess_tw0_r2, maxfev=5000, bounds=bounds_two_r2)
        k_prelec_food, s_prelec_food = k_prelec_food
        r2_prelec_food = sklearn.metrics.r2_score(All_Food_Decisions, prelec_for_curve_fit(timings, k_prelec_food,s_prelec_food),sample_weight=None)

        k_prelec_money, covar = scipy.optimize.curve_fit(prelec_for_curve_fit, timings, All_Money_Decisions,
                                                         p0=initial_guess_tw0_r2, maxfev=5000, bounds=bounds_two_r2)
        k_prelec_money, s_prelec_money = k_prelec_money
        r2_prelec_money = sklearn.metrics.r2_score(All_Money_Decisions, prelec_for_curve_fit(timings,k_prelec_money,s_prelec_money ),sample_weight=None)

        def green_meyerson_discounting_for_curve_fitting(x, k, s):
            SV = []
            for i in x:
                SV.append(40 / np.power((1 + k * i), s))
            return SV

        GM_K_Food, covar = scipy.optimize.curve_fit(green_meyerson_discounting_for_curve_fitting, timings,
                                                    All_Food_Decisions, maxfev=5000, p0=initial_guess_tw0_r2,
                                                    bounds=bounds_two_r2)
        k_GM_Food, s_GM_Food = GM_K_Food
        r2_GM_food = sklearn.metrics.r2_score(All_Food_Decisions,
                                              green_meyerson_discounting_for_curve_fitting(timings, k_GM_Food, s_GM_Food),sample_weight=None)

        GM_K_Money, covar = scipy.optimize.curve_fit(green_meyerson_discounting_for_curve_fitting, timings,
                                                     All_Money_Decisions, maxfev=5000, p0=initial_guess_tw0_r2,
                                                     bounds=bounds_two_r2)
        k_GM_Money, s_GM_Money = GM_K_Money
        r2_GM_money = sklearn.metrics.r2_score(All_Money_Decisions,green_meyerson_discounting_for_curve_fitting(timings, k_GM_Money,s_GM_Money ),sample_weight=None)

        def mazur_for_curve_fitting(x, k, s):
            SV = []
            for i in x:
                SV.append(40 / (1 + k * np.power(i, s)))
            return SV

        mazur_K_Food, covar = scipy.optimize.curve_fit(mazur_for_curve_fitting, timings, All_Food_Decisions,
                                                       p0=initial_guess_tw0_r2, maxfev=5000, bounds=bounds_two_r2)
        k_mazur_food, s_mazur_food = mazur_K_Food
        r2_mazur_food = sklearn.metrics.r2_score(All_Food_Decisions, mazur_for_curve_fitting(timings,k_mazur_food, s_mazur_food),
                                                 sample_weight=None)

        mazur_K_Money, covar = scipy.optimize.curve_fit(mazur_for_curve_fitting, timings, All_Money_Decisions,
                                                        p0=initial_guess_tw0_r2, maxfev=5000, bounds=bounds_two_r2)
        k_mazur_money, s_mazur_money = mazur_K_Money
        r2_mazur_money = sklearn.metrics.r2_score(All_Money_Decisions, mazur_for_curve_fitting(timings, k_mazur_money, s_mazur_money ),
                                                  sample_weight=None)

        # if Offer_Food_later.__len__() >= 15:#
        Food_is_calculatable = True
        Coin_is_calculatable = True
        if Food_is_calculatable and Coin_is_calculatable:
            all_data.append([float(counter), DD_Food_Fit_hyperbol.x[0], DD_Food_Fit_hyperbol.x[1], aic_food_hyperbol,
                         DD_Money_Fit_hyperbol.x[0], DD_Money_Fit_hyperbol.x[1], aic_money_hyperbol,
                         DD_Food_Fit_exponential.x[0], DD_Food_Fit_exponential.x[1], aic_food_exponential,
                         DD_Money_Fit_exponential.x[0], DD_Money_Fit_exponential.x[1], aic_money_exponential,
                         DD_Food_Fit_meyer_green.x[0], DD_Food_Fit_meyer_green.x[1], DD_Food_Fit_meyer_green.x[2],
                         aic_food_meyer_green,
                         DD_Money_Fit_meyer_green.x[0], DD_Money_Fit_meyer_green.x[1], DD_Money_Fit_meyer_green.x[2],
                         aic_money_meyer_green,
                         DD_Food_Fit_mazur.x[0], DD_Food_Fit_mazur.x[1], DD_Food_Fit_mazur.x[2], aic_food_mazur,
                         DD_Money_Fit_mazur.x[0], DD_Money_Fit_mazur.x[1], DD_Money_Fit_mazur.x[2], aic_money_mazur,
                         DD_Food_Fit_prelec.x[0], DD_Food_Fit_prelec.x[1], DD_Food_Fit_prelec.x[2], aic_food_prelec,
                         DD_Money_Fit_prelec.x[0], DD_Money_Fit_prelec.x[1], DD_Money_Fit_prelec.x[2], aic_money_prelec,
                         Coin_Later_RT, Coin_Now_RT, Food_Later_RT, Food_Now_RT, participant_code])
        if Food_is_calculatable and Coin_is_calculatable: rsquared_data.append(
            [r2_exponential_food, r2_exponential_money, r2_hyberbole_food, r2_hyberbole_money, r2_mazur_food,
             r2_mazur_money, r2_GM_food, r2_GM_money, r2_prelec_food, r2_prelec_money, participant_code])
        rsquared_data_unfiltered.append([r2_exponential_food, r2_exponential_money, r2_hyberbole_food, r2_hyberbole_money, r2_mazur_food,
             r2_mazur_money, r2_GM_food, r2_GM_money, r2_prelec_food, r2_prelec_money, participant_code, Coin_is_calculatable, Food_is_calculatable])
        rsquared_parameters.append([k_exponential_food,k_exponential_money,k_hyperbole_food,k_hyperbole_money,k_GM_Food,s_GM_Food,k_GM_Money,s_GM_Money,k_mazur_food,s_mazur_food,k_mazur_money,s_mazur_money,k_prelec_food,s_prelec_food,k_prelec_money,s_prelec_money])
        choice_probability_food = []

        for fun_counter in range(np.size(array, axis=0)):
            Choosen_Reward_D = Food[fun_counter, 4]
            immediate_offer = Food[fun_counter, 5]
            Delay = Food[fun_counter, 2]
            p_delayed = two_parameter_softmax_prelec(Choosen_Reward_D, immediate_offer, Delay, DD_Food_Fit_mazur.x[1],
                                                     DD_Food_Fit_mazur.x[0], DD_Food_Fit_mazur.x[2])
            choice_probability_food.append(p_delayed)

        choice_probability_Money = []

        for fun_counter in range(np.size(array, axis=0)):
            Choosen_Reward_D = Money[fun_counter, 4]
            immediate_offer = Money[fun_counter, 5]
            Delay = Money[fun_counter, 2]
            p_delayed = two_parameter_softmax_prelec(Choosen_Reward_D, immediate_offer, Delay,
                                                     DD_Money_Fit_meyer_green.x[1], DD_Money_Fit_meyer_green.x[0],
                                                     DD_Money_Fit_meyer_green.x[2])
            choice_probability_Money.append(p_delayed)

        choice_probability_Coin_Later = []
        for fun_counter in range(np.size(indifference_liste_Coin_Later, axis=0)):
            Choosen_Reward_D = indifference_liste_Coin_Later[fun_counter, 4]
            immediate_offer = indifference_liste_Coin_Later[fun_counter, 5]
            Delay = indifference_liste_Coin_Later[fun_counter, 2]
            p_delayed = two_parameter_softmax_mazur(Choosen_Reward_D, immediate_offer, Delay,
                                                    DD_Money_Fit_meyer_green.x[1], DD_Money_Fit_meyer_green.x[0],
                                                    DD_Money_Fit_meyer_green.x[2])
            choice_probability_Coin_Later.append(p_delayed)

        choice_probability_Coin_Now = []
        for fun_counter in range(np.size(indifference_liste_Coin_Now, axis=0)):
            Choosen_Reward_D = indifference_liste_Coin_Now[fun_counter, 4]
            immediate_offer = indifference_liste_Coin_Now[fun_counter, 5]
            Delay = indifference_liste_Coin_Now[fun_counter, 2]
            p_delayed = two_parameter_softmax_mazur(Choosen_Reward_D, immediate_offer, Delay,
                                                    DD_Money_Fit_meyer_green.x[1], DD_Money_Fit_meyer_green.x[0],
                                                    DD_Money_Fit_meyer_green.x[2])
            choice_probability_Coin_Now.append(p_delayed)

        choice_probability_Food_Later = []
        for fun_counter in range(np.size(indifference_liste_Food_Later, axis=0)):
            Choosen_Reward_D = indifference_liste_Food_Later[fun_counter, 4]
            immediate_offer = indifference_liste_Food_Later[fun_counter, 5]
            Delay = indifference_liste_Food_Later[fun_counter, 2]
            p_delayed = two_parameter_softmax_prelec(Choosen_Reward_D, immediate_offer, Delay,
                                                     DD_Food_Fit_meyer_green.x[1], DD_Food_Fit_meyer_green.x[0],
                                                     DD_Food_Fit_meyer_green.x[2])
            choice_probability_Food_Later.append(p_delayed)

        choice_probability_Food_Now = []
        for fun_counter in range(np.size(indifference_liste_Food_Now, axis=0)):
            Choosen_Reward_D = indifference_liste_Food_Now[fun_counter, 4]
            immediate_offer = indifference_liste_Food_Now[fun_counter, 5]
            Delay = indifference_liste_Food_Now[fun_counter, 2]
            p_delayed = two_parameter_softmax_prelec(Choosen_Reward_D, immediate_offer, Delay,
                                                     DD_Food_Fit_meyer_green.x[1], DD_Food_Fit_meyer_green.x[0],
                                                     DD_Food_Fit_meyer_green.x[2])
            choice_probability_Food_Now.append(p_delayed)

        def movingaverage(interval, window_size):
            window = np.ones(int(window_size)) / float(window_size)
            return np.convolve(interval, window, 'same')

        p_delayed_money = []
        SV_All = []
        immediate_offer = np.arange(0.01,30,0.01)
        for i in range(np.size(Money, axis=0)):
            Choosen_Reward_D = Money[i, 4]
            immediate_offer = Money[i, 5]
            Delay = Money[i, 2]
            SV_All.append(immediate_offer/mazur(Delay,DD_Money_Fit_mazur.x[1], DD_Money_Fit_mazur.x[2]))
            p_delayed_money.append(two_parameter_softmax_mazur(0,immediate_offer,Delay,DD_Money_Fit_mazur.x[1],DD_Money_Fit_mazur.x[0],DD_Money_Fit_mazur.x[2]))

        #
        #
        #p_delayed_money =movingaverage(p_delayed_money,7)
        data = np.column_stack([SV_All,p_delayed_money,Money[:, 4]])
        data.sort(axis=0)
        # databas, indices = np.unique(data[:,0], axis=0, return_index=True)
        # data = data[indices,:]
        # new_y = UnivariateSpline(data[:, 0],data[:, 1], k=3,w=2) #was -> interp1d , kind='nearest'
        new_y = uniform_filter1d(data[:, 1], size=14, mode='reflect')
        color = np.where(data[:, 2] <= 0.5, 'b', 'r')  # blue for delayed, red for immediate
        data[:, 2] = np.where(data[:, 2] == 0, 0, 1)  # after this 1 -> immediate 0 -> delayed
        # xnew = np.linspace(data[:,0].min(),data[:,0].max(), num=10000000, endpoint=True)
        plt.plot(data[:, 0], new_y, color='black')  # data[:, 1]
        plt.scatter(data[:, 0], data[:, 2], c=color, linewidth=0)
        #,linestyle='' , marker='o'
        #for i in range(np.size(data, axis=0)):
        #    if data[i,1]>0.5 and data[i,2]==0:#wenn predicted -> delayed und echt --> immediate
        #        #print("test")
        #        x1= data[i,0]
        #        y1= data[:,2].max()
        #        x2=data[i,0]
        #        y2=data[:,2].min()
        #        x = [x1,x2]
        #        y= [y1,y2]
        #        plt.plot(x,y, linestyle='solid',linewidth=0.5,color='green')
        #    elif  data[i,1]<0.5 and data[i,2]==1:
        #        x1 = data[i, 0]
        #        y1 = data[:, 2].min()
        #        x2 = data[i, 0]
        #        y2 = data[:, 2].max()
        #        x = [x1, x2]
        #        y = [y1, y2]
        #        plt.plot(x, y, linestyle='solid', linewidth=0.5, color='green')
        plt.xlabel('SV(immediate reward)/SV(delayed reward)')
        plt.ylabel('Estimated Softmax propability P(choose immediate)')
        os.chdir('C:/Users/mariu/Documents/Arbeit/Doktorarbeit/Fany_Analysen/prop_Money_fancy_plot/')
        name = 'money_' + participant_code + '.png'
        plt.savefig(name)
        plt.close()

        p_delayed_food = []
        SV_food = []
        immediate_offer = np.arange(0.01, 30, 0.01)
        for i in range(np.size(Money, axis=0)):
            Choosen_Reward_D = Food[i, 4]
            immediate_offer = Food[i, 5]
            Delay = Food[i, 2]
            SV_food.append(immediate_offer / prelec(Delay, DD_Food_Fit_prelec.x[1], DD_Food_Fit_prelec.x[2]))
            p_delayed_food.append(
                two_parameter_softmax_mazur(0, immediate_offer, Delay, DD_Food_Fit_prelec.x[1], DD_Food_Fit_prelec.x[0],
                                            DD_Food_Fit_prelec.x[2]))


        data = np.column_stack([SV_food, p_delayed_food, Food[:, 4]])
        data.sort(axis=0)
        #databas, indices = np.unique(data[:,0], axis=0, return_index=True)
        #data = data[indices,:]
        #new_y = UnivariateSpline(data[:, 0],data[:, 1], k=3,w=2) #was -> interp1d , kind='nearest'
        new_y = uniform_filter1d(data[:, 1],size=14,mode='reflect')
        color = np.where(data[:,2] <= 0.5, 'b', 'r')  # blue for delayed, red for immediate
        data[:, 2] = np.where(data[:, 2] == 0, 0, 1)# after this 1 -> immediate 0 -> delayed
        #xnew = np.linspace(data[:,0].min(),data[:,0].max(), num=10000000, endpoint=True)
        plt.plot(data[:, 0], new_y, color='black') #data[:, 1]
        plt.scatter(data[:, 0], data[:, 2], c=color, linewidth=0)
        # ,linestyle='' , marker='o'
        #for i in range(np.size(data, axis=0)):
        #    if data[i, 1] > 0.5 and data[i, 2] == 0:  # wenn predicted -> delayed und echt --> immediate
        #        # print("test")
        #        x1 = data[i, 0]
        #        y1 = data[:, 2].max()
        #        x2 = data[i, 0]
        #        y2 = data[:, 2].min()
        #        x = [x1, x2]
        #        y = [y1, y2]
        #        plt.plot(x, y, linestyle='solid', linewidth=0.5, color='green')
        #    elif data[i, 1] < 0.5 and data[i, 2] == 1:
        #        x1 = data[i, 0]
        #        y1 = data[:, 2].min()
        #        x2 = data[i, 0]
        #        y2 = data[:, 2].max()
        #        x = [x1, x2]
        #        y = [y1, y2]
        #        plt.plot(x, y, linestyle='solid', linewidth=0.5, color='green')
        plt.xlabel('SV(immediate reward)/SV(delayed reward)')
        plt.ylabel('Estimated Softmax propability P(choose immediate)')
        os.chdir('C:/Users/mariu/Documents/Arbeit/Doktorarbeit/Fany_Analysen/prop_Food_fancy_plot/')
        name = 'food_' + participant_code + '.png'
        plt.savefig(name)
        plt.close()

        os.chdir('C:/Users/mariu/Documents/Arbeit/Doktorarbeit/Fany_Analysen/Log_files')
        export_info = [Offer_Coin, Trial_duration_Coin, choice_probability_Money, [DD_Money_Fit_meyer_green.x[1]],
                       [DD_Money_Fit_meyer_green.x[0]], [DD_Money_Fit_meyer_green.x[2]],
                       Offer_Food, Trial_duration_Food, choice_probability_food, [DD_Food_Fit_mazur.x[1]],
                       [DD_Food_Fit_mazur.x[0]], [DD_Food_Fit_mazur.x[2]],
                       feedback_trials_coin,
                       feedback_trials_food,
                       error_trials_coin,
                       error_trials_foos]
        export_info = pd.DataFrame(export_info)
        name_creator = file[:-4] + 'probably_log.xlsx'
        export_info.to_excel(name_creator, float_format='%.12f')
        #all_data['prelec_Food_s']
        os.chdir('C:/Users/mariu/Documents/Arbeit/Doktorarbeit/Fany_Analysen/log_files_choice_split')
        export_info = [Offer_Coin_later, Trial_duration_Coin_Later, choice_probability_Coin_Later,
                       Offer_Coin_now, Trial_duration_Coin_now, choice_probability_Coin_Now,
                       Offer_Food_later, Trial_duration_Food_Later, choice_probability_Food_Later,
                       Offer_Food_now, Trial_duration_Food_Now, choice_probability_Food_Now,
                       [DD_Money_Fit_meyer_green.x[1]], [DD_Money_Fit_meyer_green.x[0]],
                       [DD_Money_Fit_meyer_green.x[2]],
                       [DD_Food_Fit_mazur.x[1]], [DD_Food_Fit_mazur.x[0]], [DD_Food_Fit_mazur.x[2]],
                       feedback_trials_coin,
                       feedback_trials_food,
                       error_trials_coin,
                       error_trials_foos]

        export_info = pd.DataFrame(export_info)
        export_info.to_excel(name_creator, float_format='%.10f')
        if Food_is_calculatable and Coin_is_calculatable:
            log_likelihood_food.append([log_likelihood_food_exponential,log_likelihood_food_hyperbol,log_likelihood_Food_mazur,log_likelihood_Food_meyer_green,log_likelihood_Food_prelec])
            log_likelihood_money.append([log_likelihood_money_exponential, log_likelihood_money_hyperbol,log_likelihood_Money_mazur,log_likelihood_Money_meyer_green,log_likelihood_Money_prelec])

        base_directory_BIDS = 'C:/Users/mariu/Documents/Arbeit/DelayDiscountingFood/BIDS_Database/'
        subject_directory_BIDS = base_directory_BIDS + participant_code + '/'
        func_directory = subject_directory_BIDS + 'ses-01/func/'
        isExist = os.path.exists(func_directory)
        if not isExist:
            os.makedirs(func_directory)
        os.chdir(func_directory)

        events_script = pd.DataFrame(events_script,columns=['onset','duration','Delay','trial_type',"Offer_Immediate","Decision"])
        events_script.to_csv('events.tsv', sep='\t')

        # tmp_array = np.zeros(shape=(np.size(array,axis=0),2))
        # tmp_array[:,0] = immediate_offer_smooth
        # tmp_array[:,1] = choice_probability

    all_data = pd.DataFrame(all_data, columns=['id', 'hyperbol_Food_beta', 'hyperbol_Food_k', 'aic_Food_hyperbol',
                                               'hyperbol_Money_beta', 'hyperbol_Money_k', 'aic_Money_hyperbol',
                                               'exponential_Food_beta', 'exponential_Food_k', 'aic_Food_exponential',
                                               'exponential_Money_beta', 'exponential_Money_k', 'aic_Money_exponential',
                                               'meyer_green_Food_beta', 'meyer_green_Food_k', 'meyer_Food_near_s',
                                               'aic_Food_meyer_green', 'meyer_Money_self_beta', 'meyer_Money_self_k',
                                               'meyer_green_Money_s', 'aic_Money_meyer_green',
                                               'mazur_Food_beta', 'mazur_Food_k', 'mazur_Food_s', 'aic_Food_mazur',
                                               'mazur_Money_beta', 'mazur_Money_k', 'mazur_Money_s', 'aic_Money_mazur',
                                               'prelec_Food_beta', 'prelec_Food_k', 'prelec_Food_s', 'aic_Food_prelec',
                                               'prelec_Money_beta', 'prelec_Money_k', 'prelec_Money_s',
                                               'aic_Money_prelec', 'Coin_Later_RT', 'Coin_Now_RT', 'Food_Later_RT',
                                               'Food_Now_RT', 'participant_code'])

    all_data['prelec_Food_s'].mean()
    scipy.stats.ttest_1samp(all_data['prelec_Food_s'],1)
    os.chdir('C:/Users/mariu/Documents/Arbeit/DelayDiscountingFood/FancyTabellen/')
    all_data.to_excel('fancy_tabelle.xlsx')
    mean_s_prelec = all_data['prelec_Food_s'].median()
    max_s_prelec = np.percentile(all_data['prelec_Food_s'],75)
    #max_s_prelec = all_data['prelec_Food_s'].max()
    min_s_prelec = np.percentile(all_data['prelec_Food_s'],25)
    #min_s_prelec = all_data['prelec_Food_s'].min()

    mean_k_prelec =  all_data['prelec_Food_k'].median()
    #max_k_prelec = all_data['prelec_Food_k'].max()
    max_k_prelec = np.percentile(all_data['prelec_Food_k'],75)
    #min_k_prelec = all_data['prelec_Food_k'].min()
    min_k_prelec= np.percentile(all_data['prelec_Food_k'],25)
    mean_b_prelec = all_data['prelec_Food_beta'].median()

    #max_b_prelec = all_data['prelec_Food_beta'].max()
    max_b_prelec = np.percentile(all_data['prelec_Food_beta'], 75)
    #min_b_prelec = all_data['prelec_Food_beta'].min()
    min_b_prelec =np.percentile(all_data['prelec_Food_beta'],25)



    mean_s_mg = all_data['meyer_green_Money_s'].median()
    max_s_mg = all_data['meyer_green_Money_s'].max()
    max_s_mg= np.percentile(all_data['meyer_green_Money_s'],75)
    min_s_mg = all_data['meyer_green_Money_s'].min()
    min_s_mg= np.percentile(all_data['meyer_green_Money_s'],25)
    mean_k_mg = all_data['meyer_Money_self_k'].median()
    max_k_mg = all_data['meyer_Money_self_k'].max()
    max_k_mg =np.percentile(all_data['meyer_Money_self_k'],75)
    min_k_mg = all_data['meyer_Money_self_k'].min()
    min_k_mg = np.percentile(all_data['meyer_Money_self_k'],25)
    mean_b_mg = all_data['meyer_Money_self_beta'].median()
    max_b_mg = all_data['meyer_Money_self_beta'].max()
    max_b_mg = np.percentile(all_data['meyer_Money_self_beta'], 75)
    min_b_mg = all_data['meyer_Money_self_beta'].min()
    min_b_mg = np.percentile(all_data['meyer_Money_self_beta'], 25)


    r2_data = pd.DataFrame(rsquared_data, columns=['r2_exponential_food', 'r2_exponential_money', 'r2_hyberbole_food',
                                                   'r2_hyberbole_money', 'r2_mazur_food', 'r2_mazur_money',
                                                   'r2_GM_food', 'r2_GM_money', 'r2_prelec_food', 'r2_prelec_money','participant_code'])
    rsquared_data_unfiltered = pd.DataFrame(rsquared_data_unfiltered, columns=['r2_exponential_food', 'r2_exponential_money', 'r2_hyberbole_food',
                                                   'r2_hyberbole_money', 'r2_mazur_food', 'r2_mazur_money',
                                                   'r2_GM_food', 'r2_GM_money', 'r2_prelec_food', 'r2_prelec_money','participant_code','Coin_is_calculatable', 'Food_is_calculatable'])
    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen')
    log_likelihood_food = pd.DataFrame(log_likelihood_food, columns=['log_likelihood_food_exponential','log_likelihood_food_hyperbol','log_likelihood_Food_mazur','log_likelihood_Food_meyer_green','log_likelihood_Food_prelec'])
    log_likelihood_food.to_excel('log_likelihood_food.xlsx')

    log_likelihood_money = pd.DataFrame(log_likelihood_money, columns=['log_likelihood_money_exponential', 'log_likelihood_money_hyperbol','log_likelihood_Money_mazur','log_likelihood_Money_meyer_green','log_likelihood_Money_prelec'])
    log_likelihood_money.to_excel('log_likelihood_monex.xlsx')
    rsquared_parameters = np.array(rsquared_parameters, dtype=float)
    rsquared_parameters = pd.DataFrame(rsquared_parameters, columns=['k_exponential_food','k_exponential_money','k_hyperbole_food','k_hyperbole_money','k_GM_Food','s_GM_Food','k_GM_Money','s_GM_Money','k_mazur_food','s_mazur_food','k_mazur_money','s_mazur_money','k_prelec_food','s_prelec_food','k_prelec_money','s_prelec_money'])

    rsquared_parameters = pd.DataFrame(rsquared_parameters, columns=['k_exponential_food','k_exponential_money','k_hyperbole_food','k_hyperbole_money','k_GM_Food','s_GM_Food','k_GM_Money','s_GM_Money','k_mazur_food','s_GM_Food','k_mazur_food','s_mazur_food','k_mazur_money','s_mazur_money','k_prelec_food','s_prelec_food','k_prelec_money','s_prelec_money'])
    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen/')
    rsquared_data_unfiltered.to_excel('Food_Money_r2_parameters.xlsx')
    rsquared_parameters['k_mazur_food'].mean()
    rsquared_parameters['k_mazur_food'].std()

    rsquared_parameters['s_mazur_food'].mean()
    rsquared_parameters['s_mazur_food'].std()
    Food_is_calculatable_rsquared = rsquared_data_unfiltered[rsquared_data_unfiltered['Food_is_calculatable'] == True]
    Coin_is_calculatable_rsquared = rsquared_data_unfiltered[rsquared_data_unfiltered['Coin_is_calculatable'] == True]
    rsquared_data_unfiltered[['r2_exponential_food', 'r2_hyberbole_food', 'r2_mazur_food', 'r2_GM_food', 'r2_prelec_food']].median()

    scipy.stats.wilcoxon(all_data[''])

    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_prelec_money'], rsquared_data_unfiltered['r2_GM_money'])
    print(r2_data['r2_prelec_money'].mean())
    print(r2_data['r2_prelec_money'].std())

    rsquared_data_unfiltered[['r2_exponential_money', 'r2_hyberbole_money', 'r2_mazur_money', 'r2_GM_money', 'r2_prelec_money']].median()
    rsquared_data_unfiltered.boxplot(column=['r2_exponential_money', 'r2_hyberbole_money', 'r2_mazur_money', 'r2_GM_money', 'r2_prelec_money'], sym='')

    r2_data[['r2_exponential_food', 'r2_hyberbole_food', 'r2_mazur_food', 'r2_GM_food', 'r2_prelec_food']].median()
    r2_data[['r2_exponential_food', 'r2_hyberbole_food', 'r2_mazur_food', 'r2_GM_food', 'r2_prelec_food']].boxplot()
    sns.boxplot(x=r2_data.iloc[['r2_exponential_food', 'r2_hyberbole_food', 'r2_mazur_food', 'r2_GM_food', 'r2_prelec_food']])
    r2_data = r2_data[r2_data['r2_exponential_money']>0]
    plt.boxplot(r2_data[['r2_exponential_food', 'r2_hyberbole_food', 'r2_mazur_food', 'r2_GM_food', 'r2_prelec_food']])

    plt.axis([0,5, 0, 1])

    r2_data[['r2_exponential_money', 'r2_hyberbole_money', 'r2_mazur_money', 'r2_GM_money', 'r2_prelec_money']].median()
    r2_data.boxplot(column=['r2_exponential_money', 'r2_hyberbole_money', 'r2_mazur_money', 'r2_GM_money', 'r2_prelec_money'], grid=False, color = ['pink', 'lightblue', 'lightgreen', 'black' , 'black'])
    food_rsquared_heatmap = get_wilcoxon_rank_and_make_fancy_graphics(r2_data[['r2_exponential_food','r2_hyberbole_food','r2_mazur_food','r2_GM_food','r2_prelec_food']],significance_value=0.05)
    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen')
    food_rsquared_heatmap.savefig('heatmap_food_rsquared.png')


    money_rsquared_heatmap = get_wilcoxon_rank_and_make_fancy_graphics(r2_data[['r2_exponential_money', 'r2_hyberbole_money', 'r2_mazur_money', 'r2_GM_money', 'r2_prelec_money']],significance_value=0.05)

    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen')
    money_rsquared_heatmap.savefig('heatmap_money_rsquared.png')
    all_data.boxplot(column=['delta_aic_food_meyer_green', 'delta_aic_food_mazur','delta_aic_food_prelec'],  grid=True, rot=0,
                patch_artist=True)
    all_data.boxplot(column=['aic_Food_hyperbol','aic_Food_exponential','aic_Food_meyer_green','aic_Food_mazur','aic_Food_prelec'], grid=True, rot=0, patch_artist=True)
    my_pal_money = {"r2_exponential_money": "darkmagenta", "r2_hyberbole_money": "limegreen", "r2_mazur_money": "fuchsia" ,"r2_GM_money":"darkgoldenrod",'r2_prelec_money':'royalblue'}
    sns.set(font_scale=3)
    ax = sns.boxplot(data= r2_data[[ 'r2_exponential_money','r2_hyberbole_money', 'r2_GM_money', 'r2_mazur_money','r2_prelec_money']],  palette=my_pal_money, showfliers=False)
    ax.set_xticklabels(
        ['Exponential', 'Hyperbole', 'Hyperbole with Time scaling (MG)', 'Hyperbole with Time scaling (Mazur)',
         'Exponential with time scaling'])
    ax.set(ylim=(0, 1))
    #'r2_exponential_money', 'r2_hyberbole_money',
    ax.set_xlabel("Model Fit for the Money condition")
    ax.set_ylabel('R\u00b2')
    ax.patches[2].set_edgecolor('red')
    ax.patches[2].set_linewidth(5)
    ax.axhline(y=2, color='black', linestyle='--', linewidth=3)
    ax.set(ylim=(0, 11))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    r2_data_food = r2_data[r2_data['r2_hyberbole_food'] > 0]

    my_pal_food = {"r2_exponential_food": "firebrick", "r2_hyberbole_food": "darkgoldenrod", "r2_mazur_food": "limegreen" ,"r2_GM_food":"darkcyan",'r2_prelec_food':'darkmagenta'}
    ax = sns.boxplot(data= r2_data_food[['r2_exponential_food', 'r2_hyberbole_food', 'r2_mazur_food', 'r2_GM_food', 'r2_prelec_food']],  palette=my_pal_food, showfliers=False)
    ax.set_xticklabels(
        ['Exponential', 'Hyperbole', 'Hyperbole with Time scaling (MG)', 'Hyperbole with Time scaling (Mazur)',
         'Exponential with time scaling'])
    ax.set_xlabel("Model Fit for the Food condition")
    ax.set_ylabel('R\u00b2')
    rsquared_data_unfiltered = rsquared_data_unfiltered[rsquared_data_unfiltered['r2_hyberbole_food'] > 0]


    sns.set(rc={'figure.figsize': (22, 11)})
    sns.set(font_scale=2.2)
    my_pal_food = {"r2_exponential_food": "firebrick", "r2_hyberbole_food": "darkgoldenrod",
                   "r2_mazur_food": "darkcyan", "r2_GM_food": "limegreen",
                   'r2_prelec_food': 'darkmagenta'}
    ax = sns.barplot(data=rsquared_data_unfiltered[
        ['r2_exponential_food', 'r2_hyberbole_food', 'r2_GM_food',
         'r2_mazur_food', 'r2_prelec_food']], palette=my_pal_food, errorbar='se')
    ax.set_xticklabels(
        ['Exponential', 'Hyperbole', 'Hyperbole \n with time scaling \n Delay', 'Hyperbole \n with scaling \n Delay & Discounting',
         'Exponential \n with time scaling'])
    ax.patches[3].set_edgecolor('red')
    ax.patches[3].set_linewidth(5)
    ax.patches[4].set_edgecolor('red')
    ax.patches[4].set_linewidth(5)
    # ax.axhline(y=2, color='black', linestyle='--')
    ax.set(ylim=(0, 1))


    sns.set(rc={'figure.figsize': (22, 11)})
    sns.set(font_scale=2.2)
    my_pal_food = {"r2_exponential_money": "firebrick", "r2_hyberbole_money": "darkgoldenrod",
                   "r2_mazur_money": "darkcyan", "r2_GM_money": "limegreen",
                   'r2_prelec_money': 'darkmagenta'}
    ax = sns.barplot(data=rsquared_data_unfiltered[
        ['r2_exponential_money', 'r2_hyberbole_money', 'r2_GM_money',
         'r2_mazur_money', 'r2_prelec_money']], palette=my_pal_food, errorbar='se')
    ax.set_xticklabels([]
        ) #['Exponential', 'Hyperbole', 'Hyperbole \n with Time scaling (MG)', 'Hyperbole \n with Time scaling (Mazur)',
         #'Exponential \n with time scaling']
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.patches[3].set_edgecolor('red')
    ax.patches[3].set_linewidth(5)
    # ax.axhline(y=2, color='black', linestyle='--')
    ax.set(ylim=(0, 1))
    food_rsquared_heatmap = get_wilcoxon_rank_and_make_fancy_graphics(
        rsquared_data_unfiltered[['r2_exponential_food', 'r2_hyberbole_food', 'r2_mazur_food', 'r2_GM_food', 'r2_prelec_food']],
        significance_value=0.05)

    get_wilcoxon_rank_and_make_fancy_graphics(rsquared_data_unfiltered[
        ['r2_exponential_money', 'r2_hyberbole_money', 'r2_GM_money',
         'r2_mazur_money', 'r2_prelec_money']], significance_value=0.05)

    scipy.stats.normaltest(rsquared_data_unfiltered['r2_exponential_food'])
    scipy.stats.normaltest(rsquared_data_unfiltered['r2_hyberbole_food'])
    scipy.stats.normaltest(rsquared_data_unfiltered['r2_mazur_food'])
    scipy.stats.normaltest(rsquared_data_unfiltered['r2_GM_food'])
    scipy.stats.normaltest(rsquared_data_unfiltered['r2_prelec_food'])

    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_exponential_food'], rsquared_data_unfiltered['r2_hyberbole_food'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_exponential_food'], rsquared_data_unfiltered['r2_mazur_food'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_exponential_food'], rsquared_data_unfiltered['r2_GM_food'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_exponential_food'], rsquared_data_unfiltered['r2_prelec_food'])

    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_hyberbole_food'], rsquared_data_unfiltered['r2_mazur_food'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_hyberbole_food'], rsquared_data_unfiltered['r2_GM_food'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_hyberbole_food'], rsquared_data_unfiltered['r2_prelec_food'])

    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_mazur_food'], rsquared_data_unfiltered['r2_GM_food'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_mazur_food'], rsquared_data_unfiltered['r2_prelec_food'])

    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_GM_food'], rsquared_data_unfiltered['r2_prelec_food'])

    scipy.stats.normaltest(rsquared_data_unfiltered['r2_exponential_money'])
    scipy.stats.normaltest(rsquared_data_unfiltered['r2_hyberbole_money'])
    scipy.stats.normaltest(rsquared_data_unfiltered['r2_mazur_money'])
    scipy.stats.normaltest(rsquared_data_unfiltered['r2_GM_money'])
    scipy.stats.normaltest(rsquared_data_unfiltered['r2_prelec_money'])


    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_exponential_money'], rsquared_data_unfiltered['r2_hyberbole_money'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_exponential_money'], rsquared_data_unfiltered['r2_mazur_money'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_exponential_money'], rsquared_data_unfiltered['r2_GM_money'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_exponential_money'], rsquared_data_unfiltered['r2_prelec_money'])

    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_hyberbole_money'], rsquared_data_unfiltered['r2_mazur_money'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_hyberbole_money'], rsquared_data_unfiltered['r2_GM_money'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_hyberbole_money'], rsquared_data_unfiltered['r2_prelec_money'])

    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_mazur_money'], rsquared_data_unfiltered['r2_GM_money'])
    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_mazur_money'], rsquared_data_unfiltered['r2_prelec_money'])

    scipy.stats.wilcoxon(rsquared_data_unfiltered['r2_GM_money'], rsquared_data_unfiltered['r2_prelec_money'])


    #scipy.stats.wilcoxon(all_data['aic_Food_prelec'], all_data['aic_Money_prelec'])
    min_food = all_data[['aic_Food_hyperbol','aic_Food_exponential','aic_Food_meyer_green','aic_Food_mazur','aic_Food_prelec']].min(axis=1)
    all_data['delta_aic_food_hyperbol'] =  all_data['aic_Food_hyperbol'] - min_food
    all_data['delta_aic_food_exponential'] = all_data['aic_Food_exponential'] - min_food
    all_data['delta_aic_food_meyer_green'] = all_data['aic_Food_meyer_green'] -min_food
    all_data['delta_aic_food_mazur'] = all_data['aic_Food_mazur'] - min_food
    all_data['delta_aic_food_prelec'] = all_data['aic_Food_prelec'] -min_food
    all_data['delta_aic_food_exponential'].mean()
    all_data['delta_aic_food_prelec'].mean()
    all_data['delta_aic_food_prelec'].std()
    food_decider = all_data[['aic_Food_hyperbol','aic_Food_exponential','aic_Food_meyer_green','aic_Food_mazur','aic_Food_prelec']]
    plt.scatter(all_data['aic_Food_meyer_green'],all_data['aic_Food_prelec'])
    plt.sca
    scipy.stats.wilcoxon(all_data['delta_aic_food_mazur'],all_data['delta_aic_food_prelec'], )
    sns.set(rc={'figure.figsize': (22, 11.5)})
    sns.set(font_scale=3)
    my_pal_food = {"delta_aic_food_exponential": "darkmagenta", "delta_aic_food_hyperbol": "limegreen", "delta_aic_food_mazur": "fuchsia" ,"delta_aic_food_meyer_green":"darkgoldenrod",'delta_aic_food_prelec':'royalblue'}

    ax = sns.barplot(data= all_data[['delta_aic_food_exponential', 'delta_aic_food_hyperbol', 'delta_aic_food_mazur', 'delta_aic_food_meyer_green', 'delta_aic_food_prelec']],  palette=my_pal_food,errorbar='se') #, showfliers=False
    ax.set_xticklabels(['Exponential', 'Hyperbole', 'Hyperbole \n with scaling \n Delay',
                        'Hyperbole \n with scaling \n Delay & Discounting',
                        'Exponential \n with time scaling'])
    #ax.set_xlabel("Model Fit for the Food condition")
    #ax.set_ylabel(u'ΔAIC')
    ax.patches[4].set_edgecolor('red')
    ax.patches[4].set_linewidth(5)
    ax.axhline(y=2, color='black',linestyle='--',linewidth=3)
    ax.set(ylim=(0, 11))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.figsize()
    sns.barplot()
    delta_aic_heat_map_food = get_wilcoxon_rank_and_make_fancy_graphics(all_data[['delta_aic_food_exponential', 'delta_aic_food_hyperbol',
                                                                                'delta_aic_food_mazur', 'delta_aic_food_meyer_green', 'delta_aic_food_prelec']],significance_value=0.05)
    #delta_aic_heat_map_food.plot()
    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen')
    delta_aic_heat_map_food.savefig('heatmap_food_delta_aic.png')

    plt.close()
    delta_aic_heat_map_food.plot()

    mean_delta_aic_food = all_data[['delta_aic_food_exponential', 'delta_aic_food_hyperbol',
              'delta_aic_food_mazur', 'delta_aic_food_meyer_green', 'delta_aic_food_prelec']].mean()

    plt.bar([0, 1,2,3,4],mean_delta_aic_food, tick_label=['delta_aic_food_exponential', 'delta_aic_food_hyperbol',
              'delta_aic_food_mazur', 'delta_aic_food_meyer_green', 'delta_aic_food_prelec'])
    plt.boxplot( )
    all_data[['delta_aic_food_hyperbol', 'delta_aic_food_hyperbol',
              'delta_aic_food_mazur', 'delta_aic_food_meyer_green', 'delta_aic_food_prelec']].boxplot()
    all_data['delta_aic_food_prelec'].mean()
    all_data['delta_aic_food_mazur'].mean()
    scipy.stats.ttest_ind(all_data['prelec_Food_s'],all_data['meyer_green_Money_s'])
    scipy.stats.ttest_ind(all_data['prelec_Food_k'],all_data['meyer_Money_self_k'])
    scipy.stats.ttest_ind(all_data['prelec_Food_beta'],all_data['meyer_Money_self_beta'])
    plt.savefig(results_methods_dir + '/delta_aic_food.png')
    plt.close()

    scipy.stats.pearsonr(all_data['prelec_Food_s'],all_data['mazur_Money_s'])
    scipy.stats.pearsonr(all_data['prelec_Food_k'],all_data['mazur_Money_k'])
    scipy.stats.pearsonr(all_data['prelec_Food_beta'],all_data['mazur_Money_beta'])



    get_correlation_strength_and_make_fancy_graphics(rsquared_parameters[['k_exponential_food','k_hyperbole_food','k_GM_Food','k_mazur_food','k_prelec_food']],0.05,['k_exponential_food','k_hyperbole_food','k_GM_Food','k_mazur_food','k_prelec_food'])
    ax = get_correlation_strength_and_make_fancy_graphics(
        rsquared_parameters[['k_exponential_food','k_hyperbole_food','k_GM_Food','k_mazur_food','k_prelec_food']], #,'k_exponential_money', 'k_hyperbole_money', 'k_GM_Money', 'k_mazur_money', 'k_prelec_money'
        0.05/rsquared_parameters[['k_exponential_food','k_hyperbole_food','k_GM_Food','k_mazur_food','k_prelec_food']].shape[1], #,'k_exponential_money', 'k_hyperbole_money', 'k_GM_Money', 'k_mazur_money', 'k_prelec_money'
        ['Exponential Food','Hyperbole_food','Green & Myerson Food','Mazur Food','Prelec Food']) #,'Exponential Money', 'Hyperbole Money', 'Green & Myerson Money', 'Mazur Money', 'Prelec Money'

    #ax.set_xticklabels(['Green Meyerson Food','Hyperbole','Mazur','Meyerson & Green', 'Prelec'])
    #ax.set_xlabel("Model Fit for the Food condition")
    #ax.set_ylabel('Spearman Correlation of k for the "Type of Reward study')
    ax.ax_col_dendrogram.set_title('Spearman Correlation of k for the "Type of Reward" study, Food Condition')
    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen')
    ax.savefig('heatmap_k_rsquared_typeofreward_food.png')

    ax = get_correlation_strength_and_make_fancy_graphics(
        rsquared_parameters[['s_GM_Money', 's_mazur_money', 's_prelec_money']], #, 's_GM_Food','s_mazur_food','s_prelec_food'
        0.05/rsquared_parameters[['s_GM_Money', 's_mazur_money', 's_prelec_money']].shape[1], #, 's_GM_Food','s_mazur_food','s_prelec_food'
        ['Green & Myerson Money', 'Mazur Money', 'Prelec Money']) #, 'Green & Myerson Food','Mazur Food','Prelec Food'

    ax.ax_col_dendrogram.set_title('Spearman Correlation of s for the "Recipient of Reward" study, Money Condition')
    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen')
    ax.savefig('heatmap_s_rsquared_type_money.png')

    ax = get_correlation_strength_and_make_fancy_graphics(
        all_data[
            ['meyer_green_Money_s', 'mazur_Money_s', 'prelec_Money_s', 'meyer_Food_near_s', 'mazur_Food_s',
             'prelec_Food_s']],
        0.05/5,
        ['Green & Myerson Money', 'Mazur Money', 'Prelec Money', 'Green & Myerson Food', 'Mazur Food', 'Prelec Food'])
    ax.ax_col_dendrogram.set_title('Spearman Correlation of s for the "Type of Reward" study')
    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen')
    ax.savefig('heatmap_s_delta_aic_type.png')
    scipy.stats.spearmanr(rsquared_parameters['s_GM_Food'],rsquared_parameters['s_mazur_food'])

    #columns=['id', 'hyperbol_Food_beta', 'hyperbol_Food_k', 'aic_Food_hyperbol',
    #                                           'hyperbol_Money_beta', 'hyperbol_Money_k', 'aic_Money_hyperbol',
    #                                           'exponential_Food_beta', 'exponential_Food_k', 'aic_Food_exponential',
    #                                           'exponential_Money_beta', 'exponential_Money_k', 'aic_Money_exponential',
    #                                           'meyer_green_Food_beta', 'meyer_green_Food_k', 'meyer_Food_near_s',
    #                                           'aic_Food_meyer_green', 'meyer_Money_self_beta', 'meyer_Money_self_k',
    #                                           'meyer_green_Money_s', 'aic_Money_meyer_green',
    #                                           'mazur_Food_beta', 'mazur_Food_k', 'mazur_Food_s', 'aic_Food_mazur',
    #                                           'mazur_Money_beta', 'mazur_Money_k', 'mazur_Money_s', 'aic_Money_mazur',
    #                                           'prelec_Food_beta', 'prelec_Food_k', 'prelec_Food_s', 'aic_Food_prelec',
    #                                           'prelec_Money_beta', 'prelec_Money_k', 'prelec_Money_s',
    #                                           'aic_Money_prelec', 'Coin_Later_RT', 'Coin_Now_RT', 'Food_Later_RT',
    #                                           'Food_Now_RT', 'participant_code'])

    ax = get_correlation_strength_and_make_fancy_graphics(
        all_data[
            ['exponential_Money_k', 'hyperbol_Money_k', 'exponential_Money_k', 'mazur_Money_k', 'prelec_Money_k',
             'exponential_Food_k', 'hyperbol_Food_k', 'mazur_Food_k', 'mazur_Food_k', 'prelec_Food_k']],
        0.05/10, ['Exponential Money', 'Hyperbole Money', 'Green & Myerson Money', 'Mazur Money', 'Prelec Money',
               'Exponential Food', 'Hyperbole_food', 'Green & Myerson Food', 'Mazur Food', 'Prelec Food'])
    ax.ax_col_dendrogram.set_title('Spearman Correlation of k for the "Type of Reward" study')
    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen')
    ax.savefig('heatmap_k_delta_aic_type.png')

    ax = get_correlation_strength_and_make_fancy_graphics(
        all_data[
            ['meyer_Money_self_beta', 'mazur_Money_beta', 'prelec_Money_beta', 'meyer_green_Food_beta', 'mazur_Food_beta',
             'prelec_Food_beta']],
        0.05/6,
        ['Green & Myerson Money', 'Mazur Money', 'Prelec Money', 'Green & Myerson Food', 'Mazur Food', 'Prelec Food'])
    ax.ax_col_dendrogram.set_title('Spearman Correlation of beta for the "Type of Reward" study')
    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen')
    ax.savefig('heatmap_beta_delta_aic_type.png')

    scipy.stats.spearmanr(rsquared_parameters['k_exponential_food'],rsquared_parameters['k_hyperbole_food'])
    scipy.stats.spearmanr(rsquared_parameters['k_hyperbole_food'],rsquared_parameters['k_GM_Food'])
    scipy.stats.spearmanr(rsquared_parameters['k_hyperbole_money'], rsquared_parameters['k_GM_Money'])

    #all_data.boxplot(column=['delta_aic_food_exponential', 'delta_aic_food_hyperbol', 'delta_aic_food_meyer_green', 'delta_aic_food_mazur', 'delta_aic_food_prelec'],  grid=True, rot=0,
    #             patch_artist=True)

    min_money = all_data[['aic_Money_hyperbol','aic_Money_exponential','aic_Money_meyer_green','aic_Money_mazur','aic_Money_prelec']].min(axis=1)
    all_data['delta_aic_Money_hyperbol'] =  all_data['aic_Money_hyperbol'] - min_money
    all_data['delta_aic_Money_exponential'] = all_data['aic_Money_exponential'] - min_money
    all_data['delta_aic_Money_meyer_green'] = all_data['aic_Money_meyer_green'] -min_money
    all_data['delta_aic_Money_mazur'] = all_data['aic_Money_mazur'] - min_money
    all_data['delta_aic_Money_prelec'] = all_data['aic_Money_prelec'] -min_money
    all_data['delta_aic_Money_prelec'].mean()
    all_data['delta_aic_Money_prelec'].std()
    scipy.stats.wilcoxon(all_data['delta_aic_Money_mazur'],  all_data['delta_aic_Money_prelec'] )
    mean_delta_aic_Money = all_data[['delta_aic_Money_exponential', 'delta_aic_Money_hyperbol',  'delta_aic_Money_meyer_green', 'delta_aic_Money_mazur', 'delta_aic_Money_prelec']].mean()

    plt.bar([0, 1,2,3,4],mean_delta_aic_Money, tick_label=['delta_aic_Money_exponential', 'delta_aic_Money_hyperbol',  'delta_aic_Money_meyer_green', 'delta_aic_Money_mazur', 'delta_aic_Money_prelec'])
    plt.savefig(results_methods_dir + '/delta_aic_Money.png')
    plt.close()
    sns.set(rc={'figure.figsize': (22, 8.27)})
    sns.set(font_scale=3)
    my_pal_food = {"delta_aic_Money_exponential": "darkmagenta", "delta_aic_Money_hyperbol": "limegreen", "delta_aic_Money_mazur": "darkgoldenrod" ,"delta_aic_Money_meyer_green":"fuchsia",'delta_aic_Money_prelec':'royalblue'}
    ax = sns.barplot(data= all_data[['delta_aic_Money_exponential', 'delta_aic_Money_hyperbol',  'delta_aic_Money_meyer_green', 'delta_aic_Money_mazur','delta_aic_Money_prelec']],  palette=my_pal_food, errorbar='se')
    ax.set_xticklabels(['Exponential', 'Hyperbole', 'Hyperbole \n with scaling \n Delay',
                        'Hyperbole \n with scaling \n Delay & Discounting',
                        'Exponential \n with time scaling'])
        #['Exponential', 'Hyperbole', 'Hyperbole \n with Time scaling', 'Hyperbole \n with Time scaling (Mazur)',
         #'Exponential \n with time scaling'])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.patches[3].set_edgecolor('red')
    ax.patches[3].set_linewidth(5)
    ax.axhline(y=2, color='black',linestyle='--',linewidth=3)
    ax.set(ylim=(0, 11))


    ax.set_xlabel("Model Fit for the Money condition")
    ax.set_ylabel(u'ΔAIC')

    delta_Aic_heat_map_money = get_wilcoxon_rank_and_make_fancy_graphics(all_data[['delta_aic_Money_exponential', 'delta_aic_Money_hyperbol',
                                                                                'delta_aic_Money_meyer_green', 'delta_aic_Money_mazur', 'delta_aic_Money_prelec']],significance_value=0.05)

    os.chdir('C:/Users/mariu/Pictures/DoktorarbeitAbbildungen')
    delta_Aic_heat_map_money.savefig('heatmap_money_delta_aic.png')

    plt.close()

    #heat_map_near.savefig(results_methods_dir + '/heatmap_aic_money.png')
    plt.close()

    all_data['meyer_Food_near_s'].mean()
    all_data['meyer_green_Money_s'].mean()
    all_data['meyer_green_Food_beta'].mean()
    all_data['meyer_Money_self_beta'].mean()


    #all_data.boxplot(column=['delta_aic_Money_exponential', 'delta_aic_Money_hyperbol',
    #                                                                            'delta_aic_Money_meyer_green', 'delta_aic_Money_mazur', 'delta_aic_Money_prelec'],  grid=True, rot=0,
    #             patch_artist=True)
    plt.scatter()
    #all_data[['delta_aic_Money_exponential', 'delta_aic_Money_hyperbol',  'delta_aic_Money_meyer_green', 'delta_aic_Money_mazur', 'delta_aic_Money_prelec']].mean().plot.bar()
    plt.scatter(all_data['Food_Later_RT'], all_data['aic_Money_mazur'])
    all_data_pearson = all_data[['Food_Later_RT', 'aic_Money_mazur']].dropna()
    scipy.stats.pearsonr(all_data_pearson['Food_Later_RT'], all_data_pearson['aic_Money_mazur'])
    all_data_t_test = all_data[['meyer_Food_near_s','meyer_green_Money_s']]
    all_data_t_test2 = all_data[['meyer_green_Food_k', 'meyer_Money_self_k']]
    scipy.stats.ttest_rel(all_data_t_test['meyer_Food_near_s'],all_data_t_test['meyer_green_Money_s'])
    scipy.stats.ttest_rel(all_data_t_test2['meyer_green_Food_k'], all_data_t_test2['meyer_Money_self_k'])
    scipy.stats.ttest_rel(all_data[['Food_Now_RT']])

    days = [2, 7, 14, 21, 31, 62, 93, 128, 256, 365]
    days_short = [0.5, 1, 2, 3, 4, 5]
    k = 0.01
    exponential_values = []
    for day in days:
        y_value = exponential_discounting(day, k)
        exponential_values.append(y_value)

    plt.plot(days, exponential_values)
    plt.xlabel("Tage")
    plt.ylabel("Subjektiver Werte")

    hyperbolic_values = []
    for day in days:
        y_value = hyperbole_function(day, k)
        hyperbolic_values.append(y_value)


    plt.plot(days, exponential_values)
    plt.plot(days, hyperbolic_values)
    plt.xlabel("Tage")
    plt.ylabel("Subjektiver Werte")
    plt.legend(['Exponential Funktion', 'Hyperbole Funktion'])

    days_short = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 21, 31, 64, 128, 192, 256, 356, 712, 1068]
    days_real= [2, 14,31,91,182.5,365]
    os.chdir('C:/Users/mariu/Documents/Arbeit/DelayDiscountingFood/indiff_plot_winning_model')
    for i in range(len(rsquared_parameters)):

        k_mazur_rsquared = rsquared_parameters['k_mazur_money'][i]
        s_mazur_rsquared = rsquared_parameters['s_mazur_money'][i]
        k_prelec_rsquared = rsquared_parameters['k_prelec_food'][i]
        s_prelec_food =  rsquared_parameters['s_prelec_food'][i]

        plot_mazur = []
        plot_prelec = []

        for day in days_real:
            plot_mazur.append(mazur(day,k_mazur_rsquared,s_mazur_rsquared))
            plot_prelec.append(prelec(day,k_prelec_rsquared,s_prelec_food))

        plt.plot(days_real,plot_mazur, label='Hyperbole with time scaling for Money', color='lime', linewidth = 6)
        plt.plot(days_real,plot_prelec, label='Exponential with time scaling for Food', color='orchid', linewidth = 6)
        plt.legend()
        plt.rc('xtick', labelsize=40)
        plt.rc('ytick', labelsize=40)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        #plt.xlabel('Delay in days')
        #plt.ylabel('indifference points')
        #plt.legend()
        name = str(i) + '_plot.png'
        plt.savefig(name)
        plt.close()

    k = 0.01
    s = 1.5
    exponential_values = []
    for day in days_short:
        y_value = exponential_discounting(day, k)
        exponential_values.append(y_value)

    # plt.plot(days_short, exponential_values)
    # plt.xlabel("Tage")
    # plt.ylabel("Subjektiver Werte")

    hyperbolic_values = []
    for day in days_short:
        y_value = hyperbole_function(day, k)
        hyperbolic_values.append(y_value)

    green_myerson_values = []

    for day in days_short:
        y_value = green_meyerson_discounting(day, k, s)
        green_myerson_values.append(y_value)

    mazur_values = []

    for day in days_short:
        y_value = mazur(day, k, s)
        mazur_values.append(y_value)


    prelec_values = []

    for day in days_short:
        y_value = prelec(day,k,s)
        prelec_values.append(y_value)




    plt.plot(days_short, exponential_values)
    plt.plot(days_short, hyperbolic_values)
    plt.plot(days_short,green_myerson_values)
    plt.plot(days_short, mazur_values)
    plt.plot(days_short,prelec_values)
    plt.xlabel("days", color='black')
    plt.ylabel("Subjective value at the same offer", color='black')
    plt.legend(['exponential function', 'hyperbolic function','green_myerson','mazur','prelec'], labelcolor='linecolor')
    plt.figtext(0.7, 0.7, "k = 0.01", c='black')
    plt.rcParams.update({'text.color': "black"})

    immediate_offer = [50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
    indifference_line = np.ones(50)
    indifference_line = indifference_line*0.5
    #immediate_offer = 2
    Choosen_Reward_D = 1
    Delay = 265
    k = 0.0001
    beta = 2
    s = 2
    likeness_k_0001_beta_2_s_2 = []
    for offer in immediate_offer:
        print(offer)
        likeness_k_0001_beta_2_s_2.append(two_parameter_softmax_green(Choosen_Reward_D, offer, Delay, k, beta, s))
        print(likeness_k_0001_beta_2_s_2)

    all_data
    k=0.001
    beta=2
    s=2
    likeness_k_001_beta_2_s_2 = []
    for offer in immediate_offer:
        print(offer)
        likeness_k_001_beta_2_s_2.append(two_parameter_softmax_green(Choosen_Reward_D, offer, Delay, k, beta, s))
        print(likeness_k_001_beta_2_s_2)
    #altering s
    s=0.1
    k = 0.0001
    beta = 2
    likeness_k_0001_beta_2_s_05 = []
    for offer in immediate_offer:
        print(offer)
        likeness_k_0001_beta_2_s_05.append(two_parameter_softmax_green(Choosen_Reward_D, offer, Delay, k, beta, s))
        print(likeness_k_0001_beta_2_s_05)

    money = np.zeros(shape=(7))
    money_min = np.zeros(shape=(7))
    money_max =  np.zeros(shape=(7))
    food = np.zeros(shape=(7))
    food_min = np.zeros(shape=(7))
    food_max = np.zeros(shape=(7))

    days_real =  [0,2,14,31,91,182.5,365]
    money[0] = 40
    money[1] = calculate_indiff_bayesian_mazur(2,mean_k_mg,mean_b_mg,mean_s_mg)[1]# 35.5705# two_parameter_softmax_green(1, 35.5705, 2,mean_k_mg, mean_b_mg, mean_s_mg)
    money[2] = calculate_indiff_bayesian_mazur(14,mean_k_mg,mean_b_mg,mean_s_mg)[1]#23.62# two_parameter_softmax_green(1, 23.62, 14, mean_k_mg, mean_b_mg, mean_s_mg)
    money[3] = calculate_indiff_bayesian_mazur(31,mean_k_mg,mean_b_mg,mean_s_mg)[1]#17.612#two_parameter_softmax_green(1, 17.612, 31, mean_k_mg, mean_b_mg, mean_s_mg)
    money[4] = calculate_indiff_bayesian_mazur(91,mean_k_mg,mean_b_mg,mean_s_mg)[1]#10.957# two_parameter_softmax_green(1, 10.957, 91, mean_k_mg, mean_b_mg, mean_s_mg)
    money[5] =calculate_indiff_bayesian_mazur(182.5,mean_k_mg,mean_b_mg,mean_s_mg)[1]#7.86# two_parameter_softmax_green(1, 7.86, 182.5, mean_k_mg, mean_b_mg, mean_s_mg)
    money[6] = calculate_indiff_bayesian_mazur(365,mean_k_mg,mean_b_mg,mean_s_mg)[1]#5.59# two_parameter_softmax_green(1, 5.59, 365, mean_k_mg, mean_b_mg, mean_s_mg)


    money_min[0] = 40
    money_min[1] = 39.92# two_parameter_softmax_green(1, 39.92, 2,min_k_mg, min_b_mg, min_s_mg)
    money_min[2] = 39.413#two_parameter_softmax_green(1, 39.413, 14,min_k_mg, min_b_mg, min_s_mg)
    money_min[3] = 38.775#two_parameter_softmax_green(1, 38.775,  31, min_k_mg, min_b_mg, min_s_mg)
    money_min[4] = 36.99#two_parameter_softmax_green(1, 36.99, 91, min_k_mg, min_b_mg, min_s_mg)
    money_min[5] =35.091# two_parameter_softmax_green(1, 35.091, 182.5, min_k_mg, min_b_mg, min_s_mg)
    money_min[6] = 32.677#two_parameter_softmax_green(1, 32.677, 365, min_k_mg, min_b_mg, min_s_mg)

    money_max[0] = 40
    money_max[1] = 32.65# two_parameter_softmax_green(1, 32.65, 2,max_k_mg, max_b_mg, max_s_mg)
    money_max[2] = 18.17#two_parameter_softmax_green(1,18.17, 14, max_k_mg, max_b_mg, max_s_mg)
    money_max[3] = 12.5792#two_parameter_softmax_green(1,12.5792,  31, max_k_mg, max_b_mg, max_s_mg)
    money_max[4] = 7.18#two_parameter_softmax_green(1, 7.18, 91, max_k_mg, max_b_mg, max_s_mg)
    money_max[5] = 4.897#two_parameter_softmax_green(1, 4.897, 182.5, max_k_mg, max_b_mg, max_s_mg)
    money_max[6] =  3.33#two_parameter_softmax_green(1, 3.33, 365, max_k_mg, max_b_mg, max_s_mg)
    plt.plot(days_real,money)
    plt.plot(days_real,money_min)
    plt.plot(days_real,money_max)
    food[0] = 40
    food[1] = calculate_indiff_bayesian_prelec(2,mean_k_prelec,mean_b_prelec,mean_s_prelec)[1]
    food[2] = calculate_indiff_bayesian_prelec(14,mean_k_prelec,mean_b_prelec,mean_s_prelec)[1]

    food[3] = calculate_indiff_bayesian_prelec(31, mean_k_prelec, mean_b_prelec, mean_s_prelec)[1]

    food[4] = calculate_indiff_bayesian_prelec(91, mean_k_prelec, mean_b_prelec, mean_s_prelec)[1]
    food[5] = calculate_indiff_bayesian_prelec(182.5, mean_k_prelec, mean_b_prelec, mean_s_prelec)[1]

    food[6] = calculate_indiff_bayesian_prelec(365, mean_k_prelec, mean_b_prelec, mean_s_prelec)[1]

    food_max[0] = 40
    food_max[1] = calculate_indiff_bayesian_prelec(2,max_k_prelec,max_b_prelec,max_s_prelec)[1]
    food_max[2] = calculate_indiff_bayesian_prelec(14,max_k_prelec,max_b_prelec,max_s_prelec)[1]
    food_max[3] = calculate_indiff_bayesian_prelec(31,max_k_prelec,max_b_prelec,max_s_prelec)[1]
    food_max[4] = calculate_indiff_bayesian_prelec(91,max_k_prelec,max_b_prelec,max_s_prelec)[1]
    food_max[5] = calculate_indiff_bayesian_prelec(182.5, max_k_prelec, max_b_prelec, max_s_prelec)[1]
    food_max[6] = calculate_indiff_bayesian_prelec(365, max_k_prelec, max_b_prelec, max_s_prelec)[1]
    food_prop2_days, index_list = calculate_indiff_bayesian_prelec(2,mean_k_prelec,mean_b_prelec,mean_s_prelec,True)
    plt.bar(food_prop2_days, height=index_list)
    food_min[0] = 40
    food_min[1] = calculate_indiff_bayesian_prelec(2,min_k_prelec,min_b_prelec,min_s_prelec)[1]
    food_min[2] = calculate_indiff_bayesian_prelec(14, min_k_prelec, min_b_prelec, min_s_prelec)[1]
    food_min[3] = calculate_indiff_bayesian_prelec(31, min_k_prelec, min_b_prelec, min_s_prelec)[1]
    food_min[4]= calculate_indiff_bayesian_prelec(91, min_k_prelec, min_b_prelec, min_s_prelec)[1]
    food_min[5]= calculate_indiff_bayesian_prelec(182.5, min_k_prelec, min_b_prelec, min_s_prelec)[1]
    food_min[6] = calculate_indiff_bayesian_prelec(365, min_k_prelec, min_b_prelec, min_s_prelec)[1]
    plt.hist(food_prop2_days)
    plt.plot(days_real,money)
    plt.plot(days_real,money_min)
    plt.plot(days_real,money_max)
    plt.plot(days_real,food)
    plt.plot(days_real,food_max)
    plt.plot(days_real,food_min)
    plt.legend()

    plt.plot(days_real, money)
    plt.plot(days_real, food)
    sv = [38, 25, 21, 15, 10]
    days = [1, 14, 32, 64, 128]
    x = [0,0,0,0,0]
    plt.plot(days,sv)
    plt.xlabel('days')
    plt.ylabel('indifference points')
    plt.fill_between(days, sv, color="lightseagreen", alpha=0.4)

    s=0.1
    k = 0.001
    beta = 2
    likeness_k_001_beta_2_s_05 = []
    for offer in immediate_offer:
        print(offer)
        likeness_k_001_beta_2_s_05.append(two_parameter_softmax_green(Choosen_Reward_D, offer, Delay, k, beta, s))
        print(likeness_k_001_beta_2_s_05)

    #altering beta
    Delay = 265
    k = 0.0001
    beta = 4
    s =2
    likeness_k_0001_beta_4_s_2 = []
    for offer in immediate_offer:
        print(offer)
        likeness_k_0001_beta_4_s_2.append(two_parameter_softmax_green(Choosen_Reward_D, offer, Delay, k, beta, s))
        print(likeness_k_0001_beta_4_s_2)


    k=0.001
    beta=4
    s=2
    likeness_k_001_beta_4_s_2 = []
    for offer in immediate_offer:
        print(offer)
        likeness_k_001_beta_4_s_2.append(two_parameter_softmax_green(Choosen_Reward_D, offer, Delay, k, beta, s))
        print(likeness_k_001_beta_4_s_2)

    SMALL_SIZE = 18
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)



    fig, axs = plt.subplots(2, 3)
    fig.suptitle(
        'Systematic alterations of k, beta and s, at a Delay of 265 days, given the Meyer Green equation for subjective value')
    axs[0, 0].plot(immediate_offer, likeness_k_0001_beta_2_s_2)
    axs[0, 0].plot(immediate_offer, indifference_line, '--')
    axs[0, 0].set_title('k=0.0001, beta = 2, s=2')
    axs[0, 0].set_xlabel('Offer Delayed Reward')
    axs[0, 0].set_ylabel('p(delayed)')
    axs[1, 0].plot(immediate_offer, likeness_k_001_beta_2_s_2)
    axs[1, 0].plot(immediate_offer, indifference_line, '--')
    axs[1, 0].set_title('k = 0.01, beta = 2, s=2')
    axs[1, 0].set_xlabel('Offer Delayed Reward')
    axs[1, 0].set_ylabel('p(delayed)')
    axs[0, 1].plot(immediate_offer, likeness_k_0001_beta_2_s_05)
    axs[0, 1].plot(immediate_offer, indifference_line, '--')
    axs[0, 1].set_title('k=0.0001, beta = 2, s=0.1')
    axs[0, 1].set_xlabel('Offer Delayed Reward')
    axs[0, 1].set_ylabel('p(delayed)')
    axs[1, 1].plot(immediate_offer, likeness_k_001_beta_2_s_05)
    axs[1, 1].plot(immediate_offer, indifference_line, '--')
    axs[1, 1].set_title('k=0.01, beta = 2, s=0.1')
    axs[1, 1].set_xlabel('Offer Delayed Reward')
    axs[1, 1].set_ylabel('p(delayed)')
    axs[0, 2].plot(immediate_offer, likeness_k_0001_beta_4_s_2)
    axs[0, 2].plot(immediate_offer, indifference_line, '--')
    axs[0, 2].set_title('k=0.0001,  beta = 4, s=2')
    axs[0, 2].set_xlabel('Offer Delayed Reward')
    axs[0, 2].set_ylabel('p(delayed)')
    axs[1, 2].plot(immediate_offer, likeness_k_001_beta_4_s_2)
    axs[1, 2].plot(immediate_offer, indifference_line, '--')
    axs[1, 2].set_title('k=0.001, beta = 4, s=2')
    axs[1, 2].set_xlabel('Offer Delayed Reward')
    axs[1, 2].set_ylabel('p(delayed)')

    print(likeness_k_001_beta_2_s_05)
    print(likeness_k_001_beta_2_s_2)
    delay = np.arange(1,365,0.1)
    s = np.arange(0,4,0.01)

    prelecthree_d = np.zeros((len(delay), len(s)))
    meyer_green_three_D = np.zeros((len(delay), len(s)))
    mazur_three_D = np.zeros((len(delay), len(s)))
    day_counter = 0
    s_counter = 0
    for day in delay:
        s_counter = 0

        for scaling in s:
            prelecthree_d[day_counter,s_counter] = prelec(day,0.001,scaling)
            meyer_green_three_D[day_counter,s_counter] = green_meyerson_discounting(day,0.001,scaling)
            mazur_three_D[day_counter, s_counter] = mazur(day, 0.001, scaling)
            s_counter = s_counter + 1

        day_counter = day_counter + 1

    SMALL_SIZE = 12
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    x_2d, y_2d = np.meshgrid( s,delay)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x_2d,y_2d,prelecthree_d,cmap=cm.coolwarm,
                    linewidth=0, antialiased=True)
    ax.set_ylabel('Delay in days')
    ax.set_xlabel('s value')
    ax.set_zlabel('Subjective value')
    ax.set_title("Subjective Value according to Prelecs equation")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x_2d,y_2d,meyer_green_three_D,cmap=cm.coolwarm,
                    linewidth=100, antialiased=True)
    ax.set_ylabel('Delay in days')
    ax.set_xlabel('s value')
    ax.set_zlabel('Subjective value')
    ax.set_title("Subjective Value according to Meyer & Greens equation")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x_2d,y_2d,mazur_three_D,cmap=cm.coolwarm,
                    linewidth=0, antialiased=True)
    ax.set_ylabel('Delay in days')
    ax.set_xlabel('s value')
    ax.set_zlabel('Subjective value')
    ax.set_title("Subjective Value according to Rachlins equation")


if __name__ == "__main__":
    single_result('SLSQP')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
