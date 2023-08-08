import numpy as np
from Discounting_Functions import itch


def log_likelihood_single_parameter(params,condition, function):
    beta, k = params
    log_likelihood = 0
    for fun_counter in range(np.size(condition, axis=0)):
        Choosen_Reward_D = condition[fun_counter, 4]
        immediate_offer = condition[fun_counter, 5]
        Delay = condition[fun_counter, 2]
        likeness =np.log(function(Choosen_Reward_D, immediate_offer, Delay, k, beta))
        log_likelihood = likeness + log_likelihood
        #log_likelihood = log_likelihood

    return 1/log_likelihood


def log_likelihood_two_parameter(params,condition, function):
    beta, k, s = params
    log_likelihood = 0
    for fun_counter in range(np.size(condition, axis=0)):
        Choosen_Reward_D = condition[fun_counter, 4]
        immediate_offer = condition[fun_counter, 5]
        Delay = condition[fun_counter, 2]
        likeness = np.log(function(Choosen_Reward_D, immediate_offer, Delay, k, beta, s))
        log_likelihood = likeness + log_likelihood
        #log_likelihood = log_likelihood

    return 1 / log_likelihood

def log_likelihood_itch(params,condition):
    intercept, bxa, bxr, bta, btr = params
    log_likelihood = 0
    for fun_counter in range(np.size(condition, axis=0)):
        Choosen_Reward_D = condition[fun_counter, 4]
        immediate_offer = condition[fun_counter, 5]
        Delay = condition[fun_counter, 2]
        likeness = np.log(itch(Choosen_Reward_D, immediate_offer, Delay, intercept, bxa, bxr, bta, btr ))
        log_likelihood = likeness + log_likelihood
        #log_likelihood = log_likelihood

    return 1 / log_likelihood