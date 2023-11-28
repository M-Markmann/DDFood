import arviz
import numpy as np
import pylab
import pymc as pm
import json
from math import e
#import pymc3
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import arviz as az
from pymc.distributions.dist_math import normal_lcdf
from sklearn.metrics import RocCurveDisplay, accuracy_score, auc, roc_curve, precision_score
import pytensor
from functools import partial
import pytensor.tensor as tensor
import pymc as pm
import numpy as np

az.rcParams["plot.matplotlib.show"] = True
print(f"Running on PyMC v{pm.__version__}")
def inv_logit(p):
    #This is th
    return (pm.math.exp(p))/(1+pm.math.exp(p))

def Phi(x,error):
    #Error function that returns a probability in range [0+error,1-error] using the inverse logit,
    return (inv_logit(x)*(1-2*error))+error

def cnv(g,x):
    return np.log(1+x*g)/x

def dimensionality_tester(distribution):
    #This is a helper function, for any xarray distribution it returns the dimensionality of the dataframe
    #This is helpful for quickly debugging PYMC models
    rng = np.random.default_rng(seed=sum(map(ord, "dimensionality")))
    draw = partial(pm.draw, random_seed=rng)
    normal_draw = draw(distribution)
    normal_draw, normal_draw.ndim
    return normal_draw





def  hyperbol_discounting(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Naive:
        k_mean = pm.Uniform('k_mean', lower=-10,upper=-2,dims=['Subject'], initval=np.repeat(-2.5,n_subj))
        k_sd = pm.Uniform('k_sd',lower=0.01, upper=3, dims=['Subject'], initval=np.repeat(1,n_subj))

        k = pm.Normal('k',mu=k_mean,sigma=k_sd,dims=['Subject'])

        error = pm.Uniform('error', lower=0.00001,upper=0.2,dims= ['Subject'], initval=np.repeat(0.001,n_subj))

        V_B = pm.Deterministic("V_B", B[:, :] /(1+ pytensor.tensor.power(math.e, k[:]) * DB[:, :]),
                               dims=['Trial_NR', 'Subject'])

        all = pm.Deterministic('all',V_B-A[:, :],dims=['Trial_NR', 'Subject'])

        p_delayed = pm.Deterministic('p_delayed',Phi(all,error))
        h = pm.Bernoulli('h', p=p_delayed, observed=R, dims=['Trial_NR', 'Subject'])
        object = pm.model_to_graphviz(formatting='plain_with_params')
        object.render(filename='graph')
        pylab.savefig('graph.png')
        weird_traces = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  init='adapt_diag', #
                                 return_inferencedata = True,  discard_tuned_samples=True, idata_kwargs={"log_likelihood": True}, nuts={'target_accept':0.9})#
        print(az.rhat(weird_traces,var_names=['k','error','k_mean'], method='rank'))

        weird_traces.extend(pm.sample_posterior_predictive(weird_traces))
        p_test_pred = weird_traces.posterior_predictive["h"].mean(dim=["chain", "draw"])
        y_test_pred = (p_test_pred >= 0.5).astype("int")
        y_test_pred = y_test_pred.to_numpy().flatten()
        y_test = R.flatten()
        y_test = (y_test >= 0.5).astype("int")
        print(f"accuracy = {accuracy_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        print(f"precision = {precision_score(y_true=y_test,y_pred=y_test_pred): 0.3f}")
        return weird_traces

def exponential_discounting(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Naive_exponential:
        k_mean = pm.Uniform('k_mean', lower=-10,upper=-2,dims=['Subject'], initval=np.repeat(-2.5,n_subj))
        k_sd = pm.Uniform('k_sd', lower=0.01, upper=3, dims=['Subject'], initval=np.repeat(1, n_subj))

        k = pm.Normal('k', mu=k_mean, sigma=k_sd, dims=['Subject'])

        error = pm.Uniform('error', lower=0.00001, upper=0.99999, dims=['Subject'], initval=np.repeat(0.001, n_subj))

        V_B = pm.Deterministic("V_B", B[:,:]*pm.math.exp(-1*pytensor.tensor.power(math.e,k[:])*DB[:,:]),
                               dims=['Trial_NR', 'Subject'])

        all = pm.Deterministic('all',V_B-A[:,:],dims=['Trial_NR', 'Subject'])
        p_delayed = pm.Deterministic('p_delayed',Phi(all,error))
        h = pm.Bernoulli('h', p=p_delayed, observed=R, dims=['Trial_NR', 'Subject'])
        weird_traces = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  init='adapt_diag',
                                         return_inferencedata = True,  discard_tuned_samples=True, idata_kwargs={"log_likelihood": True}, nuts={'target_accept':0.9}) #

        print(az.rhat(weird_traces, var_names=['k_mean', 'error'], method='rank'))
        weird_traces.extend(pm.sample_posterior_predictive(weird_traces))
        p_test_pred = weird_traces.posterior_predictive["h"].mean(dim=["chain", "draw"])
        y_test_pred = (p_test_pred >= 0.5).astype("int")
        y_test_pred = y_test_pred.to_numpy().flatten()
        y_test = R.flatten()
        y_test = (y_test >= 0.5).astype("int")
        print(f"accuracy = {accuracy_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        print(f"precision = {precision_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        return weird_traces


def hyperbol_discounting_sc_of_denominator(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Naive:
        k_mean = pm.Uniform('k_mean', lower=-10,upper=-2,dims=['Subject'], initval=np.repeat(-2.5,n_subj))
        k_sd = pm.Uniform('k_sd', lower=0.01, upper=3, dims=['Subject'], initval=np.repeat(1, n_subj))

        s_mean = pm.Uniform('s_mean', lower=0.001, upper=4, dims=['Subject'], initval=np.repeat(1, n_subj))
        s_sd = pm.Uniform('s_sd', lower=0.01, upper=3, dims=['Subject'], initval=np.repeat(1, n_subj))

        k = pm.Normal('k', mu=k_mean, sigma=k_sd, dims=['Subject'])

        error = pm.Uniform('error', lower=0.00001, upper=0.2, dims=['Subject'], initval=np.repeat(0.001, n_subj))
        s = pm.Normal('s', mu=s_mean,sigma=s_sd, dims=['Subject'], initval=np.repeat(1,n_subj))

        V_B = pm.Deterministic("V_B", B[:, :] /pytensor.tensor.power((1+ pytensor.tensor.power(math.e, k[:]) * DB[:, :]),s),
                               dims=['Trial_NR', 'Subject'])

        all = pm.Deterministic('all',V_B-A[:,:],dims=['Trial_NR', 'Subject'])
        p_delayed = pm.Deterministic('p_delayed',Phi(all,error))
        h = pm.Bernoulli('h', p=p_delayed, observed=R, dims=['Trial_NR', 'Subject'])
        weird_traces = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  init='adapt_diag',
                                         return_inferencedata = True,  discard_tuned_samples=True, idata_kwargs={"log_likelihood": True}, nuts={'target_accept':0.9})

        print(az.rhat(weird_traces, var_names=['k_mean', 's_mean', 'error'], method='rank'))
        weird_traces.extend(pm.sample_posterior_predictive(weird_traces))
        p_test_pred = weird_traces.posterior_predictive["h"].mean(dim=["chain", "draw"])
        y_test_pred = (p_test_pred >= 0.5).astype("int")
        y_test_pred = y_test_pred.to_numpy().flatten()
        y_test = R.flatten()
        y_test = (y_test >= 0.5).astype("int")
        print(f"accuracy = {accuracy_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        print(f"precision = {precision_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")

        return weird_traces


def hyperbol_discounting_sc_of_delay(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Naive:
        k_mean = pm.Uniform('k_mean', lower=-10,upper=-2,dims=['Subject'], initval=np.repeat(-2.5,n_subj))
        k_sd = pm.Uniform('k_sd', lower=0.01, upper=3, dims=['Subject'], initval=np.repeat(1, n_subj))

        s_mean = pm.Uniform('s_mean', lower=0.001, upper=4, dims=['Subject'], initval=np.repeat(1, n_subj))
        s_sd = pm.Uniform('s_sd', lower=0.01, upper=3, dims=['Subject'], initval=np.repeat(1, n_subj))

        k = pm.Normal('k', mu=k_mean, sigma=k_sd, dims=['Subject'])

        error = pm.Uniform('error', lower=0.00001, upper=0.99999, dims=['Subject'], initval=np.repeat(0.001, n_subj))
        s = pm.Normal('s', mu=s_mean, sigma=s_sd, dims=['Subject'], initval=np.repeat(1, n_subj))

        V_B = pm.Deterministic("V_B", B[:, :] /(1+ pytensor.tensor.power(math.e, k[:]) * pytensor.tensor.power((DB[:, :]),s[:])),
                               dims=['Trial_NR', 'Subject'])
        all = pm.Deterministic('all', V_B - A[:,:], dims=['Trial_NR', 'Subject'])
        p_delayed = pm.Deterministic('p_delayed', Phi(all,error))
        h = pm.Bernoulli('h', p=p_delayed, observed=R, dims=['Trial_NR', 'Subject'])
        weird_traces = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True, init='adapt_diag',
                                 return_inferencedata=True, discard_tuned_samples=True,
                                 idata_kwargs={"log_likelihood": True}, nuts={'target_accept':0.9})

        print(az.rhat(weird_traces, var_names=['k_mean', 's_mean', 'error'], method='rank'))
        weird_traces.extend(pm.sample_posterior_predictive(weird_traces))
        p_test_pred = weird_traces.posterior_predictive["h"].mean(dim=["chain", "draw"])
        y_test_pred = (p_test_pred >= 0.5).astype("int")
        y_test_pred = y_test_pred.to_numpy().flatten()
        y_test = R.flatten()
        y_test = (y_test >= 0.5).astype("int")
        print(f"accuracy = {accuracy_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        print(f"precision = {precision_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        name_trace = str(i) + '_Hyperbole_with_scaling_delay.nc'
        weird_traces.to_netcdf(name_trace)

        return weird_traces

def exponential_with_scaling_discounting(chains, samples, n_subj, A, B, DB, R, DA, condition, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Naive:
        k_mean = pm.Uniform('k_mean', lower=-10, upper=-2, dims=['Subject'], initval=np.repeat(-2.5, n_subj))
        k_sd = pm.Uniform('k_sd', lower=0.01, upper=3, dims=['Subject'], initval=np.repeat(1, n_subj))

        s_mean = pm.Uniform('s_mean',lower=0.001, upper=4, dims=['Subject'], initval=np.repeat(1, n_subj))
        s_sd = pm.Uniform('s_sd', lower=0.01, upper=3, dims=['Subject'], initval=np.repeat(1, n_subj))

        k = pm.Normal('k', mu=k_mean, sigma=k_sd, dims=['Subject'])

        error = pm.Uniform('error', lower=0.00001, upper=0.2, dims=['Subject'], initval=np.repeat(0.001, n_subj))
        s = pm.Normal('s', mu=s_mean, sigma=s_sd, dims=['Subject'], initval=np.repeat(1, n_subj))

        V_B = pm.Deterministic("V_B", B*pm.math.exp(-1*tensor.math.pow(tensor.math.exp((k[:]))*DB,(s[:]))),dims=['Trial_NR', 'Subject'])
        all = pm.Deterministic('all', V_B - A[:,:], dims=['Trial_NR', 'Subject'])
        p_delayed = pm.Deterministic('p_delayed', Phi(all, error))
        h = pm.Bernoulli('h', p=p_delayed, observed=R, dims=['Trial_NR', 'Subject'])
        weird_traces = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True, init='adapt_diag',
                                  #step=pm.Metropolis(),
                                 return_inferencedata=True, discard_tuned_samples=True,
                                 idata_kwargs={"log_likelihood": True}, nuts={'target_accept':0.9})
        name = str(i) + '_exponential_with_scaling_trace.nc'
        weird_traces.to_netcdf(name)
        print(az.rhat(weird_traces, var_names=['k_mean', 's_mean', 'error'], method='rank'))
        weird_traces.extend(pm.sample_posterior_predictive(weird_traces))
        p_test_pred = weird_traces.posterior_predictive["h"].mean(dim=["chain", "draw"])
        y_test_pred = (p_test_pred >= 0.5).astype("int")
        y_test_pred = y_test_pred.to_numpy().flatten()
        y_test = R.flatten()
        y_test = (y_test >= 0.5).astype("int")
        print(f"accuracy = {accuracy_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        print(f"precision = {precision_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")


        return weird_traces

def discounting_itch(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Naive:
        r_d_weight_sd = pm.Uniform('r_d_weight_sd',lower=0, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        r_r_weight_sd = pm.Uniform('r_r_weight_sd',lower=0, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        d_d_weight_sd = pm.Uniform('d_d_weight_sd',lower=0, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        d_r_weight_sd = pm.Uniform('d_r_weight_sd',lower=0, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))


        r_d_weight_mean = pm.Uniform('r_d_weight_mean',lower=-1, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        r_r_weight_mean = pm.Uniform('r_r_weight_mean',lower=-1, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        d_d_weight_mean = pm.Uniform('d_d_weight_mean',lower=-1, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        d_r_weight_mean = pm.Uniform('d_r_weight_mean',lower=-1, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))

        r_d_weight= pm.Normal('r_d_weight',mu=r_d_weight_mean, sigma=r_d_weight_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        r_r_weight = pm.Normal('r_r_weight',mu=r_r_weight_mean, sigma=r_r_weight_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        d_d_weight = pm.Normal('d_d_weight',mu=d_d_weight_mean, sigma=d_d_weight_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        d_r_weight = pm.Normal('d_r_weight',mu=d_r_weight_mean, sigma=d_r_weight_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))

        error = pm.Uniform('error', lower=0.00001, upper=0.2, dims=['Subject'], initval=np.repeat(0.001, n_subj))


        r_d = pm.Deterministic('r_d', (B - A) * r_d_weight[:], dims=['Trial_NR', 'Subject'])
        r_r = pm.Deterministic('r_r', ((B - A) / (((A + B) / 2))) * r_r_weight[:], dims=['Trial_NR', 'Subject'])
        d_d = pm.Deterministic('d_d', (DB - DA) * d_d_weight[:], dims=['Trial_NR', 'Subject'])
        d_r = pm.Deterministic('d_r', ((DB - DA) / (((DA + DB) / 2))) * d_r_weight[:], dims=['Trial_NR', 'Subject'])

        all_data = pm.Deterministic('all_data',  (r_d + r_r + d_d + d_r), dims=['Trial_NR', 'Subject']) #scaling *
        all = pm.Deterministic('all',all_data,dims=['Trial_NR', 'Subject'])
        p_delayed = pm.Deterministic('p_delayed',Phi(all,error))
        h = pm.Bernoulli('h', p=p_delayed, observed=R, dims=['Trial_NR', 'Subject'])
        weird_traces = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  discard_tuned_samples=True, init='adapt_diag', #step=pm.Metropolis(), DEMetropolisZ
                                         return_inferencedata = True,  idata_kwargs={"log_likelihood": True})#, nuts={'target_accept':0.85},
        print(az.rhat(weird_traces, var_names=['r_d_weight', 'r_r_weight','d_d_weight','d_r_weight'],method='rank'))
        weird_traces.extend(pm.sample_posterior_predictive(weird_traces))
        name = str(i) + 'traces_itch.nc'
        weird_traces.to_netcdf(name)

        p_test_pred = weird_traces.posterior_predictive["h"].mean(dim=["chain", "draw"])
        y_test_pred = (p_test_pred >= 0.5).astype("int")
        y_test_pred = y_test_pred.to_numpy().flatten()
        y_test = R.flatten()
        y_test = (y_test >= 0.5).astype("int")
        print(f"accuracy = {accuracy_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        print(f"precision = {precision_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")

        return weird_traces


def discounting_DRIFT(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Naive:
        scaling_1_mean = pm.Uniform('scaling_1_mean',lower=-1, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_2_mean = pm.Uniform('scaling_2_mean',lower=-1, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_3_mean = pm.Uniform('scaling_3_mean',lower=-1, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_4_mean = pm.Uniform('scaling_4_mean',lower=-1, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_5_mean = pm.Uniform('scaling_5_mean', lower=0.5, upper=3, dims=['Subject'], initval=np.repeat(1, n_subj))

        scaling_1_sd = pm.Uniform('scaling_1_sd',lower=0.01, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_2_sd = pm.Uniform('scaling_2_sd',lower=0.01, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_3_sd = pm.Uniform('scaling_3_sd',lower=0.01, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_4_sd = pm.Uniform('scaling_4_sd',lower=0.01, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_5_sd = pm.Uniform('scaling_5_sd', lower=0.01, upper=5, dims=['Subject'], initval=np.repeat(1, n_subj))

        scaling_1 = pm.Normal('scaling_1',mu=scaling_1_mean, sigma=scaling_1_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_2 = pm.Normal('scaling_2',mu=scaling_2_mean, sigma=scaling_2_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_3 = pm.Normal('scaling_3',mu=scaling_3_mean, sigma=scaling_3_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_4 = pm.Normal('scaling_4',mu=scaling_4_mean, sigma=scaling_4_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_5 = pm.Normal('scaling_5', mu=scaling_5_mean, sigma=scaling_5_sd, dims=['Subject'], initval=np.repeat(1, n_subj))


        driftD = pm.Deterministic('driftD', scaling_1[:] * (B - A), dims=['Trial_NR', 'Subject'])
        driftR = pm.Deterministic('driftR', scaling_2[:] * ((B - A) / A), dims=['Trial_NR', 'Subject'])
        driftI = pm.Deterministic('driftI', scaling_3[:] * tensor.math.pow((B / A), (1 / (DB - DA) - 1)),
                                  dims=['Trial_NR', 'Subject'])
        driftT = pm.Deterministic('driftT', scaling_4[:] * (DB - DA), dims=['Trial_NR', 'Subject'])
        error = pm.Uniform('error', lower=0.00001, upper=0.2, dims=['Subject'], initval=np.repeat(0.001, n_subj))
        all_data = pm.Deterministic('all_data', (driftD + driftR + driftI + driftT) * scaling_5[:], dims=['Trial_NR', 'Subject'])
        all = pm.Deterministic('all', all_data, dims=['Trial_NR', 'Subject'])
        p_delayed = pm.Deterministic('p_delayed', Phi(all,error))
        h = pm.Bernoulli('h', p=p_delayed, observed=R, dims=['Trial_NR', 'Subject'])
        weird_traces = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True, discard_tuned_samples=True, init='adapt_diag', #step=pm.Metropolis(), DEMetropolisZ
                                         return_inferencedata = True,  idata_kwargs={"log_likelihood": True})#, nuts={'target_accept':0.85},
        weird_traces.extend(pm.sample_posterior_predictive(weird_traces))

        p_test_pred = weird_traces.posterior_predictive["h"].mean(dim=["chain", "draw"])
        y_test_pred = (p_test_pred >= 0.5).astype("int")
        y_test_pred = y_test_pred.to_numpy().flatten()
        y_test = R.flatten()
        y_test = (y_test >= 0.5).astype("int")
        print(f"accuracy = {accuracy_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        print(f"precision = {precision_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        return weird_traces


def discounting_TRADE(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Naive:

        scaling_1_mean = pm.Uniform('scaling_1_mean',lower=0, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_2_mean = pm.Uniform('scaling_2_mean',lower=0, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_3_mean = pm.Uniform('scaling_3_mean',lower=0, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))

        scaling_1_sd = pm.Uniform('scaling_1_sd',lower=0.01, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_2_sd = pm.Uniform('scaling_2_sd',lower=0.01, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_3_sd = pm.Uniform('scaling_3_sd',lower=0.01, upper=1, dims=['Subject'], initval=np.repeat(0.5,n_subj))


        scaling_1 = pm.Normal('scaling_1',mu=scaling_1_mean, sigma=scaling_1_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_2 = pm.Normal('scaling_2',mu=scaling_2_mean, sigma=scaling_2_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))
        scaling_3 = pm.Normal('scaling_3',mu=scaling_3_mean, sigma=scaling_3_sd, dims=['Subject'], initval=np.repeat(0.5,n_subj))

        a1 = pm.Deterministic('a1', cnv(B,scaling_2))
        a2 = pm.Deterministic('a2', cnv(A,scaling_2))
        a3 = pm.Deterministic('a3', cnv(DB,scaling_3))
        a4 = pm.Deterministic('a4', cnv(DA,scaling_3))
        error = pm.Uniform('error', lower=0.00001, upper=0.2, dims=['Subject'], initval=np.repeat(0.001, n_subj))
        all_data = pm.Deterministic('all_data',(a1-a2)-scaling_1[:]*(a3-a4))
        all = pm.Deterministic('all',all_data,dims=['Trial_NR', 'Subject'])
        p_delayed = pm.Deterministic('p_delayed',Phi(all,error))
        h = pm.Bernoulli('h', p=p_delayed, observed=R, dims=['Trial_NR', 'Subject'])
        weird_traces = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True, discard_tuned_samples=True, init='adapt_diag',  #step=pm.Metropolis(), #
                                         return_inferencedata = True,  idata_kwargs={"log_likelihood": True}) #, nuts={'target_accept':0.85},

        name = str(i) + 'traces_TRADE.nc'
        weird_traces.to_netcdf(name)
        weird_traces.extend(pm.sample_posterior_predictive(weird_traces))
        p_test_pred = weird_traces.posterior_predictive["h"].mean(dim=["chain", "draw"])
        y_test_pred = (p_test_pred >= 0.5).astype("int")
        y_test_pred = y_test_pred.to_numpy().flatten()
        y_test = R.flatten()
        y_test = (y_test >= 0.5).astype("int")
        print(f"accuracy = {accuracy_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
        print(f"precision = {precision_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")

        return weird_traces, Naive
