import arviz
import numpy as np
import pymc as pm
#import pymc3
import pandas as pd
import matplotlib.pyplot as plt
import os
import arviz as az
import pytensor
from pymc.distributions.dist_math import normal_lcdf
from functools import partial
import pytensor.tensor as tensor
import pymc as pm
import numpy as np
#import pytensor.tensor as pt
#import theano
#import theano.tensor as tt
#from theano import tensor as T
#os.environ["QT_API"] = "PyQt6"
az.rcParams["plot.matplotlib.show"] = True
print(f"Running on PyMC v{pm.__version__}")

def Phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    # A.k.a the probit transform
    return(0.5 + 0.5 * pm.math.erf(x/pm.math.sqrt(2)))

def cnv(x,g):
    return pm.math.log(1+x*g)/g

def dimensionality_tester(distribution):
    rng = np.random.default_rng(seed=sum(map(ord, "dimensionality")))
    draw = partial(pm.draw, random_seed=rng)
    normal_draw = draw(distribution)
    normal_draw, normal_draw.ndim
    return normal_draw

def trying_tester(chains,samples,n_subj,A,B,DB,R,DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Discounting:
        # Priors for Hyperbolic Discounting Parameter K
        k_mu = pm.Normal('K_mu', mu=0, sigma=10)
        print(dimensionality_tester(k_mu.shape))
        k_sigma = pm.Uniform('K_sigma', lower=0, upper=10)
        print(dimensionality_tester(k_sigma.shape))
        # Priors for Comparison Acuity (alpha)
        alpha_mu = pm.Uniform('Alpha_mu', lower=0, upper=5)
        print(dimensionality_tester(alpha_mu.shape))
        alpha_sigma = pm.Uniform('Alpha_sigma', lower=0, upper=5)
        print(dimensionality_tester(alpha_sigma.shape))
        # Participant Parameters
        #BoundedNormal = pm.Bound(pm.Normal, lower=0.0, upper=)
        k = pm.Normal('k', mu=k_mu, sigma=k_sigma, dims=['Subject'], initval=np.repeat(1,n_subj))
        print(dimensionality_tester(k.shape))
        alpha = pm.Normal('alpha', mu=alpha_mu, sigma=alpha_sigma, dims=['Subject'], initval=np.repeat(1,n_subj))
        print(dimensionality_tester(alpha.shape))

        # Subjective Values
        V_A = pm.Deterministic("V_A", A / (1 + k * DA))
        print(dimensionality_tester(V_A.shape))
        V_B = pm.Deterministic("V_B", B / (1 + k * DB))
        print(dimensionality_tester(V_B.shape))

        # Psychometric function
        log_P = normal_lcdf(0, 1, (V_B - V_A) / alpha[:])  # log(p)
        log_1m_P = pm.math.log1mexp(-log_P)  # log(1-p)
        logit_P = pm.Deterministic('logit_P', log_P - log_1m_P)
        # Observed
        h = pm.Bernoulli('h', logit_p=logit_P, observed=R)
        print(dimensionality_tester(h.shape))
        # Sample
        Discounting_trace = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  init='adapt_diag',
                                         return_inferencedata = True, nuts = {'target_accept': 0.75}, discard_tuned_samples=False, idata_kwargs={"log_likelihood": True, })
        return Discounting_trace


def testing_exponential(chains,samples,n_subj,A,B,DB,R,DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Exponential:
        #k_mu = pm.Normal('K_mu', mu=0, sigma=10, dims=['Subject'])

        #k_sigma = pm.Uniform('K_sigma', lower=0, upper=4, dims=['Subject'])
        #alpha_mu = pm.Normal('alpha_mu', mu=2, sigma=5, dims=['Subject'])

        #alpha_sigma = pm.Uniform('alpha_sigma', lower=0, upper=10, dims=['Subject'])

        k = pm.Uniform('k',  lower=0, upper=5, dims=['Subject'], initval=np.repeat(1,n_subj)) #mu=k_mu, sigma=k_sigma,
        alpha = pm.Uniform('alpha', lower=0.001, upper=5, dims=['Subject'], initval=np.repeat(1,n_subj)) #mu=alpha_mu, sigma=alpha_sigma, , shape=(n_subj,1)
        # Subjective Values
        V_A = pm.Deterministic("V_A", A +k[:]*DA, dims=['Trial_NR','Subject'])
        V_B = pm.Deterministic("V_B", B*pm.math.exp(pytensor.tensor.log10(k[:])*DB), dims=['Trial_NR','Subject'])

        #softmax = pm.Deterministic('softmax', pm.math.exp(V_B)/((pm.math.exp(V_B))+pm.math.exp(V_A)), dims=['Trial_NR','Subject'])#, dims=['Subject','Trial_NR','prob']) #*alpha[:]

        #softmax = pm.Deterministic('softmax', pm.math.exp(V_B / alpha[:]) / (
        #            (pm.math.exp(V_B / alpha[:])) + pm.math.exp(V_A / alpha[:])),
        #                               dims=['Trial_NR', 'Subject'])  # , dims=['Subject','Trial_NR','prob'])
        p_delayed = pm.Deterministic('p_delayed', ((1)/(1+pm.math.exp(-1*(V_B-V_A))))-0.5) #alpha[:]*
        softmax_scaled =pm.Deterministic('softmax_scaled', 0.001 * 0.5 + ((1 - 0.001) * p_delayed))
        h = pm.Bernoulli('h', p=softmax_scaled, observed=R, dims=['Trial_NR','Subject']) #was logit_p #, shape=(n_subj,n_trials) , shape=(n_trials,n_subj)
        # Sample

        #log_P = normal_lcdf(0, 1, (V_B - V_A) / alpha[:])  # log(p)
        #log_1m_P = pm.math.log1mexp(-log_P)  # log(1-p)
        #logit_P = pm.Deterministic('logit_P', log_P - log_1m_P)
        # Observed
        #h = pm.Bernoulli('h', logit_p=logit_P, observed=R)
        Discounting_trace_exponential =  pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  init='adapt_diag', #nuts = {'target_accept': 0.75},
                                         return_inferencedata = True, discard_tuned_samples=False, idata_kwargs={"log_likelihood": True, }) #,  ,return_inferencedata=True , nuts={'target_accept':0.9}, step=pm.Metropolis(), , init='adapt_diag'jitter_max_retries=0, idata_kwargs={"log_likelihood": True, }, , idata_kwargs={"log_likelihood": True, }

        #

        parameters_k = pd.DataFrame(az.hdi(Discounting_trace_exponential, var_names=["k"], hdi_prob=0.95).k) #.to_array()
        #print(az.hdi(Discounting_trace_exponential, var_names=["k"], hdi_prob=0.95).k.to_array())  # .to_array() values " , "alpha"
        #parameters_alpha = pd.DataFrame(az.hdi(Discounting_trace_exponential, var_names=["alpha"], hdi_prob=0.95).alpha)
        parameters = pd.concat([parameters_k], axis=1) #,parameters_alpha
        parameters.columns = ["lower_bound_k","higher_bound_k"] #,"lower_bound_alpha","higher_bound_alpha"
        #parameters = parameters.rename(index={0:"lower_bound_k",1:"higher_bound_k",2:"lower_bound_alpha",3:"higher_bound_alpha"})
        #print(parameters_k)
        #parameters = pd.DataFrame(az.summary(Discounting_trace_exponential, round_to=2, var_names=["k", "alpha"], kind='stats'))
        name = str(i) + 'Exponential_without_scaling.csv'
        parameters.to_csv(name)
        #arviz.plot_posterior(Discounting_trace_exponential, var_names=['alpha','k'])
        return Discounting_trace_exponential


def testing_hyperbol(chains,samples,n_subj,A,B,DB,R,DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Hyperbol:
        #k_mu = pm.Normal('K_mu', mu=1, sigma=5, dims=['Subject'])

        #k_sigma = pm.Uniform('K_sigma', lower=0, upper=4, dims=['Subject'])
        #alpha_mu = pm.Normal('alpha_mu', mu=2, sigma=5, dims=['Subject'])

        #alpha_sigma = pm.Uniform('alpha_sigma', lower=0, upper=10, dims=['Subject'])

        k = pm.Uniform('k', lower=0, upper=5, dims=['Subject'], initval=np.repeat(1, n_subj))  # mu=k_mu, sigma=k_sigma,
        alpha = pm.Uniform('alpha', lower=0.001, upper=5, dims=['Subject'], initval=np.repeat(1, n_subj))
        #k = pm.Uniform('k', lower=0.000001, upper=5, dims=['Subject'],
        #               initval=np.repeat(1, n_subj))  # mu=k_mu, sigma=k_sigma,
        #alpha = pm.Uniform('alpha', lower=0.5, upper=10, dims=['Subject'],
        #                   initval=np.repeat(1, n_subj))  # mu=alpha_mu, sigma=alpha_sigma,
        # Subjective Values
        # V_A = pm.Deterministic("V_A", A / (1 + k[:, None] * DA))
        V_A = pm.Deterministic("V_A", A +k[:]*DA, dims=['Trial_NR','Subject'])
        V_B = pm.Deterministic("V_B", B / (1 + k[:]*DB), dims=['Trial_NR','Subject'])
        # V_B = pm.Deterministic("V_B", B*pm.math.exp(k[:, None]*-DB)) #[:, None]
        softmax = pm.Deterministic('softmax', pm.math.exp(V_B / alpha[:]) / (
                (pm.math.exp(V_B / alpha[:])) + pm.math.exp(V_A / alpha[:])),
                                   dims=['Trial_NR', 'Subject'])  # , dims=['Subject','Trial_NR','prob'])

        h = pm.Bernoulli('h', p=softmax, observed=R, dims=['Trial_NR', 'Subject']) # , dims=['Subject','Trial_NR','prob']) #*alpha[:]
        # Observed
         # was logit_p #, shape=(n_subj,n_trials) , shape=(n_trials,n_subj)
        #log_P = normal_lcdf(0, 1, (V_B - V_A) / alpha[:])  # log(p)
        #log_1m_P = pm.math.log1mexp(-log_P)  # log(1-p)
        #logit_P = pm.Deterministic('logit_P', log_P - log_1m_P)
        # Observed
        #h = pm.Bernoulli('h', logit_p=logit_P, observed=R)
        # Observed
        #h = pm.Bernoulli('h', p=P, observed=R) #was pm.Bernoulli
        #Hyperbol.debug()
        # Sample

        initvals = np.array([{'k': 3, 'alpha': 1} for k in range(chains)])
        Discounting_trace_hyperbol =pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  init='adapt_diag', #n_init=5000, nuts = {'target_accept': 0.90, 'max_treedepth':25},
                                         return_inferencedata = True,  discard_tuned_samples=False, idata_kwargs={"log_likelihood": True }) #,  ,return_inferencedata=True , nuts={'target_accept':0.9}, step=pm.Metropolis(), , init='adapt_diag'jitter_max_retries=0, idata_kwargs={"log_likelihood": True, },, idata_kwargs={"log_likelihood": True, }

        #
        parameters_k = pd.DataFrame(az.hdi(Discounting_trace_hyperbol, var_names=["k"], hdi_prob=0.95).k) #.to_array()
        #print(az.hdi(Discounting_trace_exponential, var_names=["k"], hdi_prob=0.95).k.to_array())  # .to_array() values " , "alpha"
        #parameters_alpha = pd.DataFrame(az.hdi(Discounting_trace_hyperbol, var_names=["alpha"], hdi_prob=0.95).alpha)
        parameters = pd.concat([parameters_k], axis=1) #,parameters_alpha
        parameters.columns = ["lower_bound_k","higher_bound_k"] #,"lower_bound_alpha","higher_bound_alpha"
        name = str(i) + 'hyperbole_without_scaling.csv'
        parameters.to_csv(name)
        #arviz.plot_posterior(Discounting_trace_hyperbol, var_names=['alpha','k'])
        return Discounting_trace_hyperbol


def testing_hyperbol_scaling_delay(chains,samples,n_subj,A,B,DB,R,DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Hyperbol:
        #k_mu = pm.Normal('K_mu', mu=0, sigma=10, dims=['Subject'])

        #k_sigma = pm.Uniform('K_sigma', lower=0, upper=10, dims=['Subject'])
        #alpha_mu = pm.Normal('alpha_mu', mu=2, sigma=5, dims=['Subject'])

        #alpha_sigma = pm.Uniform('alpha_sigma', lower=0, upper=10, dims=['Subject'])
        #s_mu = pm.Normal('s_mu', mu=1, sigma=1, dims=['Subject'])

        #s_sigma = pm.Uniform('s_sigma', lower=0, upper=10, dims=['Subject'])

        k = pm.Uniform('k', lower=0, upper=5, dims=['Subject'], initval=np.repeat(1, n_subj))  # mu=k_mu, sigma=k_sigma,
        alpha = pm.Uniform('alpha', lower=0.01, upper=5, dims=['Subject'], initval=np.repeat(1, n_subj))
        s = pm.Uniform('s', lower=0.01, upper=5, dims=['Subject'], initval=np.repeat(1,n_subj))
        # Subjective Values
        V_B = pm.Deterministic("V_B", (B / (1 + k[:] * (DB ** s[:]))), dims=['Trial_NR','Subject']) #[:, None]
        V_A = pm.Deterministic("V_A", (A / (1 + k[:]) * DA ), dims=['Trial_NR','Subject'])
        #V_B = pm.Deterministic("V_B", B * pm.math.exp(-k[:, None] * DB))

        # Psychometric function
        # P = pm.Deterministic('P', ((Phi((V_B) -A) / alpha[:, None])))
        # P = pm.Deterministic('P', )
        softmax = pm.Deterministic('softmax', pm.math.exp(V_B / alpha[:]) / (
                (pm.math.exp(V_B / alpha[:])) + pm.math.exp(V_A / alpha[:])),
                                   dims=['Trial_NR', 'Subject'])  # , dims=['Subject','Trial_NR','prob'])

        h = pm.Bernoulli('h', p=softmax, observed=R, dims=['Trial_NR', 'Subject']) # , dims=['Subject','Trial_NR','prob']) #*alpha[:]

        # Observed
        # h = pm.Bernoulli('h', p=P, observed=R) #was pm.Bernoulli
        initvals = {
                    "k_interval__": np.repeat(0.05,n_subj),
                    "alpha_interval__": np.repeat(3,n_subj),
                    "s_interval__": np.repeat(1,n_subj)}

        # Sample
        Discounting_trace_hyperbol_with_scaling = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  init='adapt_diag',
                                         return_inferencedata = True,  discard_tuned_samples=False, idata_kwargs={"log_likelihood": True, }) #,  ,return_inferencedata=True , nuts={'target_accept':0.9}, step=pm.Metropolis(), , init='adapt_diag'jitter_max_retries=0, idata_kwargs={"log_likelihood": True, },, idata_kwargs={"log_likelihood": True, }
        #nuts = {'target_accept': 0.90, 'max_treedepth':25},
        parameters_k = pd.DataFrame(az.hdi(Discounting_trace_hyperbol_with_scaling, var_names=["k"], hdi_prob=0.95).k)  # .to_array()
        # print(az.hdi(Discounting_trace_exponential, var_names=["k"], hdi_prob=0.95).k.to_array())  # .to_array() values " , "alpha"
        #parameters_alpha = pd.DataFrame(az.hdi(Discounting_trace_hyperbol_with_scaling, var_names=["alpha"], hdi_prob=0.95).alpha)
        parameters_s = pd.DataFrame(
            az.hdi(Discounting_trace_hyperbol_with_scaling, var_names=["s"], hdi_prob=0.95).s)  # .to_array()
        parameters = pd.concat([parameters_k, parameters_s], axis=1) #parameters_alpha,
        parameters.columns = ["lower_bound_k", "higher_bound_k","lower_bound_s", "higher_bound_s"] #, "lower_bound_alpha", "higher_bound_alpha"
        name = str(i) + 'hyperbol_scaling_delay.csv'
        parameters.to_csv(name)
        return Discounting_trace_hyperbol_with_scaling



def testing_hyperbol_scaling_both(chains,samples,n_subj,A,B,DB,R,DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Hyperbol:
        k_mu = pm.Normal('K_mu', mu=0, sigma=10, dims=['Subject'])

        k_sigma = pm.Uniform('K_sigma', lower=0, upper=10, dims=['Subject'])
        alpha_mu = pm.Normal('alpha_mu', mu=2, sigma=5, dims=['Subject'])

        alpha_sigma = pm.Uniform('alpha_sigma', lower=0, upper=10, dims=['Subject'])
        s_mu = pm.Normal('s_mu', mu=1, sigma=1, dims=['Subject'])

        s_sigma = pm.Uniform('s_sigma', lower=0, upper=10, dims=['Subject'])

        k = pm.Normal('k', mu=k_mu, sigma=k_sigma, dims=['Subject'],
                      initval=np.repeat(1, n_subj))  # mu=k_mu, sigma=k_sigma,
        alpha = pm.Normal('alpha', mu=alpha_mu, sigma=alpha_sigma, dims=['Subject'], initval=np.repeat(1, n_subj))
        s = pm.Normal('s', mu=s_mu, sigma=s_sigma, dims=['Subject'], initval=np.repeat(1, n_subj))
        # Subjective Values
        V_A = pm.Deterministic("V_A", A / (1 + -k[:] * DA),dims=['Trial_NR', 'Subject'])
        #V_B = pm.Deterministic("V_B", (B / tensor.math.pow(1 + (-k[:]*(DB),s[:]))),dims=['Trial_NR', 'Subject']) #[:, None]
        #V_B = pm.Deterministic("V_B", (B / tensor.math.pow((1+ (-k[:]*(DB))),s[:])), dims=['Trial_NR', 'Subject'])
        V_B = pm.Deterministic("V_B", (B / tensor.math.exp(s[:]*tensor.math.log(1+(k[:]*(DB))))), dims=['Trial_NR', 'Subject'])
        #V_B = pm.Deterministic("V_B", B * pm.math.exp(-k[:, None] * DB))

        # Psychometric function
        # P = pm.Deterministic('P', ((Phi((V_B) -A) / alpha[:, None])))
        # P = pm.Deterministic('P', )
        #softmax = pm.Deterministic('softmax', pm.math.exp(V_B / alpha[:]) / (
        #        (pm.math.exp(V_B / alpha[:])) + pm.math.exp(V_A / alpha[:])),
        #                           dims=['Trial_NR', 'Subject'])  # , dims=['Subject','Trial_NR','prob']) #*alpha[:]
        # Observed
        softmax = pm.Deterministic('softmax', pm.math.exp(V_B / alpha[:]) / (
                (pm.math.exp(V_B / alpha[:])) + pm.math.exp(V_A / alpha[:])),
                                   dims=['Trial_NR', 'Subject'])  # , dims=['Subject','Trial_NR','prob'])

        h = pm.Bernoulli('h', p=softmax, observed=R, dims=['Trial_NR', 'Subject'])  # , dims=['Subject','Trial_NR','prob']) #*alpha[:]

        # Sample
        Discounting_trace_hyperbol_with_scaling_both = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  init='adapt_diag',
                                         return_inferencedata = True, nuts = {'target_accept': 0.90, 'max_treedepth':25}, discard_tuned_samples=False, idata_kwargs={"log_likelihood": True, }) #,  ,return_inferencedata=True , nuts={'target_accept':0.9}, step=pm.Metropolis(), , init='adapt_diag'jitter_max_retries=0, idata_kwargs={"log_likelihood": True, },, idata_kwargs={"log_likelihood": True, }

        #pm.plot_trace(Discounting_trace_hyperbol_with_scaling_both, var_names=["k", "alpha"])
        parameters_k = pd.DataFrame(
            az.hdi(Discounting_trace_hyperbol_with_scaling_both, var_names=["k"], hdi_prob=0.95).k)  # .to_array()
        # print(az.hdi(Discounting_trace_exponential, var_names=["k"], hdi_prob=0.95).k.to_array())  # .to_array() values " , "alpha"
        #parameters_alpha = pd.DataFrame(
        #    az.hdi(Discounting_trace_hyperbol_with_scaling_both, var_names=["alpha"], hdi_prob=0.95).alpha)
        parameters_s = pd.DataFrame(
            az.hdi(Discounting_trace_hyperbol_with_scaling_both, var_names=["s"], hdi_prob=0.95).s)  # .to_array()
        parameters = pd.concat([parameters_k, parameters_s], axis=1) #, parameters_alpha
        parameters.columns = ["lower_bound_k", "higher_bound_k", #, "lower_bound_alpha", "higher_bound_alpha"
                              "lower_bound_s", "higher_bound_s"]
        name = str(i) + 'hyperbol_scaling_both.csv'
        parameters.to_csv(name)
        return Discounting_trace_hyperbol_with_scaling_both


def testing_exponential_with_scaling(chains,samples,n_subj,A,B,DB,R,DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as Exponential:
        k_mu = pm.Normal('K_mu', mu=0, sigma=10, dims=['Subject'])

        k_sigma = pm.Uniform('K_sigma', lower=0, upper=10, dims=['Subject'])
        alpha_mu = pm.Normal('alpha_mu', mu=2, sigma=5, dims=['Subject'])

        alpha_sigma = pm.Uniform('alpha_sigma', lower=0, upper=10, dims=['Subject'])
        s_mu = pm.Normal('s_mu', mu=1, sigma=1, dims=['Subject'])

        s_sigma = pm.Uniform('s_sigma', lower=0, upper=10, dims=['Subject'])

        k = pm.Normal('k', mu=k_mu, sigma=k_sigma, dims=['Subject'],
                      initval=np.repeat(1, n_subj))  # mu=k_mu, sigma=k_sigma,
        alpha = pm.Normal('alpha', mu=alpha_mu, sigma=alpha_sigma, dims=['Subject'], initval=np.repeat(1, n_subj))
        s = pm.Normal('s', mu=s_mu, sigma=s_sigma, dims=['Subject'], initval=np.repeat(1, n_subj))
        # Subjective Values
        V_A = pm.Deterministic("V_A", A / (1 + k[:] * DA),dims=['Trial_NR', 'Subject'])

        V_B = pm.Deterministic("V_B", B*pm.math.exp(-1*tensor.math.pow(k[:]*DB,s[:])),dims=['Trial_NR', 'Subject']) #[:, None]
        #P_chooseB = pm.Deterministic('P', (V_B*alpha[:, None]) / pm.math.sum(pm.math.exp([V_A*alpha[:, None],V_B*alpha[:, None]])))
        softmax = pm.Deterministic('softmax', pm.math.exp(V_B / alpha[:]) / (
                (pm.math.exp(V_B / alpha[:])) + pm.math.exp(V_A / alpha[:])),
                                   dims=['Trial_NR', 'Subject'])  # , dims=['Subject','Trial_NR','prob'])

        h = pm.Bernoulli('h', p=softmax, observed=R, dims=['Trial_NR', 'Subject']) # , dims=['Subject','Trial_NR','prob']) #*alpha[:]

        # Observed
        #h = pm.Bernoulli('h', p=P_chooseB, observed=R)

        # Observed

        #Exponential.debug(verbose=True)
        Discounting_trace_exponential_with_scaling = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  init='adapt_diag',
                                         return_inferencedata = True, nuts = {'target_accept': 0.90, 'max_treedepth':25}, discard_tuned_samples=False, idata_kwargs={"log_likelihood": True, }) #,  ,return_inferencedata=True , nuts={'target_accept':0.9}, step=pm.Metropolis(), , init='adapt_diag'jitter_max_retries=0, idata_kwargs={"log_likelihood": True, },, idata_kwargs={"log_likelihood": True, }

        #pm.plot_trace(Discounting_trace_exponential_with_scaling, var_names=["k", "alpha"])
        #parameters_alpha = pd.DataFrame(
        #    az.hdi(Discounting_trace_exponential_with_scaling, var_names=["alpha"], hdi_prob=0.95).alpha)
        parameters_s = pd.DataFrame(
            az.hdi(Discounting_trace_exponential_with_scaling, var_names=["s"], hdi_prob=0.95).s)  # .to_array()
        parameters_k = pd.DataFrame(
            az.hdi(Discounting_trace_exponential_with_scaling, var_names=["k"], hdi_prob=0.95).k)  # .to_array()
        parameters = pd.concat([parameters_k,parameters_s], axis=1) # parameters_alpha,
        parameters.columns = ["lower_bound_k", "higher_bound_k", #"lower_bound_alpha", "higher_bound_alpha",
                              "lower_bound_s", "higher_bound_s"]
        name = str(i) + 'exponential_with_scaling.csv'
        parameters.to_csv(name)
        return Discounting_trace_exponential_with_scaling



def testing_ITCH(chains,samples,n_subj,A,B,DB,R,DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as ITCH:
        r_d_mu = pm.Normal('r_d_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        r_d_sigma = pm.Uniform('r_d_sigma', lower=0, upper=0.16, dims=['Subject'])

        r_r_mu = pm.Normal('r_r_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        r_r_sigma = pm.Uniform('r_r_sigma', lower=0, upper=0.16, dims=['Subject'])

        d_d_mu = pm.Normal('d_d_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        d_d_sigma = pm.Uniform('d_d_sigma', lower=0, upper=0.16, dims=['Subject'])

        d_r_mu = pm.Normal('d_r_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        d_r_sigma = pm.Uniform('d_r_sigma', lower=0, upper=0.16, dims=['Subject'])

        scaling_mu = pm.Normal('scaling_mu', mu = 0.5, sigma=0.1667, dims=['Subject'])
        scaling_sigma = pm.Uniform('scaling_sigma', lower=0, upper=0.16, dims=['Subject'])

        #alpha_mu = pm.Normal('alpha_mu', mu=2, sigma=5, dims=['Subject'])
        #alpha_sigma = pm.Uniform('alpha_sigma', lower=0, upper=10, dims=['Subject'])
        # Participant Parameters
        r_d_weight = pm.Normal('r_d_weight',  mu= r_d_mu,sigma=r_d_sigma , dims=['Subject'], initval=np.repeat(0.8, n_subj))
        r_r_weight = pm.Normal('r_r_weight', mu= r_r_mu,sigma=r_r_sigma, dims=['Subject'], initval=np.repeat(0.8, n_subj))
        d_d_weight = pm.Normal('d_d_weight',  mu= d_d_mu,sigma=d_d_sigma, dims=['Subject'], initval=np.repeat(0.8, n_subj))
        d_r_weight = pm.Normal('d_r_weight', mu= d_r_mu,sigma=d_r_sigma, dims=['Subject'], initval=np.repeat(0.8, n_subj))
        scaling = pm.Normal('scaling', mu=scaling_mu, sigma=scaling_sigma, dims=['Subject'], initval=np.repeat(1, n_subj))
        #alpha = pm.Normal('alpha', mu=alpha_mu, sigma=alpha_sigma, dims=['Subject'], initval=np.repeat(1, n_subj))
        r_d = pm.Deterministic('r_d', (B-A)*r_d_weight[:],dims=['Trial_NR', 'Subject'])
        r_r = pm.Deterministic('r_r', ((B - A)/((A+B)/2)) * r_r_weight[:],dims=['Trial_NR', 'Subject'])
        d_d = pm.Deterministic('d_d', (DB-DA)*d_d_weight[:],dims=['Trial_NR', 'Subject'])
        d_r = pm.Deterministic('d_r', ((DB - DA)/((DA+DB)/2)) * d_r_weight[:],dims=['Trial_NR', 'Subject'])

        all = pm.Deterministic('all', scaling*(r_d+r_r+d_d+d_r),dims=['Trial_NR', 'Subject'])

        #alpha = pm.Uniform('alpha', lower=0.1, upper=10, shape=n_subj) #mu=alpha_mu, sigma=alpha_sigma,
        #s = pm.Uniform('s', lower=0.1, upper=10,shape=n_subj)
        # Subjective Values
        #V_A = pm.Deterministic("V_A", A / (1 + k[:, None] * DA))
        #V_B = pm.Deterministic("V_B", B / (1 + k[:, None]*DB))
        #V_B = pm.Deterministic("V_B", B*(pm.math.exp(pm.math.log(k[:, None])*-DB)**s[:, None])) #[:, None]
        #softmax = pm.Deterministic('softmax', pm.math.exp(all) / (
        #        (pm.math.exp(all)) + pm.math.exp(1-all)),
        #                           dims=['Trial_NR', 'Subject'])

        p_delayed = pm.Deterministic('p_delayed', ((1) / (1 + pm.math.exp(-1 * (all))))) #alpha[:]
        softmax_scaled = pm.Deterministic('softmax_scaled', 0.05 * 0.5 + ((1 - 0.05) * p_delayed))
        h = pm.Bernoulli('h', p=softmax_scaled, observed=R,
                         dims=['Trial_NR', 'Subject'])  # , dims=['Subject','Trial_NR','prob']) #*alpha[:]

        # Observed

        discounting_trace_ITCH =  pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,  init='adapt_diag',
                                         return_inferencedata = True, nuts = {'target_accept': 0.90, 'max_treedepth':25}, discard_tuned_samples=False, idata_kwargs={"log_likelihood": True, }) #,  ,return_inferencedata=True , nuts={'target_accept':0.9}, step=pm.Metropolis(), , init='adapt_diag'jitter_max_retries=0, idata_kwargs={"log_likelihood": True, },, idata_kwargs={"log_likelihood": True, }
        #pm.plot_trace(discounting_trace_ITCH, var_names=["k", "alpha"])
        parameters_r_d_weight = pd.DataFrame(
            az.hdi(discounting_trace_ITCH, var_names=["r_d_weight"], hdi_prob=0.95).r_d_weight)
        parameters_r_r_weight = pd.DataFrame(
            az.hdi(discounting_trace_ITCH, var_names=["r_r_weight"], hdi_prob=0.95).r_r_weight)  # .to_array()
        parameters_d_d_weight = pd.DataFrame(
            az.hdi(discounting_trace_ITCH, var_names=["d_d_weight"], hdi_prob=0.95).d_d_weight)  # .to_array()
        parameters_d_r_weight = pd.DataFrame(
            az.hdi(discounting_trace_ITCH, var_names=["d_r_weight"], hdi_prob=0.95).d_r_weight)# .to_array()
        parameters_scaling = pd.DataFrame(
            az.hdi(discounting_trace_ITCH, var_names=["scaling"], hdi_prob=0.95).scaling)# .to_array()
        parameters = pd.concat([parameters_r_d_weight, parameters_r_r_weight, parameters_d_d_weight, parameters_d_r_weight, parameters_scaling], axis=1)
        parameters.columns = ["lower_bound_r_d_weight", "higher_bound_r_d_weight", "lower_bound_r_r_weight", "higher_bound_r_r_weight",
                              "lower_bound_r_r_weight", "higher_bound_r_r_weight","lower_bound_d_r_weight", "higher_bound_d_r_weight", "lower_bound_scaling","higher_bound_scaling"]
        name = str(i) + 'discounting_trace_ITCH.csv'
        parameters.to_csv(name)
        return discounting_trace_ITCH

def testing_DRIFT(chains,samples,n_subj,A,B,DB,R,DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as DRIFT:
        scaling_1_mu = pm.Normal('scaling_1_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        scaling_1_sigma = pm.Uniform('scaling_1_sigma', lower=0, upper=0.16, dims=['Subject'])

        scaling_2_mu = pm.Normal('scaling_2_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        scaling_2_sigma = pm.Uniform('scaling_2_sigma', lower=0, upper=0.16, dims=['Subject'])

        scaling_3_mu = pm.Normal('scaling_3_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        scaling_3_sigma = pm.Uniform('scaling_3_sigma', lower=0, upper=0.16, dims=['Subject'])

        scaling_4_mu = pm.Normal('scaling_4_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        scaling_4_sigma = pm.Uniform('scaling_4_sigma', lower=0, upper=0.16, dims=['Subject'])

        scaling_5_mu = pm.Normal('scaling_5_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        scaling_5_sigma = pm.Uniform('scaling_5_sigma', lower=0, upper=0.16, dims=['Subject'])

        #alpha_mu = pm.Normal('alpha_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        #alpha_sigma = pm.Uniform('alpha_sigma', lower=0, upper=0.16, dims=['Subject'])

        scaling_1 = pm.Normal('scaling_1', mu=scaling_1_mu, sigma=scaling_1_sigma, dims=['Subject'],
                               initval=np.repeat(0.8, n_subj))
        scaling_2 = pm.Normal('scaling_2', mu=scaling_2_mu, sigma=scaling_2_sigma, dims=['Subject'],
                               initval=np.repeat(0.8, n_subj))
        scaling_3 = pm.Normal('scaling_3', mu=scaling_3_mu, sigma=scaling_3_sigma, dims=['Subject'],
                               initval=np.repeat(0.8, n_subj))
        scaling_4 = pm.Normal('scaling_4', mu=scaling_4_mu, sigma=scaling_4_sigma, dims=['Subject'],
                               initval=np.repeat(0.8, n_subj))
        scaling_5 = pm.Normal('scaling_5', mu=scaling_5_mu, sigma=scaling_5_sigma, dims=['Subject'],
                               initval=np.repeat(0.8, n_subj))
        #alpha = pm.Normal('alpha', mu=alpha_mu, sigma=alpha_sigma, dims=['Subject'],
        #                       initval=np.repeat(0.8, n_subj))

        driftD = pm.Deterministic('driftD', scaling_1[:]*(B-A),dims=['Trial_NR', 'Subject'])
        driftR = pm.Deterministic('driftR', scaling_2[:]*((B-A)/A),dims=['Trial_NR', 'Subject'])
        driftI = pm.Deterministic('driftI', scaling_3[:]*tensor.math.pow((B/A),(1/(DB-DA)-1)),dims=['Trial_NR', 'Subject'])
        driftT = pm.Deterministic('driftT', scaling_4[:]*(DB-DA),dims=['Trial_NR', 'Subject'])

        all = pm.Deterministic('all', (driftD+driftR + driftI + driftT)*scaling_5[:],dims=['Trial_NR', 'Subject'])
        p_delayed = pm.Deterministic('p_delayed', ((1) / (1 + pm.math.exp(-1 * (all))))) #alpha[:]
        softmax_scaled = pm.Deterministic('softmax_scaled', 0.05 * 0.5 + ((1 - 0.05) * p_delayed))
        h = pm.Bernoulli('h', p=softmax_scaled, observed=R,
                         dims=['Trial_NR', 'Subject'])  # , dims=['Subject','Trial_NR','prob']) #*alpha[:]
        discounting_trace_DRIFT = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,
                                           init='adapt_diag',
                                           return_inferencedata=True, nuts={'target_accept': 0.75, 'max_treedepth': 30},
                                           discard_tuned_samples=False, idata_kwargs={
                "log_likelihood": True, })  # ,  ,return_inferencedata=True , nuts={'target_accept':0.9}, step=pm.Metropolis(), , init='adapt_diag'jitter_max_retries=0, idata_kwargs={"log_likelihood": True, },, idata_kwargs={"log_likelihood": True, }

        parameters_driftD = pd.DataFrame(
            az.hdi(discounting_trace_DRIFT, var_names=["scaling_1"], hdi_prob=0.95).scaling_1)
        parameters_driftR = pd.DataFrame(
            az.hdi(discounting_trace_DRIFT, var_names=["scaling_2"], hdi_prob=0.95).scaling_2)
        parameters_driftI = pd.DataFrame(
            az.hdi(discounting_trace_DRIFT, var_names=["scaling_3"], hdi_prob=0.95).scaling_3)
        parameters_driftT = pd.DataFrame(
            az.hdi(discounting_trace_DRIFT, var_names=["scaling_4"], hdi_prob=0.95).scaling_4)
        parameters_scaling = pd.DataFrame(
            az.hdi(discounting_trace_DRIFT, var_names=["scaling_5"], hdi_prob=0.95).scaling_5)
        #parameters_alpha = pd.DataFrame(
        #    az.hdi(discounting_trace_DRIFT, var_names=["alpha"], hdi_prob=0.95).alpha)
        parameters = pd.concat([parameters_driftD, parameters_driftR, parameters_driftI, parameters_driftT, parameters_scaling, parameters_alpha], axis=1)
        parameters.columns = ["lower_bound_driftD", "higher_bound_driftD", "lower_bound_driftR",
                              "higher_bound_driftR",
                              "lower_bound_driftI", "higher_bound_driftI", "lower_bound_scaling",
                              "higher_bound_scaling" ] #,"lower_bound_alpha", "higher_bound_alpha"
        name = str(i) + 'discounting_trace_DRIFT.csv'
        parameters.to_csv(name)
        return discounting_trace_DRIFT

def testing_TRADE(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials):
    with pm.Model(coords=dict) as TRADE:
        scaling_1_mu = pm.Normal('scaling_1_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        scaling_1_sigma = pm.Uniform('scaling_1_sigma', lower=0, upper=0.16, dims=['Subject'])

        scaling_2_mu = pm.Normal('scaling_2_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        scaling_2_sigma = pm.Uniform('scaling_2_sigma', lower=0, upper=0.16, dims=['Subject'])

        scaling_3_mu = pm.Normal('scaling_3_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        scaling_3_sigma = pm.Uniform('scaling_3_sigma', lower=0, upper=0.16, dims=['Subject'])

        #alpha_mu = pm.Normal('alpha_mu', mu=0.5, sigma=0.1667, dims=['Subject'])
        #alpha_sigma = pm.Uniform('alpha_sigma', lower=0, upper=0.16, dims=['Subject'])


        scaling_1 = pm.Normal('scaling_1', mu=scaling_1_mu, sigma=scaling_1_sigma, dims=['Subject'],
                              initval=np.repeat(0.8, n_subj))
        scaling_2 = pm.Normal('scaling_2', mu=scaling_2_mu, sigma=scaling_2_sigma, dims=['Subject'],
                              initval=np.repeat(0.8, n_subj))
        scaling_3 = pm.Normal('scaling_3', mu=scaling_3_mu, sigma=scaling_3_sigma, dims=['Subject'],
                              initval=np.repeat(0.8, n_subj))

        #alpha = pm.Normal('alpha', mu=alpha_mu, sigma=alpha_sigma, dims=['Subject'],
        #                       initval=np.repeat(0.8, n_subj))


        a1 = pm.Deterministic('a1', cnv(B,scaling_2))
        a2 = pm.Deterministic('a2', cnv(A,scaling_2))
        a3 = pm.Deterministic('a3', cnv(DB,scaling_3))
        a4 = pm.Deterministic('a4', cnv(DA,scaling_3))

        all = pm.Deterministic('all',((a1-a2)-scaling_1[:]*(a3-a4)))

        p_delayed = pm.Deterministic('p_delayed', ((1) / (1 + pm.math.exp(-1 * (all))))) #alpha[:]
        softmax_scaled = pm.Deterministic('softmax_scaled', 0.05 * 0.5 + ((1 - 0.05) * p_delayed))
        h = pm.Bernoulli('h', p=softmax_scaled, observed=R,
                         dims=['Trial_NR', 'Subject'])  # , dims=['Subject','Trial_NR','prob']) #*alpha[:]
        discounting_trace_TRADE = pm.sample(samples, chains=chains, cores=cores, tune=tune, progressbar=True,
                                            init='adapt_diag',
                                            return_inferencedata=True,
                                            nuts={'target_accept': 0.75, 'max_treedepth': 30},
                                            discard_tuned_samples=False, idata_kwargs={
                "log_likelihood": True, })  # ,  ,return_inferencedata=True , nuts={'target_accept':0.9}, step=pm.Metropolis(), , init='adapt_diag'jitter_max_retries=0, idata_kwargs={"log_likelihood": True, },, idata_kwargs={"log_likelihood": True, }

        parameters_scaling_1 = pd.DataFrame(
            az.hdi(discounting_trace_TRADE, var_names=["scaling_1"], hdi_prob=0.95).scaling_1)
        parameters_scaling_2 = pd.DataFrame(
            az.hdi(discounting_trace_TRADE, var_names=["scaling_2"], hdi_prob=0.95).scaling_2)
        parameters_scaling_3 = pd.DataFrame(
            az.hdi(discounting_trace_TRADE, var_names=["scaling_3"], hdi_prob=0.95).scaling_3)

        parameters = pd.concat([parameters_scaling_1, parameters_scaling_2, parameters_scaling_3], axis=1)
        parameters.columns = ["lower_bound_scaling_1", "higher_bound_scaling_1", "lower_bound_scaling_2", "higher_bound_scaling_2",
                              "lower_bound_scaling_3", "higher_bound_scaling_3"]
        name = str(i) + 'discounting_trace_TRADE.csv'
        parameters.to_csv(name)
        return discounting_trace_TRADE

