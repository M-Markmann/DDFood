import numpy as np
import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import os
import arviz as az
import pytensor.tensor as tensor
import numpyro

#from pymc3.distributions.dist_math import normal_lcdf
#import xarray as xr
from pymc_model_collection import testing_TRADE, testing_DRIFT, testing_exponential, testing_hyperbol, testing_hyperbol_scaling_delay, testing_hyperbol_scaling_both, testing_exponential_with_scaling, testing_ITCH, trying_tester
#from pymc_model_collection_sample_pro import testing_TRADE, testing_DRIFT, testing_exponential, testing_hyperbol, testing_hyperbol_scaling_delay, testing_hyperbol_scaling_both, testing_exponential_with_scaling, testing_ITCH, trying_tester

#from theano import tensor as T
#os.environ["QT_API"] = "PyQt6"
az.rcParams["plot.matplotlib.show"] = True

def main():
    # Import data
    #dat = pd.read_csv(
    #    "https://raw.githubusercontent.com/psy-farrell/computational-modelling/master/codeFromBook/Chapter9/hierarchicalITC.dat",
    #    sep="\t")
    numpyro.set_platform('cpu')
    #numpyro
    #numpyro.set_host_device_count(1)
    chains = 8
    cores = 8
    samples = 2000
    tune = 2000
    #data_ours = pd.read_csv('/Users/mariusmarkmann/Documents/Data/DD_Food/all_choices.csv')
    #data_ours['B'] = np.ones((len(data_ours),1,))*40
    #data_ours = data_ours.rename(columns={'X1':'A','T1':'DA','T2':'DB','LaterOptionChosen':'R'})

    #
    for i in [0,1]: #1
        data_ours = pd.read_csv('/home/team-tesla-linux/Documents/DDFood/all_choices.csv')
        data_ours['B'] = np.ones((len(data_ours), 1,)) * 40
        #data_ours = data_ours.drop('R', axis=0)
        data_ours = data_ours.rename(columns={'X1': 'A', 'T1': 'DA', 'T2': 'DB', 'LaterOptionChosen': 'R','Unnamed: 0':'Trial_NR', 'R':'for_different'})
        data_ours['Subject'] = data_ours['Subject'].astype('category')
        data_ours = data_ours.loc[data_ours['Condition']==i]
        dict = {'Subject':np.arange(start=1,stop=28,step=1),'Trial_NR':data_ours['Question'].unique(),'prob':['delayed','now']}

        #Data = xr.DataArray(data_ours,
        #            dims=("user", "day"),
        #            coords={"Question": [data_ours['Question'].unique()],
        #                    "Condition": [0,1],
        #                    "DB":[data_ours['DB'].unique()],
        #                   }
        #           )
        #actions = tensor.as_tensor_variable(np.array(data_ours['R']).flatten())
        data_ours['R'] = data_ours['R'].replace([0, 1], [1, 0])
        #data_ours = data_ours.loc[data_ours['Subject'] == 23]
        # Experiment info
        n_subj = len(data_ours['Subject'].unique())
        n_trials = int(data_ours.shape[0] / n_subj)

        # Convenience lambda to reshape the data
        reshape = lambda var: data_ours[var].values.reshape((n_trials,n_subj))

        # Extract variables in n * t matrices
        A, DA = reshape('A'), reshape('DA')
        A, DA = tensor.as_tensor_variable(A), tensor.as_tensor_variable(DA)
        B, DB = reshape('B'), reshape('DB')
        B, DB = tensor.as_tensor_variable(B), tensor.as_tensor_variable(DB)
        R = np.array(reshape('R'))
        R = tensor.as_tensor_variable(R)
        #R = pd.DataFrame(R,c)
        os.chdir('/home/team-tesla-linux/Documents/DDFood/')
        #trying_trace = trying_tester(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials)
        traces_exponential = testing_exponential(chains,samples,n_subj,A,B,DB,R,DA,cores, tune, i, dict, n_trials) #runs without fail
        #traces_hyperbol = testing_hyperbol(chains,samples,n_subj,A,B,DB,R,DA,cores, tune, i, dict, n_trials) #runs without fail
        #traces_hyperbol_with_scaling = testing_hyperbol_scaling_delay(chains,samples,n_subj,A,B,DB,R,DA,cores, tune, i, dict, n_trials) #runs without fail
        #traces_hyperbol_with_scaling_both = testing_hyperbol_scaling_both(chains,samples,n_subj,A,B,DB,R,DA,cores, tune, i, dict, n_trials)
        #traces_testing_exponential_with_scaling = testing_exponential_with_scaling(chains,samples,n_subj,A,B,DB,R,DA,cores, tune, i, dict, n_trials)
        #traces_testing_ITCH = testing_ITCH(chains,samples,n_subj,A,B,DB,R,DA,cores, tune, i, dict, n_trials) #runs without fail
        #traces_testing_DRIFT = testing_DRIFT(chains,samples,n_subj,A,B,DB,R,DA,cores, tune, i, dict, n_trials)
        #traces_testing_TRADE = testing_TRADE(chains,samples,n_subj,A,B,DB,R,DA,cores, tune, i, dict, n_trials)
        #df_comp_loo = az.compare({"Exponential": traces_exponential,"Hyperbol": traces_hyperbol, "ITCH": traces_testing_ITCH,   "Hyperbol sDelay": traces_hyperbol_with_scaling, "Hyperbol sA":traces_hyperbol_with_scaling_both, "Exponential wS":traces_testing_exponential_with_scaling,"ITCH": traces_testing_ITCH, "DRIFT": traces_testing_DRIFT, "TRADE": traces_testing_TRADE}, ic="waic") # "testing":trying_trace
        #df_comp_loo = pd.DataFrame(df_comp_loo)
        name = str(i) + '_model_comparison.csv'
        df_comp_loo.to_csv(name)
        print(df_comp_loo)
        #
        #idata = pm.Model.fit(draws=3000)
        #az.plot_annotated_trace(idata)
        #plt.savefig('annotated_trace.png')

if __name__ == '__main__':
    #pm.freeze_support()
    main()