import numpy as np
import os
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import os
import arviz as az
import pytensor.tensor as tensor
import numpyro

from pymc_model_collection import  discounting_TRADE,discounting_DRIFT, discounting_itch,hyperbol_discounting, exponential_discounting, hyperbol_discounting_sc_of_denominator, hyperbol_discounting_sc_of_delay, exponential_with_scaling_discounting

az.rcParams["plot.matplotlib.show"] = True

def main():

    numpyro.set_platform('cpu')

    chains = 6
    cores = 8
    samples = 1500
    tune = 60000


    #
    for i in [0,1]: #,1
        data_ours = pd.read_csv('') #Add your file here
        data_ours['B'] = np.ones((len(data_ours), 1,)) * 40
        data_ours = data_ours.rename(columns={'X1': 'A', 'T1': 'DA', 'T2': 'DB', 'LaterOptionChosen': 'R','Unnamed: 0':'Trial_NR', 'R':'for_different'})
        data_ours = data_ours.loc[data_ours['Condition']==i]

        dict = {'Subject':np.arange(start=data_ours['Subject'].min(),stop=data_ours['Subject'].max()+1,step=1),'Trial_NR':data_ours['Question'].unique(),'prob':['delayed','now']}
        data_ours['Subject'] = data_ours['Subject'].astype('category')

        # Experiment info
        n_subj =len(data_ours['Subject'].unique())
        n_trials = int(data_ours.shape[0] / n_subj)

        # Convenience lambda to reshape the data
        reshape = lambda var: data_ours[var].values.reshape((n_trials,n_subj))

        # Extract variables in n * t matrices
        A, DA = reshape('A'), reshape('DA')

        B, DB = reshape('B'), reshape('DB')

        R = np.array(reshape('R'))
        condition = np.array(reshape('Condition'))

        os.chdir('/home/team-tesla-linux/Documents/DDFood/')
        hyperbol_trace = hyperbol_discounting(chains,samples,n_subj,A,B,DB,R,DA,cores, tune, i, dict, n_trials)
        exponential_trace= exponential_discounting(chains,samples,n_subj,A,B,DB,R,DA,cores, tune, i, dict, n_trials)
        hyperbol_sc_denominator_trace = hyperbol_discounting_sc_of_denominator(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials)
        hyperbol_sc_delay_trace = hyperbol_discounting_sc_of_delay(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials)
        exponential_w_sc_trace = exponential_with_scaling_discounting(chains, samples, n_subj, A, B, DB, R, DA, condition, cores, tune, i, dict, n_trials)
        itch_traces = discounting_itch(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials)
        drift_traces = discounting_DRIFT(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials)
        trade_traces = discounting_TRADE(chains, samples, n_subj, A, B, DB, R, DA, cores, tune, i, dict, n_trials)

        df_comp = az.compare({"Exponential":exponential_trace,"Hyperbol":hyperbol_trace,"Hyperbol with scaling of denominator":hyperbol_sc_denominator_trace, 'Hyperbol with scaling of the delay':hyperbol_sc_delay_trace,'Exponential with scaling':exponential_w_sc_trace,'ITCH':itch_traces,'DRIFT':drift_traces,'TRADE':trade_traces}, ic="waic",scale='negative_log') #waic
        df_comp.to_csv(str(i) + '_model_comparison.csv')

        name3 = str(i) + '_model_comparison.png'
        fig, ax = plt.subplots(1,1)

        az.plot_compare(df_comp, legend=True, title=True, textsize=14,plot_kwargs={'color_ic':'Red','ls_min_ic':'--','fontsize':12}, ax=ax, show=False)
        fig.set_figwidth(10)
        plt.tight_layout()
        ax.set_yticks([])
        fig.show()

        fig.figure(figsize=(100,100))
        fig.savefig(name3)

if __name__ == '__main__':

    main()