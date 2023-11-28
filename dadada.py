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

#winner Food

arviz.load_tr

def Phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    # A.k.a the probit transform
    error = 0.001
    inv_logit_value = inv_logit(x)
    return (error + (1.0-2.0*error)*inv_logit_value)
    #return pm.math.sigmoid(x*10)
pytensor
def cnv(x,g):
    return np.log(1+x*g)/g


df = pd.read_csv('0.csv')
plt.hist(df['lower_bound_r_d_weight'])
plt.show()

plt.hist(df['lower_bound_s'])
plt.show()
plt.hist(df['higher_bound_s'])
plt.show()

df_money = pd.read_csv('1discounting_trace_ITCH.csv')
stats.ttest_rel(df['lower_bound_r_d_weight'],df_money['lower_bound_r_d_weight'])
stats.ttest_rel(df['higher_bound_r_d_weight'],df_money['higher_bound_r_d_weight'])

stats.ttest_rel(df['lower_bound_r_r_weight'],df_money['lower_bound_r_r_weight'])
stats.ttest_rel(df['higher_bound_r_r_weight'],df_money['higher_bound_r_r_weight'])

stats.ttest_rel(df['lower_bound_r_r_weight.1'],df_money['lower_bound_r_r_weight.1'])
stats.ttest_rel(df['higher_bound_r_r_weight.1'],df_money['higher_bound_r_r_weight.1'])

stats.ttest_rel(df['lower_bound_d_r_weight'],df_money['lower_bound_d_r_weight'])
stats.ttest_rel(df['higher_bound_d_r_weight'],df_money['higher_bound_d_r_weight'])

stats.ttest_rel(df['lower_bound_scaling'],df_money['lower_bound_scaling'])
stats.ttest_rel(df['higher_bound_scaling'],df_money['higher_bound_scaling'])


plt.hist(df_money['lower_bound_scaling_1'])
plt.show()
plt.hist(df_money['lower_bound_scaling_2'])
plt.show()
plt.hist(df_money['lower_bound_scaling_3'])
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df_money['lower_bound_scaling_1'],df_money['lower_bound_scaling_2'],df_money['lower_bound_scaling_3'])
scipy.stats.pearsonr(df_money['lower_bound_scaling_1'],df_money['lower_bound_scaling_2'])
scipy.stats.pearsonr(df_money['lower_bound_scaling_1'],df_money['lower_bound_scaling_3'])
scipy.stats.pearsonr(df_money['lower_bound_scaling_2'],df_money['lower_bound_scaling_3'])

trace = arviz.from_netcdf('0traces_TRADE_with_Normal.nc')
all_paramters = pd.read_csv('random_exp_w_sc_parameters.csv')
trace = arviz.from_netcdf('235_exponential_w_sc_trace.nc')
trace_food = arviz.from_netcdf('100traces_TRADE.nc')
scipy.stats.spearmanr(all_paramters['k'],trace.posterior['k'])

arviz.plot_trace(trace_food,var_names=['scaling_1_mean'])
q = trace_food.posterior['scaling_1_mean'].values
list_nd_arrays = []
for i in range(99):
    print(i)
    list_nd_arrays.append(hpd_grid(trace_food.posterior['scaling_1_mean'][:,:,i].values)[0])

list_nd_arrays = np.array(list_nd_arrays)
pd.DataFrame(list_nd_arrays)

hpd_grid(trace_food.posterior['scaling_1_mean'].stack(stacked=[...]).values, alpha=0.05)
trace_money = arviz.from_netcdf('1traces_TRADE.nc')
name = i + '0_exponential_w_sc_trace.nc'

trace_money = arviz.from_netcdf('0_exponential_w_sc_trace.nc')
trace_food = arviz.from_netcdf('1_exponential_w_sc_trace.nc')

data_waic_food = arviz.waic(trace_food,pointwise=True)
data_waic_food.waic_i.values.mean(axis=0)

data_waic_money = arviz.waic(trace_money,pointwise=True)
data_waic_money.waic_i.values.mean(axis=0)

stats.pearsonr(np.exp(trace_food.posterior['k'].values.mean(axis=(0,1))),data_waic_food.waic_i.values.mean(axis=0))
stats.pearsonr(np.exp(trace_food.posterior['k'].values.mean(axis=(0,1))),data_waic_money.waic_i.values.mean(axis=0))

trace_food

trace_food.log_likelihood.to_array()

hpd_grid(trace_money.posterior['k'])
hpd_grid(trace_food.posterior['k'])

x = np.exp(trace_food.posterior['k_mean'][:,:,:].values).flatten() - np.exp(trace_money.posterior['k_mean'][:,:,:].values).flatten()
#x = trace_food.posterior['k_mean'].values.flatten() - trace_money.posterior['k_mean'].values.flatten()
hdi, trash,trash2, mean = hpd_grid(x,alpha=0.05,roundto=1)
result1  = plt.hist(x, color='k', edgecolor='k', alpha=0.8, bins=14)
mean_value = np.mean(x)
box_width = result1[0][1] - result1[0][0]
new_valueMin = np.interp(hdi[0][0],[x.min(),x.max()],[0,1]) #
new_valueMax = np.interp(hdi[0][1],[x.min(),x.max()],[0,1]) #[result1[1][0],result1[1][-1]]
plt.axhline(xmin=new_valueMin,xmax=new_valueMax-0.01, linewidth=6, label='95%HDI', color='r') #
plt.axvline(x=mean_value, label='mean', color='b', ls='--')
plt.legend()
plt.title('Difference between estimates for k')


aaaah = trace_food.posterior['k_mean'] - trace_money.posterior['k_mean']
hpd_grid(difference_scaling_1,alpha=0.05,roundto=1)
hpd_grid(aaaah)
difference_scaling_2 = trace_food.posterior['s_mean'] - trace_money.posterior['s_mean']
hpd_grid(difference_scaling_2)
list_nd_arrays = []
for i in range(26):
    try:
        list_nd_arrays.append(hpd_grid(difference_scaling_1[:,:,i])[0])
    except:
        print(i)

hpd_grid(weird_traces_exponential_with_scaling.posterior['condition_effect_k_mean'])

x = trace_food.posterior['scaling_1_mean'].values.mean(axis=(0,1)) - all_paramters['scaling_delay']
plt.hist(x)
hdi, trash,trash2, mean = hpd_grid(x)

result1  = plt.hist(x, color='k', edgecolor='k', alpha=0.65, bins=35)
mean_value = np.mean(x)
box_width = result1[1][34] - result1[1][33]
new_valueMin = np.interp(hdi[0][0],[result1[1][0]-box_width,result1[1][35]+box_width],[0,1]) #
new_valueMax = np.interp(hdi[0][1],[result1[1][0]-box_width,result1[1][35]+box_width],[0,1]) #
plt.axhline(xmin=new_valueMin,xmax=new_valueMax, linewidth=6, label='95%HDI', color='r') #
plt.axvline(x=mean_value, label='mean', color='b', ls='--')
plt.legend()
plt.title('Relevancy of delay')




y = trace_food.posterior['scaling_2_mean'].values.mean(axis=(0,1)) - all_paramters['rel_rew']
hdi, trash,trash2, mean = hpd_grid(y)

result1  = plt.hist(y, color='k', edgecolor='k', alpha=0.65, bins=35)
mean_value = np.mean(y)
box_width = result1[1][34] - result1[1][33]
new_valueMin = np.interp(hdi[0][0],[result1[1][0]-box_width,result1[1][35]+box_width],[0,1]) #
new_valueMax = np.interp(hdi[0][1],[result1[1][0]-box_width,result1[1][35]+box_width],[0,1]) #
plt.axhline(xmin=new_valueMin,xmax=new_valueMax, linewidth=6, label='95%HDI', color='r') #
plt.axvline(x=mean_value, label='mean', color='b', ls='--')
plt.title('Relative reward')
plt.legend()


z = trace_food.posterior['scaling_3_mean'].values.mean(axis=(0,1)) - all_paramters['rel_del']
hdi, trash,trash2, mean = hpd_grid(z)

result1  = plt.hist(z, color='k', edgecolor='k', alpha=0.65, bins=35)
mean_value = np.mean(x)
box_width = result1[1][34] - result1[1][33]
new_valueMin = np.interp(hdi[0][0],[result1[1][0]-box_width,result1[1][35]+box_width],[0,1]) #
new_valueMax = np.interp(hdi[0][1],[result1[1][0]-box_width,result1[1][35]+box_width],[0,1]) #
plt.axhline(xmin=new_valueMin,xmax=new_valueMax, linewidth=6, label='95%HDI', color='r') #
plt.axvline(x=mean_value, label='mean', color='b', ls='--')
plt.title('Relative delay')
plt.legend()

error_diff = trace_food.posterior['error'].values.mean(axis=(0,1)) - all_paramters['error']
hdi, trash,trash2, mean = hpd_grid(error_diff)

result1  = plt.hist(error_diff, color='k', edgecolor='k', alpha=0.65, bins=35)
mean_value = np.mean(error_diff)
box_width = result1[1][34] - result1[1][33]
new_valueMin = np.interp(hdi[0][0],[result1[1][0]-box_width,result1[1][35]+box_width],[0,1]) #
new_valueMax = np.interp(hdi[0][1],[result1[1][0]-box_width,result1[1][35]+box_width],[0,1]) #
plt.axhline(xmin=new_valueMin,xmax=new_valueMax, linewidth=6, label='95%HDI', color='r') #
plt.axvline(x=mean_value, label='mean', color='b', ls='--')
plt.legend('Error')

value = az.from_netcdf('1_exponential_w_sc_trace_smaller_error.nc')
value.posterior['s'].values.mean(axis=(0,1))
value.posterior['k'].values.mean(axis=(0,1))

value.posterior['error'].values.mean(axis=(0,1))
trace_food.posterior['scaling_1']

all_parameters = pd.read_csv('random_exp_w_sc_parameters.csv')
estimate_shizzle = az.from_netcdf('666_exponential_w_sc_trace_smaller_error.nc')
import scipy.stats as stats
import matplotlib.pyplot as plt
stats.normaltest(np.exp(estimate_shizzle.posterior['k'].values.mean(axis=(0,1))))
stats.pearsonr(np.exp(estimate_shizzle.posterior['k'].values.mean(axis=(0,1))),np.exp(all_parameters['k']))
stats.pearsonr(estimate_shizzle.posterior['s_mean'].values.mean(axis=(0,1)),all_parameters['s'])
stats.pearsonr(estimate_shizzle.posterior['error'].values.mean(axis=(0,1)),all_parameters['error'])

plt.scatter(np.exp(estimate_shizzle.posterior['k'].values.mean(axis=(0,1)))[1:],np.exp(all_parameters['k'])[1:])
plt.scatter(estimate_shizzle.posterior['s'].values.mean(axis=(0,1)),all_parameters['s'])


plt.scatter(trace_food.posterior['scaling_1_mean'].values.mean(axis=(0,1)),all_paramters['scaling_delay'])
check_structure = trace_food.posterior['scaling_1_mean'].values

trace_food.posterior['scaling_1_mean'].values.mean(axis=(0,1))
trace_food.posterior['scaling_2_mean'].values.mean()
trace_food.posterior['scaling_3_mean'].values.mean()

trace_money.posterior['scaling_1_mean'].values.mean()
trace_money.posterior['scaling_2_mean'].values.mean()
trace_money.posterior['scaling_3_mean'].values.mean()

difference_scaling_1 = trace_food.posterior['scaling_1_mean'] - trace_money.posterior['scaling_1_mean']
difference_scaling_2 = trace_food.posterior['scaling_2_mean'] - trace_money.posterior['scaling_2_mean']
difference_scaling_3 = trace_food.posterior['scaling_3_mean'] - trace_money.posterior['scaling_3_mean']
difference_error = trace_food.posterior['error'] - trace_money.posterior['error']
difference_scaling_1 = trace.posterior['scaling_1_condition_effect']
difference_scaling_2 = trace.posterior['scaling_2_condition_effect']
difference_scaling_3 = trace.posterior['scaling_3_condition_effect']

difference_scaling_1.stack(stacked=[...]).values.max()
difference_scaling_1.stack(stacked=[...]).values.min()

hdi, trash,trash2, mean = hpd_grid(difference_scaling_1.stack(stacked=[...]).values, alpha=0.05)
hdi2, trash2, trash22, mean2 = hpd_grid(difference_scaling_2.stack(stacked=[...]).values, alpha=0.05)
hdi3, trash3, trash23, mean3 = hpd_grid(difference_scaling_3.stack(stacked=[...]).values, alpha=0.05)
hdi_error, trash_error, trash_error2, mean_error = hpd_grid(trace.posterior['error'].stack(stacked=[...]).values, alpha=0.05)
print(hdi)

for i in range(27):
    hdi, trash, trash2, mean = hpd_grid(difference_scaling_1[:,:1000,i].stack(stacked=[...]).values, alpha=0.05)
    print('Subject Number: ',i,'HDI:',hdi)
plt.hist(difference_scaling_1[:,:,0].stack(stacked=[...]).values, edgecolor='k', alpha=0.65, bins=35)

result  = plt.hist(difference_scaling_1.stack(stacked=[...]).values, color='k', edgecolor='k', alpha=0.65, bins=35)
mean_value = np.mean(difference_scaling_1.stack(stacked=[...]).values)
box_width = result[1][34] - result[1][33]
new_valueMin = np.interp(hdi[0][0],[result[1][0]-box_width,result[1][35]+box_width],[0,1]) #
new_valueMax = np.interp(hdi[0][1],[result[1][0]-box_width,result[1][35]+box_width],[0,1]) #
plt.axhline(xmin=new_valueMin,xmax=new_valueMax, linewidth=6, label='95%HDI', color='r') #
plt.axvline(x=mean_value, label='mean', color='b', ls='--')
plt.legend()
plt.title('Estimates for the change of scaling 1 between conditions')

result  = plt.hist(difference_scaling_2.stack(stacked=[...]).values, color='k', edgecolor='k', alpha=0.65, bins=35)
mean_value = np.mean(difference_scaling_2.stack(stacked=[...]).values)
box_width = result[1][34] - result[1][33]
new_valueMin = np.interp(hdi2[0][0],[result[1][0]-box_width,result[1][35]+box_width],[0,1]) #
new_valueMax = np.interp(hdi2[0][1],[result[1][0]-box_width,result[1][35]+box_width],[0,1]) #
plt.axhline(xmin=new_valueMin,xmax=new_valueMax, linewidth=6, label='95%HDI', color='r') #
plt.axvline(x=mean_value, label='mean', color='b', ls='--')
plt.legend()
plt.title('Estimates for the change of scaling 2 between conditions')

result  = plt.hist(difference_scaling_3.stack(stacked=[...]).values, color='k', edgecolor='k', alpha=0.65, bins=35)
mean_value = np.mean(difference_scaling_3.stack(stacked=[...]).values)
box_width = result[1][34] - result[1][33]
new_valueMin = np.interp(hdi3[0][0],[result[1][0]-box_width,result[1][35]+box_width],[0,1]) #
new_valueMax = np.interp(hdi3[0][1],[result[1][0]-box_width,result[1][35]+box_width],[0,1]) #
plt.axhline(xmin=new_valueMin,xmax=new_valueMax, linewidth=6, label='95%HDI', color='r') #
plt.axvline(x=mean_value, label='mean', color='b', ls='--')
plt.legend()
plt.title('Estimates for the change of scaling 3 between conditions')

difference_scaling_3 = trace_food.posterior['scaling_3'] - trace_money.posterior['scaling_3']
plt.hist(difference_scaling_3.stack(stacked=[...]).values)
np.mean(difference_scaling_3)

print()

np.sum(difference_scaling_1.stack(stacked=[...]).values>0)/len(difference_scaling_1.stack(stacked=[...]).values)
np.sum(difference_scaling_2.stack(stacked=[...]).values<0)/len(difference_scaling_2.stack(stacked=[...]).values)
np.sum(difference_scaling_3.stack(stacked=[...]).values>0)/len(difference_scaling_3.stack(stacked=[...]).values)


arviz.plot_trace(trace, var_names=['interaction_scaling1_hippocampus', 'interaction_scaling1_rdlpfc',
                                          'interaction_scaling1_l_insula', 'interaction_scaling1_hypothalamus',
                                          'interaction_scaling1_rputamen'], combined=True, compact=True)

print(np.mean(trace.posterior['interaction_scaling1_rputamen'][:]>1))
print(np.mean(trace.posterior['interaction_scaling1_hippocampus'][:]>1))
print(np.mean(trace.posterior['interaction_scaling1_rdlpfc'][:]>1))
print(np.mean(trace.posterior['interaction_scaling1_l_insula'][:]>1))
print(np.mean(trace.posterior['interaction_scaling1_hypothalamus'][:]>1))

print(np.mean(trace.posterior['interaction_scaling2_rputamen'][:]>1))
print(np.mean(trace.posterior['interaction_scaling2_hippocampus'][:]>1))
print(np.mean(trace.posterior['interaction_scaling2_rdlpfc'][:]>1))
print(np.mean(trace.posterior['interaction_scaling2_l_insula'][:]>1))
print(np.mean(trace.posterior['interaction_scaling2_hypothalamus'][:]>1))

print(np.mean(trace.posterior['interaction_scaling3_rputamen'][:]>1))
print(np.mean(trace.posterior['interaction_scaling3_hippocampus'][:]>1))
print(np.mean(trace.posterior['interaction_scaling3_rdlpfc'][:]>1))
print(np.mean(trace.posterior['interaction_scaling3_l_insula'][:]>1))
print(np.mean(trace.posterior['interaction_scaling3_hypothalamus'][:]>1))

print(np.mean(trace.posterior['interaction_scaling1_rputamen'][:]<1))
print(np.mean(trace.posterior['interaction_scaling1_hippocampus'][:]<1))
print(np.mean(trace.posterior['interaction_scaling1_rdlpfc'][:]<1))
print(np.mean(trace.posterior['interaction_scaling1_l_insula'][:]<1))
print(np.mean(trace.posterior['interaction_scaling1_hypothalamus'][:]<1))

print(np.mean(trace.posterior['interaction_scaling2_rputamen'][:]<1))
print(np.mean(trace.posterior['interaction_scaling2_hippocampus'][:]<1))
print(np.mean(trace.posterior['interaction_scaling2_rdlpfc'][:]<1))
print(np.mean(trace.posterior['interaction_scaling2_l_insula'][:]<1))
print(np.mean(trace.posterior['interaction_scaling2_hypothalamus'][:]<1))

print(np.mean(trace.posterior['interaction_scaling3_rputamen'][:]<1))
print(np.mean(trace.posterior['interaction_scaling3_hippocampus'][:]<1))
print(np.mean(trace.posterior['interaction_scaling3_rdlpfc'][:]<1))
print(np.mean(trace.posterior['interaction_scaling3_l_insula'][:]<1))
print(np.mean(trace.posterior['interaction_scaling3_hypothalamus'][:]<1))
print(trace.posterior_predictive.p_delayed.data[:][:].max())
print(trace.posterior.p_delayed.T.T[0,499])
pd.DataFrame()

parameters_k = pd.DataFrame(arviz.hdi(trace, var_names=["r_d_weight"], hdi_prob=0.50).r_d_weight)
trace.posterior.p_delayed.data[3][499].min()
arviz.plot_posterior(trace)
arviz.plot_ppc(trace)
pm.find_MAP(model=trace)
'traces_hyperbol.nc'



for condition in [0, 1]:  # 1
    data_ours = pd.read_csv('/home/team-tesla-linux/Documents/DDFood/all_choices.csv')
    data_ours['B'] = np.ones((len(data_ours), 1,)) * 40
    # data_ours = data_ours.drop('R', axis=0)
    data_ours = data_ours.rename(
        columns={'X1': 'A', 'T1': 'DA', 'T2': 'DB', 'LaterOptionChosen': 'R', 'Unnamed: 0': 'Trial_NR',
                 'R': 'for_different'})
    if condition == 0:
        data_ours = data_ours.loc[data_ours['Condition'] == condition]
        probability = []
        name = str(condition) + 'traces_itch2.nc'
        trace_ITCH = arviz.from_netcdf(name)
        d_d_weight = trace_ITCH.posterior.mean(dim=['chain', 'draw']).d_d_weight.data
        d_r_weight = trace_ITCH.posterior.mean(dim=['chain', 'draw']).d_r_weight.data
        r_d_weight = trace_ITCH.posterior.mean(dim=['chain', 'draw']).r_d_weight.data
        r_r_weight = trace_ITCH.posterior.mean(dim=['chain', 'draw']).r_r_weight.data
        #scaling_weight = trace.posterior.mean(dim=['chain', 'draw']).scaling.data

        for i in range(len(data_ours)):
            #print(i)
            current_row = data_ours.iloc[i]
            B = current_row.B
            A = current_row.A
            DB = current_row.DB
            DA = current_row.DA

                current_d_d_weight = d_d_weight[int(current_row['Subject'])-1]
                current_d_r_weight = d_r_weight[int(current_row['Subject'])-1]
                current_r_d_weight = r_d_weight[int(current_row['Subject'])-1]
                current_r_r_weight = r_r_weight[int(current_row['Subject'])-1]
                #scaling_weight_current = scaling_weight[int(current_row['Subject'])-1]
                r_d = (B - A) * current_r_d_weight
                r_r = ((B - A) / (((A + B) / 2))) * current_r_r_weight
                d_d = (DB - DA) * current_d_d_weight
                d_r = ((DB - DA) / (((DA + DB) / 2))) * current_d_r_weight
                all_data = (r_d + r_r + d_d + d_r) #scaling_weight_current *
                error = 0.001
                probability.append(error + (1.0 - 2.0 * error) * pm.math.invlogit(all_data))


    data_ours['probability'] = probability
    name = str(condition) + '_choices.csv'
    data_ours.to_csv(name)


data_ours = pd.read_csv('/home/team-tesla-linux/Documents/DDFood/all_choices.csv')
data_ours['B'] = np.ones((len(data_ours), 1,)) * 40
# data_ours = data_ours.drop('R', axis=0)
data_ours = data_ours.rename(
    columns={'X1': 'A', 'T1': 'DA', 'T2': 'DB', 'LaterOptionChosen': 'R', 'Unnamed: 0': 'Trial_NR',
                 'R': 'for_different'})
data_ours['R'] = data_ours['R'].replace([0, 1], [1, 0])
trace_ITCH = arviz.from_netcdf('1traces_itch2.nc')
traces_hyperbol = arviz.from_netcdf('0_Hyperbole_with_scaling_delay.nc')

trace_food_TRADE = arviz.from_netcdf('0traces_TRADE.nc')
trace_money_TRADE = arviz.from_netcdf('1traces_TRADE.nc')

traces_itch_food = arviz.from_netcdf('0traces_itch2.nc')
d_d_weight = trace_ITCH.posterior.d_d_weight_mean.data[:,1999,:].mean(axis=0)
d_r_weight = trace_ITCH.posterior.d_r_weight_mean.data[:,1999,:].mean(axis=0)
r_d_weight = trace_ITCH.posterior.r_d_weight_mean.data[:,1999,:].mean(axis=0)
r_r_weight = trace_ITCH.posterior.r_r_weight_mean.data[:,1999,:].mean(axis=0)
k = traces_hyperbol.posterior.k_mean.data[:,1999,:].mean(axis=0)
s = traces_hyperbol.posterior.s_mean.data[:,1999,:].mean(axis=0)

d_d_weight_food = traces_itch_food.posterior.d_d_weight_mean.data[:,1999,:].mean(axis=0)
d_r_weight_food = traces_itch_food.posterior.d_r_weight_mean.data[:,1999,:].mean(axis=0)
r_d_weight_food = traces_itch_food.posterior.r_d_weight_mean.data[:,1999,:].mean(axis=0)
r_r_weight_food = traces_itch_food.posterior.r_r_weight_mean.data[:,1999,:].mean(axis=0)

scaling_1_food = trace_food_TRADE.posterior.mean(dim=['chain', 'draw']).scaling_1
scaling_2_food = trace_food_TRADE.posterior.mean(dim=['chain', 'draw']).scaling_2
scaling_3_food = trace_food_TRADE.posterior.mean(dim=['chain', 'draw']).scaling_3
scaling_4_food = trace_food_TRADE.posterior.mean(dim=['chain', 'draw']).scaling_4

scaling_1_money = trace_money_TRADE.posterior.mean(dim=['chain', 'draw']).scaling_1
scaling_2_money = trace_money_TRADE.posterior.mean(dim=['chain', 'draw']).scaling_2
scaling_3_money = trace_money_TRADE.posterior.mean(dim=['chain', 'draw']).scaling_3
scaling_4_money = trace_money_TRADE.posterior.mean(dim=['chain', 'draw']).scaling_4




probability_coin = []
probability_food = []
for i in range(len(data_ours)):
    # print(i)
    current_row = data_ours.iloc[i]
    B = current_row.B
    A = current_row.A
    DB = current_row.DB
    DA = current_row.DA
    condition = current_row.Condition
    if condition == 0:
        # current_k = k[int(current_row['Subject']) - 1]
        # current_s = s[int(current_row['Subject']) - 1]
        # V_A = A/(1+ pytensor.tensor.power(10, current_k) * pytensor.tensor.power(DA,current_s))
        # V_B = B /(1+ pytensor.tensor.power(10, current_k) * pytensor.tensor.power(DB,current_s))
        # all_data = V_B - V_A
        # error = 0.001
        # probability_food.append(error + (1.0 - 2.0 * error) * (1/(1+np.exp(-all_data))))
        #current_d_d_weight = d_d_weight_food[int(current_row['Subject']) - 1]
        #current_d_r_weight = d_r_weight_food[int(current_row['Subject']) - 1]
        #current_r_d_weight = r_d_weight_food[int(current_row['Subject']) - 1]
        #current_r_r_weight = r_r_weight_food[int(current_row['Subject']) - 1]
        current_scaling_1 = float(scaling_1_food[int(current_row['Subject']) - 1].data)
        current_scaling_2 = float(scaling_2_food[int(current_row['Subject']) - 1].data)
        current_scaling_3 = float(scaling_3_food[int(current_row['Subject']) - 1].data)
        current_scaling_4 = float(scaling_4_food[int(current_row['Subject']) - 1].data)

        a1 = cnv(B,current_scaling_2)
        a2 = cnv(A,current_scaling_2)
        a3 = cnv(DB,current_scaling_3)
        a4 = cnv(DA,current_scaling_3)

        all_data = current_scaling_4*((a1-a2)-current_scaling_1*(a3-a4))
        probability_food.append(Phi(all_data))
        #r_d = (B - A) * current_r_d_weight
        #r_r = ((B - A) / (((A + B) / 2))) * current_r_r_weight
        #d_d = (DB - DA) * current_d_d_weight
        #d_r = ((DB - DA) / (((DA + DB) / 2))) * current_d_r_weight
        #all_data = (r_d + r_r + d_d + d_r)  # scaling_weight_current *
        #error = 0.001
        #probability_food.append(error + (1.0 - 2.0 * error) * (1 / (1 + np.exp(-all_data))))
    else:
        current_scaling_1 = float(scaling_1_money[int(current_row['Subject']) - 1].data)
        current_scaling_2 = float(scaling_2_money[int(current_row['Subject']) - 1].data)
        current_scaling_3 = float(scaling_3_money[int(current_row['Subject']) - 1].data)
        current_scaling_4 = float(scaling_4_money[int(current_row['Subject']) - 1].data)

        a1 = cnv(B, current_scaling_2)
        a2 = cnv(A, current_scaling_2)
        a3 = cnv(DB, current_scaling_3)
        a4 = cnv(DA, current_scaling_3)

        all_data = current_scaling_4 * ((a1 - a2) - current_scaling_1 * (a3 - a4))
        probability_coin.append(Phi(all_data))
        #current_d_d_weight = d_d_weight[int(current_row['Subject']) - 1]
        #current_d_r_weight = d_r_weight[int(current_row['Subject']) - 1]
        #current_r_d_weight = r_d_weight[int(current_row['Subject']) - 1]
        #current_r_r_weight = r_r_weight[int(current_row['Subject']) - 1]
        #r_d = (B - A) * current_r_d_weight
        #r_r = ((B - A) / (((A + B) / 2))) * current_r_r_weight
        #d_d = (DB - DA) * current_d_d_weight
        #d_r = ((DB - DA) / (((DA + DB) / 2))) * current_d_r_weight
        #all_data = (r_d + r_r + d_d + d_r)  # scaling_weight_current *
        #error = 0.001
        #probability_coin.append(error + (1.0 - 2.0 * error) * (1/(1+np.exp(-all_data))))

data_ours_food = data_ours[data_ours['Condition'] == 0].copy(deep=True)
data_ours_coin = data_ours[data_ours['Condition'] == 1].copy(deep=True)
data_ours_food['probability'] = probability_food
data_ours_coin['probability'] = probability_coin

pl.hist(probability_food)
pl.hist2d(data_ours_food['R'],probability_food)
data_ours_food.to_csv('0_choices.csv')
data_ours_coin.to_csv('1_choices.csv')
subject= 22
fig, ax = plt.subplots(4,2)
ax = arviz.plot_trace(trace_food_TRADE,  axes=ax, compact=False, combined=True,var_names=['scaling_1','scaling_2','scaling_3','scaling_4'], coords={'Subject':[subject]}, figsize=(20,14), show=False)
fig.suptitle('Example trace: Food condition')
ax[0,0].set_title('scaling 1')
ax[0,1].set_title('Trace: scaling 1')
ax[1,0].set_title('scaling 2')
ax[1,1].set_title('Trace: scaling 2')
ax[2,0].set_title('scaling 3')
ax[2,1].set_title('Trace: scaling 3')
ax[3,0].set_title('scaling 4')
ax[3,1].set_title('Trace: scaling 4')
fig.tight_layout(pad=1)
arviz.plot_trace(trace_money_TRADE, compact=False, combined=True, var_names=['scaling_1','scaling_2','scaling_3','scaling_4'], coords={'Subject':[subject]})

subject= 22
fig, ax = plt.subplots(4,2)
ax = arviz.plot_trace(trace_money_TRADE,  axes=ax, compact=False, combined=True,var_names=['scaling_1','scaling_2','scaling_3','scaling_4'], coords={'Subject':[subject]}, figsize=(20,14), show=False)
fig.suptitle('Example trace: Money condition')
ax[0,0].set_title('scaling 1')
ax[0,1].set_title('Trace: scaling 1')
ax[1,0].set_title('scaling 2')
ax[1,1].set_title('Trace: scaling 2')
ax[2,0].set_title('scaling 3')
ax[2,1].set_title('Trace: scaling 3')
ax[3,0].set_title('scaling 4')
ax[3,1].set_title('Trace: scaling 4')
fig.tight_layout(pad=1)


stats.ttest_rel(scaling_1_money.data,scaling_1_food.data)  #modulates the effect of delay discrepancy
stats.ttest_rel(scaling_2_money.data,scaling_2_food.data) #scales  reward
stats.ttest_rel(scaling_3_money.data,scaling_3_food.data) #scales delay
stats.ttest_rel(scaling_4_money.data,scaling_4_food.data) #modulates the effect of reward discrepancy

import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
i = 1
name3 = str(i) + '_model_comparison.png'
model_comparison = pd.read_csv('1_model_comparison.csv')
model_comparison.index = model_comparison['Unnamed: 0']
fig, ax = plt.subplots(1,1)
arviz.plot_compare(model_comparison, show=False,  title=True, legend=True,  textsize=14,plot_kwargs={'color_ic':'Red','ls_min_ic':'--','fontsize':12}, ax=ax, plot_ic_diff=False) #
#ax.yaxis.label = []
ax.axes.set_yticklabels(ax.get_yticklabels(), horizontalalignment='right')
ax.axes.set_ylabel(None)
#ax.tick_params(axis='y', which='major', pad=200)
fig.set_figwidth(15)
plt.tight_layout()

