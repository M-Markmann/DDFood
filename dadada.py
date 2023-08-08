import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
#winner Food
df = pd.read_csv('0hyperbol_scaling_delay.csv')
plt.hist(df['lower_bound_k'])
plt.show()

plt.hist(df['lower_bound_s'])
plt.show()
plt.hist(df['higher_bound_s'])
plt.show()

df_money = pd.read_csv('1discounting_trace_TRADE.csv')

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