import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import scipy
from matplotlib.ticker import StrMethodFormatter

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



def statistics_and_plotting_aic(all_data):
    min_food = all_data[
        ['aic_Food_hyperbol', 'aic_Food_exponential', 'aic_Food_meyer_green', 'aic_Food_mazur', 'aic_Food_prelec']].min(
        axis=1)
    all_data['delta_aic_food_hyperbol'] = all_data['aic_Food_hyperbol'] - min_food
    all_data['delta_aic_food_exponential'] = all_data['aic_Food_exponential'] - min_food
    all_data['delta_aic_food_meyer_green'] = all_data['aic_Food_meyer_green'] - min_food
    all_data['delta_aic_food_mazur'] = all_data['aic_Food_mazur'] - min_food
    all_data['delta_aic_food_prelec'] = all_data['aic_Food_prelec'] - min_food

    min_money = all_data[['aic_Money_hyperbol','aic_Money_exponential','aic_Money_meyer_green','aic_Money_mazur','aic_Money_prelec']].min(axis=1)
    all_data['delta_aic_Money_hyperbol'] =  all_data['aic_Money_hyperbol'] - min_money
    all_data['delta_aic_Money_exponential'] = all_data['aic_Money_exponential'] - min_money
    all_data['delta_aic_Money_meyer_green'] = all_data['aic_Money_meyer_green'] -min_money
    all_data['delta_aic_Money_mazur'] = all_data['aic_Money_mazur'] - min_money
    all_data['delta_aic_Money_prelec'] = all_data['aic_Money_prelec'] -min_money

    sns.set(rc={'figure.figsize': (22, 11.5)})
    sns.set(font_scale=3)
    my_pal_food = {"delta_aic_food_exponential": "darkmagenta", "delta_aic_food_hyperbol": "limegreen",
                   "delta_aic_food_mazur": "fuchsia", "delta_aic_food_meyer_green": "darkgoldenrod",
                   'delta_aic_food_prelec': 'royalblue'}

    ax = sns.barplot(data=all_data[
        ['delta_aic_food_exponential', 'delta_aic_food_hyperbol', 'delta_aic_food_mazur', 'delta_aic_food_meyer_green',
         'delta_aic_food_prelec']], palette=my_pal_food) #, errorbar='se')  # , showfliers=False
    ax.set_xticklabels(['Exponential', 'Hyperbole', 'Hyperbole \n with scaling \n Delay',
                        'Hyperbole \n with scaling \n Delay & Discounting',
                        'Exponential \n with time scaling'])
    # ax.set_xlabel("Model Fit for the Food condition")
    # ax.set_ylabel(u'ΔAIC')
    ax.patches[4].set_edgecolor('red')
    ax.patches[4].set_linewidth(5)
    ax.axhline(y=2, color='black', linestyle='--', linewidth=3)
    ax.set(ylim=(0, 11))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


    sns.set(rc={'figure.figsize': (22, 8.27)})
    sns.set(font_scale=3)
    my_pal_food = {"delta_aic_Money_exponential": "darkmagenta", "delta_aic_Money_hyperbol": "limegreen",
                   "delta_aic_Money_mazur": "darkgoldenrod", "delta_aic_Money_meyer_green": "fuchsia",
                   'delta_aic_Money_prelec': 'royalblue'}
    ax = sns.barplot(data=all_data[
        ['delta_aic_Money_exponential', 'delta_aic_Money_hyperbol', 'delta_aic_Money_meyer_green',
         'delta_aic_Money_mazur', 'delta_aic_Money_prelec']], palette=my_pal_food) #, errorbar='se')
    ax.set_xticklabels(['Exponential', 'Hyperbole', 'Hyperbole \n with scaling \n Delay',
                        'Hyperbole \n with scaling \n Delay & Discounting',
                        'Exponential \n with time scaling'])
    # ['Exponential', 'Hyperbole', 'Hyperbole \n with Time scaling', 'Hyperbole \n with Time scaling (Mazur)',
    # 'Exponential \n with time scaling'])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.patches[3].set_edgecolor('red')
    ax.patches[3].set_linewidth(5)
    ax.axhline(y=2, color='black', linestyle='--', linewidth=3)
    ax.set(ylim=(0, 11))

    return True

def statistis_and_plotting_r2(r2_data):
    my_pal_money = {"r2_exponential_money": "darkmagenta", "r2_hyberbole_money": "limegreen", "r2_mazur_money": "fuchsia" ,"r2_GM_money":"darkgoldenrod",'r2_prelec_money':'royalblue'}
    sns.set(font_scale=3)
    ax = sns.barplot(data=r2_data[
        ['r2_exponential_money','r2_hyberbole_money', 'r2_GM_money', 'r2_mazur_money','r2_prelec_money']],  palette=my_pal_money)
    ax.set_xticklabels(
        ['Exponential', 'Hyperbole', 'Hyperbole \n with time scaling \n Delay',
         'Hyperbole \n with scaling \n Delay & Discounting',
         'Exponential \n with time scaling'])
    ax.set(ylim=(0, 1))
    #'r2_exponential_money', 'r2_hyberbole_money',
    ax.set_xlabel("Model Fit for the Money condition")
    ax.set_ylabel('R\u00b2')
    ax.patches[2].set_edgecolor('red')
    ax.patches[2].set_linewidth(5)
    ax.set(ylim=(0, 1))

    sns.set(rc={'figure.figsize': (22, 11)})
    sns.set(font_scale=2.2)
    my_pal_food = {"r2_exponential_food": "firebrick", "r2_hyberbole_food": "darkgoldenrod",
                   "r2_mazur_food": "darkcyan", "r2_GM_food": "limegreen",
                   'r2_prelec_food': 'darkmagenta'}
    ax = sns.barplot(data=r2_data[
        ['r2_exponential_food', 'r2_hyberbole_food', 'r2_GM_food',
         'r2_mazur_food', 'r2_prelec_food']], palette=my_pal_food)
    ax.set_xticklabels(
        ['Exponential', 'Hyperbole', 'Hyperbole \n with time scaling \n Delay',
         'Hyperbole \n with scaling \n Delay & Discounting',
         'Exponential \n with time scaling'])
    ax.set_xlabel("Model Fit for the Money condition")
    ax.set_ylabel('R\u00b2')
    ax.patches[3].set_edgecolor('red')
    ax.patches[3].set_linewidth(5)
    ax.patches[4].set_edgecolor('red')
    ax.patches[4].set_linewidth(5)
    # ax.axhline(y=2, color='black', linestyle='--')
    ax.set(ylim=(0, 1))

    get_wilcoxon_rank_and_make_fancy_graphics(r2_data[['r2_exponential_food', 'r2_hyberbole_food', 'r2_GM_food',
         'r2_mazur_food', 'r2_prelec_food']],0.05)
    get_wilcoxon_rank_and_make_fancy_graphics(r2_data[['r2_exponential_money','r2_hyberbole_money', 'r2_GM_money', 'r2_mazur_money','r2_prelec_money']], 0.05)

    return True



def calculate_indifference_list(List_of_decisions):
    decisions_run_1 = []
    decisions_run2 = []
    for i in range(int(len(List_of_decisions))):
        if i % 2 == 0:
            if List_of_decisions[i, 0] > List_of_decisions[i + 1, 0]:
                decisions_run_1.append(List_of_decisions[i + 1, 3])
                decisions_run2.append(List_of_decisions[i, 3])
            else:
                decisions_run_1.append(List_of_decisions[i, 3])
                decisions_run2.append(List_of_decisions[i + 1, 3])
    indifference_points_list = (np.array(decisions_run_1) + np.array(decisions_run2))/2

    return indifference_points_list