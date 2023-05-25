import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter


def statistics_and_plotting(all_data):
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
    ax.savefig('DAIC_FOOD.png')

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