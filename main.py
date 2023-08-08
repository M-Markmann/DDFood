#import seaborn
import os
#os.environ["QT_API"] = "PyQt6"
from Discounting_Functions import *

from create_behavioral_log import behavioral_analaysis, calculate_indifference_point_list
from log_likelihood_estimation import log_likelihood_single_parameter, log_likelihood_two_parameter, log_likelihood_itch
from statistics_and_plots import statistics_and_plotting_aic, calculate_indifference_list, statistis_and_plotting_r2
import os
import scipy

import pymc as pm

print(f"Running on PyMC v{pm.__version__}")

import seaborn as sns
#import matplotlib

def analyze_and_do_statistics():
    #define all variables necessary:
    method ='SLSQP' #SLSQ
    image_folder = 'C:/Users/mariu/Documents/Arbeit/Doktorarbeit/Abbildungen/'
    results_dir = os.path.dirname(image_folder)
    results_methods_dir = os.path.join(results_dir, method)
    if not os.path.exists(results_methods_dir):
        os.makedirs(results_methods_dir)
    timings = [2, 14, 31, 91, 182.5, 365]
    data_path = '/Users/mariusmarkmann/Documents/Data/DD_Food/BehavioralDATA/'
    log_files = os.listdir(data_path)
    maxiter = 10000000
    all_data = []
    all_data_r2 = []
    rsquared_parameter_estimates = []
    rsquared_data = []
    rsquared_data_unfiltered = []
    rsquared_parameters = []
    log_likelihood_food = []
    log_likelihood_money = []
    use_simulated_data = False
    counter = 0
    all_choices = np.zeros(shape=(0,7))
    for file in log_files:
    #for i in range(1, 1000):
            counter +=1
            file_name = data_path + file
            events_script, indifference_liste, food_chooser, entry_counter = behavioral_analaysis(file_name, file)
            if use_simulated_data:
                os.chdir('/Users/mariusmarkmann/Documents/Data/DD_Food/simulated_data')
                name = 'simulated_data_' + str(counter) + '.npy'
                indifference_liste = np.load(name)
            indifference_liste = np.array(indifference_liste)
            indifference_liste[np.where(indifference_liste[:,4]==0),4] = -1
            indifference_liste[np.where(indifference_liste[:, 4] == 1),4] = 0
            indifference_liste[np.where(indifference_liste[:, 4] == -1),4] = 1
            subject_id = np.full(shape=(indifference_liste.shape[0],1),fill_value=counter)
            fancy_all_including_list = np.concatenate((indifference_liste,subject_id),axis=1)
            all_choices = np.concatenate((all_choices,fancy_all_including_list),axis=0)
            #all_choices.append(fancy_all_including_list)
            #all_choices = np.array(all_choices)
            indifference_liste_with_all = indifference_liste
            Food = indifference_liste[indifference_liste[:, 1] < 0.5]
            Money = indifference_liste[indifference_liste[:, 1] > 0.5]
            list_of_indifference_points, did_change_happened = calculate_indifference_point_list(food_chooser, entry_counter,indifference_liste_with_all)
            list_of_indifference_points = np.array(list_of_indifference_points)
            list_of_indifference_points = list_of_indifference_points[list_of_indifference_points[:, 2].argsort()]
            Food_indifference_list = list_of_indifference_points[
               list_of_indifference_points[:, 1] < 0.5]  # Create List for all Food Indifference Points
            Money_indifference_list = list_of_indifference_points[
               list_of_indifference_points[:, 1] > 0.5]  # Create List for all Money Decisions

            initial_guess_one_r2 = (0.00001)  # k
            initial_guess_tw0_r2 = [0.0055, 0.403]
            bounds_one_r2 = (0.00001, 1)  # k
            bounds_two_r2 = ((0.00001, 0.01), (1, 3))  # k,s
            bounds_two = [(0.1, 10), (0.00001, 5)]  # beta, k
            bounds_itch = [(0.3,1),(0,1),(0,1),(0,1),(0,1)]
            initial_guess_two = np.array([2, 0.055])  # beta, k
            initial_guess_itch = np.array([0.8,0.8,0.8,0.8,0.8])
            bounds_three = [(0.1, 10), (0.00001, 5), (0.01,  3)]  # beta, k, s
            initial_guess_three = np.array([5, 3, 1])


            discounting_functions_softmax = { single_parameter_softmax_hyperbol:2,single_parameter_softmax_exponential:2,two_parameter_softmax_green: 3, two_parameter_softmax_mazur: 3, two_parameter_softmax_prelec: 3}
            discounting_functions_r = {hyberbolic_discounting:1, exponential_discounting: 1, green_meyerson_discounting: 2, mazur: 2,prelec: 2}
            conditions = [Food, Money]
            conditions_r = [Food_indifference_list, Money_indifference_list]
            list_of_parameters_softmax = []
            list_of_parameters_r = []
            for condition in conditions:
               #return_value=  scipy.optimize.minimize(log_likelihood_itch,x0=initial_guess_itch,bounds = bounds_itch,method='SLSQP',args=(condition),options={'maxiter': maxiter})
                for key, val in discounting_functions_softmax.items():
                    if val == 2:
                        Model_Fit = scipy.optimize.minimize(log_likelihood_single_parameter,
                                                                           x0=initial_guess_two,
                                                                           bounds=bounds_two,
                                                                           method=method,
                                                                           args=(condition,key),
                                                                           options={'maxiter': maxiter})
                        params = [Model_Fit.x[0], Model_Fit.x[1]]
                        log_likelihood = 1/log_likelihood_single_parameter(params,condition,key)
                        aic = -2*log_likelihood+2*2
                        #aic = 2*np.log(condition.__len__())-2*log_likelihood #eigentlich BIC
                        list_of_parameters_softmax.append(Model_Fit.x[0])
                        list_of_parameters_softmax.append(Model_Fit.x[1])
                        list_of_parameters_softmax.append(aic)
                    elif val == 3:
                        Model_Fit = scipy.optimize.minimize(log_likelihood_two_parameter,
                                                                           x0=initial_guess_three,
                                                                           bounds=bounds_three,
                                                                           method=method,
                                                                           args=(condition,key),
                                                                           options={'maxiter': maxiter})
                        params = [Model_Fit.x[0], Model_Fit.x[1], Model_Fit.x[2]]
                        log_likelihood = 1 / log_likelihood_two_parameter(params, condition, key)
                        aic = -2*log_likelihood+2*3
                        #aic = 3 * np.log(condition.__len__()) - 2 * log_likelihood  # eigentlich BIC
                        list_of_parameters_softmax.append(Model_Fit.x[0])
                        list_of_parameters_softmax.append(Model_Fit.x[1])
                        list_of_parameters_softmax.append(Model_Fit.x[2])
                        list_of_parameters_softmax.append(aic)
            all_data.append(list_of_parameters_softmax)
            #print(counter)
            for condition in conditions_r:
                for key, val in discounting_functions_r.items():
                    indifference_points = calculate_indifference_list(condition)
                    #list_of_parameters_r.append(counter)

                    if val == 1:
                        def discounting_for_curve_fitting(x, k):
                            SV = []
                            for i in x:
                                SV.append(key(i,k))
                            return SV

                        k_estimate, covar = curve_fit(discounting_for_curve_fitting, timings,
                                                               indifference_points,
                                                               p0=0.0055, maxfev=5000, bounds=bounds_one_r2)
                        r2 = sklearn.metrics.r2_score(indifference_points,
                                                                        discounting_for_curve_fitting(timings,k_estimate),
                                                                        sample_weight=None)
                        k_estimate = float(k_estimate)
                        list_of_parameters_r.append(k_estimate)
                        list_of_parameters_r.append(r2)
                    elif val == 2:
                        def discounting_for_curve_fitting(x, k, s):
                            SV = []
                            for i in x:
                                SV.append(key(i, k,s))
                            return SV

                        parameter_estimate, covar = curve_fit(discounting_for_curve_fitting, timings,
                                                      indifference_points,
                                                      maxfev=5000, bounds=bounds_two_r2, p0=initial_guess_tw0_r2)
                        k_estimate, s_estimate = parameter_estimate
                        r2 = sklearn.metrics.r2_score(indifference_points,
                                                      discounting_for_curve_fitting(timings, k_estimate, s_estimate),
                                                      sample_weight=None)
                        list_of_parameters_r.append(k_estimate)
                        list_of_parameters_r.append(s_estimate)
                        list_of_parameters_r.append(r2)
            all_data_r2.append(list_of_parameters_r)
            #print(counter)






    all_data = np.array(all_data)
    all_data_r2 = np.array(all_data_r2)
    all_data =pd.DataFrame(all_data, columns=['hyperbol_Food_beta', 'hyperbol_Food_k', 'aic_Food_hyperbol',
                                              'exponential_Food_beta', 'exponential_Food_k', 'aic_Food_exponential',
                                              'meyer_green_Food_beta', 'meyer_green_Food_k', 'meyer_Food_near_s','aic_Food_meyer_green',
                                              'mazur_Food_beta', 'mazur_Food_k', 'mazur_Food_s', 'aic_Food_mazur',
                                              'prelec_Food_beta', 'prelec_Food_k', 'prelec_Food_s', 'aic_Food_prelec',
                                              'hyperbol_Money_beta', 'hyperbol_Money_k', 'aic_Money_hyperbol',
                                              'exponential_Money_beta', 'exponential_Money_k', 'aic_Money_exponential',
                                              'meyer_Money_self_beta', 'meyer_Money_self_k', 'meyer_green_Money_s', 'aic_Money_meyer_green',
                                              'mazur_Money_beta', 'mazur_Money_k', 'mazur_Money_s', 'aic_Money_mazur',
                                              'prelec_Money_beta', 'prelec_Money_k', 'prelec_Money_s', 'aic_Money_prelec'
                                              ])

   scipy.stats.kendalltau(all_data['prelec_Food_s'],parameter_list['s_food'])
   matplotlib.pyplot.hist(parameter_list['s_food'])
    matplotlib.pyplot.hist(parameter_list['k_food'])
    matplotlib.pyplot.scatter(all_data['prelec_Food_s'],parameter_list['s_food'])
matplotlib.pyplot.scatter(all_data['prelec_Food_k'],parameter_list['k_food'])


    all_choices = pd.DataFrame(all_choices, columns=['Question','Condition','T2','Wayne','LaterOptionChosen','X1','Subject'])
    #all_choices['G'] = 40 - all_choices['X1']
    #all_choices['R'] =  40 / all_choices['X1']
    #all_choices['D'] = all_choices['T2']
    all_choices['T1'] = 0
    all_choices['X2'] = 40
    os.chdir('/Users/mariusmarkmann/Documents/Data/DD_Food/')
    all_choices.to_csv('all_choices.csv')


    scipy.stats.pearsonr(all_data['prelec_Food_k'],all_data['prelec_Money_k'])
    scipy.stats.pearsonr(all_data['mazur_Food_k'],all_data['mazur_Money_k'])
    scipy.stats.pearsonr(all_data['mazur_Food_s'],all_data['mazur_Money_s'])
    scipy.stats.pearsonr(all_data['prelec_Food_s'], all_data['prelec_Money_s'])
    matplotlib.pyplot.scatter(all_data['prelec_Food_s'], all_data['prelec_Money_s'])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_trisurf(all_data['prelec_Food_k'], parameter_list['k_food'], (all_data['prelec_Food_s']-parameter_list['s_food']), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_trisurf(all_data['mazur_Money_k'], parameter_list['k_money'],
                       (all_data['mazur_Money_s'] - parameter_list['s_money']), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    r2_data = pd.DataFrame(all_data_r2, columns=['k_food_hyperbol','r2_hyberbole_food','k_food_exponential','r2_exponential_food',
                                                 'k_food_gm','s_food_gm','r2_GM_food','k_food_mazur','s_food_mazur','r2_mazur_food','k_food_prelec','s_food_prelec','r2_prelec_food',
                                                       'k_money_hyperbol','r2_hyberbole_money','k_money_exponential','r2_exponential_money',
                                                     'k_money_gm','s_money_gm','r2_GM_money','k_money_mazur','s_money_mazur','r2_mazur_money','k_money_prelec','s_money_prelec','r2_prelec_money',
                                                      ])
        'r2_exponential_money', 'r2_hyberbole_money', 'r2_GM_money', 'r2_mazur_money', 'r2_prelec_money'
        statistics_and_plotting_aic(all_data)
        statistis_and_plotting_r2(r2_data)




            if __name__ == '__main__':
    analyze_and_do_statistics()


