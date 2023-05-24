from Discounting_Functions import *
from  plotting_results import *
from create_behavioral_log import behavioral_analaysis, calculate_indifference_point_list
from log_likelihood_estimation import log_likelihood_single_parameter, log_likelihood_two_parameter
import scipy

def print_hi(name):
    #define all variables necessary:
    method ='SLSQP'#
    image_folder = 'C:/Users/mariu/Documents/Arbeit/Doktorarbeit/Abbildungen/'
    results_dir = os.path.dirname(image_folder)
    results_methods_dir = os.path.join(results_dir, method)
    if not os.path.exists(results_methods_dir):
        os.makedirs(results_methods_dir)
    data_path = 'C:/Users/mariu/Documents/Arbeit/DelayDiscountingFood/BehavioralDATA/'
    log_files = os.listdir(data_path)
    maxiter = 10000000
    all_data = []
    rsquared_parameter_estimates = []
    rsquared_data = []
    rsquared_data_unfiltered = []
    rsquared_parameters = []
    log_likelihood_food = []
    log_likelihood_money = []
    counter = 0
    for file in log_files:
        counter +=1
        file_name = data_path + file
        events_script, indifference_liste, food_chooser, entry_counter = behavioral_analaysis(file_name, file)
        indifference_liste = np.array(indifference_liste)
        indifference_liste_with_all = indifference_liste
        # indifference_liste = indifference_liste[indifference_liste[:,2]< 180]
        #indifference_liste_Coin_Later = np.array(indifference_liste_Coin_Later)
        #indifference_liste_Coin_Now = np.array(indifference_liste_Coin_Now)
        #indifference_liste_Food_Later = np.array(indifference_liste_Food_Later)
        #indifference_liste_Food_Now = np.array(indifference_liste_Food_Now)
        Food = indifference_liste[indifference_liste[:, 1] < 0.5]
        Money = indifference_liste[indifference_liste[:, 1] > 0.5]
        list_of_indifference_points, did_change_happened = calculate_indifference_point_list(food_chooser, entry_counter,indifference_liste_with_all)
        list_of_indifference_points = np.array(list_of_indifference_points)
        list_of_indifference_points = list_of_indifference_points[list_of_indifference_points[:, 2].argsort()]
        Food_indifference_list = list_of_indifference_points[
           list_of_indifference_points[:, 1] < 0.5]  # Create List for all Food Indifference Points
        Money_indifference_list = list_of_indifference_points[
           list_of_indifference_points[:, 1] > 0.5]  # Create List for all Money Decisions

        bounds_two = [(0.1, 10), (0.000001, 1)]  # beta, k
        initial_guess_two = np.array([2, 0.055])  # beta, k
        bounds_three = [(0.1, 10), (0.000001, 1), (0.01,  1.5)]  # beta, k, s
        initial_guess_three = np.array([2, 0.055, 0.42])

        discounting_functions = { single_parameter_softmax_hyperbol:2,single_parameter_softmax_exponential:2,two_parameter_softmax_green: 3, two_parameter_softmax_mazur: 3, two_parameter_softmax_prelec: 3}
        conditions = [Food, Money]
        list_of_parameters = []
        for condition in conditions:
            for key, val in discounting_functions.items():
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
                    list_of_parameters.append(Model_Fit.x[0])
                    list_of_parameters.append(Model_Fit.x[1])
                    list_of_parameters.append(aic)
                if val == 3:
                    Model_Fit = scipy.optimize.minimize(log_likelihood_two_parameter,
                                                                       x0=initial_guess_three,
                                                                       bounds=bounds_three,
                                                                       method=method,
                                                                       args=(condition,key),
                                                                       options={'maxiter': maxiter})
                    params = [Model_Fit.x[0], Model_Fit.x[1], Model_Fit.x[2]]
                    log_likelihood = 1 / log_likelihood_two_parameter(params, condition, key)
                    aic = -2*log_likelihood+2*3
                    list_of_parameters.append(Model_Fit.x[0])
                    list_of_parameters.append(Model_Fit.x[1])
                    list_of_parameters.append(Model_Fit.x[2])
                    list_of_parameters.append(aic)

        all_data.append(list_of_parameters)
    all_data = np.array(all_data)
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





        if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
