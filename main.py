from Discounting_Functions import *
from import plotting_results import *
from create_behavioral_log import behavioral_analaysis

def print_hi(name):
    #define all variables necessary:
    method = 'SLSQP'
    image_folder = 'C:/Users/mariu/Documents/Arbeit/Doktorarbeit/Abbildungen/'
    results_dir = os.path.dirname(image_folder)
    results_methods_dir = os.path.join(results_dir, method)
    if not os.path.exists(results_methods_dir):
        os.makedirs(results_methods_dir)
    data_path = 'C:/Users/mariu/Documents/Arbeit/DelayDiscountingFood/BehavioralDATA/'
    log_files = os.listdir(data_path)
    maxiter = 100000
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
        events_script, indifference_liste = behavioral_analaysis(file_name, file)
        indifference_liste = np.array(indifference_liste)
        indifference_liste_with_all = indifference_liste
        # indifference_liste = indifference_liste[indifference_liste[:,2]< 180]
        indifference_liste_Coin_Later = np.array(indifference_liste_Coin_Later)
        indifference_liste_Coin_Now = np.array(indifference_liste_Coin_Now)
        indifference_liste_Food_Later = np.array(indifference_liste_Food_Later)
        indifference_liste_Food_Now = np.array(indifference_liste_Food_Now)
        Food = indifference_liste[indifference_liste[:, 1] < 0.5]
        Money = indifference_liste[indifference_liste[:, 1] > 0.5]


if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
