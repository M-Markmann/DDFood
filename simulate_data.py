import random
from Maximum_Likelihood.Discounting_Functions import *

def simulate_data():
    random.random()
    days = [2,14, 31, 91,182.5,365]


    model_food = two_parameter_softmax_prelec
    model_money = two_parameter_softmax_mazur



    reduction_list = [10,5,2.5,1,0]
    parameter_list = []
    conditions = {model_food:0,model_money:1}
    for i in range(1,1000):
        parameters_money = [(random.random() * 4), (random.random() * 10), (random.random() * 2)]  # k, beta, s
        parameters_food = [(random.random() * 4), (random.random() * 10), (random.random() * 2)]  # k, beta, s
        parameter_current = [parameters_food[0],parameters_food[1],parameters_food[2],parameters_money[0],parameters_money[1],parameters_money[2]]
        parameter_list.append(parameter_current)
        part_code = i
        choice_list = []
        day_list = []
        condition_list = []
        counter_list = []
        big_counter_list = []
        offer_list = []
        big_counter = 1
        for key, val in conditions.items():
            if val =='Food':
                parameters = parameters_food
            else:
                parameters = parameters_money
            for repetition_in_block in [1,2]:
                for day in days:
                    counter = 0
                    offer = 20
                    for i in range(0,5):
                        probability_delayed = key(0,offer,day,parameters[0],parameters[1],parameters[2])
                        if probability_delayed <=random.random():
                            choice=0#immediate
                        else: choice=1 #delayed
                        counter_list.append(counter)
                        offer_list.append(offer)
                        choice_list.append(choice)
                        day_list.append(day)
                        condition_list.append(val)
                        big_counter_list.append(big_counter)
                        if choice==1:
                            offer = offer + reduction_list[counter]
                        else:
                            offer = offer - reduction_list[counter]
                        counter+=1
                        big_counter+=1
        big_counter_list = np.array(big_counter_list)
        condition_list = np.array(condition_list)
        day_list = np.array(day_list)
        counter_list = np.array(counter_list)
        choice_list = np.array(choice_list)
        offer_list = np.array(offer_list)

        indifference_liste = np.vstack((big_counter_list, condition_list, day_list, counter_list, choice_list, offer_list))
        indifference_liste = np.transpose(indifference_liste)
        os.chdir('/Users/mariusmarkmann/Documents/Data/DD_Food/simulated_data')
        name = 'simulated_data_' + str(part_code)
        np.save(name, indifference_liste)

    parameter_list = np.array(parameter_list)
    os.chdir('/Users/mariusmarkmann/Documents/Data/DD_Food/')
    parameter_list =pd.DataFrame(parameter_list, columns=['k_food','beta_food','s_food','k_money','beta_money','s_money'])# k, beta, s
    parameter_list.to_csv("this_is_the_data.csv")


    print(big_counter_list)
    print(condition_list)
    print(day_list)
    print(counter_list)
    print(choice_list)
    print(offer_list)












