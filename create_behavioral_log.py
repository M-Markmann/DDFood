def behavioral_analaysis(file_name,file):
        f = open(file_name, "r")
        participant_code = file[:7]
        # indifference_list = np.ndarray(shape=(2,2,6,5),dtype=int)
        indifference_liste = []
        Coin_Task = 0  # 0 = food
        Counter_Decisions = 0;
        Decision = 0  # later = 0 now = 1
        f1 = f.readlines()

        act_vol_time = 0
        pulse_counter = 0
        Offer_Food = []
        Offer_Food_now = []
        Offer_Food_later = []
        Offer_Coin = []
        Offer_Coin_now = []
        Offer_Coin_later = []
        indifference_liste_Coin_Now = []
        indifference_liste_Coin_Later = []
        indifference_liste_Food_Now = []
        indifference_liste_Food_Later = []
        error_trials_foos = []
        error_trials_coin = []
        Trial_duration_Food = []
        Trial_duration_Food_Now = []
        Trial_duration_Food_Later = []
        Trial_duration_Coin = []
        Trial_duration_Coin_now = []
        Trial_duration_Coin_Later = []
        end_offer_timing = 0
        feedback_trials_food = []
        feedback_trials_coin = []
        events_script = []
        start_point = 0
        startl_point = 0

        f1 = [x for x in f1 if not str('last trial') in x]
        f1 = [x for x in f1 if not str('start trial') in x]
        entry_counter = 0
        immediate_offer = 20
        immediate_offer_next = 20
        choice_counter = 1
        for x in f1:  # This analyses the logfile line by line, extracting all Decisions made,the Conditiona and the Time displayed
            split_string = x.split("\t");
            # print(split_string)
            if "Pulse" in split_string:
                act_vol_time = float(split_string[4]) / 10000 - startl_point
                pulse_counter += 1
                if pulse_counter == 1:
                    start_point = 0  # act_vol_time-2.5
                    startl_point = float(split_string[4]) / 10000
            elif "Response" in split_string and pulse_counter > 1:
                diftime_response = float(split_string[4]) / 10000 - act_vol_time
                ons_response = (pulse_counter * 2.5 + diftime_response - start_point)
                reaction_time = ons_response - end_offer_timing
            elif "coin.jpg" in str(split_string) and pulse_counter > 1:
                Coin_Task = 1
                diftime_offer = float(split_string[4]) / 10000 - act_vol_time
                ons_offer_coin = (pulse_counter * 2.5 + diftime_offer - start_point)
                end_offer_timing = ons_offer_coin + float(split_string[5]) / 10000

            elif "food" in str(split_string) and pulse_counter > 1 and not "foodchoice" in str(
                    split_string) and ".jpg" in str(split_string):
                Coin_Task = 0
                if "food1.jpg" in split_string:
                    food_chooser = 1
                elif "food2.jpg" in split_string:
                    food_chooser = 2
                elif "food3.jpg" in split_string:
                    food_chooser = 3
                elif "food4.jpg" in split_string:
                    food_chooser = 4
                diftime_offer = float(split_string[4]) / 10000 - act_vol_time
                ons_offer_food = (pulse_counter * 2.5 + diftime_offer - start_point)
                end_offer_timing = ons_offer_food + float(split_string[5]) / 10000
                # split_string = split_string[3].split("/t")
            elif "Picture" in split_string and ".jpg" not in str(split_string) and "foodchoice" not in str(
                    split_string) and pulse_counter > 1:
                # print(split_strin1g[3])
                # print(Coin_Task)
                # print(split_string)
                diftime_fdb = float(split_string[4]) / 10000 - act_vol_time
                relevant_data = split_string[3].split(",")[0]
                # print(split_string)
                ons_fdb = ((pulse_counter * 2.5) + diftime_fdb - start_point)

                split_string_timing = split_string[3].split(";")
                split_string_decision = split_string_timing[0].split(",")
                split_string_timing = split_string_timing[1]
                split_string_decision = split_string_decision[1]

                # print(split_string_timing[1])
                if "in 2 Tagen" in str(split_string_timing):
                    day = 2
                elif "in 2 Wochen" in str(split_string_timing):
                    day = 14
                elif "in 1 Monat" in str(split_string_timing):
                    day = 31
                elif "in 3 Monaten" in str(split_string_timing):
                    day = 91
                elif "in 6 Monaten" in str(split_string_timing):
                    day = 182.5
                elif "in 1 Jahr" in str(split_string_timing):
                    day = 365
                Counter_Decisions += 1

                if Counter_Decisions > 5:
                    Counter_Decisions = 1

                if split_string_decision == split_string_timing:
                    Decision = 0  # später
                else:
                    Decision = 1  # jetzt
                # print(Coin_Task,First_Round,Counter_Decisions,Decision)
                # print(Coin_Task)
                if choice_counter > 5:
                    choice_counter = 1

                if (choice_counter == 1 and Decision == 0):  # later
                    immediate_offer = 20
                    immediate_offer_next = immediate_offer + 10
                elif (choice_counter == 1 and Decision == 1):  # now
                    immediate_offer = 20
                    immediate_offer_next = immediate_offer - 10
                elif (choice_counter == 2 and Decision == 0):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer + 5
                elif (choice_counter == 2 and Decision == 1):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer - 5
                elif (choice_counter == 3 and Decision == 0):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer + 2.5
                elif (choice_counter == 3 and Decision == 1):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer - 2.5
                elif (choice_counter == 4 and Decision == 0):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer + 1
                elif (choice_counter == 4 and Decision == 1):
                    immediate_offer = immediate_offer_next
                    immediate_offer_next = immediate_offer - 1
                elif (choice_counter == 5):
                    immediate_offer = immediate_offer_next

                if Coin_Task == 0:
                    Offer_Food.append(ons_offer_food)  # was ons_offer_food
                    Trial_duration_Food.append(ons_response - ons_offer_food)
                    feedback_trials_food.append(ons_fdb)
                elif Coin_Task == 1:
                    Offer_Coin.append(ons_offer_coin)
                    Trial_duration_Coin.append(ons_response - ons_offer_coin)
                    feedback_trials_coin.append(ons_fdb)

                if Coin_Task == 0 and Decision == 0:
                    if reaction_time<8:
                        Trial_duration_Food_Later.append(ons_response - ons_offer_food)
                        Offer_Food_later.append(ons_response)
                    else:
                        error_trials_foos.append(ons_response)
                    indifference_liste_Food_Later.append(
                        (entry_counter, Coin_Task, day, Counter_Decisions, Decision, immediate_offer))
                elif Coin_Task == 0 and Decision == 1:
                    if reaction_time < 8:
                        Trial_duration_Food_Now.append(ons_response - ons_offer_food)
                        Offer_Food_now.append(ons_response)
                    else:
                        error_trials_foos.append(ons_response)
                    indifference_liste_Food_Now.append(
                        (entry_counter, Coin_Task, day, Counter_Decisions, Decision, immediate_offer))
                elif Coin_Task == 1 and Decision == 0:
                    if reaction_time < 8:
                        Trial_duration_Coin_Later.append(ons_response - ons_offer_coin)
                        Offer_Coin_later.append(ons_response)
                    else:
                        error_trials_coin.append(ons_response)
                    indifference_liste_Coin_Later.append(
                        (entry_counter, Coin_Task, day, Counter_Decisions, Decision, immediate_offer))
                elif Coin_Task == 1 and Decision == 1:
                    if reaction_time < 8:
                        Trial_duration_Coin_now.append(ons_response - ons_offer_coin)
                        Offer_Coin_now.append(ons_response)
                    else:
                        error_trials_coin.append(ons_response)
                    indifference_liste_Coin_Now.append(
                        (entry_counter, Coin_Task, day, Counter_Decisions, Decision, immediate_offer))

                choice_counter = choice_counter + 1
                entry_counter = entry_counter + 1
                if Coin_Task == 0:
                    duration = ons_response - ons_offer_food
                    trial_start = ons_offer_food
                    condition = "Food"

                else:
                    duration = ons_response -ons_offer_coin
                    trial_start = ons_offer_coin
                    condition = "Coin"

                if Decision== 0:
                    choice = "Later"
                else:
                    choice = "Now"
                events_script.append((trial_start,duration,day,condition,immediate_offer,choice))
                indifference_liste.append((entry_counter, Coin_Task, day, Counter_Decisions, Decision, immediate_offer))

        return events_script, indifference_liste, food_chooser, entry_counter

def calculate_indifference_point_list(food_chooser,entry_counter, indifference_liste_with_all):
    list_of_indifference_points = []
    did_change_happened = []
    food_chooser = [food_chooser]
    for i in range(int(entry_counter / 5)):
         # print((i+1)*5)
        start = ((i + 1) * 5) - 5
        end = ((i + 1) * 5)
            # print(start,end)
        indifference_point, change_happened = indifference_point_calculator(indifference_liste_with_all[start:end][:][:][:][:], start, end)
        list_of_indifference_points.append(
             (start, indifference_liste_with_all[start, 1], indifference_liste_with_all[start, 2], indifference_point))
        did_change_happened.append(change_happened)

    return list_of_indifference_points, did_change_happened




def indifference_point_calculator(numbers, start, end):
    indif = int(20)
    if numbers[0, 4] == 1:
        indif += -10
    elif numbers[0, 4] == 0:
        indif += 10
    if numbers[1, 4] == 1:
        indif += -5
    elif numbers[1, 4] == 0:
        indif += 5
    if numbers[2, 4] == 1:
        indif += -2.5
    elif numbers[2, 4] == 0:
        indif += 2.5
    if numbers[3, 4] == 1:
        indif += -1
    elif numbers[3, 4] == 0:
        indif += 1

    change_happened = any(numbers[:,4])
    return indif, change_happened
