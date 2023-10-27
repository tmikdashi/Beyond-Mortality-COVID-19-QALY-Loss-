from classes.model import AllStates
from data_preprocessing.support_functions import generate_county_data_csv
from definitions import ROOT_DIR

# Create the model and populate it with data from the provided csv files
generate_county_data_csv('cases')
all_states = AllStates(county_cases_csvfile=ROOT_DIR + '/csv_files/county_cases.csv')
all_states.populate('cases')


# TESTING:
# To get the overall QALY loss across states and time
all_states_overall_qaly_loss = all_states.get_overall_qaly_loss(case_weight=0.1)
print(f"Overall QALY Loss for all states: {all_states_overall_qaly_loss}")

# To get the weekly QALY loss across states:
all_states_weekly_qaly_loss = all_states.get_weekly_qaly_loss(case_weight=0.1)
print(f"Weekly QALY Loss for all states: {all_states_weekly_qaly_loss}")

# To get the total QALY loss for by state:
overall_qaly_loss_by_state = all_states.get_overall_qaly_loss_by_state(case_weight=0.1)
for state_name, weekly_qaly_loss in overall_qaly_loss_by_state.items():
    print(f"Overall QALY Loss for {state_name}: {weekly_qaly_loss}")

# To get the weeklyQALY loss for by state -- REMOVE # TO PRINT
weekly_qaly_loss_by_state = all_states.get_weekly_qaly_loss_by_state(case_weight=0.1)
#for state_name, weekly_qaly_loss in weekly_qaly_loss_by_state.items():
    #print(f"Weekly QALY Loss for {state_name}: {weekly_qaly_loss}")


# TESTING: Pulling Specific Data
# To get the overall QALY loss for one county
overall_qaly_loss_for_county=all_states.get_overall_qaly_loss_for_a_county(county_name="Autauga", state_name="AL",case_weight=0.1)
print(f"Overall QALY Loss for Autauga, AL : {overall_qaly_loss_for_county}")

# To get the weekly QALY loss for one county
weekly_qaly_loss_for_county = all_states.get_weekly_qaly_loss_for_a_county(county_name="Autauga", state_name="AL", case_weight=0.1)
print(f"Weekly QALY Loss for Autauga, AL: {weekly_qaly_loss_for_county}")

# To get the overall QALY loss for one state
overall_qaly_loss_for_state = all_states.get_overall_qaly_loss_for_a_state(state_name="AL", case_weight=0.1)
print(f"Overall QALY Loss for AL: {overall_qaly_loss_for_state}")

# To get the weekly QALY loss for one state
weekly_qaly_loss_for_state = all_states.get_weekly_qaly_loss_for_a_state(state_name="AL", case_weight=0.1)
print(f"Weekly QALY Loss for AL: {weekly_qaly_loss_for_state}")


# TESTING PLOTS
# To plot weeklyQALY Loss by state
all_states.plot_weekly_qaly_loss_by_state(case_weight=0.1)

# To plot national weeklyQALY Loss
all_states.plot_weekly_qaly_loss(case_weight=0.1)

# To map weeklyQALY Loss per county
all_states.plot_map_of_qaly_loss_by_county(case_weight=0.1)