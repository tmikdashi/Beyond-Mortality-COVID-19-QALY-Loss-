from classes.model import AllStates
from data_preprocessing.support_functions import generate_county_data_csv
from definitions import ROOT_DIR

##################
# HOSPITALIZATIONS
##################
# Create the model and populate it with data from the provided csv files
generate_county_data_csv('hospitalizations')

#all_states = AllStates(county_data_csvfile=ROOT_DIR + '/csv_files/county_hospitalizations.csv')
all_states = AllStates(county_data_csvfile=ROOT_DIR + f'/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_hospitalizations.csv')
all_states.populate('hospitalizations')


# TESTING:
# To get the overall QALY loss across states and time
all_states_overall_qaly_loss_hosp = all_states.get_overall_qaly_loss(case_weight=0.15)
print(f"Overall QALY Loss for all states from hospitalizations: {all_states_overall_qaly_loss_hosp}")

# To get the weekly QALY loss across states:
all_states_weekly_qaly_loss_hosp = all_states.get_weekly_qaly_loss(case_weight=0.15)
print(f"Weekly QALY Loss for all states: {all_states_weekly_qaly_loss_hosp}")

# To get the total QALY loss for by state:
overall_qaly_loss_by_state_hosp = all_states.get_overall_qaly_loss_by_state(case_weight=0.15)
for state_name, weekly_qaly_loss_hosp in overall_qaly_loss_by_state_hosp.items():
    print(f"Overall QALY Loss for {state_name}: {weekly_qaly_loss_hosp}")

# To get the weeklyQALY loss for by state -- REMOVE # TO PRINT
weekly_qaly_loss_by_state_hosp = all_states.get_weekly_qaly_loss_by_state(case_weight=0.15)
#for state_name, weekly_qaly_loss in weekly_qaly_loss_by_state.items():
    #print(f"Weekly QALY Loss for {state_name}: {weekly_qaly_loss}")


# TESTING: Pulling Specific Data
# To get the overall QALY loss for one county
overall_qaly_loss_for_county_hosp=all_states.get_overall_qaly_loss_for_a_county(county_name="Autauga", state_name="AL",case_weight=0.15)
print(f"Overall QALY Loss for Autauga, AL : {overall_qaly_loss_for_county_hosp}")

# To get the weekly QALY loss for one county
weekly_qaly_loss_for_county_hosp = all_states.get_weekly_qaly_loss_for_a_county(county_name="Autauga", state_name="AL", case_weight=0.15)
print(f"Weekly QALY Loss for Autauga, AL: {weekly_qaly_loss_for_county_hosp}")

# To get the overall QALY loss for one state
overall_qaly_loss_for_state_hosp = all_states.get_overall_qaly_loss_for_a_state(state_name="AL", case_weight=0.15)
print(f"Overall QALY Loss for AL: {overall_qaly_loss_for_state_hosp}")

# To get the weekly QALY loss for one state
weekly_qaly_loss_for_state_hosp = all_states.get_weekly_qaly_loss_for_a_state(state_name="AL", case_weight=0.15)
print(f"Weekly QALY Loss for AL: {weekly_qaly_loss_for_state_hosp}")


# TESTING PLOTS
# To plot weeklyQALY Loss by state
all_states.plot_weekly_qaly_loss_by_state(data_type ='hospitalizations',case_weight=0.15)

# To plot national weeklyQALY Loss
all_states.plot_weekly_qaly_loss(data_type ='hospitalizations',case_weight=0.15)

# To map weeklyQALY Loss per county
all_states.plot_map_of_qaly_loss_by_county(case_weight=0.15)



##################
# DEATHS
##################

# Create the model and populate it with data from the provided csv files
generate_county_data_csv('deaths')

#all_states = AllStates(county_data_csvfile=ROOT_DIR + '/csv_files/county_deaths.csv')
all_states = AllStates(county_data_csvfile=ROOT_DIR + f'/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_deaths.csv')

all_states.populate('deaths')


# TESTING:
# To get the overall QALY loss across states and time
all_states_overall_qaly_loss_deaths = all_states.get_overall_qaly_loss(data_type='deaths', case_weight=0.15)
print(f"Overall QALY Loss for all states from deaths: {all_states_overall_qaly_loss_deaths}")

# To get the weekly QALY loss across states:
all_states_weekly_qaly_loss_deaths = all_states.get_weekly_qaly_loss(data_type='deaths',case_weight=0.15)
print(f"Weekly QALY Loss for all states: {all_states_weekly_qaly_loss_deaths}")

# To get the total QALY loss for by state:
overall_qaly_loss_by_state_deaths = all_states.get_overall_qaly_loss_by_state(data_type='deaths',case_weight=0.15)
for state_name, weekly_qaly_loss_deaths in overall_qaly_loss_by_state_deaths.items():
    print(f"Overall QALY Loss for {state_name}: {weekly_qaly_loss_deaths}")

# To get the weeklyQALY loss for by state -- REMOVE # TO PRINT
weekly_qaly_loss_by_state_deaths = all_states.get_weekly_qaly_loss_by_state(data_type='deaths',case_weight=0.15)
#for state_name, weekly_qaly_loss in weekly_qaly_loss_by_state.items():
    #print(f"Weekly QALY Loss for {state_name}: {weekly_qaly_loss}")


# TESTING: Pulling Specific Data
# To get the overall QALY loss for one county
overall_qaly_loss_for_county_deaths=all_states.get_overall_qaly_loss_for_a_county(county_name="Autauga", state_name="AL",case_weight=0.15)
print(f"Overall QALY Loss for Autauga, AL : {overall_qaly_loss_for_county_deaths}")

# To get the weekly QALY loss for one county
weekly_qaly_loss_for_county_deaths = all_states.get_weekly_qaly_loss_for_a_county(county_name="Autauga", state_name="AL", case_weight=0.15)
print(f"Weekly QALY Loss for Autauga, AL: {weekly_qaly_loss_for_county_deaths}")

# To get the overall QALY loss for one state
overall_qaly_loss_for_state_deaths = all_states.get_overall_qaly_loss_for_a_state(state_name="AL", case_weight=0.15)
print(f"Overall QALY Loss for AL: {overall_qaly_loss_for_state_deaths}")

# To get the weekly QALY loss for one state
weekly_qaly_loss_for_state_deaths = all_states.get_weekly_qaly_loss_for_a_state(state_name="AL", case_weight=0.15)
print(f"Weekly QALY Loss for AL: {weekly_qaly_loss_for_state_deaths}")


# TESTING PLOTS
# To plot weeklyQALY Loss by state
all_states.plot_weekly_qaly_loss_by_state(data_type ='deaths',case_weight=0.15)

# To plot national weeklyQALY Loss
all_states.plot_weekly_qaly_loss(data_type ='deaths',case_weight=0.15)

# To map weeklyQALY Loss per county
all_states.plot_map_of_qaly_loss_by_county(case_weight=0.15)


