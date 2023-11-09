from classes.model_v2 import AllStates
from data_preprocessing.support_functions import generate_county_data_csv
from definitions import ROOT_DIR

# Create the model and populate it with data from the provided csv files
generate_county_data_csv('cases')
generate_county_data_csv('deaths')
generate_county_data_csv('hospitalizations')

all_states = AllStates(
    county_case_csvfile=ROOT_DIR + '/csv_files/county_cases.csv',
    county_death_csvfile=ROOT_DIR + '/csv_files/county_deaths.csv',
    county_hosp_csvfile=ROOT_DIR + '/csv_files/county_hospitalizations.csv')


all_states.populate(case_weight=0.1, death_weight=0.3, hosp_weight=0.2)

#ALL STATES AND COUNTIES
all_states.get_overall_qaly_loss()
all_states.get_weekly_qaly_loss()
all_states.get_weekly_qaly_loss_by_outcome()

#TODO: many of these are just to demonstrate that the functions work and therefore can be deleted later

# STATE
all_states.get_overall_qaly_loss_by_state()
all_states.get_weekly_qaly_loss_by_state()

# COUNTY
all_states.get_overall_qaly_loss_by_county()
#all_states.get_weekly_qaly_loss_by_county()


# STATE-SPECIFIC DATA
all_states.get_overall_qaly_loss_for_a_state("AL")
all_states.get_weekly_qaly_loss_for_a_state("AL")


# COUNTY-SPECIFIC DATA
all_states.get_overall_qaly_loss_for_a_county("Autauga", "AL")
all_states.get_weekly_qaly_loss_for_a_county("Autauga", "AL")

# PLOTTING
all_states.plot_map_of_qaly_loss_by_county()
all_states.plot_weekly_qaly_loss_by_state()
all_states.plot_weekly_qaly_loss()
all_states.plot_weekly_qaly_loss_by_outcome()
all_states.plot_qaly_loss_by_state_and_by_outcome()


