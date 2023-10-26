from classes.model import AllDataTypes
from data_preprocessing.support_functions import generate_county_data_csv, get_dict_of_county_data_by_type
from definitions import ROOT_DIR


generate_county_data_csv('cases')
generate_county_data_csv('deaths')
generate_county_data_csv('hospitalizations')


data_types = AllDataTypes(
    cases_csvfile=ROOT_DIR + '/csv_files/county_cases.csv',
    hospitalizations_csvfile=ROOT_DIR + '/csv_files/county_hospitalizations.csv',
    deaths_csvfile=ROOT_DIR + '/csv_files/county_deaths.csv')

# Setting the weights
case_weight = 0.1
hospitalizations_weight = 0.15
deaths_weight = 0.2

# Calculate the total QALY loss for all states
total_qaly_loss = data_types.get_total_qaly_loss(case_weight, hospitalizations_weight, deaths_weight)
print(f"Total QALY Loss for all states: {total_qaly_loss}")

# Calculate the weekly QALY loss
total_weekly_qaly_loss = data_types.get_total_weekly_qaly_loss(case_weight, hospitalizations_weight, deaths_weight)
print(f"Total Weekly QALY Loss for all states: {total_weekly_qaly_loss}")

# Calculate the weekly QALY loss by state
total_overall_qaly_loss_by_state = data_types.get_total_overall_qaly_loss_by_state(case_weight, hospitalizations_weight, deaths_weight)
for state_name, total_qaly_loss in total_overall_qaly_loss_by_state.items():
    print(f"Total QALY Loss for {state_name}: {total_qaly_loss}")



# PLOTTING
data_types.plot_weekly_qaly_loss_by_data_type(case_weight, hospitalizations_weight, deaths_weight)

data_types.plot_total_weekly_qaly_loss(case_weight, hospitalizations_weight, deaths_weight)

data_types.plot_qaly_loss_by_state_and_type(case_weight, hospitalizations_weight, deaths_weight)

