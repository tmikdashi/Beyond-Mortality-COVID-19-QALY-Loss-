from classes.model import AllStates
from data_preprocessing.support_functions import generate_county_data_csv, get_dict_of_county_data_by_type
from definitions import ROOT_DIR


# Create the model and populate it with data from the provided csv files
generate_county_data_csv('cases')
generate_county_data_csv('deaths')
generate_county_data_csv('hospitalizations')

#all_states = AllStates(county_cases_csvfile=ROOT_DIR + '/csv_files/county_cases.csv')
#all_states = AllStates(county_data_csvfile=ROOT_DIR + '/csv_files/county_deaths.csv')
all_states_cases = AllStates(county_data_csvfile=ROOT_DIR + f'/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_cases.csv')
all_states_deaths = AllStates(county_data_csvfile=ROOT_DIR + f'/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_deaths.csv')
all_states_hospitalizations = AllStates(county_data_csvfile=ROOT_DIR + f'/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_hospitalizations.csv')

all_states_cases.populate('cases')
all_states_deaths.populate('deaths')
all_states_hospitalizations.populate('hospitalizations')



total_overall_qaly_loss = all_states_cases.get_overall_qaly_loss(case_weight=0.1) + all_states_hospitalizations.get_overall_qaly_loss(case_weight=0.15) + all_states_deaths.get_overall_qaly_loss(case_weight=0.2)
print(f"Total QALY Loss for all states from cases, hospitalizations and deaths: {total_overall_qaly_loss}")

total_weekly_qaly_loss = all_states_cases.get_weekly_qaly_loss(case_weight=0.1) + all_states_hospitalizations.get_weekly_qaly_loss(case_weight=0.15) + all_states_deaths.get_weekly_qaly_loss(case_weight=0.2)
print(f"Total weekly QALY Loss for all states from cases, hospitalizations and deaths: {total_weekly_qaly_loss}")


total_overall_qaly_loss_by_state_cases = all_states_cases.get_overall_qaly_loss_by_state(case_weight=0.1)
total_overall_qaly_loss_by_state_hospitalizations = all_states_hospitalizations.get_overall_qaly_loss_by_state(case_weight=0.15)
total_overall_qaly_loss_by_state_deaths = all_states_deaths.get_overall_qaly_loss_by_state(case_weight=0.2)
total_overall_qaly_loss_by_state = {}
for state_name in total_overall_qaly_loss_by_state_cases:
    total_overall_qaly_loss_by_state[state_name] = (
        total_overall_qaly_loss_by_state_cases[state_name] +
        total_overall_qaly_loss_by_state_hospitalizations[state_name] +
        total_overall_qaly_loss_by_state_deaths[state_name]
    )

for state_name, weekly_qaly_loss in total_overall_qaly_loss_by_state.items():
    print(f"Total QALY Loss for {state_name}: {weekly_qaly_loss}")



## PLOTTING WEEKLY QALY LOSS BY TYPE
import matplotlib.pyplot as plt
import numpy as np

# Define the case weight, hospitalizations weight, and deaths weight
case_weight = 0.1
hospitalizations_weight = 0.15
deaths_weight = 0.2

# Calculate the weekly total QALY loss for each data type
total_weekly_cases = all_states_cases.get_weekly_qaly_loss(case_weight)
total_weekly_hospitalizations = all_states_hospitalizations.get_weekly_qaly_loss(hospitalizations_weight)
total_weekly_deaths = all_states_deaths.get_weekly_qaly_loss(deaths_weight)

# Get the dates from your data
county_data, dates = get_dict_of_county_data_by_type('cases')

# Determine the maximum length among the data
max_length = max(len(total_weekly_cases), len(total_weekly_hospitalizations), len(total_weekly_deaths))

# Pad the arrays with zeros to match the maximum length
total_weekly_cases = np.pad(total_weekly_cases, (0, max_length - len(total_weekly_cases)))
total_weekly_hospitalizations = np.pad(total_weekly_hospitalizations, (0, max_length - len(total_weekly_hospitalizations)))
total_weekly_deaths = np.pad(total_weekly_deaths, (0, max_length - len(total_weekly_deaths)))

# Create a line plot with dates on the x-axis
plt.figure(figsize=(12, 6))
x = dates[:max_length]  # Take the first 'max_length' dates
plt.plot(x, total_weekly_cases, label="Cases")
plt.plot(x, total_weekly_hospitalizations, label="Hospitalizations")
plt.plot(x, total_weekly_deaths, label="Deaths")

plt.xlabel("Date")
plt.ylabel("QALY Loss")
plt.title("Weekly QALY Loss by Data Type")
plt.legend()
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()
