import csv
import numpy as np
from collections import defaultdict
from deampy.in_out_functions import write_csv, read_csv_rows
from definitions import ROOT_DIR


# PART 1: Calculating County-level QALY Loss (no csv file generated, but printed results)

# Loading the data
# When I use read_csv_rows, it appears to skip through the first 1715 counties. Any advice on why that could be?
#cases_csv_file = read_csv_rows(file_name=ROOT_DIR + '/analyses/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/analyses/county_cases_time_series.csv',
                    #if_ignore_first_row = True)
cases_csv_file = ROOT_DIR + '/analyses/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/analyses/county_cases_time_series.csv'

# Create a dictionary to store the time series of cases for each county
county_cumulative_cases = defaultdict(list)

# Open and read the CSV file
with open(cases_csv_file, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)

    # Skip the header row
    next(csvreader)

    for row in csvreader:
        county = row[0]
        state = row[1]
        # if case values are missing replace with 0
        cases_data = [0 if value == 'NA' else float(value) for value in row[2:]]

        # Append the data to the respective county's time series
        county_state_key = f"{county}, {state}"
        county_cumulative_cases[county_state_key] = cases_data


# Creating a list to store CountyQALYLoss instances
county_qaly_losses = []

from classes.county_qaly_loss import CountyQALYLoss

for county_state_key, cases_data in county_cumulative_cases.items():
    county, state = county_state_key.split(', ')
    total_cases = int(sum(cases_data))

    # Creating a CountyQALYLoss instance for the county and state combination
    county_qaly_loss = CountyQALYLoss(state=state, county=county)

    # Adding the cases_data directly to the CountyQALYLoss instance
    county_qaly_loss.add_traj(cases_data)

    # Calculating the QALY loss
    county_qaly_loss.calculate_weekly_qaly_loss(case_weigh=0.1)  # Adjust the case_weight as needed

    # Appending the CountyQALYLoss instance to the list
    county_qaly_losses.append(county_qaly_loss)

    # Print county name, state, and QALY loss
    print(f'County: {county}, State: {state}, QALY Loss: {county_qaly_loss.qalyLoss}')




# PART 2: Calculating State-level QALY Loss

# Creating a dictionary to store state-level QALY losses
state_qaly_losses = defaultdict(float)

# Getting the sum of QALY losses for each state
for county_qaly_loss in county_qaly_losses:
    state_qaly_losses[county_qaly_loss.state] += county_qaly_loss.qalyLoss

# Converting state_qaly_losses to a list of lists to use write_csv
state_qaly_loss_rows = [[state, qaly_loss] for state, qaly_loss in state_qaly_losses.items()]

# Write into a CSV file
write_csv(rows=state_qaly_loss_rows, file_name=ROOT_DIR + '/Analysis/state_qalys.csv')

