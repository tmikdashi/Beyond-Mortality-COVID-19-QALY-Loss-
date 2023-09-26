import csv
import os
from collections import defaultdict
import numpy as np
from deampy.in_out_functions import write_csv
from definitions import ROOT_DIR

# csv filename where the data is located
csv_file_path = ROOT_DIR + '/data/county_time_data_all_dates.csv'

# Creating a dictionary to store the time series of cases for each county
county_cases_time_series = defaultdict(list)

# Opening and reading the CSV file
with open(csv_file_path, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)

    # Skip the header row
    next(csvreader)

    for row in csvreader:
        county = row[3]
        state = row[12]  # State abbreviation
        date = row[2]
        cases = row[5]

        # Specify abbreviation for Puerto Rico
        if state == 'NA':
            state = 'PR'
        # Replace missing values with 'NA'
        cases = float(cases) if cases != '' else np.nan

        # Append the data to the respective county's time series
        # TODO: CHANGE COMPLETE, note that country is not a unique identifier (there are multiple counties with the
        #  same name).So here you need to use the state as well to uniquely identify the county.
        #  For example, you can use the state abbreviation and county name as a tuple.
        #   county_cases_time_series[(county, state)].append((date, cases))
        county_cases_time_series[(county,state)].append((date, cases))

# Create a list of unique dates across all counties
all_dates = sorted(set(date for time_series in county_cases_time_series.values() for date, _ in time_series))

# Create a list of unique county and state combinations
unique_county_state_combinations = sorted(set((county, state) for county, state in county_cases_time_series.keys()))

# Create the header row with dates
header_row = ['County', 'State'] + all_dates

# Create a list of data rows for each county, ensuring length of all the rows is the same
county_cases_rows = []

for county, state in unique_county_state_combinations:
    time_series = county_cases_time_series[(county, state)]
    data = []
    for date in all_dates:
        found = False
        for time_date, time_cases in time_series:
            if time_date == date:
                data.append(time_cases)
                found = True
                break
        if not found:
            # If data is missing for a date, fill with np.nan
            data.append(np.nan)
    county_cases_rows.append([county, state] + data)

# Write into a CSV file using the write_csv function
write_csv(rows=[header_row] + county_cases_rows, file_name=ROOT_DIR + '/data/summary/county_cases.csv')


