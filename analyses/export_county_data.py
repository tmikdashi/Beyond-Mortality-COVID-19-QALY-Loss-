from collections import defaultdict

import numpy as np
from deampy.in_out_functions import write_csv, read_csv_rows

from definitions import ROOT_DIR

# Read the data
rows = read_csv_rows(file_name=ROOT_DIR + '/data/county_time_data_all_dates.csv',
                     if_ignore_first_row=True)

# Creating a dictionary to store the time series of cases for each county
county_cases_time_series = defaultdict(list)
for row in rows:
    county = row[3]
    state = row[12]  # State abbreviation
    date = row[2]
    cases = row[5]

    # Specify abbreviation for Puerto Rico
    if state == 'NA':
        state = 'PR'
    # Replace missing values with 'NA'
    cases = float(cases)*7 if cases != '' else np.nan

    # Append the data to the respective county's time series
    county_cases_time_series[(county, state)].append((date, cases))

# Create a list of unique dates across all counties
unique_dates = sorted(set(date for time_series in county_cases_time_series.values() for date, _ in time_series))


# Create the header row with dates
header_row = ['County', 'State'] + unique_dates

# Create a list of data rows for each county, ensuring length of all the rows is the same
county_cases_rows = []
for key, time_series in county_cases_time_series.items():
    data = []
    for date in unique_dates:
        found = False
        for time_date, time_cases in time_series:
            if time_date == date:
                data.append(time_cases)
                found = True
                break
        if not found:
            # If data is missing for a date, fill with np.nan
            data.append(np.nan)
    county_cases_rows.append([key[0], key[1]] + data)

# Write into a CSV file using the write_csv function
write_csv(rows=[header_row] + county_cases_rows, file_name= ROOT_DIR + '/data/summary/county_cases.csv')
