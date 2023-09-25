import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from deampy.plots.plot_support import output_figure
from collections import defaultdict
from deampy.in_out_functions import write_csv, read_csv_rows
from definitions import ROOT_DIR

# Creating a dictionary to store the time series of cases for each county
county_cases_time_series = defaultdict(list)

# Specifying the file path. NOTE: please use this file path: '/data/county_time_data_all_dates.csv'
csv_file_path = ROOT_DIR + '/data/county_time_data_all_dates_small.csv'

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
        cases = cases if cases != '' else 'NA'

        # Append the data to the respective county's time series
        county_cases_time_series[county].append((date, cases, state))

# Create a list of unique dates across all counties
all_dates = sorted(set(date for time_series in county_cases_time_series.values() for date, _, _ in time_series))

# Create the header row with dates
header_row = ['County', 'State'] + all_dates

# Create a list of data rows for each county, ensuring length of all the rows is the same
county_cases_rows = []

for county, time_series in county_cases_time_series.items():
    data = []
    for date in all_dates:
        for time_date, time_cases, time_state in time_series:
            if time_date == date:
                data.append(time_cases)
                break
        else:
            #If data is missing for a date, fill with 'NA'
            data.append('NA')
    county_cases_rows.append([county, time_series[0][2]] + data)

# Write into a CSV file using the write_csv function
write_csv(rows=[header_row] + county_cases_rows, file_name=ROOT_DIR + '/data/summary/county_cases.csv')


# PART 2

# Get the first 9 counties from county_cases_time_series
counties_to_plot = list(county_cases_time_series.keys())[:9]

# Define a marker for missing data points
missing_data_marker = 'x'

# Loop through the selected counties
for county in counties_to_plot:
    # Extract the date and cases data for the county
    time_series = county_cases_time_series[county]
    dates = [date for date, _, _ in time_series]
    cases = [float(cases) if cases != 'NA' else np.nan for _, cases, _ in time_series]

    # Create a new figure and plot the data
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(dates, cases, label=f'{county}, {time_series[0][2]}', marker=missing_data_marker, markevery=[i for i, c in enumerate(cases) if np.isnan(c)])
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.title(f'Cases Time Series for {county}, {time_series[0][2]}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.legend()

    # For dates with missing data, the date is labeled in red
    for i, case in enumerate(cases):
        if np.isnan(case):
            plt.text(dates[i], 0, dates[i], color='red', verticalalignment='bottom', horizontalalignment='center')

    # Save each plot with a unique filename, e.g., county_cases_countyname.png
    filename = os.path.join(ROOT_DIR + '/data/summary', f'county_cases_{county}.png')

    output_figure(plt, filename)

# Show the plots (optional)
plt.show()
