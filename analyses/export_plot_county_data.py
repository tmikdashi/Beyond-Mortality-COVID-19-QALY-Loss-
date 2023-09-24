import matplotlib.pyplot as plt
from deampy.plots.plot_support import output_figure
import csv
import os
import sys
from collections import defaultdict
from deampy.in_out_functions import write_csv, read_csv_rows

# I'm unable to import ROOT_DIR without adding the parent directory containing 'definitions.py' to the Python path
script_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.dirname(script_directory)
sys.path.append(project_directory)

from definitions import ROOT_DIR

# TODO: for the initial analysis, all we need is the number of cases. So we don't want to recreate this data
#  every time we run the analysis. We should export and save whatever we need from the source dataset
#  in a separate files.
#  So here it would be great to write the code that reads the data
#  and saves the number of cases for each county in a separate file.
#  Each column corresponds to a county (the first cell in the column is the county name or some unique ID)
#  you could store these files under data/summary/county_cases.csv
# PART 1: CREATING 4 DIFFERENT CSV DATA FILES:
# 1: county_cases_time_series.csv: saves the number of cases for each county, with corresponding state
# 2: county_deaths_time_series.csv: saves the number of death for each county, with corresponding state
# 3: county_hospitalization_time_series.csv: saves the number of hospitalizations for each county, with corresponding state
# 4: county_icu_time_series.csv: saves the number of icu admissions for each county, with corresponding state


# Create a dictionary to store the time series of cases for each county
county_cases_time_series = defaultdict(list)
county_deaths_time_series = defaultdict(list)
county_hospitalizations_time_series = defaultdict(list)
county_icu_time_series = defaultdict(list)

# Specify the file path
#Note: I tried using read_csv_rows but it would consistently skip many rows
csv_file_path = ROOT_DIR + '/data/county_time_data_all_dates_small.csv'

# Open and read the CSV file
with open(csv_file_path, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)

    # Skip the header row
    next(csvreader)

    for row in csvreader:
        county = row[2]
        state = row[6]  # State abbreviation
        date = row[1]
        cases = row[3]
        deaths = row[4]
        hospitalizations = row[10]
        icu_hospitalizations = row[11]

        # Specify abbreviation for Puerto Rico
        if state == 'NA':
            state = 'PR'
        # Replace missing values with 'NA'
        cases = cases if cases != '' else 'NA'
        deaths = deaths if deaths != '' else 'NA'
        hospitalizations = hospitalizations if hospitalizations != '' else 'NA'
        icu_hospitalizations = icu_hospitalizations if icu_hospitalizations != '' else 'NA'

        # Append the data to the respective county's time series
        county_cases_time_series[county].append((date, cases, state))
        county_deaths_time_series[county].append((date, deaths, state))
        county_hospitalizations_time_series[county].append((date, hospitalizations, state))
        county_icu_time_series[county].append((date, icu_hospitalizations, state))

# Create CSV files for cases, deaths, hospitalizations, and ICU data
# Because I wanted to create 3 csv files automatically, create_csv was more intuitive to use.
# I used write_csv in the other files though.
def create_csv(filename, data_dict):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['County', 'State'] + [date for date, _, _ in data_dict[next(iter(data_dict))]]
        csvwriter.writerow(header)

        # Write the time series of data for each county with state
        for county, time_series in data_dict.items():
            data = [value for _, value, _ in time_series]
            csvwriter.writerow([county, time_series[0][2]] + data)

# Specify output file paths for cases, deaths, hospitalizations, and ICU data
cases_csv_file = 'county_cases_time_series.csv'
deaths_csv_file = 'county_deaths_time_series.csv'
hospitalizations_csv_file = 'county_hospitalizations_time_series.csv'
icu_csv_file = 'county_icu_time_series.csv'



# TODO: The next step is to plot them without dealing with missing values. In other words,
#  if there is a week with missing values, it should be clear from the plot.
#  we need a separate plot for each county to check if everything makes sense.


# PART 2: CREATING PLOTS:

# Note: there are over 1,000 different counties so plotting them all may take up a lot of space.
# This code creates plots for the fist 9 counties

import matplotlib.pyplot as plt
from deampy.plots.plot_support import output_figure
import numpy as np

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
    filename = f'county_cases_{county}.png'
    output_figure(plt, filename)