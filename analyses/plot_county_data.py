# TODO: please move the code to plot cases by county here.
#  It should read the csv file county_cases.csv and plot the cases

import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from deampy.plots.plot_support import output_figure
from definitions import ROOT_DIR

# PART 1: Loading the data for plotting

csv_file_path = ROOT_DIR + '/analyses/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/data/summary/county_cases.csv'

# Create a dictionary to store the data from the CSV file
county_cases_data = {}

# Open and read the CSV file
with open(csv_file_path, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)

    for row in csvreader:
        county = row[0]
        state = row[1]
        cases = row[2:]

        # Convert cases to a list of floats, handling missing values
        cases = [float(case) if case != 'NA' else np.nan for case in cases]

        county_cases_data[(county, state)] = cases

# PART 2: Finding counties and states with missing values and counting missing values
counties_with_missing_values = {}

for (county, state), cases in county_cases_data.items():
    num_missing_values = sum(np.isnan(cases))
    if num_missing_values > 0:
        counties_with_missing_values[(county, state)] = num_missing_values


# PART 3: PLOTTING the first 9 counties with missing data
counties_to_plot = list(counties_with_missing_values.keys())[:9]

# Loop through counties to plot
for (county, state) in counties_to_plot:
    # Extract the date and cases data for the county
    cases = county_cases_data[(county, state)]
    dates = header[2:]

    # Creating a new figure and plot the data
    plt.figure(figsize=(12, 6))

    # Plotting available data as blue circles for contrast
    plt.plot(dates, cases, label=f'{county}, {state}', marker='o', linestyle='-', markersize=4, linewidth=1.5, color='blue')

    # For missing data, we will convert NaN to 0 and plot it as a red 'x'
    missing_data_marker = {'marker': 'x', 'color': 'red'}
    for i, is_missing in enumerate(cases):
        if np.isnan(is_missing):
            plt.scatter(dates[i], 0, marker='x', color='red')

    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.title(f'Cases Time Series for {county}, {state}')
    plt.xticks(rotation=90, fontsize=6)  # Rotate x-axis labels for better visibility
    plt.legend()

    # Save each plot with a unique filename, e.g., county_cases_countyname.png
    filename = os.path.join(ROOT_DIR + '/data/summary', f'county_cases_{county}_{state}.png')
    output_figure(plt, filename)
