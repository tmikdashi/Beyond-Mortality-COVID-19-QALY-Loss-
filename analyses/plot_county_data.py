import os

import matplotlib.pyplot as plt
import numpy as np
from deampy.in_out_functions import write_csv
from deampy.plots.plot_support import output_figure

from definitions import ROOT_DIR
from support.support_functions import get_dict_of_county_cases_and_dates

# get the dictionary of county cases
county_cases_data, dates = get_dict_of_county_cases_and_dates()

# find counties and states with missing values and counting missing values
counties_with_missing_values = {}
for (county, state), cases in county_cases_data.items():
    num_missing_values = sum(np.isnan(cases))
    if num_missing_values > 0:
        counties_with_missing_values[(county, state)] = num_missing_values

# report counties with missing values
rows = [['County', 'State', 'Number of Missing Values']]
for (county, state), num_missing_values in counties_with_missing_values.items():
    rows.append([county, state, num_missing_values])
write_csv(rows=rows, file_name=ROOT_DIR + '/data/summary/counties_with_missing_values.csv')


# PART 3: PLOTTING the first 9 counties with missing data
counties_to_plot = list(counties_with_missing_values.keys())[:9]

# Loop through counties to plot
for (county, state) in counties_to_plot:
    # Extract the date and cases data for the county
    cases = county_cases_data[(county, state)]

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
    filename = os.path.join(ROOT_DIR + '/figs/cases', f'county_cases_{county}_{state}.png')
    output_figure(plt, filename)
