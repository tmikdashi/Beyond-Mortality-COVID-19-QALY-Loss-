import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from deampy.in_out_functions import write_csv
from deampy.plots.plot_support import output_figure

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
        cases = cases if cases != '' else 'NA'

        # Append the data to the respective county's time series
        # TODO: note that country is not a unique identifier (there are multiple counties with the same name).
        #  So here you need to use the state as well to uniquely identify the county.
        #  For example, you can use the state abbreviation and county name as a tuple.
        #   county_cases_time_series[(county, state)].append((date, cases))
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
            # TODO: it would be better to use math.nan or np.nan instead of 'NA' to represent missing data.
            data.append('NA')
    county_cases_rows.append([county, time_series[0][2]] + data)

# Write into a CSV file using the write_csv function
write_csv(rows=[header_row] + county_cases_rows, file_name=ROOT_DIR + '/data/summary/county_cases.csv')


# TODO: Could you please move the code below to a separate file called plot_county_data.py?

# PART 2

# Get the first 9 counties from county_cases_time_series
# TODO: I think it makes sense to only look at a limited number of counties at this stage
#  (there are more 3000 counties in the US I think!). However, let's make sure to include
#  counties with missing observations as well to see if the code is working properly.
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
    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
    # TODO: I am not sure what markevery does. Could you please explain?
    #   Also, I think if plot function encounters a nan value, it will not plot anything for that date (which is good).
    #   In other word, the line will be broken for dates with missing data.
    plt.plot(dates, cases, label=f'{county}, {time_series[0][2]}',
             marker=missing_data_marker, markevery=[i for i, c in enumerate(cases) if np.isnan(c)])
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.title(f'Cases Time Series for {county}, {time_series[0][2]}')
    plt.xticks(rotation=90, fontsize=6)  # Rotate x-axis labels for better visibility
    plt.legend()

    # For dates with missing data, the date is labeled in red
    # TODO: if the line could be broken for dates with missing data, we probably don't need to do this.
    for i, case in enumerate(cases):
        if np.isnan(case):
            plt.text(dates[i], 0, dates[i], color='red', verticalalignment='bottom', horizontalalignment='center')

    # Save each plot with a unique filename, e.g., county_cases_countyname.png
    filename = os.path.join(ROOT_DIR + '/data/summary', f'county_cases_{county}.png')

    output_figure(plt, filename)

# Show the plots (optional)
plt.show()
