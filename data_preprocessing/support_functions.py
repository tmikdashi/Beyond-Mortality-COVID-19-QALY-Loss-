from collections import defaultdict

import numpy as np
import pandas as pd

from deampy.in_out_functions import write_csv, read_csv_rows
from definitions import ROOT_DIR
from datetime import datetime


def get_dict_of_county_data_by_type(data_type):
    """
    This function reads the county data CSV file and returns a dictionary of county data by the specified data type.
    :param data_type: The data type to extract ('cases', 'deaths', 'hospitalizations', 'icu admissions', etc.)
    :return: (dictionary, list) a dictionary with (county, state) as keys and a list of data values as values,
            and a list of dates
    """

    # Construct the file path based on the data type
    #file_path = ROOT_DIR + f'/csv_files/county_{data_type.replace(" ", "_")}.csv'
    file_path = ROOT_DIR + f'/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_{data_type.replace(" ", "_")}.csv'

    # Read the data
    data_rows = read_csv_rows(file_name=file_path, if_ignore_first_row=False)

    # Remove the population value from the dates
    dates = data_rows[0][4:]

    county_data_by_type = {}
    for row in data_rows[1:]:
        county = row[0]
        state = row[1]
        fips = row[2]
        population = row[3]
        data_values = row[4:]

        # Convert data values to a list of floats, handling missing values
        data_values = [float(data) if data != 'NA' else np.nan for data in data_values]

        county_data_by_type[(county, state, fips, population)] = data_values

    return county_data_by_type, dates




def generate_county_data_csv(data_type='cases'):
    """
    This function reads the county data CSV file and creates a CSV of county data over time for a specified data type.
    :param data_type: The data type to extract ('cases', 'deaths', 'hospitalizations', 'icu admissions', etc.)
    :return: (.csv file) a CSV file with data per county, over time, where each row corresponds to a county, identified
    by county name, state, fips, and population.
    """
    # Define a dictionary to map data types to column indices
    data_type_mapping = {
        'cases': 5,
        'deaths': 6,
        'hospitalizations': 20,
        'icu admissions': 23,
        'cases per 100,000': 17,
        'deaths per 100,000': 18,
        'hospitalizations per 100,000': 22,
        'icu admissions per 100,000': 24,
    }

    # Ensure the specified data_type is valid
    if data_type not in data_type_mapping:
        raise ValueError(
            "Invalid data_type. Choose from 'cases', 'deaths', 'hospitalizations', "
            "'icu admissions', 'cases per 100,000', "
            "'deaths per 100,000', 'hospitalizations per 100,000', 'icu admissions per 100,000'.")

    # Read the data
    #rows = read_csv_rows(file_name=ROOT_DIR + '/data/county_time_data_all_dates.csv',
                         #if_ignore_first_row=True)

    #rows = read_csv_rows(file_name='/Users/fm478/Downloads/county_time_data_all_dates.csv',
                         #if_ignore_first_row=True)

    rows = read_csv_rows(file_name='/Users/timamikdashi/Downloads/county_time_data_all_dates.csv',
                         if_ignore_first_row=True)


    # Creating a dictionary to store the time series of data for each county
    county_data_time_series = defaultdict(list)
    for row in rows:
        fips = row[1]
        county = row[3]
        state = row[12]  # State abbreviation
        date_str = row[2]
        population = row[10]  # Add population to this section

        # Check if the date is before or on November 2, 2022
        date = datetime.strptime(date_str, "%Y-%m-%d")
        if date <= datetime(2022, 11, 2):
            data_value = row[data_type_mapping[data_type]]

            # Removing PR from analysis
            if state == 'NA':
                data_value = np.nan
            # Check if data_value is empty or 'NA' and assign np.nan
            if data_value == '' or data_value == 'NA':
                data_value = np.nan
            else:
                # Convert other values to float
                data_value = float(data_value) * 7

            # Append the data to the respective county's time series
            county_data_time_series[(county, state, fips, population)].append((date_str, data_value))

    # Create a list of unique dates across all counties
    unique_dates = sorted(set(date for time_series in county_data_time_series.values() for date, _ in time_series))

    # Exclude dates after November 2, 2022
    unique_dates = [date for date in unique_dates if datetime.strptime(date, "%Y-%m-%d") <= datetime(2022, 11, 2)]

    # Generate the output file name based on data_type
    output_file = f'/csv_files/county_{data_type.replace(" ", "_")}.csv'

    # Create the header row with dates
    header_row = ['County', 'State', 'FIPS', 'Population'] + unique_dates

    # Create a list of data rows for each county, ensuring the length of all the rows is the same
    county_data_rows = []
    for key, time_series in county_data_time_series.items():
        data = []
        for date in unique_dates:
            found = False
            for time_date, time_data in time_series:
                if time_date == date:
                    data.append(time_data)
                    found = True
                    break
            if not found:
                # If data is missing for a date, fill with np.nan
                data.append(np.nan)
        # Check if the state name is 'NA' and skip adding the row
        if key[1] != 'NA':
            county_data_rows.append([key[0], key[1], key[2], key[3]] + data)

    # Write into a CSV file using the write_csv function
    write_csv(rows=[header_row] + county_data_rows, file_name=ROOT_DIR + output_file)


def generate_combined_county_data_csv():
    # Define a dictionary to map data types to column indices
    data_type_mapping = {
        'cases': 5,
        'deaths': 6,
        'hospitalizations': 20,
    }

    # Read the data
    rows = read_csv_rows(file_name=ROOT_DIR + '/data/county_time_data_all_dates.csv',
                         if_ignore_first_row=True)

    # Creating a dictionary to store the time series of data for each county
    county_data_time_series = defaultdict(list)
    for row in rows:
        fips = row[1]
        county = row[3]
        state = row[12]  # State abbreviation
        date = row[2]
        population = row[10]

        for data_type, data_column in data_type_mapping.items():
            data_value = row[data_column]

            # Check if data_value is empty or 'NA' and assign np.nan
            if data_value == '' or data_value == 'NA':
                data_value = np.nan
            else:
                # Convert other values to float
                data_value = float(data_value) * 7  # Adjust as needed

            # Append the data to the respective county's time series for the specific data type
            county_data_time_series[(county, state, fips, population, data_type)].append((date, data_value))

    # Create a list of unique dates across all counties
    unique_dates = sorted(set(date for time_series in county_data_time_series.values() for date, _ in time_series))

    # Generate the output file name for the combined data
    output_file = ROOT_DIR + '/csv_files/county_data_combined.csv'

    # Create the header row with dates for each data type
    header_row = ['County', 'State', 'FIPS', 'Population']
    for data_type in data_type_mapping.keys():
        header_row += [f'{data_type} {date}' for date in unique_dates]

    # Create a list of data rows for each county and data type
    combined_county_data_rows = []
    for key, time_series in county_data_time_series.items():
        data = {date: np.nan for date in unique_dates}
        for date, data_value in time_series:
            data[date] = data_value
        combined_county_data_rows.append([key[0], key[1], key[2], key[3]] + [data[date] for date in unique_dates])

    # Write into a CSV file using the write_csv function
    write_csv(rows=[header_row] + combined_county_data_rows, file_name=output_file)


def generate_deaths_by_age_group():
    """
    This function generates a csv containing information on the number of deaths associated with each age group.
    A crucial step in this process is redefining the age bands to match the dQALY age groups in the Briggs paper.
    Calculation for the number of deaths in each age band are based on Briggs spreadsheet tool

    :return: A csv of COVID-19 deaths by age group.
    """

    data = pd.read_csv(ROOT_DIR + '/data_deaths/Provisional_COVID-19_Deaths_by_Sex_and_Age.csv')

    deaths_by_age = data.groupby(['Age Group'])['COVID-19 Deaths'].sum().reset_index()

    age_band_mapping = {
        '0-9': ['Under 1 year', '1-4 years', '5-14 years'],
        '10-19': ['5-14 years','15-24 years'],
        '20-29': ['15-24 years', '25-34 years'],
        '30-39': ['25-34 years', '35-44 years'],
        '40-49': ['35-44 years', '45-54 years'],
        '50-59': ['45-54 years', '55-64 years'],
        '60-69': ['55-64 years','65-74 years'],
        '70-79': ['65-74 years','75-84 years'],
        '80-90': ['75-84 years' , '85 years and over'],
        '90-100': ['85 years and over']
    }

    new_age_data = {'Age Group': [], 'COVID-19 Deaths': []}

    for age_band, age_groups in age_band_mapping.items():
        total_deaths = sum(deaths_by_age[deaths_by_age['Age Group'].isin(age_groups)]['COVID-19 Deaths'])

        if age_band == '0-9':
            # Sum 'Under 1 year' and '1-4 years'
            total_deaths = sum(
                deaths_by_age[deaths_by_age['Age Group'].isin(['Under 1 year', '1-4 years'])]['COVID-19 Deaths'])
            # Add half of '5-14 years'
            total_deaths += 0.5 * sum(deaths_by_age[deaths_by_age['Age Group'] == '5-14 years']['COVID-19 Deaths'])
        else:
            total_deaths /= 2

        new_age_data['Age Group'].append(age_band)
        new_age_data['COVID-19 Deaths'].append(total_deaths)

    # Create a DataFrame for the new age bands
    new_age_df = pd.DataFrame(new_age_data)
    new_age_df['COVID-19 Deaths'] = pd.to_numeric(new_age_df['COVID-19 Deaths'], errors='coerce').fillna(0)

    # Select relevant columns
    deaths_by_age_group = new_age_df[['Age Group', 'COVID-19 Deaths']]

    # save the data as a csv file
    deaths_by_age_group.to_csv(ROOT_DIR + '/csv_files/deaths_by_age.csv', index=False)


