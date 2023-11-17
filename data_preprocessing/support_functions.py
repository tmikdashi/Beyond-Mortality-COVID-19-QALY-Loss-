from collections import defaultdict

import numpy as np
import pandas as pd
from deampy.in_out_functions import write_csv, read_csv_rows

from definitions import ROOT_DIR


def get_dict_of_county_data_by_type(data_type):
    """
    This function reads the county data CSV file and returns a dictionary of county data by the specified data type.
    :param data_type: The data type to extract ('cases', 'deaths', 'hospitalizations', 'icu admissions', etc.)
    :return: (dictionary, list) a dictionary with (county, state) as keys and a list of data values as values,
            and a list of dates
    """

    # Construct the file path based on the data type
    file_path = ROOT_DIR + f'/csv_files/county_{data_type.replace(" ", "_")}.csv'
    #file_path = ROOT_DIR + f'/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_{data_type.replace(" ", "_")}.csv'

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
    rows = read_csv_rows(file_name=ROOT_DIR + '/data/county_time_data_all_dates.csv',
                         if_ignore_first_row=True)


    # Creating a dictionary to store the time series of data for each county
    county_data_time_series = defaultdict(list)
    for row in rows:
        fips = row[1]
        county = row[3]
        state = row[12]  # State abbreviation
        date = row[2]
        population = row[10]  # Add population to this section
        data_value = row[data_type_mapping[data_type]]


        # Specify abbreviation for Puerto Rico
        if state == 'NA':
            state = 'PR'
        # Check if data_value is empty or 'NA' and assign np.nan
        if data_value == '' or data_value == 'NA':
            data_value = np.nan
        else:
            # Convert other values to float
            data_value = float(data_value) * 7

        # Append the data to the respective county's time series
        county_data_time_series[(county, state, fips, population)].append((date, data_value))

    # Create a list of unique dates across all counties
    unique_dates = sorted(set(date for time_series in county_data_time_series.values() for date, _ in time_series))

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
        county_data_rows.append([key[0], key[1], key[2], key[3]] + data)

    # Write into a CSV file using the write_csv function
    write_csv(rows=[header_row] + county_data_rows, file_name=ROOT_DIR + output_file)


from collections import defaultdict


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

def generate_prop_deaths_by_age_group_and_sex():
    # Read the data
    rows = read_csv_rows(file_name=ROOT_DIR + '/data/Provisional_COVID-19_Deaths_by_Sex_and_Age (3).csv',
                         if_ignore_first_row=True)

    # Create a DataFrame from the list of rows
    data = pd.DataFrame(rows, columns=["Data As Of", "Start Date", "End Date", "Group", "Year", "Month", "State", "Sex",
                                       "Age Group", "COVID-19 Deaths", "Total Deaths", "Pneumonia Deaths",
                                       "Pneumonia and COVID-19 Deaths", "Influenza Deaths",
                                       "Pneumonia, Influenza, or COVID-19 Deaths", "Footnote"])
    data['COVID-19 Deaths'] = pd.to_numeric(data['COVID-19 Deaths'], errors='coerce').fillna(0)

    # Calculate the total deaths
    total_deaths = data['COVID-19 Deaths'].sum()

    # Calculate proportions for male and female
    data['Proportion of Deaths Male'] = data.apply(
        lambda row: row['COVID-19 Deaths'] / total_deaths if row['Sex'] == 'Male' else 0, axis=1)
    data['Proportion of Deaths Female'] = data.apply(
        lambda row: row['COVID-19 Deaths'] / total_deaths if row['Sex'] == 'Female' else 0, axis=1)

    # Group by 'Age Group' to get unique rows for each age group
    result = data.groupby('Age Group')[
        ['Age Group', 'Proportion of Deaths Male', 'Proportion of Deaths Female']].max().reset_index(drop=True)

    # Save the new data with proportions to a new CSV file
    output_file_path = ROOT_DIR + '/csv_files/prop_deaths_by_age_and_sex.csv'
    header_row = ['Age Group', 'Proportion of Deaths Male', 'Proportion of Deaths Female']
    combined_rows = [header_row] + result.values.tolist()
    write_csv(rows=combined_rows, file_name=output_file_path)



def generate_life_exp_by_age_group_and_sex():
    rows_m = read_csv_rows(file_name=ROOT_DIR + '/data/PerLifeTables_M_Alt2_TR2017.csv',
                         if_ignore_first_row=True)
    rows_f = read_csv_rows(file_name=ROOT_DIR + '/data/PerLifeTables_F_Alt2_TR2017.csv',
                           if_ignore_first_row=True)

    # Create a DataFrame from the list of rows
    data_m = pd.DataFrame(rows_m)
    data_f = pd.DataFrame(rows_f)

    # Creating average life expectancy values across different years
    window_size =5 # specifying that we are looking at average every 5 years
    data_m.loc[data_m['Age'] <1,'Age group'] = "Under 1 year"
    data_m.loc[data_m['Age'] >= 1 & data_m['Age'] < 5,'Age group'] = "1-4 years"
    data_m.loc[data_m['Age'] >= 5 & data_m['Age'] < 15,'Age group'] = "5-14 years"
    data_m.loc[data_m['Age'] >= 15 & data_m['Age'] < 25, 'Age group'] = "15-24 years"
    data_m.loc[data_m['Age'] >= 25 & data_m['Age'] < 35, 'Age group'] = '25-34 years'
    data_m.loc[data_m['Age'] >= 35 & data_m['Age'] < 45, 'Age group'] = '35-44 years'
    data_m.loc[data_m['Age'] >= 45 & data_m['Age'] < 55, 'Age group'] = '45-54 years'
    data_m.loc[data_m['Age'] >= 55 & data_m['Age'] < 65, 'Age group'] = '55-64 years'
    data_m.loc[data_m['Age'] >= 65 & data_m['Age'] < 75, 'Age group'] = '65-74 years'
    data_m.loc[data_m['Age'] >= 75 & data_m['Age'] < 85, 'Age group'] = '75-84 years'
    data_m.loc[data_m['Age'] >= 85, 'Age group'] = '85 years and over'

    data_m['Average LE Male'] = data_m.group_by('Age Group')[data_m[' Expectation of life at age x'].mean()]
    data_f['Average LE Male'] = data_m.group_by('Age Group')[data_f[' Expectation of life at age x'].mean()]


# Generate the output file name for the combined data
    output_file = ROOT_DIR + '/csv_files/prop_deaths_by_age_and_sex.csv'

    # Create the header row with dates for each data type
    header_row = ['Age Group', 'Average LE Male', 'Average LE Female']
    combined_rows = [header_row] + data_m.values.tolist() + data_f.values.tolist()
    write_csv(rows=combined_rows, file_name=output_file)

def calculate_nb_deaths_per_age_group_by_sex():

    rows = read_csv_rows(file_name=ROOT_DIR + '/csv_files/prop_deaths_by_age_and_sex.csv',
                         if_ignore_first_row=True)

    rows['Nb of deaths male'] = rows['Proportion of Deaths Male'] * AllStates.deaths.weeklyQALYLoss
    rows['Nb of deaths male'] = rows['Proportion of Deaths Female'] * AllStates.deaths.weeklyQALYLoss