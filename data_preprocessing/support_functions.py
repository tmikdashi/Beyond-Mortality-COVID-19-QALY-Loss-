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


def generate_deaths_by_age_group_and_sex():
    """
    This function generate a csv containing information on the number of deaths associated with each age group and sex

    :return: creates a csv that describes the number total COVID deaths come from each age group and sex.
    The outputted csv is organized as 3 columns: Age Group, Sex and COVID-19 Deaths.
    """

    data = pd.read_csv(ROOT_DIR + '/data_deaths/Provisional_COVID-19_Deaths_by_Sex_and_Age.csv')
    data = data.rename(columns={'Age Group': 'Age group'}) # needed for merging in extract_LE_and_prop_death_arrays

    # Calculate the total number of deaths
    data['COVID-19 Deaths'] = pd.to_numeric(data['COVID-19 Deaths'], errors='coerce').fillna(0)

    # Select relevant columns
    deaths_by_age_group_and_sex = data[['Age group', 'Sex', 'COVID-19 Deaths']]

    # save the data as a csv file
    deaths_by_age_group_and_sex.to_csv(ROOT_DIR + '/csv_files/deaths_by_age_and_sex.csv', index=False)



def process_life_expectancy_data(data, sex):
    """
    Because life expectancy data comes in separate but parallely-organized files for males and females,
    this function describes as general approach to processing life expectancy data and preparing it for later analysis.
    This processing step specifically consists of (1) reformatting the age data, (2) designating age groups
    that match those from the proportion of deaths data and (3) calculating average life expectancy for that age group.

    :param data: csv containing life expectancy data for a specific sec
    :param sex: describes which sex the file is for
    :return: This function taken in data in the form of a csv file for a designated sex and transforms the data into 3
    columns 'Age group', 'Sex' and 'Life Expectancy'
    """

    # Re-formatting age data: Age data is currently presented as 0?1 to designate life-expectancy at age 0
    data['Age (years)'] = data['Age (years)'].str.split('?').str[0]
    data['Age (years)'] = pd.to_numeric(data['Age (years)'], errors='coerce')

    mask = data['Age (years)'] == '100 and over'
    data.loc[mask, 'Age (years)'] = 100  # Set 'Age (years)' to 100 for '100 and over'

    # Creating age groups that match the ones from the proportion of deaths data
    age_cutoffs = [0, 1, 4, 14, 24, 34, 44, 54, 64, 74, 84, float('inf')]
    labels = ['Under 1 year', '1-4 years', '5-14 years', '15-24 years', '25-34 years',
              '35-44 years', '45-54 years', '55-64 years', '65-74 years', '75-84 years', '85 years and over']

    data['Age group'] = pd.cut(data['Age (years)'], bins=age_cutoffs, labels=labels, right=False)
    data['Life Expectancy'] = data['Expectation of life at age x'].astype(str)
    data['Life Expectancy'] = pd.to_numeric(data['Life Expectancy'].str.replace(r'\D', ''), errors='coerce')

    # Converting inputted sex to a new column
    data['Sex'] = sex

    # Calculate average life expectancy by age group
    avg_life_expectancy = data.groupby(['Age group', 'Sex'])['Life Expectancy'].mean().reset_index()

    return avg_life_expectancy


def generate_life_expectancy_by_sex_age():
    """
    This function creates a csv file that describes the average expected years of life by age group and sex.
    2 csv files are analyzed(male and female) and the average_LE_by_age_and_sex.csv is used to clean up the data.

    :return: a csv file that describes the average expected years of life by age group and sex.
    """

    # Calculate life-expectancy for males by age group
    le_data_male = pd.read_csv(
        ROOT_DIR + '/data_deaths/Life_table_male_2019_USA.csv',
        skiprows=[0, 2], skipfooter=2)

    processed_le_data_male = process_life_expectancy_data(le_data_male, 'Male')

    # Calculate life-expectancy for females by age group
    le_data_female = pd.read_csv(
        ROOT_DIR + '/data_deaths/Life_table_female_2019_USA.csv',
        skiprows=[0, 2], skipfooter=4)

    processed_le_data_female = process_life_expectancy_data(le_data_female, 'Female')

    # Combine the male and female life expectancy by age group
    average_le_data_by_age_group_and_sex = pd.concat([processed_le_data_male, processed_le_data_female])

    # save to csv file
    average_le_data_by_age_group_and_sex.to_csv(
        ROOT_DIR + '/csv_files/average_LE_by_age_and_sex.csv', index=False)


def extract_LE_and_death_arrays():
    """
    This function generates combines the life expectancy and number of deaths data in one csv file
    to be able to easily extract the life expectancy and number of deaths arrays that could later serve as inputs for
    Dirichlet.

    :return: a csv file with combined data and life expectancy and number of death arrays  that could later serve as
    inputs for  Dirichlet
    """

    # Combine data by Age group and sex
    average_LE_data_by_age_group_and_sex = pd.read_csv(ROOT_DIR + '/csv_files/average_LE_by_age_and_sex.csv')
    deaths_by_age_group_and_sex = pd.read_csv(ROOT_DIR + '/csv_files/deaths_by_age_and_sex.csv')

    combined_data = pd.merge(
        average_LE_data_by_age_group_and_sex,
        deaths_by_age_group_and_sex,
        on=['Age group', 'Sex'], how='inner')


    # Extract arrays for life expectancy and proportion of deaths
    life_expectancy_array = combined_data['Life Expectancy'].to_numpy()
    nb_deaths_array = combined_data['COVID-19 Deaths'].to_numpy()

    combined_data.to_csv(
        ROOT_DIR + '/csv_files/average_LE_and deaths_data_by_age_group_and_sex', index=False)

    # TODO: These many no longer be necessary
    #return life_expectancy_array, nb_deaths_array

    #print("\nLife Expectancy Array:")
    #print(life_expectancy_array)

    #print("\nNumber of Deaths Array:")
    #print(nb_deaths_array)

