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
    '''
    This function generate a csv containing information on the proportion of deaths associated with each age group and sex

    :return: creates a csv that describes the proportion of total COVID deaths come from each age group and sex.
    The outputted csv is organized as 3 columns: Age Group, Males: Proportion of Deaths and Females: Proportion of Deaths.

    Note: 'Males: Proportion of Deaths' does not represent the proportion of deaths by age group among total male deaths
    Instead, all the values in Males: Proportion of Deaths and in Females: Proportion of Deaths sum to 1.
    '''


    rows = read_csv_rows(file_name=ROOT_DIR + '/data/Provisional_COVID-19_Deaths_by_Sex_and_Age (3).csv',
                         if_ignore_first_row=True)

    # Create a DataFrame from the list of rows: Some formatting issues with columns so took extra step to rename some columns
    data = pd.DataFrame(rows, columns=["Data As Of", "Start Date", "End Date", "Group", "Year", "Month", "State", "Sex",
                                       "Age group", "COVID-19 Deaths", "Total Deaths", "Pneumonia Deaths",
                                       "Pneumonia and COVID-19 Deaths", "Influenza Deaths",
                                       "Pneumonia, Influenza, or COVID-19 Deaths", "Footnote"])


    # Calculate the total number of deaths
    data['COVID-19 Deaths'] = pd.to_numeric(data['COVID-19 Deaths'], errors='coerce').fillna(0)
    total_deaths = data['COVID-19 Deaths'].sum()

    # Calculate proportions of deaths associated with each age group and sex
    data['Prop of Deaths'] = data['COVID-19 Deaths'] / total_deaths

    # Select relevant columns
    prop_deaths_by_age_group_and_sex = data[['Age group', 'Sex', 'Prop of Deaths']]
    output_file_path = ROOT_DIR + '/csv_files/prop_deaths_by_age_and_sex.csv'
    prop_deaths_by_age_group_and_sex.to_csv(output_file_path, index=False)

    return prop_deaths_by_age_group_and_sex
'''
    # Calculate proportions of deaths associated with each age group, while keeping track of sex
    # (Note: we are not calculating the proportion of deaths within a sex, but simply separating them)
    data['Males: Proportion of Deaths'] = data.apply(
        lambda row: row['COVID-19 Deaths'] / total_deaths if row['Sex'] == 'Male' else 0, axis=1)
    data['Females: Proportion of Deaths'] = data.apply(
        lambda row: row['COVID-19 Deaths'] / total_deaths if row['Sex'] == 'Female' else 0, axis=1)

    # Group by 'Age Group'
    result = data.groupby('Age group')[
        ['Age group', 'Males: Proportion of Deaths', 'Females: Proportion of Deaths']].max().reset_index(drop=True)
'''
    # Save the new proportions to a new CSV file




def process_life_expectancy_data(data, sex):
    '''
    Because life expectancy data comes in separate but parallely-organized files for males and females,
    this function describes as general approach to processing life expectancy data and preparing it for later analysis.
    This processing step specifically consists of taking in a data file for a specific sex and (1) designating age groups
    that match those from the death data and (2) calculating average life expectancy for that age group.

    :param data: csv data file pathway
    :param sex: describes which sex the file is for
    :return: This function taken in data in the form of a csv file for a designated sex and transforms the data into 3
    columns 'Age group', 'Expectation of life at age x' and Sex'
    '''

    header = ["Age (years)", "Probability of dying between ages x and x + 1",
              "Number surviving to age x", "Number dying between ages x and x + 1",
              "Person-years lived between ages x and x + 1", "Total number of person-years lived above age x",
              "Expectation of life at age x"]
    data.columns = header

    # Re-formatting age data: Age data is currently presented as 0?1 to designate life-expectancy at age 0
    data['Age (years)'] = data['Age (years)'].str.split('?').str[0]
    data['Age (years)'] = pd.to_numeric(data['Age (years)'], errors='coerce')

    mask = data['Age (years)'] == '100 and over'
    data.loc[mask, 'Age (years)'] = 100  # Set 'Age (years)' to 100 for '100 and over'
    data['Age (years)'] = data['Age (years)'].astype(int)  # Convert 'Age (years)' to integer

    # Creating age groups that match those above
    age_cutoffs = [0, 1, 5, 15, 25, 35, 45, 55, 65, 75, 85, float('inf')]
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

    # Calculate average life expectancy by age group
    #data['Avg Life Expectancy'] = data.groupby('Age group')['Life Expectancy'].mean().reset_index()

    # Calculate average life expectancy by age group
    #avg_life_expectancy = data.groupby('Age group')['Life Expectancy'].mean().reset_index()

    # Merge back the average life expectancy to the original DataFrame
    #data = pd.merge(data, avg_life_expectancy, on='Age group', how='left', suffixes=('', '_avg'))

    #return data[['Age group', 'Avg Life Expectancy', 'Sex']]

def generate_combined_life_expectancy():
    '''
    This function creates a csv file that describes the average expected years of life by age group and sex.
    2 csv files are analyzed(male and female) and the process_average_expected_years_of_life.csv is used to clean up the data.

    :return: a csv file that describes the average expected years of life by age group and sex.
    '''

    LE_data_male = pd.read_csv(ROOT_DIR + '/data/Life_table_male_2019_USA.csv', skiprows=2, skipfooter=3, engine='python')
    processed_LE_data_male = process_life_expectancy_data(LE_data_male, 'Male')

    LE_data_female = pd.read_csv(ROOT_DIR + '/data/Life_table_female_2019_USA.csv', skiprows=2, skipfooter=5)
    processed_LE_data_female = process_life_expectancy_data(LE_data_female, 'Female')

    # Calculate average life expectancy by age group
    average_LE_data_by_age_group_and_sex = pd.concat([processed_LE_data_male, processed_LE_data_female])



    print("average LE both sexes:", average_LE_data_by_age_group_and_sex)


    return average_LE_data_by_age_group_and_sex


def extract_LE_and_prop_death_arrays(average_LE_data_by_age_and_sex,prop_deaths_by_age_group_and_sex):
    '''
    This function generates the life expectancy and proportion of death arrays that could later serve as inputs for  Dirichlet.

    :param life_expectancy_data:
    :param prop_deaths_data:
    :return:
    '''


   # Combine data by Age group and sex
    combined_data = pd.merge(average_LE_data_by_age_and_sex, prop_deaths_by_age_group_and_sex, on=['Age group', 'Sex'], how='inner')

    # Sort the data by life expectancy
    sorted_data = combined_data.sort_values(by='Life Expectancy')

    # Extract arrays for life expectancy and proportion of deaths
    life_expectancy_array = sorted_data['Life Expectancy'].to_numpy()
    prop_deaths_array = sorted_data['Prop of Deaths'].to_numpy()

    return life_expectancy_array, prop_deaths_array

