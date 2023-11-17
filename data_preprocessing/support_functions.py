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
    :return: creates a csv that describes the proportion of total COVID deaths come from each age group and sex.
    The outputted csv is organized at 3 columns: Age Group, Males: Proportion of Deaths and Females: Proportion of Deaths.
    Note: 'Males: Proportion of Deaths' does not represente the proportion of deaths by age group among males.
    Instead, all the values in Males: Proportion of Deaths as well as Females: Proportion of Deaths sum to 1.
    '''


    rows = read_csv_rows(file_name=ROOT_DIR + '/data/Provisional_COVID-19_Deaths_by_Sex_and_Age (3).csv',
                         if_ignore_first_row=True)

    # Create a DataFrame from the list of rows: Some with columns so took extra step
    data = pd.DataFrame(rows, columns=["Data As Of", "Start Date", "End Date", "Group", "Year", "Month", "State", "Sex",
                                       "Age group", "COVID-19 Deaths", "Total Deaths", "Pneumonia Deaths",
                                       "Pneumonia and COVID-19 Deaths", "Influenza Deaths",
                                       "Pneumonia, Influenza, or COVID-19 Deaths", "Footnote"])


    # Calculate the total deaths
    data['COVID-19 Deaths'] = pd.to_numeric(data['COVID-19 Deaths'], errors='coerce').fillna(0)
    total_deaths = data['COVID-19 Deaths'].sum()

    # Calculate proportions for each group while keeping track of sex (Note: we are not looking at proportion within a sex)
    data['Males: Proportion of Deaths'] = data.apply(
        lambda row: row['COVID-19 Deaths'] / total_deaths if row['Sex'] == 'Male' else 0, axis=1)
    data['Females: Proportion of Deaths'] = data.apply(
        lambda row: row['COVID-19 Deaths'] / total_deaths if row['Sex'] == 'Female' else 0, axis=1)

    # Group by 'Age Group'
    result = data.groupby('Age group')[
        ['Age group', 'Males: Proportion of Deaths', 'Females: Proportion of Deaths']].max().reset_index(drop=True)

    # Save the new data with proportions to a new CSV file
    output_file_path = ROOT_DIR + '/csv_files/prop_deaths_by_age_and_sex.csv'
    header_row = ['Age group', 'Males: Proportion of Deaths', 'Females: Proportion of Deaths']
    combined_rows = [header_row] + result.values.tolist()
    write_csv(rows=combined_rows, file_name=output_file_path)


def process_life_expectancy_data(data, sex):
    '''
    Because life expectancy data comes in separate but parallely-organized files for males and females,
    this function describes as general approach to processing life expectancy data and preparing it for later analysis.
    This processing step specifically takes in a data file for a specific sex and (1) designates age groups that match those from the death data and (2)
    calculates average life expectancy for that age group.

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

    # Issue with the format of the data, Extract the Age (years) values before the ?
    data['Age (years)'] = data['Age (years)'].str.split('?').str[0]
    data['Age (years)'] = pd.to_numeric(data['Age (years)'], errors='coerce')

    mask = data['Age (years)'] == '100 and over'
    data.loc[mask, 'Age (years)'] = 100  # Set 'Age (years)' to 100 for '100 and over'
    data['Age (years)'] = data['Age (years)'].astype(int)  # Convert 'Age (years)' to integer

    bins = [0, 1, 5, 15, 25, 35, 45, 55, 65, 75, 85, float('inf')]

    # Define the corresponding labels for each bin
    labels = ['Under 1 year', '1-4 years', '5-14 years', '15-24 years', '25-34 years',
              '35-44 years', '45-54 years', '55-64 years', '65-74 years', '75-84 years', '85 years and over']

    # Apply the cut function to create the 'Age group' column
    data['Age group'] = pd.cut(data['Age (years)'], bins=bins, labels=labels, right=False)

    # Convert 'Expectation of life at age x' to numeric (remove non-numeric characters)
    data['Expectation of life at age x'] = data['Expectation of life at age x'].astype(str)
    data['Expectation of life at age x'] = pd.to_numeric(data['Expectation of life at age x'].str.replace(r'\D', ''),
                                                         errors='coerce')

    # Add a new column for sex
    data['Sex'] = sex

    return data[['Age group', 'Expectation of life at age x', 'Sex']]

def generate_combined_life_expectancy():
    data_male = pd.read_csv(ROOT_DIR + '/data/Life_table_male_2019_USA.csv', skiprows=2, skipfooter=3, engine='python')
    processed_data_male = process_life_expectancy_data(data_male, 'Male')

    data_female = pd.read_csv(ROOT_DIR + '/data/Life_table_female_2019_USA.csv', skiprows=2, skipfooter=5)
    processed_data_female = process_life_expectancy_data(data_female, 'Female')

    # Combine male and female datasets
    combined_data = pd.concat([processed_data_male, processed_data_female])

    # Group by 'Age group' and 'Sex', and calculate the mean for 'Expectation of life at age x'
    average_expectation = combined_data.groupby(['Age group', 'Sex'])['Expectation of life at age x'].mean()

    # Reshape the data to have 'Age group' as index and 'Sex' as columns
    reshaped_data = average_expectation.unstack(level='Sex')
    reshaped_data.columns = ['Avg LE Male', 'Avg LE Female']

    # Save the reshaped data to CSV
    output_file_path = ROOT_DIR + '/csv_files/average_expected_years_of_life.csv'
    reshaped_data.reset_index().to_csv(output_file_path, index=False)

    # Print the average expectation of life for each age group and sex
    print("Average Expected Years of Life:")
    print(average_expectation)


    return combined_data, average_expectation
def generate_combined_data_with_prop_deaths():
    # Generate life expectancy data
    combined_data, average_expectation = generate_combined_life_expectancy()

    # Generate proportion of deaths data
    generate_prop_deaths_by_age_group_and_sex()

    # Read life expectancy data
    life_expectancy_data = pd.read_csv(ROOT_DIR + '/csv_files/average_expected_years_of_life.csv')

    # Read proportion of deaths data
    prop_deaths_data = pd.read_csv(ROOT_DIR + '/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/prop_deaths_by_age_and_sex.csv')


    # Merge life expectancy and proportion of deaths data
    merged_data = pd.merge(life_expectancy_data, prop_deaths_data, on=['Age group'], how='inner')

    # Save the combined data to CSV
    combined_output_file_path = ROOT_DIR + '/csv_files/combined_data_with_prop_deaths.csv'
    merged_data.to_csv(combined_output_file_path, index=False)

    # Sort the data by life expectancy
    sorted_data = merged_data.sort_values(by='Avg LE Male')

    # Extract arrays for life expectancy and proportion of deaths
    life_expectancy_array = sorted_data[['Avg LE Male', 'Avg LE Female']].to_numpy().flatten()
    prop_deaths_array = sorted_data[['Males: Proportion of Deaths', 'Females: Proportion of Deaths']].to_numpy().flatten()

    return life_expectancy_array, prop_deaths_array



