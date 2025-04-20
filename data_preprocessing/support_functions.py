

from collections import defaultdict
import os
import numpy as np
import pandas as pd

from deampy.in_out_functions import write_csv, read_csv_rows
from definitions import ROOT_DIR
from datetime import datetime
from scipy.stats import pearsonr


def get_dict_of_county_data_by_type(data_type):
    """
    This function reads the county data CSV file and returns a dictionary of county data by the specified data type.
    :param data_type: The data type to extract ('cases', 'deaths', 'hospitalizations', 'icu admissions', etc.)
    :return: (dictionary, list) a dictionary with (county, state) as keys and a list of data values as values,
            and a list of dates
    """

    # Construct the file path based on the data type
    file_path = ROOT_DIR + f'/csv_files/county_{data_type.replace(" ", "_")}.csv'

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
        data_values = [float(data) if data not in ['NA', ''] else np.nan for data in data_values]

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
        'icu': 23,
        'cases per 100,000': 17,
        'deaths per 100,000': 18,
        'hospitalizations per 100,000': 22,
        'icu occupancy per 100,000': 24,
        'longcovid': (5, 6)
    }

    # Ensure the specified data_type is valid
    if data_type not in data_type_mapping:
        raise ValueError(
            "Invalid data_type. Choose from 'cases', 'deaths', 'hospitalizations', "
            "'icu', 'cases per 100,000', "
            "'deaths per 100,000', 'hospitalizations per 100,000', 'icu occupancy per 100,000'.")

    # Read the data
    rows = read_csv_rows(ROOT_DIR + '/Data/county_time_data_all_dates.csv',
                         if_ignore_first_row=True)


    # Creating a dictionary to store the time series of data for each county
    county_data_time_series = defaultdict(list)

    for row in rows:
        fips = row[1]
        county = row[3]
        state = row[12]  # State abbreviation
        date_str = row[2]
        population = row[10]  # Add population to this section

        # Convert the date string to a datetime object for comparison
        date = datetime.strptime(date_str, "%Y-%m-%d")

        # Check if the date is within the desired range
        #if start_date <= date <= end_date:
        if data_type == 'longcovid':
            cases_index, deaths_index = data_type_mapping[data_type]
            cases = row[data_type_mapping['cases']]
            deaths = row[data_type_mapping['deaths']]

                # Removing PR from analysis
            if state == 'NA':
                cases = np.nan
                deaths = np.nan

                # Check if data_value is empty or 'NA' and assign np.nan
            if cases == '' or cases == 'NA':
                cases = np.nan
            else:
                cases = float(cases) * 7

            if deaths == '' or deaths == 'NA':
                deaths = np.nan
            else:
                deaths = float(deaths) * 7

            longcovid = cases - deaths

                # Append the data to the respective county's time series
            county_data_time_series[(county, state, fips, population)].append((date_str, longcovid))

        else:
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

    # Create a list of unique dates across all counties within the specified range
    unique_dates = sorted(set(date for time_series in county_data_time_series.values() for date, _ in time_series))


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

    new_fips_values = [{'County': 'Cass', 'State': 'MO', 'NewFIPS': '29037'},
                       {'County': 'Clay', 'State': 'MO', 'NewFIPS': '29047'},
                       {'County': 'Jackson', 'State': 'MO', 'NewFIPS': '29095'},
                       {'County': 'Platte', 'State': 'MO', 'NewFIPS': '29165'},
                       {'County': 'Kansas City', 'State': 'MO', 'NewFIPS': '29025'},
                       {'County': 'Yakutat plus Hoonah-Angoon', 'State': 'AK', 'NewFIPS': '2282'},
                       {'County': 'Bristol Bay plus Lake and Peninsula', 'State': 'AK', 'NewFIPS': '36061'},
                       {'County': 'Joplin', 'State': 'MO', 'NewFIPS': '29011'},
                       {'County': 'New York City', 'State': 'NY', 'NewFIPS': '36061'}]

    for county_update in new_fips_values:
        county_name = county_update['County']
        state_name = county_update['State']
        new_fips = county_update['NewFIPS']

        # Update FIPS values for the specified county and state in county_data_rows
        for i, row in enumerate(county_data_rows):
            if row[0] == county_name and row[1] == state_name:
                county_data_rows[i][2] = new_fips  # Update the FIPS value

    # Create the output file name
    output_file = ROOT_DIR +f'/csv_files/county_{data_type.replace(" ", "_")}.csv'

    write_csv(rows=[header_row] + county_data_rows, file_name=output_file)


def generate_deaths_by_age_group():
    """
    This function generates a csv containing information on the number of deaths associated with each age group.
    A crucial step in this process is redefining the age bands to match the dQALY age groups in the Briggs paper.
    Calculation for the number of deaths in each age band are based on Briggs spreadsheet tool

    :return: A csv of COVID-19 deaths by age group.
    """

    data = pd.read_csv(ROOT_DIR + '/data/data_deaths/Provisional_COVID-19_Deaths_by_Sex_and_Age.csv')

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




def generate_hsa_mapped_county_hosp_data():
    # Load county hosp data
    county_hosp_data = pd.read_csv(ROOT_DIR + '/csv_files/county_hospitalizations.csv',skiprows=0)

    # Load HSA data
    hsa_data = pd.read_csv(ROOT_DIR + '/Data/county_names_HSA_number.csv', skiprows=0)


    # Ensure the FIPS column has the same data type in both dataframes
    county_hosp_data['FIPS'] = county_hosp_data['FIPS'].astype(str)
    hsa_data['county_fips'] = hsa_data['county_fips'].astype(str)

    # Define new FIPS values for specific counties
    new_fips_values = [
        {'County': 'Cass', 'State': 'MO', 'NewFIPS': '29037'},
        {'County': 'Clay', 'State': 'MO', 'NewFIPS': '29047'},
        {'County': 'Jackson', 'State': 'MO', 'NewFIPS': '29095'},
        {'County': 'Platte', 'State': 'MO', 'NewFIPS': '29165'},
        {'County': 'Kansas City', 'State': 'MO', 'NewFIPS': '29025'},
        {'County': 'Yakutat plus Hoonah-Angoon', 'State': 'AK', 'NewFIPS': '2282'},
        {'County': 'Bristol Bay plus Lake and Peninsula', 'State': 'AK', 'NewFIPS': '36061'},
        {'County': 'Joplin', 'State': 'MO', 'NewFIPS': '29011'},
        {'County': 'New York County', 'State': 'NY', 'NewFIPS': '36061'}
    ]

    # Update FIPS values for the specified counties
    for county_update in new_fips_values:
        county_name = county_update['County']
        state_name = county_update['State']
        new_fips = county_update['NewFIPS']

        # Update FIPS values for the specified county and state in county_hosp_data
        condition = (county_hosp_data['County'] == county_name) & (county_hosp_data['State'] == state_name)
        county_hosp_data.loc[condition, 'FIPS'] = new_fips


    # Merge county hosp data with HSA data based on FIPS
    merged_data = pd.merge(county_hosp_data, hsa_data, left_on='FIPS', right_on='county_fips', how='left')

    # Extract the necessary columns for computation
    selected_columns = ['County', 'State', 'FIPS', 'Population', 'health_service_area_number', 'health_service_area_population']
    selected_columns += county_hosp_data.columns[4:].to_list()  # Add the date columns

    merged_data = merged_data[selected_columns]

    # Convert 'Population' and 'health_service_area_population' columns to numeric
    merged_data['Population'] = pd.to_numeric(merged_data['Population'], errors='coerce')
    merged_data['health_service_area_population'] = pd.to_numeric(merged_data['health_service_area_population'].str.replace(',', ''), errors='coerce')

    # Calculate the Population Proportion
    merged_data['Population Proportion'] = merged_data['Population'] / merged_data['health_service_area_population']

    adjusted_weekly_hosp_values = None  # Initialize the variable

    if (merged_data[county_hosp_data.columns[4:]] == '').any().any():
        # Replace empty strings with NaN
        merged_data[county_hosp_data.columns[4:]] = merged_data[county_hosp_data.columns[4:]].replace('', np.nan)
    else:
        adjusted_weekly_hosp_values = merged_data[county_hosp_data.columns[4:]] * merged_data['Population Proportion'].values[:, None]

    # Create a new dataframe with adjusted values
    adjusted_data = pd.concat([merged_data[['County', 'State', 'FIPS', 'Population']], adjusted_weekly_hosp_values], axis=1)

    # Replace NaN values with 'nan' to match the behavior in generate_county_data
    adjusted_data = adjusted_data.where(pd.notna(adjusted_data), 'nan')

    # Save the adjusted data to a new CSV
    adjusted_data.to_csv(ROOT_DIR + '/csv_files/county_hospitalizations.csv', index=False)

    print("Hospitalization data has been updated based on HSA")

def generate_hsa_mapped_county_icu_data():
    # Load county ICU data
    county_icu_data = pd.read_csv(ROOT_DIR + '/csv_files/county_icu.csv', skiprows=0)

    # Load HSA data
    hsa_data = pd.read_csv(ROOT_DIR + '/Data/county_names_HSA_number.csv', skiprows=0)

    # Ensure the FIPS column has the same data type in both dataframes
    county_icu_data['FIPS'] = county_icu_data['FIPS'].astype(str)
    hsa_data['county_fips'] = hsa_data['county_fips'].astype(str)

    # Define new FIPS values for specific counties
    new_fips_values = [
        {'County': 'Cass', 'State': 'MO', 'NewFIPS': '29037'},
        {'County': 'Clay', 'State': 'MO', 'NewFIPS': '29047'},
        {'County': 'Jackson', 'State': 'MO', 'NewFIPS': '29095'},
        {'County': 'Platte', 'State': 'MO', 'NewFIPS': '29165'},
        {'County': 'Kansas City', 'State': 'MO', 'NewFIPS': '29025'},
        {'County': 'Yakutat plus Hoonah-Angoon', 'State': 'AK', 'NewFIPS': '2282'},
        {'County': 'Bristol Bay plus Lake and Peninsula', 'State': 'AK', 'NewFIPS': '36061'},
        {'County': 'Joplin', 'State': 'MO', 'NewFIPS': '29011'},
        {'County': 'New York County', 'State': 'NY', 'NewFIPS': '36061'}
    ]

    # Update FIPS values for the specified counties
    for county_update in new_fips_values:
        county_name = county_update['County']
        state_name = county_update['State']
        new_fips = county_update['NewFIPS']

        # Update FIPS values for the specified county and state in county ICU data
        condition = (county_icu_data['County'] == county_name) & (county_icu_data['State'] == state_name)
        county_icu_data.loc[condition, 'FIPS'] = new_fips

    # Merge county ICU data with HSA data based on FIPS
    merged_data = pd.merge(county_icu_data, hsa_data, left_on='FIPS', right_on='county_fips', how='left')

    # Extract the necessary columns for computation
    selected_columns = ['County', 'State', 'FIPS', 'Population', 'health_service_area_number', 'health_service_area_population']
    selected_columns += county_icu_data.columns[4:].to_list()  # Add the date columns

    merged_data = merged_data[selected_columns]

    # Convert 'Population' and 'health_service_area_population' columns to numeric
    merged_data['Population'] = pd.to_numeric(merged_data['Population'], errors='coerce')
    merged_data['health_service_area_population'] = pd.to_numeric(merged_data['health_service_area_population'].str.replace(',', ''), errors='coerce')

    # Calculate the Population Proportion
    merged_data['Population Proportion'] = merged_data['Population'] / merged_data['health_service_area_population']

    adjusted_weekly_hosp_values = None  # Initialize the variable

    if (merged_data[county_icu_data.columns[4:]] == '').any().any():
        # Replace empty strings with NaN
        merged_data[county_icu_data.columns[4:]] = merged_data[county_icu_data.columns[4:]].replace('', np.nan)
    else:
        adjusted_weekly_hosp_values = merged_data[county_icu_data.columns[4:]] * merged_data['Population Proportion'].values[:, None]

    # Create a new dataframe with adjusted values
    adjusted_data = pd.concat([merged_data[['County', 'State', 'FIPS', 'Population']], adjusted_weekly_hosp_values], axis=1)

    # Replace NaN values with 'nan' to match the behavior in generate_county_data
    adjusted_data = adjusted_data.where(pd.notna(adjusted_data), 'nan')

    # Save the adjusted data to a new CSV
    adjusted_data.to_csv(ROOT_DIR + '/csv_files/county_icu.csv', index=False)

    print("ICU data has been updated based on HSA")


def generate_hosps_by_age_group():
    """
    This function generates a csv containing information on the number of deaths associated with each age group.
    A crucial step in this process is redefining the age bands to match the dQALY age groups in the Briggs paper.
    Calculation for the number of deaths in each age band are based on Briggs spreadsheet tool

    :return: A csv of COVID-19 hosps by age group.
    """

    data = pd.read_csv(ROOT_DIR + '/Data/HHS_COVID_Reported_Hospital_ICU_Capacity.csv')

    age_band_mapping = {
        '0-9': ['previous_day_admission_pediatric_covid_confirmed_0_4', 'previous_day_admission_pediatric_covid_confirmed_5_11'],
        '10-19': ['previous_day_admissions_pediatric_covid_confirmed_5-11', 'previous_day_admission_pediatric_covid_confirmed_12_17','previous_day_admission_adult_covid_confirmed_18-19'],
        '20-29': ['previous_day_admission_adult_covid_confirmed_20-29'],
        '30-39': ['previous_day_admission_adult_covid_confirmed_30-39'],
        '40-49': ['previous_day_admission_adult_covid_confirmed_40-49'],
        '50-59': ['previous_day_admission_adult_covid_confirmed_50-59'],
        '60-69': ['previous_day_admission_adult_covid_confirmed_60-69'],
        '70-79': ['previous_day_admission_adult_covid_confirmed_70-79'],
        '80-90': ['previous_day_admission_adult_covid_confirmed_80+'],
        '90-100': ['previous_day_admission_adult_covid_confirmed_80+']
    }


    new_age_data = {'Age Group': [], 'COVID-19 Hosps': []}

    for age_band, age_groups in age_band_mapping.items():
        total_hosps = 0

        for age_group in age_groups:
            # Check if the columns exist in the DataFrame
            if age_group in data.columns:
                total_hosps += data.groupby(['date'])[age_group].sum().sum()

        if age_band == '0-9':
            # Sum 'previous_day_admission_pediatric_covid_confirmed_0_4' and 'previous_day_admission_pediatric_covid_confirmed_5_11'
            total_hosps = data.groupby(['date'])[['previous_day_admission_pediatric_covid_confirmed_0_4', 'previous_day_admission_pediatric_covid_confirmed_5_11']].sum().sum().sum()
            # Add 2/3 of 'previous_day_admission_pediatric_covid_confirmed_5_11'
            total_hosps += (2 / 3) * data.groupby(['date'])['previous_day_admission_pediatric_covid_confirmed_5_11'].sum().sum()
        elif age_band == '10-19':
            # Sum 'previous_day_admission_pediatric_covid_confirmed_12_17', 'previous_day_admission_adult_covid_confirmed_18-19'
            total_hosps = data.groupby(['date'])[['previous_day_admission_pediatric_covid_confirmed_12_17', 'previous_day_admission_adult_covid_confirmed_18-19']].sum().sum().sum()
            # Add 2/3 of 'previous_day_admission_pediatric_covid_confirmed_5_11'
            total_hosps += (2 / 3) * data.groupby(['date'])['previous_day_admission_pediatric_covid_confirmed_5_11'].sum().sum()
        elif age_band =='80-90':
            total_hosps= (1/2)*data.groupby(['date'])['previous_day_admission_adult_covid_confirmed_80+'].sum().sum()
        elif age_band =='90-100':
            total_hosps= (1/2)*data.groupby(['date'])['previous_day_admission_adult_covid_confirmed_80+'].sum().sum()
        else:
            total_hosps = data.groupby(['date'])[age_groups].sum().sum().sum()

        new_age_data['Age Group'].append(age_band)
        new_age_data['COVID-19 Hosps'].append(total_hosps)

    # Create a DataFrame for the new age bands
    new_age_df = pd.DataFrame(new_age_data)
    new_age_df['COVID-19 Hosps'] = pd.to_numeric(new_age_df['COVID-19 Hosps'], errors='coerce').fillna(0)

    # Select relevant columns
    hosps_by_age_group = new_age_df[['Age Group', 'COVID-19 Hosps']]

    # Save the data as a CSV file
    hosps_by_age_group.to_csv(ROOT_DIR + '/csv_files/hosps_by_age.csv', index=False)


def generate_county_info_csv():
    """
    Generates a CSV containing county information (County, State, FIPS, Population) from a source file.
    """

    # Read the county data CSV file
    rows = read_csv_rows(ROOT_DIR + '/Data/county_time_data_all_dates.csv',
                         if_ignore_first_row=True)


    # Initialize a list to store county information
    county_info_list = []

    for row in rows:
        fips = row[1]
        county = row[3]
        state = row[12]  # State abbreviation
        population = row[10]
        date_str = row[2]

        date = datetime.strptime(date_str, "%Y-%m-%d")

        # Remove rows where state is 'NA' or 'PR'
        if state == 'NA' or state == 'PR':
            continue

        # Correct the county name for Doña Ana
        if county == 'DoÃ±a Ana':
            county = 'Doña Ana'

        # Append county information to the list
        county_info_list.append((county, state, fips, population))

    # Create a DataFrame for county information
    county_info_df = pd.DataFrame(county_info_list, columns=['County', 'State', 'FIPS', 'Population'])

    # Ensure counties are not repeated by removing duplicates based on County, State, and FIPS
    county_info_df.drop_duplicates(subset=['County', 'State', 'FIPS'], inplace=True)

    # Define the list of new FIPS values for specific counties
    new_fips_values = [
        {'County': 'Cass', 'State': 'MO', 'NewFIPS': '29037'},
        {'County': 'Clay', 'State': 'MO', 'NewFIPS': '29047'},
        {'County': 'Jackson', 'State': 'MO', 'NewFIPS': '29095'},
        {'County': 'Platte', 'State': 'MO', 'NewFIPS': '29165'},
        {'County': 'Kansas City', 'State': 'MO', 'NewFIPS': '29025'},
        {'County': 'Yakutat plus Hoonah-Angoon', 'State': 'AK', 'NewFIPS': '2282'},
        {'County': 'Bristol Bay plus Lake and Peninsula', 'State': 'AK', 'NewFIPS': '36061'},
        {'County': 'Joplin', 'State': 'MO', 'NewFIPS': '29011'},
        {'County': 'New York City', 'State': 'NY', 'NewFIPS': '36061'}
    ]

    # Update FIPS values for the specified counties
    for county_update in new_fips_values:
        county_name = county_update['County']
        state_name = county_update['State']
        new_fips = county_update['NewFIPS']

        # Update FIPS values for the specified county and state in county_info_df
        condition = (county_info_df['County'] == county_name) & (county_info_df['State'] == state_name)
        county_info_df.loc[condition, 'FIPS'] = new_fips

    # Save county information to CSV
    output_path = ROOT_DIR + '/csv_files/county_info.csv'
    county_info_df.to_csv(output_path, index=False)

    return county_info_df


def distribute_infections_in_counties():
    # Load the state infections data (with dates as columns)
    state_infections_df = pd.read_csv(ROOT_DIR +'/Data/covidestim_state_infections.csv')


    # Load the county information data
    county_info_df = pd.read_csv(
        ROOT_DIR + '/csv_files/county_info.csv')

    # Count the number of counties per state in county_info_df
    counties_per_state = county_info_df.groupby('State').size().reset_index(name='county_count')

    # Convert counties_per_state DataFrame to a dictionary for quick lookup
    counties_per_state_dict = counties_per_state.set_index('State')['county_count'].to_dict()

    # Loop through each row of the state_infections_df to divide the infection values by the number of counties
    for index, row in state_infections_df.iterrows():
        state = row['state']  # Assuming the state name is in the 'state' column

        # If the state has a matching entry in counties_per_state_dict
        if state in counties_per_state_dict:
            num_counties = counties_per_state_dict[state]

            # Divide each infection value by the number of counties for the corresponding state
            state_infections_df.loc[index, state_infections_df.columns[1:]] = row[state_infections_df.columns[1:]] / num_counties

    return state_infections_df

def generate_state_divided_county_infections_csv():
    """
    Generates a CSV containing county-level infections by assigning the state-level
    infection value to all counties for each time point using efficient DataFrame operations.
    The output format will be: County, State, FIPS, Population, followed by columns of infection data for each date.
    """

    state_infections_df = distribute_infections_in_counties()

    # Load the county information data
    county_info_df = pd.read_csv(
        ROOT_DIR + '/csv_files/county_info.csv')

    if county_info_df.isnull().any().any():
        print("Missing values in county information data:")
        print(county_info_df.isnull().sum())

    # Merge state infection data with county info using the state as the common key
    county_infections_df = county_info_df.merge(
        state_infections_df, left_on='State', right_on='state', how='left'
    )

    # Drop redundant state column from the result
    county_infections_df.drop(columns=['state'], inplace=True)


    # Save the county-level infection data to a CSV
    output_path = ROOT_DIR + '/csv_files/state_divided_county_infections.csv'
    county_infections_df.to_csv(output_path, index=False)



def generate_state_cases_infections_factor():
    # Read the CSV file
    state_divided_county_infections = pd.read_csv(ROOT_DIR + '/csv_files/state_divided_county_infections.csv')
    county_cases = pd.read_csv(ROOT_DIR + '/csv_files/county_cases.csv')



    infections_grouped = state_divided_county_infections.drop(columns=['County', 'FIPS', 'Population'])
    infections_state = infections_grouped.groupby('State').sum()
    cases_grouped = county_cases.drop(columns=['County', 'FIPS', 'Population'])
    cases_state = cases_grouped.groupby('State').sum()

    # Calculate the factor of infections to cases for each state and time point
    factor = infections_state / cases_state

    # Replace infinity or NaN values (where cases are 0) with 1
    factor = factor.replace([float('inf'), float('nan')], 1)
    factor.to_csv(ROOT_DIR + '/csv_files/state_cases_infections_factor', index=True)



def generate_infections_from_cases():
    state_factors = pd.read_csv(ROOT_DIR + '/csv_files/state_cases_infections_factor',index_col='State')

    county_cases = pd.read_csv(ROOT_DIR + '/csv_files/county_cases.csv')

    case_columns = county_cases.columns[4:]  # Assuming first 4 columns are 'County', 'State', 'FIPS', 'Population'
    county_infections_from_cases = county_cases.copy()

    for state in county_infections_from_cases['State'].unique():
        state_factor = state_factors.loc[state]
        state_counties = county_infections_from_cases['State'] == state
        county_infections_from_cases.loc[state_counties, case_columns] = county_cases.loc[
            state_counties, case_columns].multiply(state_factor.values, axis=1)

    # Save the new infections estimate to a CSV file
    county_infections_from_cases.to_csv( ROOT_DIR + '/csv_files/county_infections.csv', index=False)


def generate_symptomatic_infections_vax():
    """
    Generate CSV files for symptomatic infections with lower bound (LB) and upper bound (UB) long COVID estimates.
    """
    # Get county data and dates for symptomatic infections
    county_data_by_type, dates = get_dict_of_county_data_by_type('infections')

    # Define parameters for the sigmoid function
    L_UB = 0.8  # Upper bound cap
    L_LB = 0.5  # Lower bound cap
    k = 0.3
    # Find the index of the date
    all_dates = np.array(dates, dtype=str)
    x0 = np.where(all_dates == "2021-08-04")[0][0]

    weeks = np.arange(len(dates))

    # Compute sigmoid values for both lower and upper bounds
    sigmoid_values_LB = L_LB / (1 + np.exp(-k * (weeks - x0)))
    sigmoid_values_UB = L_UB / (1 + np.exp(-k * (weeks - x0)))

    # Prepare rows for CSV output
    county_data_rows_v_LB = []
    county_data_rows_uv_LB = []
    county_data_rows_v_UB = []
    county_data_rows_uv_UB = []

    for (county, state, fips, population), weekly_obs in county_data_by_type.items():
        # Apply sigmoid transformations to generate long COVID estimates
        weekly_long_covid_v_LB = np.array(weekly_obs) * sigmoid_values_LB
        weekly_long_covid_uv_LB = np.array(weekly_obs) * (1 - sigmoid_values_LB)

        weekly_long_covid_v_UB = np.array(weekly_obs) * sigmoid_values_UB
        weekly_long_covid_uv_UB = np.array(weekly_obs) * (1 - sigmoid_values_UB)

        # Add rows for both LB and UB estimates
        county_data_rows_v_LB.append([county, state, fips, population] + weekly_long_covid_v_LB.tolist())
        county_data_rows_v_UB.append([county, state, fips, population] + weekly_long_covid_v_UB.tolist())
        county_data_rows_uv_LB.append([county, state, fips, population] + weekly_long_covid_uv_LB.tolist())
        county_data_rows_uv_UB.append([county, state, fips, population] + weekly_long_covid_uv_UB.tolist())

    # Define headers with dates
    header_row = ['County', 'State', 'FIPS', 'Population'] + dates

    output_file_v_LB = f'{ROOT_DIR}/csv_files/county_infections_v_LB.csv'
    write_csv(rows=[header_row] + county_data_rows_v_LB, file_name=output_file_v_LB)

    # Write UB data to CSV
    output_file_v_UB = f'{ROOT_DIR}/csv_files/county_infections_v_UB.csv'
    write_csv(rows=[header_row] + county_data_rows_v_UB, file_name=output_file_v_UB)

    # Write LB data to CSV
    output_file_uv_LB = f'{ROOT_DIR}/csv_files/county_infections_uv_LB.csv'
    write_csv(rows=[header_row] + county_data_rows_uv_LB, file_name=output_file_uv_LB)

    # Write UB data to CSV
    output_file_uv_UB = f'{ROOT_DIR}/csv_files/county_infections_uv_UB.csv'
    write_csv(rows=[header_row] + county_data_rows_uv_UB, file_name=output_file_uv_UB)
