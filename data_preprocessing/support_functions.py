

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
        data_values = [float(data) if data not in ['NA', ''] else np.nan for data in data_values]

        county_data_by_type[(county, state, fips, population)] = data_values

    return county_data_by_type, dates
def get_dict_of_county_data_by_type_2(data_type):
    """
    Fetch and process county data by type.
    """
    # Example implementation, adjust as needed for your actual function
    # Assuming you load the data and extract `data_values` for the given type
    # Here, we simulate the data_values for demonstration purposes.
    data_values = ['12.5', '', 'NA', '15.2', 'abc']  # Replace this with actual data loading logic

    processed_values = []
    for i, data in enumerate(data_values):
        try:
            if data != 'NA' and data != '':
                processed_value = float(data)
            else:
                processed_value = np.nan
            processed_values.append(processed_value)
        except ValueError as e:
            print(f"Error converting value '{data}' at index {i}: {e}")
            processed_values.append(np.nan)  # Or handle differently as needed

    # Continue with the rest of your function logic
    # For example, you might return the processed values or use them in further processing
    return processed_values


def generate_county_data_csv(data_type='cases'):
    """
    This function reads the county data CSV file and creates a CSV of county data over time for a specified data type.
    :param data_type: The data type to extract ('cases', 'deaths', 'hospitalizations', 'icu admissions', etc.)
    :return: (.csv file) a CSV file with data per county, over time, where each row corresponds to a county, identified
    by county name, state, fips, and population.
    """
    ROOT_DIR = '/Users/timamikdashi/PycharmProjects/covid19-qaly-loss'

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
    rows = read_csv_rows(file_name='/Users/timamikdashi/Downloads/county_time_data_all_dates.csv',
                         if_ignore_first_row=True)
    #rows = read_csv_rows(file_name='/Users/fm478/Downloads/county_time_data_all_dates.csv',
                         #if_ignore_first_row=True)

    # Creating a dictionary to store the time series of data for each county
    county_data_time_series = defaultdict(list)

    # Define the date range for filtering
    #start_date = datetime(2021, 12, 2)
    #end_date = datetime(2022, 10, 27)

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

    # Generate the output file name based on data_type
    output_file = ROOT_DIR +f'/csv_files/county_{data_type.replace(" ", "_")}.csv'

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
    output_file = f'{ROOT_DIR}/csv_files/county_{data_type.replace(" ", "_")}.csv'

    write_csv(rows=[header_row] + county_data_rows, file_name=output_file)


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




def generate_hsa_mapped_county_hosp_data():
    # Load county hosp data
    county_hosp_data = pd.read_csv(ROOT_DIR + '/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_hospitalizations.csv', skiprows=0)
    #county_hosp_data = pd.read_csv(ROOT_DIR + '/csv_files/county_hospitalizations.csv',skiprows=0)

    # Load HSA data
    hsa_data = pd.read_csv('/Users/timamikdashi/Downloads/county_names_HSA_number.csv', skiprows=0)
    #hsa_data = pd.read_csv('C:/Users/fm478/Downloads/county_names_HSA_number.csv', skiprows=0)
    #hsa_data = pd.read_csv('C:/Users/timamikdashi/Downloads/county_names_HSA_number.csv', skiprows=0)

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

    # Update "New York City" to "New York County" in county hosp data to match HSA data
    #county_hosp_data.loc[(county_hosp_data['County'] == 'New York City') & (county_hosp_data['State'] == 'NY'), 'County'] = 'New York County'

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
    #county_icu_data = pd.read_csv(ROOT_DIR + '/csv_files/county_icu.csv', skiprows=0)
    county_icu_data=pd.read_csv(ROOT_DIR +'/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_icu.csv', skiprows=0)

    # Load HSA data
    #hsa_data = pd.read_csv('C:/Users/fm478/Downloads/county_names_HSA_number.csv', skiprows=0)
    hsa_data = pd.read_csv('/Users/timamikdashi/Downloads/county_names_HSA_number.csv', skiprows=0)

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

    # Update "New York City" to "New York County" in county ICU data to match HSA data
    #county_icu_data.loc[(county_icu_data['County'] == 'New York City') & (county_icu_data['State'] == 'NY'), 'County'] = 'New York County'

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
    adjusted_data.to_csv(ROOT_DIR + f'/csv_files/county_icu.csv', index=False)

    print("ICU data has been updated based on HSA")


def generate_hosps_by_age_group():
    """
    This function generates a csv containing information on the number of deaths associated with each age group.
    A crucial step in this process is redefining the age bands to match the dQALY age groups in the Briggs paper.
    Calculation for the number of deaths in each age band are based on Briggs spreadsheet tool

    :return: A csv of COVID-19 hosps by age group.
    """

    #data = pd.read_csv('C:/Users/fm478/Downloads/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW__20240307 (1).csv')
    data = pd.read_csv('/Users/timamikdashi/Downloads/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW__20240307 (1).csv')

    #deaths_by_age = data.groupby(['state','date']).sum().reset_index() #TODO" A REVOIR TO ENSURE THAT THE data is aggregated over state and dates

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

'''IN USE BUT DESKTOP FORMATTING 

def generate_county_infections_data_extended():



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


def generate_hsa_mapped_county_hosp_data():
    # Load county hosp data

    county_hosp_data = pd.read_csv(ROOT_DIR+'/csv_files/county_hospitalizations.csv', skiprows=0)
    #county_hosp_data=pd.read_csv(ROOT_DIR +'/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_hospitalizations.csv',skiprows=0)

    # Load HSA data
    hsa_data = pd.read_csv('C:/Users/fm478/Downloads/county_names_HSA_number.csv', skiprows=0)
    #hsa_data=pd.read_csv('/Users/timamikdashi/Downloads/county_names_HSA_number.csv',skiprows=0)

    # Ensure the FIPS column has the same data type in both dataframes
    county_hosp_data['FIPS'] = county_hosp_data['FIPS'].astype(str)
    hsa_data['county_fips'] = hsa_data['county_fips'].astype(str)

    new_fips_values = [{'County': 'Cass', 'State': 'MO', 'NewFIPS': '29037'},
                       {'County': 'Clay', 'State': 'MO', 'NewFIPS': '29047'},
                       {'County': 'Jackson', 'State': 'MO', 'NewFIPS': '29095'},
                       {'County': 'Platte', 'State': 'MO', 'NewFIPS': '29165'},
                       {'County': 'Kansas City', 'State': 'MO', 'NewFIPS': '29025'},
                       {'County': 'Yakutat plus Hoonah-Angoon', 'State': 'AK', 'NewFIPS': '2282'},
                       {'County': 'Bristol Bay plus Lake and Peninsula', 'State': 'AK', 'NewFIPS': '36061'},
                       {'County': 'Joplin', 'State': 'MO', 'NewFIPS': '29011'}]


    for county_update in new_fips_values:
        county_name = county_update['County']
        state_name = county_update['State']
        new_fips = county_update['NewFIPS']

        # Update FIPS values for the specified county and state in county_hosp_data
        condition = (county_hosp_data['County'] == county_name) & (county_hosp_data['State'] == state_name)
        county_hosp_data.loc[condition, 'FIPS'] = new_fips

    # Merge county_hosp_data with hsa_data based on FIPS
    merged_data = pd.merge(county_hosp_data, hsa_data, left_on='FIPS', right_on='county_fips', how='left')

    # Extract the necessary columns for computation
    selected_columns = ['County', 'State', 'FIPS', 'Population', 'health_service_area_number',
                        'health_service_area_population']
    selected_columns += county_hosp_data.columns[4:].to_list()  # Add the date columns

    merged_data = merged_data[selected_columns]

    # Convert 'Population' and 'health_service_area_population' columns to numeric
    merged_data['Population'] = pd.to_numeric(merged_data['Population'], errors='coerce')
    merged_data['health_service_area_population'] = pd.to_numeric(
        merged_data['health_service_area_population'].str.replace(',', ''), errors='coerce')

    # Calculate the Population Proportion
    merged_data['Population Proportion'] = merged_data['Population']/ merged_data['health_service_area_population']

    adjusted_weekly_hosp_values = None  # Initialize the variable

    if (merged_data[county_hosp_data.columns[4:]] == '').any().any():
        # Replace empty strings with NaN
        merged_data[county_hosp_data.columns[4:]] = merged_data[county_hosp_data.columns[4:]].replace('', np.nan)

    else:
        adjusted_weekly_hosp_values = merged_data[county_hosp_data.columns[4:]] * merged_data[
            'Population Proportion'].values[:, None]

    # Create a new dataframe with adjusted values
    adjusted_data = pd.concat([merged_data[['County', 'State', 'FIPS', 'Population']], adjusted_weekly_hosp_values],
                              axis=1)

    # Replace NaN values with 'nan' to match the behavior in generate_county_data
    adjusted_data = adjusted_data.where(pd.notna(adjusted_data), 'nan')

    # Save the adjusted data to a new CSV
    adjusted_data.to_csv(ROOT_DIR + '/csv_files/county_hospitalizations.csv', index=False)

    print("Hospitalization data has been updated to based on HSA")

def generate_hsa_mapped_county_icu_data():
    # Load county hosp data

    county_icu_data = pd.read_csv(ROOT_DIR+'/csv_files/county_icu.csv', skiprows=0)
    #county_icu_data=pd.read_csv(ROOT_DIR +'/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_icu.csv',skiprows=0)

    # Load HSA data
    hsa_data = pd.read_csv('C:/Users/fm478/Downloads/county_names_HSA_number.csv', skiprows=0)
    #hsa_data=pd.read_csv('/Users/timamikdashi/Downloads/county_names_HSA_number.csv',skiprows=0)

    # Ensure the FIPS column has the same data type in both dataframes
    county_icu_data['FIPS'] = county_icu_data['FIPS'].astype(str)
    hsa_data['county_fips'] = hsa_data['county_fips'].astype(str)

    new_fips_values = [{'County': 'Cass', 'State': 'MO', 'NewFIPS': '29037'},
                       {'County': 'Clay', 'State': 'MO', 'NewFIPS': '29047'},
                       {'County': 'Jackson', 'State': 'MO', 'NewFIPS': '29095'},
                       {'County': 'Platte', 'State': 'MO', 'NewFIPS': '29165'},
                       {'County': 'Kansas City', 'State': 'MO', 'NewFIPS': '29025'},
                       {'County': 'Yakutat plus Hoonah-Angoon', 'State': 'AK', 'NewFIPS': '2282'},
                       {'County': 'Bristol Bay plus Lake and Peninsula', 'State': 'AK', 'NewFIPS': '36061'},
                       {'County': 'Joplin', 'State': 'MO', 'NewFIPS': '29011'}]


    for county_update in new_fips_values:
        county_name = county_update['County']
        state_name = county_update['State']
        new_fips = county_update['NewFIPS']

        # Update FIPS values for the specified county and state in county_hosp_data
        condition = (county_icu_data['County'] == county_name) & (county_icu_data['State'] == state_name)
        county_icu_data.loc[condition, 'FIPS'] = new_fips

    # Merge county_hosp_data with hsa_data based on FIPS
    merged_data = pd.merge(county_icu_data, hsa_data, left_on='FIPS', right_on='county_fips', how='left')

    # Extract the necessary columns for computation
    selected_columns = ['County', 'State', 'FIPS', 'Population', 'health_service_area_number',
                        'health_service_area_population']
    selected_columns += county_icu_data.columns[4:].to_list()  # Add the date columns

    merged_data = merged_data[selected_columns]

    # Convert 'Population' and 'health_service_area_population' columns to numeric
    merged_data['Population'] = pd.to_numeric(merged_data['Population'], errors='coerce')
    merged_data['health_service_area_population'] = pd.to_numeric(
        merged_data['health_service_area_population'].str.replace(',', ''), errors='coerce')

    # Calculate the Population Proportion
    merged_data['Population Proportion'] = merged_data['Population']/ merged_data['health_service_area_population']

    adjusted_weekly_hosp_values = None  # Initialize the variable

    if (merged_data[county_icu_data.columns[4:]] == '').any().any():
        # Replace empty strings with NaN
        merged_data[county_icu_data.columns[4:]] = merged_data[county_icu_data.columns[4:]].replace('', np.nan)

    else:
        adjusted_weekly_hosp_values = merged_data[county_icu_data.columns[4:]] * merged_data[
            'Population Proportion'].values[:, None]

    # Create a new dataframe with adjusted values
    adjusted_data = pd.concat([merged_data[['County', 'State', 'FIPS', 'Population']], adjusted_weekly_hosp_values],
                              axis=1)

    # Replace NaN values with 'nan' to match the behavior in generate_county_data
    adjusted_data = adjusted_data.where(pd.notna(adjusted_data), 'nan')

    # Save the adjusted data to a new CSV
    adjusted_data.to_csv(ROOT_DIR + '/csv_files/county_icu.csv', index=False)

    print("ICU data has been updated to based on HSA")
'''

'''
FUNCTIONS NOT IN USE IN CURRENT CODE 
def generate_hosps_by_age_group():
    """
    This function generates a csv containing information on the number of deaths associated with each age group.
    A crucial step in this process is redefining the age bands to match the dQALY age groups in the Briggs paper.
    Calculation for the number of deaths in each age band are based on Briggs spreadsheet tool

    :return: A csv of COVID-19 hosps by age group.
    """

    data = pd.read_csv('C:/Users/fm478/Downloads/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW__20240307 (1).csv')
    #data = pd.read_csv('/Users/timamikdashi/Downloads/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW__20240307 (1).csv')

    #deaths_by_age = data.groupby(['state','date']).sum().reset_index() #TODO" A REVOIR TO ENSURE THAT THE data is aggregated over state and dates

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


def generate_cases_by_age_group():
    """
    This function generates a csv containing information on the number of cases associated with each age group.
    The underlying data used is a CDC spreadsheet of Rates of COVID-19 Cases or Deaths by Age Group and Vaccination Status

    :return: A csv of COVID-19 cases by age group.
    """

    # Read the data from CSV file
    cases_by_age = pd.read_csv(
        'C:/Users/fm478/Downloads/Rates_of_COVID-19_Cases_or_Deaths_by_Age_Group_and_Vaccination_Status_20240417 (1).csv')

    # Group by Age Group and sum the values
    cases_by_age = cases_by_age.groupby(['Age group']).sum().reset_index()

    # Define age band mapping
    age_band_mapping = {
        '0-9': ['11-May'],
        '10-19': ['17-Dec', '18-29'],
        '20-29': ['18-29'],
        '30-39': ['30-49'],
        '40-49': ['30-49'],
        '50-59': ['50-64'],
        '60-69': ['50-64', '65-79'],
        '70-79': ['65-79'],
        '80-90': ['80+'],
        '90-100': ['80+']
    }

    # Initialize a dictionary to store new age data
    new_age_data = {'Age Group': [], 'COVID-19 Cases': []}

    # Iterate over age band mapping
    for age_band, age_groups in age_band_mapping.items():
        total_cases = 0

        # Iterate over age groups in each band
        for group in age_groups:
            if group in cases_by_age['Age group'].values:
                # Sum vaccinated and unvaccinated cases for each age group
                total_cases += cases_by_age.loc[
                    (cases_by_age['Age group'] == group), 'Vaccinated with outcome'].sum()
                total_cases += cases_by_age.loc[
                    (cases_by_age['Age group'] == group), 'Unvaccinated with outcome'].sum()

        # Handle specific calculations for each age band
        if age_band == '0-9':
            # Specific calculation for age band 0-9
            total_cases += (8 / 10) * cases_by_age.loc[
                (cases_by_age['Age group'] == '11-May'), 'Vaccinated with outcome'].sum()
            total_cases += (8 / 10) * cases_by_age.loc[
                (cases_by_age['Age group'] == '11-May'), 'Unvaccinated with outcome'].sum()
        elif age_band == '10-19':
            # Specific calculation for age band 10-19
            total_cases += (8 / 10) * cases_by_age.loc[
                (cases_by_age['Age group'] == '17-Dec'), 'Vaccinated with outcome'].sum()
            total_cases += (8 / 10) * cases_by_age.loc[
                (cases_by_age['Age group'] == '18-29'), 'Vaccinated with outcome'].sum()
            total_cases += (8 / 10) * cases_by_age.loc[
                (cases_by_age['Age group'] == '17-Dec'), 'Unvaccinated with outcome'].sum()
            total_cases += (8 / 10) * cases_by_age.loc[
                (cases_by_age['Age group'] == '18-29'), 'Unvaccinated with outcome'].sum()
        elif age_band == '20-29':
            # Specific calculation for age band 20-29
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '18-29'), 'Vaccinated with outcome'].sum()
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '18-29'), 'Unvaccinated with outcome'].sum()
        elif age_band == '30-39':
            # Specific calculation for age band 30-39
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '30-49'), 'Vaccinated with outcome'].sum()
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '30-49'), 'Unvaccinated with outcome'].sum()
        elif age_band == '40-49':
            # Specific calculation for age band 40-49
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '30-49'), 'Vaccinated with outcome'].sum()
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '30-49'), 'Unvaccinated with outcome'].sum()
        elif age_band == '50-59':
            # Specific calculation for age band 50-59
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '50-64'), 'Vaccinated with outcome'].sum()
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '50-64'), 'Unvaccinated with outcome'].sum()
        elif age_band == '60-69':
            # Specific calculation for age band 60-69
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '50-64'), 'Vaccinated with outcome'].sum()
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '50-64'), 'Vaccinated with outcome'].sum()
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '65-79'), 'Unvaccinated with outcome'].sum()
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '65-79'), 'Unvaccinated with outcome'].sum()
        elif age_band == '70-79':
            # Specific calculation for age band 70-79
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '65-79'), 'Vaccinated with outcome'].sum()
            total_cases += cases_by_age.loc[
                (cases_by_age['Age group'] == '65-79'), 'Unvaccinated with outcome'].sum()
        elif age_band == '80-90':
            # Specific calculation for age band 80-90
            total_cases += (1 / 2) * cases_by_age.loc[
                (cases_by_age['Age group'] == '80+'), 'Vaccinated with outcome'].sum()
            total_cases += (1 / 2) * cases_by_age.loc[
                (cases_by_age['Age group'] == '80+'), 'Unvaccinated with outcome'].sum()
        elif age_band == '90-100':
            # Specific calculation for age band 90-100
            total_cases += (1 / 2) * cases_by_age.loc[
                (cases_by_age['Age group'] == '80+'), 'Vaccinated with outcome'].sum()
            total_cases += (1 / 2) * cases_by_age.loc[
                (cases_by_age['Age group'] == '80+'), 'Unvaccinated with outcome'].sum()

        # Append total cases to the new_age_data dictionary
        new_age_data['Age Group'].append(age_band)
        new_age_data['COVID-19 Cases'].append(total_cases)

    # Create a DataFrame from new_age_data
    new_age_df = pd.DataFrame(new_age_data)

    # Select relevant columns
    cases_by_age_group = new_age_df[['Age Group', 'COVID-19 Cases']]

    # Save the data as a CSV file
    cases_by_age_group.to_csv('cases_by_age.csv', index=False)





def generate_correlation_matrix_timeseries():

    cases_data_tuple = get_dict_of_county_data_by_type('cases')
    hosps_data_tuple = get_dict_of_county_data_by_type('hospitalizations')
    deaths_data_tuple = get_dict_of_county_data_by_type('deaths')

    print('cases data', cases_data_tuple)
    print('hosps data', hosps_data_tuple)

    cases_data = cases_data_tuple[0] if cases_data_tuple else {}  # Accessing the dictionary from the tuple
    hosps_data = hosps_data_tuple[0] if hosps_data_tuple else {}  # Accessing the dictionary from the tuple
    deaths_data = deaths_data_tuple[0] if deaths_data_tuple else {}  # Accessing the dictionary from the tuple

    print('cases', cases_data)
    print('hosps', hosps_data)

    # Combine data for all counties into a single dataset
    combined_data = {'Cases': [], 'Deaths': [], 'Hospitalizations': []}
    for key in cases_data:
        cases_total = sum(cases_data[key])
        deaths_total = sum(deaths_data.get(key, [0]))  # Use 0 if no deaths data available
        hospitalizations_total = sum(hosps_data.get(key, [0]))  # Use 0 if no hospitalizations data available
        combined_data['Cases'].append(cases_total)
        combined_data['Deaths'].append(deaths_total)
        combined_data['Hospitalizations'].append(hospitalizations_total)

    # Convert combined data to DataFrame
    combined_df = pd.DataFrame(combined_data)

    # Calculate correlation matrix
    correlation_matrix = combined_df.corr()

    return correlation_matrix



def generate_correlation_matrix_total_per_capita():

    cases_data_tuple = get_dict_of_county_data_by_type('cases')
    hosps_data_tuple = get_dict_of_county_data_by_type('hospitalizations')
    deaths_data_tuple = get_dict_of_county_data_by_type('deaths')

    cases_data = cases_data_tuple[0] if cases_data_tuple else {}
    hosps_data = hosps_data_tuple[0] if hosps_data_tuple else {}
    deaths_data = deaths_data_tuple[0] if deaths_data_tuple else {}

    # Create a dictionary to store total observed values per 100K individuals for each health outcome
    total_outcomes_per_100k_data = {
        "Cases per 100K": [],
        "Hosps per 100K": [],
        "Deaths per 100K": [],
    }

    # Calculate the total observed values per 100K individuals for each health outcome
    for key in cases_data:
        population = int(key[3])  # Assuming population is the fourth element in the key
        cases_total_per_100k = (sum(cases_data[key]) / population) * 100000
        deaths_total_per_100k = (sum(deaths_data.get(key, [0])) / population) * 100000
        hosps_total_per_100k = (sum(hosps_data.get(key, [0])) / population) * 100000
        total_outcomes_per_100k_data['Cases per 100K'].append(cases_total_per_100k)
        total_outcomes_per_100k_data['Deaths per 100K'].append(deaths_total_per_100k)
        total_outcomes_per_100k_data['Hosps per 100K'].append(hosps_total_per_100k)

    # Convert total outcomes per 100K data to DataFrame
    total_outcomes_per_100k_df = pd.DataFrame(total_outcomes_per_100k_data)

    # Calculate correlation matrix for total outcomes per 100K
    correlation_matrix = total_outcomes_per_100k_df.corr()
    print('Per capita correlation matrix', correlation_matrix)

    return correlation_matrix


def generate_correlation_matrix_total():

    cases_data_tuple = get_dict_of_county_data_by_type('cases')
    hosps_data_tuple = get_dict_of_county_data_by_type('hospitalizations')
    deaths_data_tuple = get_dict_of_county_data_by_type('deaths')

    cases_data = cases_data_tuple[0] if cases_data_tuple else {}
    hosps_data = hosps_data_tuple[0] if hosps_data_tuple else {}
    deaths_data = deaths_data_tuple[0] if deaths_data_tuple else {}

    # Create a dictionary to store total observed values for each health outcome
    total_outcomes_data = {
        "Cases": [],
        "Hosps": [],
        "Deaths": [],
    }

    # Calculate the total observed values for each health outcome
    for key in cases_data:
        cases_total = sum(cases_data[key])
        deaths_total = sum(deaths_data.get(key, [0]))  # Use 0 if no deaths data available
        hospitalizations_total = sum(hosps_data.get(key, [0]))  # Use 0 if no hospitalizations data available
        total_outcomes_data['Cases'].append(cases_total)
        total_outcomes_data['Deaths'].append(deaths_total)
        total_outcomes_data['Hosps'].append(hospitalizations_total)

    # Convert total outcomes data to DataFrame
    total_outcomes_df = pd.DataFrame(total_outcomes_data)

    print (total_outcomes_df)

    # Calculate correlation matrix for total outcomes
    correlation_matrix = total_outcomes_df.corr()
    print('Total correlation matrix', correlation_matrix)

    return correlation_matrix


def generate_hosps_by_age_group():
    """
    This function generates a csv containing information on the number of deaths associated with each age group.
    A crucial step in this process is redefining the age bands to match the dQALY age groups in the Briggs paper.
    Calculation for the number of deaths in each age band are based on Briggs spreadsheet tool

    :return: A csv of COVID-19 hosps by age group.
    """

    #data = pd.read_csv('C:/Users/fm478/Downloads/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW__20240307 (1).csv')
    data = pd.read_csv('/Users/timamikdashi/Downloads/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW__20240307 (1).csv')

    #deaths_by_age = data.groupby(['state','date']).sum().reset_index() #TODO" A REVOIR TO ENSURE THAT THE data is aggregated over state and dates

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



def generate_cases_by_age_group():
    """
    This function generates a csv containing information on the number of cases associated with each age group.
    The underlying data used is a CDC spreadsheet of Rates of COVID-19 Cases or Deaths by Age Group and Vaccination Status

    :return: A csv of COVID-19 cases by age group.
    """

    # Read the data from CSV file
    data = pd.read_csv('/Users/timamikdashi/Downloads/Rates_of_COVID-19_Cases_or_Deaths_by_Age_Group_and_Vaccination_Status_20240417 (1).csv')

    # Combine values for vaccinated and unvaccinated into a new column
    data['Cases'] = data['Vaccinated with outcome'] + data['Unvaccinated with outcome']

    # Group the data by age group
    cases_by_age_grouped = data.groupby(['Age group']).sum().reset_index()

    # Print the unique age groups and their corresponding total combined cases
    for index, row in cases_by_age_grouped.iterrows():
        print(f"Age Group: {row['Age group']}, Total Combined Cases: {row['Cases']}")

    age_band_mapping = {
        '0-9': ['5-11'],
        '10-19': ['5-11', '12-17', '18-29'],
        '20-29': ['18-29'],
        '30-39': ['30-49'],
        '40-49': ['30-49'],
        '50-59': ['50-64'],
        '60-69': ['50-64', '65-79'],
        '70-79': ['65-79'],
        '80-90': ['80+'],
        '90-100': ['80+']
    }

    new_age_data = {'Age Group': [], 'COVID-19 Cases': []}

    for age_band, age_groups in age_band_mapping.items():
        total_cases = 0

        for age_group in age_groups:
            # Check if the columns exist in the DataFrame
            if age_group in cases_by_age_grouped['Age group'].values:
                total_cases += cases_by_age_grouped.loc[
                    cases_by_age_grouped['Age group'] == age_group, 'Cases'].sum()

        # Handle specific calculations for each age band
        if age_band == '0-9':
            # Specific calculation for age band 0-9
            total_cases = (6/7) * cases_by_age_grouped.loc[
                cases_by_age_grouped['Age group'] == '5-11', 'Cases'].sum()
        elif age_band == '10-19':
            # Specific calculation for age band 10-19
            total_cases = ((1 / 7) * cases_by_age_grouped.loc[cases_by_age_grouped['Age group'] == '5-11', 'Cases'].sum() +
                           cases_by_age_grouped.loc[cases_by_age_grouped['Age group'] == '12-17', 'Cases'].sum() +
                           (2 / 12) * cases_by_age_grouped.loc[cases_by_age_grouped['Age group'] == '18-29', 'Cases'].sum()
                           )
        elif age_band == '20-29':
            total_cases = ((10/12)*cases_by_age_grouped.loc[cases_by_age_grouped['Age group'] == '18-29', 'Cases'].sum())

        elif age_band == '30-39':
            total_cases = ((1/2)*cases_by_age_grouped.loc[cases_by_age_grouped['Age group'] == '30-49', 'Cases'].sum())

        elif age_band == '40-49':
            total_cases = ((1/2)*cases_by_age_grouped.loc[cases_by_age_grouped['Age group'] == '30-49', 'Cases'].sum())

        elif age_band == '50-59':
            total_cases = ((10 / 15) * cases_by_age_grouped.loc[cases_by_age_grouped['Age group'] == '50-64', 'Cases'].sum())

        elif age_band == '60-69':
            total_cases = ((5/15)* cases_by_age_grouped.loc[cases_by_age_grouped['Age group'] == '50-64', 'Cases'].sum() +
                           (5/15)*cases_by_age_grouped.loc[cases_by_age_grouped['Age group'] == '65-79', 'Cases'].sum())

        elif age_band == '70-79':
            total_cases = ((10 / 15) * cases_by_age_grouped.loc[cases_by_age_grouped['Age group'] == '65-79', 'Cases'].sum())

        elif age_band == '80-90' or age_band == '90-100':
            # Specific calculation for age bands 80-90 and 90-100
            total_cases = (1 / 2) * cases_by_age_grouped.loc[
                cases_by_age_grouped['Age group'] == '80+', 'Cases'].sum()
        else:
            # General calculation for other age bands
            total_cases = cases_by_age_grouped.loc[
                cases_by_age_grouped['Age group'].isin(age_groups), 'Cases'].sum()

        new_age_data['Age Group'].append(age_band)
        new_age_data['COVID-19 Cases'].append(total_cases)

    # Create a DataFrame for the new age bands
    new_age_df = pd.DataFrame(new_age_data)
    new_age_df['COVID-19 Cases'] = pd.to_numeric(new_age_df['COVID-19 Cases'], errors='coerce').fillna(0)

    # Save the data as a CSV file
    new_age_df.to_csv('cases_by_age.csv', index=False)


def generate_correlation_matrix_timeseries():

    cases_data_tuple = get_dict_of_county_data_by_type('cases')
    hosps_data_tuple = get_dict_of_county_data_by_type('hospitalizations')
    deaths_data_tuple = get_dict_of_county_data_by_type('deaths')

    print('cases data', cases_data_tuple)
    print('hosps data', hosps_data_tuple)

    cases_data = cases_data_tuple[0] if cases_data_tuple else {}  # Accessing the dictionary from the tuple
    hosps_data = hosps_data_tuple[0] if hosps_data_tuple else {}  # Accessing the dictionary from the tuple
    deaths_data = deaths_data_tuple[0] if deaths_data_tuple else {}  # Accessing the dictionary from the tuple

    print('cases', cases_data)
    print('hosps', hosps_data)

    # Combine data for all counties into a single dataset
    combined_data = {'Cases': [], 'Deaths': [], 'Hospitalizations': []}
    for key in cases_data:
        cases_total = sum(cases_data[key])
        deaths_total = sum(deaths_data.get(key, [0]))  # Use 0 if no deaths data available
        hospitalizations_total = sum(hosps_data.get(key, [0]))  # Use 0 if no hospitalizations data available
        combined_data['Cases'].append(cases_total)
        combined_data['Deaths'].append(deaths_total)
        combined_data['Hospitalizations'].append(hospitalizations_total)

    # Convert combined data to DataFrame
    combined_df = pd.DataFrame(combined_data)

    # Calculate correlation matrix
    correlation_matrix = combined_df.corr()

    return correlation_matrix



def generate_correlation_matrix_total_per_capita():

    cases_data_tuple = get_dict_of_county_data_by_type('cases')
    hosps_data_tuple = get_dict_of_county_data_by_type('hospitalizations')
    deaths_data_tuple = get_dict_of_county_data_by_type('deaths')

    cases_data = cases_data_tuple[0] if cases_data_tuple else {}
    hosps_data = hosps_data_tuple[0] if hosps_data_tuple else {}
    deaths_data = deaths_data_tuple[0] if deaths_data_tuple else {}

    # Create a dictionary to store total observed values per 100K individuals for each health outcome
    total_outcomes_per_100k_data = {
        "Cases per 100K": [],
        "Hosps per 100K": [],
        "Deaths per 100K": [],
    }

    # Calculate the total observed values per 100K individuals for each health outcome
    for key in cases_data:
        population = int(key[3])  # Assuming population is the fourth element in the key
        cases_total_per_100k = (sum(cases_data[key]) / population) * 100000
        deaths_total_per_100k = (sum(deaths_data.get(key, [0])) / population) * 100000
        hosps_total_per_100k = (sum(hosps_data.get(key, [0])) / population) * 100000
        total_outcomes_per_100k_data['Cases per 100K'].append(cases_total_per_100k)
        total_outcomes_per_100k_data['Deaths per 100K'].append(deaths_total_per_100k)
        total_outcomes_per_100k_data['Hosps per 100K'].append(hosps_total_per_100k)

    # Convert total outcomes per 100K data to DataFrame
    total_outcomes_per_100k_df = pd.DataFrame(total_outcomes_per_100k_data)

    # Calculate correlation matrix for total outcomes per 100K
    correlation_matrix = total_outcomes_per_100k_df.corr()
    print('Per capita correlation matrix', correlation_matrix)

    return correlation_matrix


def generate_correlation_matrix_total():

    cases_data_tuple = get_dict_of_county_data_by_type('cases')
    hosps_data_tuple = get_dict_of_county_data_by_type('hospitalizations')
    deaths_data_tuple = get_dict_of_county_data_by_type('deaths')

    cases_data = cases_data_tuple[0] if cases_data_tuple else {}
    hosps_data = hosps_data_tuple[0] if hosps_data_tuple else {}
    deaths_data = deaths_data_tuple[0] if deaths_data_tuple else {}

    # Create a dictionary to store total observed values for each health outcome
    total_outcomes_data = {
        "Cases": [],
        "Hosps": [],
        "Deaths": [],
    }

    # Calculate the total observed values for each health outcome
    for key in cases_data:
        cases_total = sum(cases_data[key])
        deaths_total = sum(deaths_data.get(key, [0]))  # Use 0 if no deaths data available
        hospitalizations_total = sum(hosps_data.get(key, [0]))  # Use 0 if no hospitalizations data available
        total_outcomes_data['Cases'].append(cases_total)
        total_outcomes_data['Deaths'].append(deaths_total)
        total_outcomes_data['Hosps'].append(hospitalizations_total)

    # Convert total outcomes data to DataFrame
    total_outcomes_df = pd.DataFrame(total_outcomes_data)

    print (total_outcomes_df)

    # Calculate correlation matrix for total outcomes
    correlation_matrix = total_outcomes_df.corr()
    print('Total correlation matrix', correlation_matrix)

    return correlation_matrix



'''

def generate_county_info_csv():
    """
    Generates a CSV containing county information (County, State, FIPS, Population) from a source file.
    """

    # Read the county data CSV file
    #rows = read_csv_rows(file_name='/Users/fm478/Downloads/county_time_data_all_dates.csv',
                         #if_ignore_first_row=True)
    rows = read_csv_rows(file_name='/Users/timamikdashi/Downloads/county_time_data_all_dates.csv',
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
    state_infections_df = pd.read_csv("/Users/timamikdashi/Downloads/infections_summary_with_new_dates.csv")
    #state_infections_df= pd.read_csv("C:/Users/fm478/Documents/infections_summary_with_new_dates.csv")


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

            # Print the state and its corresponding number of counties for debugging
            print(f"State: {state}, Number of counties: {num_counties}")

            # Divide each infection value by the number of counties for the corresponding state
            state_infections_df.loc[index, state_infections_df.columns[1:]] = row[state_infections_df.columns[1:]] / num_counties


    # Save the resulting DataFrame to a CSV file
    output_path = ROOT_DIR+ '/csv_files/state_infections_divided_by_counties.csv'
    state_infections_df.to_csv(output_path, index=False)

    return state_infections_df


def generate_county_infections_csv():
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
    output_path = ROOT_DIR + '/csv_files/county_infections.csv'
    county_infections_df.to_csv(output_path, index=False)

    print("County-level infection data saved to:", output_path)


'''
def generate_county_data_csv_2021(data_type='cases'):
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
    }

    # Ensure the specified data_type is valid
    if data_type not in data_type_mapping:
        raise ValueError(
            "Invalid data_type. Choose from 'cases', 'deaths', 'hospitalizations', "
            "'icu', 'cases per 100,000', "
            "'deaths per 100,000', 'hospitalizations per 100,000', 'icu occupancy per 100,000'.")

    # Read the data
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

        # Check if the date is within 2021
        date = datetime.strptime(date_str, "%Y-%m-%d")
        if date.year == 2021:
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

    # Create a list of unique dates across all counties within 2021
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

    # Write into a CSV file using the write_csv function
    write_csv(rows=[header_row] + county_data_rows, file_name=ROOT_DIR + output_file)
'''


def generate_state_and_county_infections_csv():
    """
    Generates a CSV file with county-level infections data, distributing state-level infections evenly
    across counties, for the date range from 2020-07-15 to 2022-12-28.
    """

    # State name to abbreviation mapping
    state_name_to_abbreviation = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
        'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }

    # Step 1: Generate county information CSV (if not already generated)
    generate_county_info_csv()
    county_info_df = pd.read_csv(ROOT_DIR + '/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_info.csv')

    # Step 2: Read state estimates CSV file
    estimates_df = pd.read_csv('/Users/timamikdashi/Downloads/state/estimates.csv')

    # Step 3: Generate county-level infection data
    county_data_list = []

    for _, row in estimates_df.iterrows():
        state_full_name = row[0]  # State name
        date_str = row[1]  # Date
        infections = row[28]  # Infections count (29th column, 0-indexed)

        # Map full state name to abbreviation
        state_abbreviation = state_name_to_abbreviation.get(state_full_name)

        # Filter counties belonging to the current state
        state_counties_df = county_info_df[county_info_df['State'] == state_abbreviation]

        if not state_counties_df.empty:
            # Distribute state-level infections across counties
            infections_per_county = infections / len(state_counties_df)

            for _, county_row in state_counties_df.iterrows():
                county_data_list.append([
                    county_row['County'], county_row['State'], county_row['FIPS'], date_str, infections_per_county
                ])

    # Step 4: Save county-level infection data to CSV
    county_data_df = pd.DataFrame(county_data_list, columns=['County', 'State', 'FIPS', 'Date', 'Infections'])
    county_data_df.to_csv(ROOT_DIR + '/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/state_mapped_county_infections.csv', index=False)


def generate_cases_infections_factor():
    # Read the CSV file
    county_infections= pd.read_csv(ROOT_DIR + '/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_infections.csv')
    county_cases = pd.read_csv(
        ROOT_DIR + '/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_cases.csv')

    #county_infections = pd.read_csv(ROOT_DIR + '/csv_files/county_infections.csv')
    #county_cases = pd.read_csv(ROOT_DIR + '/csv_files/county_cases.csv')


    # Exclude unnecessary columns (FIPS, Population, etc.)
    infections_grouped = county_infections.drop(columns=['County', 'FIPS', 'Population'])
    cases_grouped = county_cases.drop(columns=['County', 'FIPS', 'Population'])

    # Group by the 'State' column and sum the values for each state at each time point
    infections_state = infections_grouped.groupby('State').sum()
    cases_state = cases_grouped.groupby('State').sum()

    # Calculate the factor of infections to cases for each state and time point
    factor = infections_state / cases_state

    # Replace infinity or NaN values (where cases are 0) with 1
    factor = factor.replace([float('inf'), float('nan')], 1)
    factor.to_csv(ROOT_DIR + '/csv_files/cases_infections_factor', index=True)


def generate_infections_from_cases():
    #state_factors = pd.read_csv(ROOT_DIR + '/csv_files/cases_infections_factor',index_col='State')
    state_factors = pd.read_csv(ROOT_DIR + '/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/cases_infections_factor', index_col='State')


    #county_cases = pd.read_csv(
        #ROOT_DIR + '/csv_files/county_cases.csv')
    county_cases = pd.read_csv(
        ROOT_DIR + '/tests/Users/timamikdashi/PycharmProjects/covid19-qaly-loss/csv_files/county_cases.csv')

    case_columns = county_cases.columns[4:]  # Assuming first 4 columns are 'County', 'State', 'FIPS', 'Population'
    county_infections_from_cases = county_cases.copy()

    for state in county_infections_from_cases['State'].unique():
        state_factor = state_factors.loc[state]
        state_counties = county_infections_from_cases['State'] == state
        county_infections_from_cases.loc[state_counties, case_columns] = county_cases.loc[
            state_counties, case_columns].multiply(state_factor.values, axis=1)

    # Save the new infections estimate to a CSV file
    county_infections_from_cases.to_csv( ROOT_DIR + '/csv_files/county_infections_from_cases.csv', index=False)