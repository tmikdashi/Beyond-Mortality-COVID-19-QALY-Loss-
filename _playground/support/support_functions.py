import numpy as np
from deampy.in_out_functions import read_csv_rows

from definitions import ROOT_DIR


def get_dict_of_county_cases_and_dates(data_type):
    """
    This function reads the county_cases.csv file and returns a dictionary of county cases and dates.
    :return: (dictionary, list) a dictionary with (county, state) as keys and a list of cases as values,
            and a list of dates
    """

    # Read the data
    data_rows = read_csv_rows(file_name=ROOT_DIR + '/data/summary/county_cases.csv',
                              if_ignore_first_row=False)

    county_cases_data = {}
    for row in data_rows[1:]:
        county = row[0]
        state = row[1]
        cases = row[2:]

        # Convert cases to a list of floats, handling missing values
        cases = [float(case) if case != 'NA' else np.nan for case in cases]

        county_cases_data[(county, state)] = cases

    return county_cases_data, data_rows[0][2:]
