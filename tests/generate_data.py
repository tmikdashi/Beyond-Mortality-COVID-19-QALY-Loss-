from data_preprocessing.support_functions import generate_county_data_csv

# TODO: this could be run once so I moved it to a seperate file so that we don't have to run it
#  every time we run the tests

# Create the model and populate it with data from the provided csv files
generate_county_data_csv('cases')
generate_county_data_csv('deaths')
generate_county_data_csv('hospitalizations')