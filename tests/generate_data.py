from data_preprocessing.support_functions import generate_county_data_csv

# Create the model and populate it with data from the provided csv files
generate_county_data_csv('cases')
generate_county_data_csv('deaths')
generate_county_data_csv('hospitalizations')