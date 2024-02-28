from data_preprocessing.support_functions import generate_county_data_csv, generate_hsa_mapped_county_hosp_data, generate_deaths_by_age_group
# Create the model and populate it with data from the provided csv files


generate_county_data_csv('cases')
generate_county_data_csv('hospitalizations')
generate_county_data_csv('deaths')

generate_hsa_mapped_county_hosp_data()
generate_deaths_by_age_group()
