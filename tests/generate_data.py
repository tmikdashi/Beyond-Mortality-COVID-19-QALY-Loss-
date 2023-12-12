from data_preprocessing.support_functions import generate_county_data_csv, generate_life_expectancy_by_sex_age, generate_deaths_by_age_group_and_sex, extract_LE_and_death_arrays

# Create the model and populate it with data from the provided csv files
generate_county_data_csv('cases')
generate_county_data_csv('deaths')
generate_county_data_csv('hospitalizations')

generate_deaths_by_age_group_and_sex()
generate_life_expectancy_by_sex_age()
extract_LE_and_death_arrays()
