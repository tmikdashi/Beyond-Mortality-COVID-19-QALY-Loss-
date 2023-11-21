from data_preprocessing.support_functions import generate_prop_deaths_by_age_group_and_sex, generate_combined_life_expectancy, extract_LE_and_prop_death_arrays

# Defining the proportion of cases that are from age group and sex
prop_deaths_by_age_group_and_sex = generate_prop_deaths_by_age_group_and_sex()

# Calculating avg life expectancy across by age group and sex
average_LE_data_by_age_group_and_sex = generate_combined_life_expectancy()

# Determining inputs
life_expectancy_array, prop_deaths_array = extract_LE_and_prop_death_arrays(average_LE_data_by_age_group_and_sex, prop_deaths_by_age_group_and_sex)

