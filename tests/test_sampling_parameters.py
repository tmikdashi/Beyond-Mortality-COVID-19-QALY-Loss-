import numpy as np

from data_preprocessing.support_functions import generate_prop_deaths_by_age_group_and_sex, generate_life_expectancy_by_sex_age, extract_LE_and_prop_death_arrays
from classes.parameters import ParameterGenerator

# Defining the number of cases and life expectancy by  age group and sex
deaths_by_age_group_and_sex = generate_prop_deaths_by_age_group_and_sex()
average_LE_data_by_age_group_and_sex = generate_life_expectancy_by_sex_age()

# Determining Dirichlet inputs
life_expectancy_array, nb_deaths_array = extract_LE_and_prop_death_arrays(
    average_le_data_by_age_and_sex=average_LE_data_by_age_group_and_sex,
    deaths_by_age_group_and_sex=deaths_by_age_group_and_sex)


# random number generator
rng = np.random.RandomState(1)

# create a parameter generator
param_gen = ParameterGenerator(life_expectancy_array, nb_deaths_array)

# generate a parameter set
for i in range(10):
    param = param_gen.generate(rng)
    print(param)

