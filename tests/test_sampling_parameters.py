import numpy as np

from classes.parameters import ParameterGenerator
from data_preprocessing.support_functions import generate_deaths_by_age_group_and_sex, generate_life_expectancy_by_sex_age, extract_LE_and_death_arrays

# generate death and life expectancy distribution by age and sex
generate_deaths_by_age_group_and_sex()
generate_life_expectancy_by_sex_age()
extract_LE_and_death_arrays()

# random number generator
rng = np.random.RandomState(1)

# create a parameter generator
param_gen = ParameterGenerator()

# generate a parameter set
for i in range(10):
    param = param_gen.generate(rng)
    print(param)

