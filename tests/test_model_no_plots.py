import numpy as np

from classes.model import AllStates
# create the model, populate it, and calculate the QALY loss
from classes.parameters import ParameterGenerator
from data_preprocessing.support_functions import generate_deaths_by_age_group_and_sex, \
    generate_life_expectancy_by_sex_age, extract_LE_and_death_arrays

# TODO: does it make sense to move these 3 functions to generate_data.py?
#  We need to produce these only once so maybe it makes sense to have them in generate_data.py?
# generate death and life expectancy distribution by age and sex
generate_deaths_by_age_group_and_sex()
generate_life_expectancy_by_sex_age()
extract_LE_and_death_arrays()

# Generate parameter set
rng = np.random.RandomState(1)
param_gen = ParameterGenerator()


# loop to generate and analyze 10 sets of parameters
for i in range(5):
    # generate a new set of parameters
    rng = np.random.RandomState(i)  # use different seeds for different sets
    param_values = param_gen.generate(rng)

    # create the model, populate it, and calculate the QALY loss
    # TODO: this is good but what is happening here is that we need to recreate ALLStates for each set of parameters.
    #  This is not very efficient because AllStates is independent of parameter values (it is populated with data from
    #  csv files of cases, hospitalizations, and deaths).
    #  So, we should get parameter_values as an argument of .calcualte_qaly_loss() function.
    #  This way, we can create AllStates once and then call .calculate_qaly_loss() for each set of parameters.
    all_states = AllStates(param_values)
    all_states.populate()
    all_states.calculate_qaly_loss()

    # print results for each set
    print(f"\nResults for Parameter Set {i + 1}:\n")
    print(f"case_weight = {param_values.qWeightCase}")
    print(f"hosp_weight = {param_values.qWeightHosp}")
    print(f"death_weight = {param_values.qWeightDeath}")

    print('Total US population: ', '{:,}'.format(all_states.population))
    print('Total US QALY loss: ', '{:,.0f}'.format(all_states.get_overall_qaly_loss()))

    # get and print QALY loss by outcome
    qaly_loss_by_outcome = all_states.get_qaly_loss_by_outcome()
    for outcome, value in qaly_loss_by_outcome.items():
        print(f'   Due to {outcome}:', '{:,.0f}'.format(value))

    # STATE
    all_states.get_overall_qaly_loss_by_state()
    all_states.get_weekly_qaly_loss_by_state()

    # COUNTY
    all_states.get_overall_qaly_loss_by_county()

    # STATE-SPECIFIC DATA
    all_states.get_overall_qaly_loss_for_a_state("AL")
    all_states.get_weekly_qaly_loss_for_a_state("AL")

    # COUNTY-SPECIFIC DATA
    all_states.get_overall_qaly_loss_for_a_county("Autauga", "AL")
    all_states.get_weekly_qaly_loss_for_a_county("Autauga", "AL")

