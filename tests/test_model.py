from classes.model import AllStates
import numpy as np


# create the model, populate it, and calculate the QALY loss
from classes.parameters import ParameterValues, ParameterGenerator
from data_preprocessing.support_functions import generate_deaths_by_age_group_and_sex, generate_life_expectancy_by_sex_age, extract_LE_and_death_arrays

# generate death and life expectancy distribution by age and sex
generate_deaths_by_age_group_and_sex()
generate_life_expectancy_by_sex_age()
extract_LE_and_death_arrays()

# Generate parameter set
rng = np.random.RandomState(1)
param_gen = ParameterGenerator()
param_values = param_gen.generate(rng)

case_weight = param_values.qWeightCase
hosp_weight = param_values.qWeightHosp
death_weight = param_values.qWeightDeath

all_states = AllStates()
all_states.populate()
all_states.calculate_qaly_loss(case_weight, hosp_weight, death_weight)

print('Total US population: ', '{:,}'.format(all_states.population))
print('Total US QALY loss: ', '{:,.0f}'.format(all_states.get_overall_qaly_loss()))

# get and print QALY loss by outcome
qaly_loss_by_outcome = all_states.get_qaly_loss_by_outcome()
for outcome, value in qaly_loss_by_outcome.items():
    print('   Due to {}:'.format(outcome),
          '{:,.0f}'.format(value))


print('Weekly US QALY loss: ', all_states.get_weekly_qaly_loss())


#TODO: many of these are just to demonstrate that the functions work and therefore can be deleted later

# STATE
all_states.get_overall_qaly_loss_by_state()
all_states.get_weekly_qaly_loss_by_state()

# COUNTY
all_states.get_overall_qaly_loss_by_county()
#all_states.get_weekly_qaly_loss_by_county()


# STATE-SPECIFIC DATA
all_states.get_overall_qaly_loss_for_a_state("AL")
all_states.get_weekly_qaly_loss_for_a_state("AL")


# COUNTY-SPECIFIC DATA
all_states.get_overall_qaly_loss_for_a_county("Autauga", "AL")
all_states.get_weekly_qaly_loss_for_a_county("Autauga", "AL")

# PLOTTING
all_states.plot_map_of_qaly_loss_by_county()
all_states.plot_weekly_qaly_loss_by_state()
all_states.plot_weekly_qaly_loss()
all_states.plot_weekly_qaly_loss_by_outcome()
all_states.plot_qaly_loss_by_state_and_by_outcome()

print('total deaths:', all_states.pandemicOutcomes.deaths.totalObs)


