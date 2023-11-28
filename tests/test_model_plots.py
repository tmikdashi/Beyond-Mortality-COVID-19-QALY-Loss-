from classes.model import AllStates
import numpy as np
import matplotlib.pyplot as plt
import os


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


# Specify the base folder to save the figures
output_base_folder = 'figs'

all_states = AllStates()
all_states.populate()

# loop to generate and analyze 10 sets of parameters
for i in range(5):
    # generate a new set of parameters
    rng = np.random.RandomState(i)  # use different seeds for different sets
    param_values = param_gen.generate(rng)

    # create the model, populate it, and calculate the QALY loss

    case_weight = param_values.qWeightCase
    hosp_weight = param_values.qWeightHosp
    death_weight = param_values.qWeightDeath
    all_states.calculate_qaly_loss(case_weight, hosp_weight, death_weight)

    # print results for each set
    print(f"\nResults for Parameter Set {i + 1}:\n")
    print(f"case_weight = {param_values.qWeightCase}")
    print(f"hosp_weight = {param_values.qWeightHosp}")
    print(f"death_weight = {param_values.qWeightDeath}")

    print('Total US QALY loss: ', '{:,.0f}'.format(all_states.get_overall_qaly_loss()))

    # get and print QALY loss by outcome
    qaly_loss_by_outcome = all_states.get_qaly_loss_by_outcome()
    for outcome, value in qaly_loss_by_outcome.items():
        print(f'   Due to {outcome}:', '{:,.0f}'.format(value))


    # PLOTTING
    # create a folder for each parameter set
    output_folder = os.path.join(output_base_folder, f'Figs_parameter_{i + 1}')
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot map of QALY loss by county
    all_states.plot_map_of_qaly_loss_by_county()
    plt.title(f'Parameter Set {i + 1} - QALY Loss by County')
    plt.savefig(os.path.join(output_folder, f'parameter_set_{i + 1}_qaly_loss_by_county.png'))

    # Plot weekly QALY loss by state
    plt.figure(figsize=(10, 6))
    all_states.plot_weekly_qaly_loss_by_state()
    plt.title(f'Parameter Set {i + 1} - Weekly QALY Loss by State')
    plt.savefig(os.path.join(output_folder, f'parameter_set_{i + 1}_weekly_qaly_loss_by_state.png'))

    # Plot weekly QALY loss
    plt.figure(figsize=(10, 6))
    all_states.plot_weekly_qaly_loss()
    plt.title(f'Parameter Set {i + 1} - Weekly QALY Loss')
    plt.savefig(os.path.join(output_folder, f'parameter_set_{i + 1}_weekly_qaly_loss.png'))

    # Plot weekly QALY loss by outcome
    plt.figure(figsize=(10, 6))
    all_states.plot_weekly_qaly_loss_by_outcome()
    plt.title(f'Parameter Set {i + 1} - Weekly QALY Loss by Outcome')
    plt.savefig(os.path.join(output_folder, f'parameter_set_{i + 1}_weekly_qaly_loss_by_outcome.png'))

    # Plot QALY loss by state and by outcome
    plt.figure(figsize=(10, 6))
    all_states.plot_qaly_loss_by_state_and_by_outcome()
    plt.title(f'Parameter Set {i + 1} - QALY Loss by State and by Outcome')
    plt.savefig(os.path.join(output_folder, f'parameter_set_{i + 1}_qaly_loss_by_state_and_by_outcome.png'))
