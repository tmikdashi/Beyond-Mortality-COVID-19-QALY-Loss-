import numpy as np

from classes.model import AllStates,ProbabilisticAllStates
from classes.parameters import ParameterGenerator


'''
# Generate parameter set
rng = np.random.RandomState(1)
param_gen = ParameterGenerator()

all_states = AllStates()
all_states.populate()
# loop to generate and analyze 10 sets of parameters
for i in range(5):
    # generate a new set of parameters
    rng = np.random.RandomState(i)  # use different seeds for different sets
    params = param_gen.generate(rng)

    # Calculate the QALY loss
    all_states.calculate_qaly_loss(params)

    # print results for each set
    print(f"\nResults for Parameter Set {i + 1}:\n")
    print(f"case_weight = {params.qWeightCase}")
    print(f"hosp_weight = {params.qWeightHosp}")
    print(f"death_weight = {params.qWeightDeath}")

    print('Total US QALY loss: ', '{:,.0f}'.format(all_states.get_overall_qaly_loss()))

    # get and print QALY loss by outcome
    qaly_loss_by_outcome = all_states.get_qaly_loss_by_outcome()
    for outcome, value in qaly_loss_by_outcome.items():
        print(f'   Due to {outcome}:', '{:,.0f}'.format(value))

    # STATE
    all_states.get_overall_qaly_loss_by_state()

    # COUNTY
    all_states.get_overall_qaly_loss_by_county()

    # STATE-SPECIFIC DATA
    all_states.get_overall_qaly_loss_for_a_state("AL")
    all_states.get_weekly_qaly_loss_for_a_state("AL")

    # COUNTY-SPECIFIC DATA
    all_states.get_overall_qaly_loss_for_a_county("Autauga", "AL")



'''
all_states = AllStates()
all_states.populate()
probabilistic_states=ProbabilisticAllStates()
probabilistic_states.allStates =all_states


probabilistic_states.simulate(5)


