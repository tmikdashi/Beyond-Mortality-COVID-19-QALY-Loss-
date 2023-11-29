import numpy as np

from classes.model import AllStates,ProbabilisticAllStates
from classes.parameters import ParameterGenerator


probabilistic_states=ProbabilisticAllStates()
probabilistic_states.simulate(5)

#KEY FUNCTIONS
probabilistic_states.get_overall_qaly_loss()
probabilistic_states.get_weekly_qaly_loss()
probabilistic_states.plot_weekly_qaly_loss()

#ADDITIONAL FUNCTIONS
probabilistic_states.get_overall_qaly_loss_by_state()


