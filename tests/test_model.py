from classes.model import ProbabilisticAllStates, AllStates
import numpy as np

probabilisticModel = ProbabilisticAllStates()
probabilisticModel.simulate(n=5)


probabilisticModel.print_overall_qaly_loss()
print(probabilisticModel.get_mean_ui_weekly_qaly_loss())

#PLOTS
probabilisticModel.plot_weekly_qaly_loss()
probabilisticModel.plot_weekly_qaly_loss_by_outcome()
probabilisticModel.plot_map_of_avg_qaly_loss_by_county()



# QALY LOSS BY STATE: FORMATTING ALTERNATIVE
probabilisticModel.plot_weekly_qaly_loss_by_state()
probabilisticModel.plot_weekly_qaly_loss_by_state_100K_pop()
probabilisticModel.subplot_weekly_qaly_loss_by_state_100K_pop()





#

