from classes.model import ProbabilisticAllStates, AllStates
import numpy as np

probabilisticModel = ProbabilisticAllStates()
probabilisticModel.simulate(n=5)
print("QALY Loss Cases by State:", probabilisticModel.summaryOutcomes.overallQALYlossesCasesByState)
print("QALY Loss Hosps by State:", probabilisticModel.summaryOutcomes.overallQALYlossesHospsByState)
print("QALY Loss Deathss by State:", probabilisticModel.summaryOutcomes.overallQALYlossesDeathsByState)
print(probabilisticModel.summaryOutcomes.overallQALYlossesByState)


probabilisticModel.print_overall_qaly_loss()
print(probabilisticModel.get_mean_ui_weekly_qaly_loss())

#PLOTS
probabilisticModel.plot_weekly_qaly_loss_by_outcome()
probabilisticModel.plot_map_of_avg_qaly_loss_by_county()
probabilisticModel.plot_qaly_loss_by_state_and_by_outcome()

