from classes.model_2 import ProbabilisticAllStates

probabilisticModel = ProbabilisticAllStates()
probabilisticModel.simulate(n=5)

#KEY FUNCTIONS
probabilisticModel.print_overall_qaly_loss()
print(probabilisticModel.get_mean_ui_weekly_qaly_loss())
# probabilisticModel.get_weekly_qaly_loss()
probabilisticModel.plot_weekly_qaly_loss()

#ADDITIONAL FUNCTIONS
probabilisticModel.get_overall_qaly_loss_by_state()
probabilisticModel.get_overall_qaly_loss_by_state()
probabilisticModel.plot_map_of_qaly_loss_by_county()


