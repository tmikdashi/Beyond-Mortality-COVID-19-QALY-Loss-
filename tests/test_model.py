from classes.model import ProbabilisticAllStates


probabilisticModel = ProbabilisticAllStates()
probabilisticModel.simulate(n=5)

probabilisticModel.print_overall_outcomes_and_qaly_loss()


# probabilisticModel.plot_weekly_qaly_loss_by_outcome()
# probabilisticModel.plot_map_of_avg_qaly_loss_by_county()
#
# probabilisticModel.plot_weekly_qaly_loss_by_state()
# probabilisticModel.subplot_weekly_qaly_loss_by_state_100K_pop()
# probabilisticModel.plot_qaly_loss_by_state_and_by_outcome()
# probabilisticModel.plot_weekly_qaly_loss_by_outcome()
# probabilisticModel.plot_qaly_loss_by_state_and_by_outcome()




