from classes.model import ProbabilisticAllStates

probabilisticModel = ProbabilisticAllStates()
probabilisticModel.simulate(n=5)


probabilisticModel.print_overall_qaly_loss()
print(probabilisticModel.get_mean_ui_weekly_qaly_loss())

#PLOTS
probabilisticModel.plot_weekly_qaly_loss()
probabilisticModel.plot_weekly_qaly_loss_by_outcome()
probabilisticModel.plot_map_of_avg_qaly_loss_by_county()


# QALY LOSS BY STATE: FORMATTING ALTERNATIVE
#Creates a folder with separate plots for each state
probabilisticModel.plot_weekly_qaly_loss_by_state()

# Creates single plot with all states' QALY loss per 100K pop
probabilisticModel.plot_weekly_qaly_loss_by_state_100K_pop()

# Creates subplots for each state
probabilisticModel.subplot_weekly_qaly_loss_by_state_100K_pop()



