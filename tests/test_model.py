from classes.model import ProbabilisticAllStates


probabilisticModel = ProbabilisticAllStates()
probabilisticModel.simulate(n=10)

probabilisticModel.print_overall_outcomes_and_qaly_loss()
probabilisticModel.plot_weekly_qaly_loss_by_outcome()
probabilisticModel.plot_qaly_loss_by_state_and_by_outcome()

probabilisticModel.print_state_prevax_values()

#probabilisticModel.get_state_vax_index()
#probabilisticModel.plot_prevax_postvax_qaly_loss_by_state()

#probabilisticModel.plot_map_of_hsa_outcomes_by_county_per_100K()
#probabilisticModel.plot_qaly_loss_by_state_and_vax_status_subplots()
#probabilisticModel.plot_qaly_loss_by_state_and_by_outcome_alt_2()
#probabilisticModel.print_state_prevax_values()

#probabilisticModel.plot_date_70pct_vaccinated_by_state()

'''
probabilisticModel.print_overall_outcomes_and_qaly_loss()
probabilisticModel.plot_qaly_loss_from_deaths_by_age()


probabilisticModel.plot_weekly_outcomes()
probabilisticModel.plot_weekly_qaly_loss_by_outcome()
probabilisticModel.plot_qaly_loss_by_state_and_by_outcome()

probabilisticModel.plot_map_of_avg_qaly_loss_by_county()

#vertically representing sub-maps



probabilisticModel.plot_map_of_pop_over_65_by_county()
probabilisticModel.plot_map_of_median_age_by_county()

'''


