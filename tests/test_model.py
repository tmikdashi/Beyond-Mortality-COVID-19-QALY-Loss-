from classes.model import AllStates


# create the model, populate it, and calculate the QALY loss
all_states = AllStates()
all_states.populate()
all_states.calculate_qaly_loss(case_weight=0.1, hosp_weight=0.2, death_weight=0.3)

print('Total US population: ', '{:,}'.format(all_states.population))
print('Total US QALY loss: ', '{:,.0f}'.format(all_states.get_overall_qaly_loss()))

# get and print QALY loss by outcome
qaly_loss_by_outcome = all_states.get_qaly_loss_by_outcome()
for outcome, value in qaly_loss_by_outcome.items():
    print('   Due to {}:'.format(outcome),
          '{:,.0f}'.format(value))


print('Weekly US QALY loss: ', all_states.get_weekly_qaly_loss())

# TODO: when I run in this module, I get this warning:
#     "No artists with labels found to put in legend.
#     Note that artists whose label start with an underscore are ignored when legend() is called with no argument.


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


