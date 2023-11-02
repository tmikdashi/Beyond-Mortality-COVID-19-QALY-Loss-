from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _playground.support.support_functions import get_dict_of_county_cases_and_dates

# get the dictionary of county cases
county_cases_data, dates = get_dict_of_county_cases_and_dates()

# To be able to group by date and state, create a df to manipulate the data
df = pd.DataFrame(county_cases_data)

# Using the groupby function to assess whether the value is NaN or not and
# count the number of NaNs per date and state
nan_counts = df.isna().groupby(level=1, axis=1).sum()

# Create a bar chart with stacked bars for the first 50 dates
fig, ax = plt.subplots(figsize=(12, 8))

# We want a unique color per state: get the number of states and generate a list of distinct colors
# I had some issues with this because most colormaps have less than 20 different colors
states = df.columns.get_level_values(1).unique()
num_states = len(states)
colors = plt.cm.get_cmap('tab20', num_states)
color_cycle = cycle([colors(i) for i in range(num_states)])

# We want to keep count of NaNs per state, so we first initialize an array to keep track of the values for stacking
bottom = np.zeros(len(nan_counts))

for state in states:
    state_nan_counts = nan_counts[state]

    # After looking at the full data, I only want to plot the stacked bars for the first 50 dates
    ax.bar(nan_counts.index[:30], state_nan_counts[:30], bottom=bottom[:30], label=state, color=next(color_cycle))

    # Update the bottom values for stacking
    bottom[:30] += state_nan_counts[:30]

# Setting the labels and legend
ax.set_xlabel('Dates')
ax.set_ylabel('Number of Counties w/ Missing Values')
ax.set_title('Number of Counties with Missing Data (NaNs) by Date and State')
ax.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

# Adjust subplot parameters to make room for state labels
# plt.subplots_adjust(bottom=0.25)
plt.tight_layout()

# Show or save the plot
plt.show()
