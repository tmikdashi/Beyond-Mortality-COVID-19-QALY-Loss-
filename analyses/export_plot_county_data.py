import matplotlib.pyplot as plt
from deampy.plots.plot_support import output_figure

from definitions import ROOT_DIR

# TODO: for the initial analysis, all we need is the number of cases. So we don't want to recreate this data
#  every time we run the analysis. We should export and save whatever we need from the source dataset
#  in a separate files.
#  So here it would be great to write the code that reads the data
#  and saves the number of cases for each county in a separate file.
#  Each column corresponds to a county (the first cell in the column is the county name or some unique ID)
#  you could store these files under data/summary/county_cases.csv


# TODO: The next step is to plot them without dealing with missing values. In other words,
#  if there is a week with missing values, it should be clear from the plot.
#  we need a separate plot for each county to check if everything makes sense.


# TODO: Also deampy has this function to export figures (output_figure). The advantage of using this function
#  is that it will create the directory if it does not exist and also simplifies your code a little.
#  here is an example
fig = plt.figure(figsize=(3, 3))

output_figure(plt=fig, filename=ROOT_DIR + '/results/figures/cases/county_1.png')
