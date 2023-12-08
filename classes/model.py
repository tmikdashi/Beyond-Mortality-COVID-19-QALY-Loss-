import os

import geopandas as gpd
import geoplot as gplt
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classes.parameters import ParameterGenerator
from classes.support import get_mean_ui_of_a_time_series, get_overall_mean_ui
from data_preprocessing.support_functions import get_dict_of_county_data_by_type
from deampy.plots.plot_support import output_figure
from deampy.statistics import SummaryStat
from definitions import ROOT_DIR
from shapely.geometry import Polygon


class AnOutcome:

    def __init__(self):
        self.weeklyObs = np.array([])
        self.totalObs = 0
        self.weeklyQALYLoss = np.array([])
        self.totalQALYLoss = 0

    def add_traj(self, weekly_obs):
        """
        Add weekly data to the Outcome object.
        :param weekly_obs: Weekly data as a numpy array.
        """
        if not isinstance(weekly_obs, np.ndarray):
            weekly_obs = np.array(weekly_obs)

        # replace missing values with 0
        weekly_obs = np.nan_to_num(weekly_obs, nan=0)

        # add the weekly data to the existing data
        if len(self.weeklyObs) == 0:
            self.weeklyObs = weekly_obs
        else:
            self.weeklyObs += weekly_obs

        self.totalObs += sum(weekly_obs)

    def calculate_qaly_loss(self, quality_weight):
        """
        Calculates the weekly and overall QALY
        :param quality_weight: Weight to be applied to each case in calculating QALY loss.
        :return Weekly QALY loss as a numpy array or numerical values to total QALY loss.
        """
        self.weeklyQALYLoss = quality_weight * self.weeklyObs
        self.totalQALYLoss = sum(self.weeklyQALYLoss)


class PandemicOutcomes:
    def __init__(self):

        self.cases = AnOutcome()
        self.hosps = AnOutcome()
        self.deaths = AnOutcome()

        self.weeklyQALYLoss = np.array([])
        self.totalQALYLoss = 0

    def add_traj(self, weekly_cases, weekly_hosp, weekly_deaths):
        """
        Add weekly cases, hospitalization, and deaths and calculate the total cases, hospitalizations, and deaths.
        :param weekly_cases: Weekly cases data as a numpy array.
        :param weekly_hosp: Weekly hospitalizations data as a numpy array.
        :param weekly_deaths: Weekly deaths data as a numpy array.
        """
        self.cases.add_traj(weekly_obs=weekly_cases)
        self.hosps.add_traj(weekly_obs=weekly_hosp)
        self.deaths.add_traj(weekly_obs=weekly_deaths)

    def calculate_qaly_loss(self, case_weight, hosp_weight, death_weight):
        """
        Calculates the weekly and overall QALY
        :param case_weight: cases-specific weight to be applied to each case in calculating QALY loss.
        :param hosp_weight: hosp-specific weight to be applied to each hospitalization in calculating QALY loss.
        :param death_weight: death-specific weight to be applied to each death in calculating QALY loss.
        """

        self.cases.calculate_qaly_loss(quality_weight=case_weight)
        self.hosps.calculate_qaly_loss(quality_weight=hosp_weight)
        self.deaths.calculate_qaly_loss(quality_weight=death_weight)

        self.weeklyQALYLoss = self.cases.weeklyQALYLoss + self.hosps.weeklyQALYLoss + self.deaths.weeklyQALYLoss
        self.totalQALYLoss = self.cases.totalQALYLoss + self.hosps.totalQALYLoss + self.deaths.totalQALYLoss


class County:
    def __init__(self, name, state, fips, population):
        """
         Initialize a County object.

         :param name: Name of the county.
         :param state: Name of the state to which the county belongs.
         :param fips: FIPS code of the county.
         :param population: Population of the county.
        """
        self.name = name
        self.state = state
        self.fips = fips
        self.population = int(population)
        self.pandemicOutcomes = PandemicOutcomes()

    def add_traj(self, weekly_cases, weekly_deaths, weekly_hosp):
        """
        Add weekly data to the County object.
        :param weekly_cases: Weekly cases data as a numpy array.
        :param weekly_hosp: Weekly hospitalizations data as a numpy array.
        :param weekly_deaths: Weekly deaths data as a numpy array.
        """
        self.pandemicOutcomes.add_traj(
            weekly_cases=weekly_cases, weekly_hosp=weekly_hosp, weekly_deaths=weekly_deaths)

    def calculate_qaly_loss(self, case_weight, death_weight, hosp_weight):
        """
        Calculates the weekly and total QALY loss for the County.

        :param case_weight: cases-specific weight to be applied to each case in calculating QALY loss.
        :param death_weight: death-specific weight to be applied to each death in calculating QALY loss.
        :param hosp_weight: hosp-specific weight to be applied to each hospitalization in calculating QALY loss.
        :return QALY loss for each county.
        """

        self.pandemicOutcomes.calculate_qaly_loss(
            case_weight=case_weight, hosp_weight=hosp_weight, death_weight=death_weight)

    def get_overall_qaly_loss(self):
        """
        Retrieves total QALY loss for the County, across outcomes.
        """
        return self.pandemicOutcomes.totalQALYLoss

    def get_weekly_qaly_loss(self):
        """
        Retrieves weekly QALY loss for the County, across outcomes.
        """
        return self.pandemicOutcomes.weeklyQALYLoss


class State:
    def __init__(self, name, num_weeks):
        """
        Initialize a State object.

        :param name: Name of the state.
        """
        self.name = name
        self.population = 0
        self.counties = {}  # Dictionary of county objects
        self.pandemicOutcomes = PandemicOutcomes()
        self.numWeeks = num_weeks

    def add_county(self, county):
        """
        Add a County object to the State and calculates the population size of the state

        :param county: County object to be added to the State.
        """
        self.counties[county.name] = county
        self.population += county.population
        self.pandemicOutcomes.add_traj(
            weekly_cases=county.pandemicOutcomes.cases.weeklyObs,
            weekly_hosp=county.pandemicOutcomes.hosps.weeklyObs,
            weekly_deaths=county.pandemicOutcomes.deaths.weeklyObs)

    def calculate_qaly_loss(self, case_weight, hosp_weight, death_weight):
        """
        Calculates QALY loss for the State.
        :param case_weight: cases-specific weight to be applied to each case in calculating QALY loss.
        :param hosp_weight: hosp-specific weight to be applied to each hospitalization in calculating QALY loss.
        :param death_weight: death-specific weight to be applied to each death in calculating QALY loss.
        """

        # calculate QALY loss for each county
        for county in self.counties.values():
            county.calculate_qaly_loss(
                case_weight=case_weight, hosp_weight=hosp_weight, death_weight=death_weight)

        # calculate QALY loss for the state
        self.pandemicOutcomes.calculate_qaly_loss(
            case_weight=case_weight, hosp_weight=hosp_weight, death_weight=death_weight)

    def get_overall_qaly_loss(self):
        """
        Retrieves total QALY loss for the State, across outcomes.
        """
        return self.pandemicOutcomes.totalQALYLoss

    def get_weekly_qaly_loss(self):
        """
        Retrieves weekly QALY loss for the State, across outcomes.
        """
        return self.pandemicOutcomes.weeklyQALYLoss


class AllStates:

    def __init__(self):
        """
        Initialize an AllStates object.
        """

        self.states = {}  # dictionary of state objects
        self.pandemicOutcomes = PandemicOutcomes()
        self.numWeeks = 0
        self.population = 0

    def populate(self):
        """
        Populates the AllStates object with county case data.
        """

        county_case_data, dates = get_dict_of_county_data_by_type('cases')
        county_death_data, dates = get_dict_of_county_data_by_type('deaths')
        county_hosp_data, dates = get_dict_of_county_data_by_type('hospitalizations')

        self.numWeeks = len(dates)
        self.dates=dates

        for (county_name, state, fips, population), case_values in county_case_data.items():

            self.population += int(population)

            # making sure data is available for deaths and hospitalizations
            try:
                death_values = county_death_data[(county_name, state, fips, population)]
                hosp_values = county_hosp_data[(county_name, state, fips, population)]
            except KeyError as e:
                raise KeyError(f"Data not found for {county_name}, {state}, {fips}, {population}.") from e

            # create a new county
            county = County(
                name=county_name, state=state, fips=fips, population=int(population))

            # Add weekly data to county object
            county.add_traj(
                weekly_cases=case_values, weekly_deaths=death_values, weekly_hosp=hosp_values)

            # update the nation pandemic outcomes based on the outcomes for this county
            self.pandemicOutcomes.add_traj(
                weekly_cases=county.pandemicOutcomes.cases.weeklyObs,
                weekly_hosp=county.pandemicOutcomes.hosps.weeklyObs,
                weekly_deaths=county.pandemicOutcomes.deaths.weeklyObs)

            # create a new state if not already in the dictionary of states
            if state not in self.states:
                self.states[state] = State(name=state, num_weeks=self.numWeeks)

            # add the new county to the state
            self.states[state].add_county(county)

    def calculate_qaly_loss(self, param_values):
        """
        calculates the QALY loss for all states and the nation
        :param param_values: Parameter values to be used in calculating QALY loss.
        """

        # calculate QALY loss for each state
        for state in self.states.values():
            state.calculate_qaly_loss(
                case_weight=param_values.qWeightCase,
                hosp_weight=param_values.qWeightHosp,
                death_weight=param_values.qWeightDeath)

        # calculate QALY loss for the nation
        self.pandemicOutcomes.calculate_qaly_loss(
            case_weight=param_values.qWeightCase,
            hosp_weight=param_values.qWeightHosp,
            death_weight=param_values.qWeightDeath)

    def get_overall_qaly_loss(self):
        """
        :return: Overall QALY loss summed over all states.
        """
        return self.pandemicOutcomes.totalQALYLoss

    def get_weekly_qaly_loss(self):
        """
        :return: Weekly QALY losses across all states as numpy array
        """
        return self.pandemicOutcomes.weeklyQALYLoss

    def get_qaly_loss_by_outcome(self):
        """
        :return: (dictionary) of QALY loss by outcome with keys 'cases', 'hosps', 'deaths'
        """
        dict_results = {'cases': self.pandemicOutcomes.cases.totalQALYLoss,
                        'hosps': self.pandemicOutcomes.hosps.totalQALYLoss,
                        'deaths': self.pandemicOutcomes.deaths.totalQALYLoss}

        return dict_results

    def get_overall_qaly_loss_by_county(self):
        """
        Print the overall QALY loss for each county.
        :return: (dictionary) Overall QALY loss summed across timepoints for each county
        """

        overall_qaly_loss_by_county = {}
        for state_name, state_obj in self.states.items():
            for county_name, county_obj in state_obj.counties.items():
                overall_qaly_loss_by_county[state_name, county_name] = county_obj.pandemicOutcomes.totalQALYLoss
        return overall_qaly_loss_by_county

    def get_overall_qaly_loss_by_state(self):
        """
        :return: (dictionary) Overall QALY loss by states (as dictionary key)
        """
        overall_qaly_loss_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_by_state[state_name] = state_obj.pandemicOutcomes.totalQALYLoss
        return overall_qaly_loss_by_state

    def get_weekly_qaly_loss_by_state(self):
        """
        :return: (dictionary) The weekly QALY losses by states (as dictionary key)
        """
        weekly_qaly_loss_by_state ={}
        for state_name, state_obj in self.states.items():
            weekly_qaly_loss_by_state[state_name] = state_obj.pandemicOutcomes.weeklyQALYLoss

        return weekly_qaly_loss_by_state

    def get_weekly_qaly_loss_by_county(self):
        """
        :return: (dictionary) The weekly QALY losses by county and state (as dictionary keys)
        """
        weekly_qaly_loss_by_county = {}
        for state_name, state_obj in self.states.items():
            for county_name, county_obj in state_obj.counties.items():
                weekly_qaly_loss_by_county[state_name, county_name] = county_obj.pandemicOutcomes.weeklyQALYLoss

        return weekly_qaly_loss_by_county

    def get_overall_qaly_loss_for_a_county(self, county_name, state_name):
        """
        :param county_name: Name of the county.
        :param state_name: Name of the state.
        :return: Overall QALY loss for the specified county, summed over all timepoints
        """
        return self.states[state_name].counties[county_name].pandemicOutcomes.totalQALYLoss

    def get_overall_qaly_loss_for_a_state(self, state_name):
        """
        :param state_name: Name of the state.
        :return: Overall QALY loss for the specified state, summed over all timepoints.
        """

        return self.states[state_name].pandemicOutcomes.totalQALYLoss

    def get_weekly_qaly_loss_for_a_state(self, state_name):
        """
        :param state_name: Name of the state.
        :return: Weekly QALY loss for the specified state.
        """

        return self.states[state_name].pandemicOutcomes.weeklyQALYLoss

    def get_weekly_qaly_loss_for_a_county(self, county_name, state_name):
        """
        :param county_name: Name of the county.
        :param state_name: Name of the state.
        :return: Weekly QALY loss for the specified county.
        """

        return self.states[state_name].counties[county_name].pandemicOutcomes.weeklyQALYLoss

    def plot_weekly_qaly_loss_by_state(self):
        """
        :return: Plot of weekly QALY loss per 100,000 population for each state, with each state represented as a
        uniquely colored line
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for state_name, state_obj in self.states.items():
            weeks = range(1, len(state_obj.pandemicOutcomes.weeklyQALYLoss) + 1)

            # Calculate the weekly QALY loss per 100,000 population
            state_qaly_loss_per_100k = [(qaly_loss / state_obj.population) * 100000 for qaly_loss in
                                        state_obj.pandemicOutcomes.weeklyQALYLoss]

            ax.plot(weeks, state_qaly_loss_per_100k, label=state_name)

        ax.set_title('Weekly QALY Loss per 100,000 Population by State')
        ax.set_xlabel('Week')
        ax.set_ylabel('QALY Loss per 100,000 Population')
        ax.grid(True)

        plt.xticks(rotation=90)
        ax.tick_params(axis='x', labelsize=6.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10)
        plt.subplots_adjust(top=0.45)

        output_figure(fig, filename=ROOT_DIR + '/figs/weekly_qaly_loss_by_state.png')

    def plot_weekly_qaly_loss(self):
        """
        Plots National Weekly QALY Loss from Cases, Hospitalizations and Deaths across all states
        """

        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the total weekly QALY loss
        weeks = range(1, len(self.pandemicOutcomes.weeklyQALYLoss) + 1)
        ax.plot(weeks, self.pandemicOutcomes.weeklyQALYLoss)

        ax.set_title('National Weekly QALY Loss from Cases, Hospitalizations and Deaths')
        ax.set_xlabel('Date')
        ax.set_ylabel('QALY Loss')
        ax.grid(True)

        plt.xticks(rotation=90)
        ax.tick_params(axis='x', labelsize=6.5)

        output_figure(fig, filename=ROOT_DIR + '/figs/national_qaly_loss.png')

    def plot_map_of_qaly_loss_by_county(self):
        """
        Plots a map of the QALY loss per 100,000 population for each county, considering cases, deaths, and hospitalizations.
        """

        county_qaly_loss_data = {
            "COUNTY": [],
            "FIPS": [],
            "QALY Loss per 100K": []
        }

        for state in self.states.values():
            for county in state.counties.values():
                # Calculate the QALY loss per 100,000 population
                qaly_loss = county.pandemicOutcomes.totalQALYLoss
                qaly_loss_per_100k = (qaly_loss / county.population) * 100000
                # Append county data to the list
                county_qaly_loss_data["COUNTY"].append(county.name)
                county_qaly_loss_data["FIPS"].append(county.fips)
                county_qaly_loss_data["QALY Loss per 100K"].append(qaly_loss_per_100k)

        # Create a DataFrame from the county data
        county_qaly_loss_df = pd.DataFrame(county_qaly_loss_data)

        # Merge the county QALY loss data with the geometry data
        geoData = gpd.read_file(
            "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
        )
        geoData['STATE'] = geoData['STATE'].str.lstrip('0')
        geoData['FIPS'] = geoData['STATE'] + geoData['COUNTY']
        merged_geo_data = geoData.merge(county_qaly_loss_df, left_on='FIPS', right_on='FIPS', how='left')

        # Remove counties where there is no data
        merged_geo_data = merged_geo_data.dropna(subset=["QALY Loss per 100K"])

        # Remove Alaska, HI, Puerto Rico (to be plotted later)
        stateToRemove = ["02", "15", "72"]
        merged_geo_data_mainland = merged_geo_data[~merged_geo_data.STATE.isin(stateToRemove)]

        # Explode the MultiPolygon geometries into individual polygons
        merged_geo_data_mainland = merged_geo_data_mainland.explode()

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.set_aspect('equal')

        scheme = mc.Quantiles(merged_geo_data_mainland["QALY Loss per 100K"], k=10)
        gplt.choropleth(
            merged_geo_data_mainland,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'Cumulative QALY Loss per 100K'},
            edgecolor="black",
            ax=ax,
        )
        ax.set_xlim([-170.0, 60])
        ax.set_ylim([25, 76])
        plt.title("Cumulative County QALY Loss per 100K", fontsize=24)

        stateToInclude = ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])
        akax.axis('off')
        polygon_AK = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
        scheme_AK = mc.Quantiles(merged_geo_data_AK_exploded["QALY Loss per 100K"], k=2)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme_AK,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=akax,
            extent=(-180, -90, 50, 75)
        )

        akax.get_legend().remove()

        ## Hawai'i ##
        # fig_HI, ax_HI = plt.subplots(1, 1, figsize=(16, 12))
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])
        hiax.axis('off')
        hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
        scheme_HI = mc.Quantiles(merged_geo_data_HI_exploded["QALY Loss per 100K"], k=2)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=hiax,
        )

        hiax.get_legend().remove()


        plt.tight_layout()
        output_figure(fig, filename=ROOT_DIR + '/figs/map_county_qaly_loss.png')

        return fig


    def plot_weekly_qaly_loss_by_outcome(self):
        """
        Plots national weekly QALY Loss across all states broken down by cases, hospitalizations and deaths.
        """

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the lines for each outcome
        weeks = range(1, len(self.pandemicOutcomes.cases.weeklyQALYLoss) + 1)
        ax.plot(weeks, self.pandemicOutcomes.cases.weeklyQALYLoss, label='Cases', color='blue')
        ax.plot(weeks, self.pandemicOutcomes.hosps.weeklyQALYLoss, label='Hospitalizations', color='green')
        ax.plot(weeks, self.pandemicOutcomes.deaths.weeklyQALYLoss, label='Deaths', color='red')

        ax.set_title('Weekly National QALY Loss by Outcome ')
        ax.set_xlabel('Date')
        ax.set_ylabel('QALY Loss ')
        ax.grid(True)

        plt.xticks(rotation=90)
        ax.tick_params(axis='x', labelsize=6.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10)
        plt.subplots_adjust(top=0.45)

        plt.tight_layout()
        output_figure(fig, filename=ROOT_DIR + '/figs/national_weekly_qaly_loss_by_outcome.png')

    def plot_qaly_loss_by_state_and_by_outcome(self):
        """
        Generate a bar graph of the total QALY loss per 100,000 pop for each state, with each outcome's contribution
        represented in a different color.
        """

        num_states = len(self.states)
        states_list = list(self.states.values())

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set up the positions for the bars
        bar_positions = np.arange(num_states)

        # Set up the width for each state bar
        bar_width = 0.8

        # Set up colors for each segment
        cases_color = 'blue'
        deaths_color = 'red'
        hosps_color = 'green'

        # Iterate through each state
        for i, state_obj in enumerate(states_list):
            # Calculate the heights for each segment
            cases_height = (state_obj.pandemicOutcomes.cases.totalQALYLoss / state_obj.population) * 100000
            deaths_height = (state_obj.pandemicOutcomes.deaths.totalQALYLoss / state_obj.population) * 100000
            hosps_height = (state_obj.pandemicOutcomes.hosps.totalQALYLoss / state_obj.population) * 100000

            # Plot the segments
            ax.bar(i, cases_height, color=cases_color, width=bar_width, align='center', label='Cases' if i == 0 else "")
            ax.bar(i, deaths_height, bottom=cases_height, color=deaths_color, width=bar_width, align='center',
                   label='Deaths' if i == 0 else "")
            ax.bar(i, hosps_height, bottom=cases_height + deaths_height, color=hosps_color, width=bar_width,
                   align='center', label='Hospitalizations' if i == 0 else "")

        # Set the labels for each state
        ax.set_xticks(bar_positions)
        ax.set_xticklabels([state_obj.name for state_obj in states_list], fontsize=8, rotation=45, ha='right')

        # Set the labels and title
        ax.set_xlabel('States')
        ax.set_ylabel('Total QALY Loss per 100,000')
        ax.set_title('Total QALY Loss by State and Outcome')

        # Show the legend with unique labels
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Cases', 'Deaths', 'Hospitalizations'])

        plt.tight_layout()
        output_figure(fig, filename=ROOT_DIR + '/figs/total_qaly_loss_by_state_and_outcome.png')


    def get_overall_qaly_loss_by_state_cases(self):
        """
        :return: (dictionary) Overall QALY loss from cases by states (as dictionary key)
        """
        overall_qaly_loss_cases_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_cases_by_state[state_name] = state_obj.pandemicOutcomes.cases.totalQALYLoss
        return overall_qaly_loss_cases_by_state

    def get_overall_qaly_loss_by_state_hosps(self):
        """
        :return: (dictionary) Overall QALY loss from hosps by states (as dictionary key)
        """
        overall_qaly_loss_hosps_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_hosps_by_state[state_name] = state_obj.pandemicOutcomes.hosps.totalQALYLoss
        return overall_qaly_loss_hosps_by_state

    def get_overall_qaly_loss_by_state_deaths(self):
        """
        :return: (dictionary) Overall QALY loss from deaths by states (as dictionary key)
        """
        overall_qaly_loss_deaths_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_deaths_by_state[state_name] = state_obj.pandemicOutcomes.deaths.totalQALYLoss
        return overall_qaly_loss_deaths_by_state

class SummaryOutcomes:

    def __init__(self):

        # Lists for the outcomes of interest to collect
        self.overallQALYlosses = []
        self.overallQALYlossesByState = []
        self.overallQALYlossesByCounty = []
        self.overallQALYlossesCases = []
        self.overallQALYlossesHosps = []
        self.overallQALYlossesDeaths = []

        self.weeklyQALYlosses = []
        self.weeklyQALYlossesByState = []
        self.weeklyQALYlossesCases = []
        self.weeklyQALYlossesHosps = []
        self.weeklyQALYlossesDeaths = []

        self.overallQALYlossesCasesByState = []
        self.overallQALYlossesHospsByState = []
        self.overallQALYlossesDeathsByState = []

        self.overallQALYlossessByStateandOutcome =[]

        self.statOverallQALYLoss = None

    def summarize(self):

        self.statOverallQALYLoss = SummaryStat(data=self.overallQALYlosses)


    def get_mean_ci_ui_overall_qaly_loss(self):
        """
        :return: Mean, confidence interval, and uncertainty interval for overall QALY loss summed over all states.
        """
        return (self.statOverallQALYLoss.get_mean(),
                self.statOverallQALYLoss.get_t_CI(alpha=0.05),
                self.statOverallQALYLoss.get_PI(alpha=0.05))

class ProbabilisticAllStates:

    def __init__(self):

        # Create and populate an AllStates object
        self.allStates = AllStates()
        self.allStates.populate()
        self.summaryOutcomes = SummaryOutcomes()

    def simulate(self, n):
        """
        Simulates the model n times
        :param n: (int) number of times parameters should be sampled and the model simulated
        """

        rng = np.random.RandomState(1)
        param_gen = ParameterGenerator()

        for i in range(n):

            # Generate a new set of parameters
            params = param_gen.generate(rng=rng)

            # Calculate the QALY loss for this set of parameters
            self.allStates.calculate_qaly_loss(param_values=params)

            # Store outcomes
            self.summaryOutcomes.overallQALYlosses.append(self.allStates.get_overall_qaly_loss())
            self.summaryOutcomes.overallQALYlossesByState.append(self.allStates.get_overall_qaly_loss_by_state())
            self.summaryOutcomes.overallQALYlossesByCounty.append(self.allStates.get_overall_qaly_loss_by_county())

            self.summaryOutcomes.weeklyQALYlosses.append(self.allStates.get_weekly_qaly_loss())
            self.summaryOutcomes.weeklyQALYlossesByState.append(self.allStates.get_weekly_qaly_loss_by_state())

            self.summaryOutcomes.weeklyQALYlossesCases.append(self.allStates.pandemicOutcomes.cases.weeklyQALYLoss)
            self.summaryOutcomes.weeklyQALYlossesHosps.append(self.allStates.pandemicOutcomes.hosps.weeklyQALYLoss)
            self.summaryOutcomes.weeklyQALYlossesDeaths.append(self.allStates.pandemicOutcomes.deaths.weeklyQALYLoss)

            self.summaryOutcomes.overallQALYlossesCases.append(self.allStates.pandemicOutcomes.cases.totalQALYLoss)
            self.summaryOutcomes.overallQALYlossesHosps.append(self.allStates.pandemicOutcomes.hosps.totalQALYLoss)
            self.summaryOutcomes.overallQALYlossesDeaths.append(self.allStates.pandemicOutcomes.deaths.totalQALYLoss)


            self.summaryOutcomes.overallQALYlossesCasesByState.append(self.allStates.get_overall_qaly_loss_by_state_cases())
            self.summaryOutcomes.overallQALYlossesHospsByState.append(self.allStates.get_overall_qaly_loss_by_state_hosps())
            self.summaryOutcomes.overallQALYlossesDeathsByState.append(self.allStates.get_overall_qaly_loss_by_state_deaths())

        self.summaryOutcomes.summarize()

    def print_overall_qaly_loss(self):
        """
        :return: Prints the mean, confidence interval, and the uncertainty interval for the overall QALY loss .
        """

        mean, ci, ui = self.summaryOutcomes.get_mean_ci_ui_overall_qaly_loss()

        print('Overall QALY loss:')
        print('  Mean:', mean)
        print('  95% Confidence Interval:', ci)
        print('  95% Uncertainty Interval:', ui)

    def get_mean_ui_weekly_qaly_loss(self, alpha=0.05):
        """
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss.
        """

        mean, ui = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlosses, alpha=alpha)
        return mean, ui

    def plot_weekly_qaly_loss_by_outcome(self):
        """
        :return: Plots National Weekly QALY Loss from Cases, Hospitalizations and Deaths across all states
        """
        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 6))

        [mean_cases, ui_cases, mean_hosps, ui_hosps, mean_deaths, ui_deaths] = (
            self.get_mean_ui_weekly_qaly_loss_by_outcome(alpha=0.05))

        ax.plot(self.allStates.dates, mean_cases,
                label='QALY Loss Cases', linewidth=2, color='blue')
        ax.fill_between(self.allStates.dates, ui_cases[0], ui_cases[1], color='lightblue', alpha=0.25)

        ax.plot(self.allStates.dates, mean_hosps,
                label='QALY Loss Hospitalizations', linewidth=2, color='green')
        ax.fill_between(self.allStates.dates, ui_hosps[0], ui_hosps[1], color='lightgreen', alpha=0.25)

        ax.plot(self.allStates.dates, mean_deaths,
                label='QALY Loss Deaths', linewidth=2, color='red')
        ax.fill_between(self.allStates.dates, ui_deaths[0], ui_deaths[1], color='orange', alpha=0.25)

        [mean, ui] = self.get_mean_ui_weekly_qaly_loss(alpha=0.05)

        ax.plot(self.allStates.dates, mean,
                label='All outcomes', linewidth=2, color='black')
        ax.fill_between(self.allStates.dates, ui[0], ui[1], color='grey', alpha=0.25)
        ax.axvspan("2021-06-30","2021-10-27",alpha=0.2,color="lightblue")
        ax.axvspan("2021-10-27", "2022-12-28", alpha=0.2, color="grey")

        ax.set_title('National Weekly QALY Loss by Outcome')
        ax.set_xlabel('Date')
        ax.set_ylabel('QALY Loss')
        ax.legend()

        vals_y = ax.get_yticks()
        ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_y])
        # To label every other tick on the x-axis
        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
        plt.xticks(rotation=90)
        ax.tick_params(axis='x', labelsize=6.5)

        output_figure(fig, filename=ROOT_DIR + '/figs/simulations_national_qaly_loss_by_outcome.png')


    def get_mean_ui_weekly_qaly_loss_by_outcome(self, alpha=0.05):
        """
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss.
        """
        mean_cases, ui_cases = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesCases, alpha=alpha)
        mean_hosps, ui_hosps = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesHosps, alpha=alpha)
        mean_deaths, ui_deaths = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths, alpha=alpha)

        return mean_cases, ui_cases, mean_hosps, ui_hosps, mean_deaths, ui_deaths

    def plot_map_of_avg_qaly_loss_by_county(self):
        """
        Plots a map of the QALY loss per 100,000 population for each county, considering cases, deaths, and hospitalizations.
        """

        # TODO: is it possible to format the legends so that the numbers in the legend are whole numbers?

        county_qaly_loss_data = {
            "COUNTY": [],
            "FIPS": [],
            "QALY Loss per 100K": []
        }

        for state in self.allStates.states.values():
            for county in state.counties.values():
                # Calculate the QALY loss per 100,000 population
                mean, ui = self.get_mean_ui_overall_qaly_loss_by_county(state.name, county.name)
                qaly_loss = mean
                qaly_loss_per_100k = (qaly_loss / county.population) * 100000
                # Append county data to the list
                county_qaly_loss_data["COUNTY"].append(county.name)
                county_qaly_loss_data["FIPS"].append(county.fips)
                county_qaly_loss_data["QALY Loss per 100K"].append(qaly_loss_per_100k)

        # Create a DataFrame from the county data
        county_qaly_loss_df = pd.DataFrame(county_qaly_loss_data)

        # Merge the county QALY loss data with the geometry data
        geoData = gpd.read_file(
            "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
        )
        geoData['STATE'] = geoData['STATE'].str.lstrip('0')
        geoData['FIPS'] = geoData['STATE'] + geoData['COUNTY']
        merged_geo_data = geoData.merge(county_qaly_loss_df, left_on='FIPS', right_on='FIPS', how='left')

        # Remove counties where there is no data
        merged_geo_data = merged_geo_data.dropna(subset=["QALY Loss per 100K"])

        # Remove Alaska, HI, Puerto Rico (to be plotted later)
        stateToRemove = ["2", "15", "72"]
        merged_geo_data_mainland = merged_geo_data[~merged_geo_data.STATE.isin(stateToRemove)]

        # Explode the MultiPolygon geometries into individual polygons
        merged_geo_data_mainland = merged_geo_data_mainland.explode()

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))

        ax.axis = ('off')


        ax.set_title('Cumulative County QALY Loss per 100K', fontsize=42)

        scheme = mc.Quantiles(merged_geo_data_mainland["QALY Loss per 100K"], k=10)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'Cumulative QALY Loss per 100K', 'bbox_to_anchor': (1, 0.5)},
            edgecolor="black",
            ax=ax
        )


        plt.tight_layout()

        ## Alaska ##
        stateToInclude = ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])
        akax.axis('off')
        polygon_AK = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
        scheme_AK = mc.Quantiles(merged_geo_data_AK_exploded["QALY Loss per 100K"], k=2)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme_AK,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=akax,
            extent=(-180, -90, 50, 75)
        )

        akax.get_legend().remove()

        ## Hawai'i ##
        # fig_HI, ax_HI = plt.subplots(1, 1, figsize=(16, 12))
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])
        hiax.axis('off')
        hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
        scheme_HI = mc.Quantiles(merged_geo_data_HI_exploded["QALY Loss per 100K"], k=2)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=hiax,
        )

        hiax.get_legend().remove()

        output_figure(fig, filename=ROOT_DIR + '/figs/map_avg_county_qaly_loss_all_simulations.png')

        return fig

    def get_mean_ui_overall_qaly_loss_by_county(self, state_name, county_name, alpha=0.05):
        """
        :param state_name: Name of the state.
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss for a specific state.
        """

        county_qaly_losses = [qaly_losses[state_name, county_name] for qaly_losses in self.summaryOutcomes.overallQALYlossesByCounty]
        mean,ui = get_overall_mean_ui(county_qaly_losses, 0.05)
        return mean, ui

    def plot_weekly_qaly_loss_by_state(self):
        """
        :return: Folder containing independent plots of the weekly QALY loss for each state
        """

        # Create a folder to save individual state plots if it doesn't exist
        output_folder = ROOT_DIR + '/figs/weekly_qaly_loss_by_state_plots/'
        os.makedirs(output_folder, exist_ok=True)

        for i, (state_name, state_obj) in enumerate(self.allStates.states.items()):
            # Calculate the weekly QALY loss per 100,000 population
            mean, ui = self.get_mean_ui_weekly_qaly_loss_by_state(state_name, alpha=0.05)

            # Create a plot for the current state
            fig, ax = plt.subplots(figsize=(12, 6))
            self.format_weekly_qaly_plot(ax, state_name, mean, ui)

            # Set common labels and title
            ax.set_xlabel('Week')
            plt.xticks(rotation=90)
            ax.tick_params(axis='x', labelsize=6.5)

            # Save the plot with the state name in the filename
            filename = f"{output_folder}/{state_name}_weekly_qaly_loss.png"
            plt.savefig(filename)

    def subplot_weekly_qaly_loss_by_state_100K_pop(self):
        """
        :return: Plot composed of 52 state subplots of weekly QALY loss per 100K population
        """

        fig, axes =plt.subplots(nrows = 13, ncols = 4,figsize=(18,25))

        axes = np.ravel(axes)

        for i, (state_name, state_obj) in enumerate(self.allStates.states.items()):

            # Calculate the weekly QALY loss per 100,000 population
            mean, ui = self.get_mean_ui_weekly_qaly_loss_by_state(state_name, alpha=0.05)

            mean_per_100K_pop = np.array(mean)/state_obj.population
            ui_per_100K_pop = np.array(ui)/state_obj.population

            self.format_weekly_qaly_plot(axes[i], state_name, mean_per_100K_pop, ui_per_100K_pop)


        axes[-1].set_xlabel('Week')
        axes[-1].tick_params(axis='x', labelsize =6.5)

        plt.tight_layout()

        # Save the plot with the state name in the filename
        filename = ROOT_DIR + f"/figs/subplots_all_states_weekly_qaly_loss.png"
        output_figure(fig, filename)

    def get_mean_ui_weekly_qaly_loss_by_state(self, state_name, alpha=0.05):
        """
        :param state_name: Name of the state.
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss for a specific state.
        """

        state_qaly_losses = np.array(
            [qaly_losses[state_name] for qaly_losses in self.summaryOutcomes.weeklyQALYlossesByState])
        mean = np.mean(state_qaly_losses, axis=0)
        ui = np.percentile(state_qaly_losses, q=[alpha / 2 * 100, 100 - alpha / 2 * 100], axis=0)
        return mean, ui

    def format_weekly_qaly_plot(self, ax, state_name, mean, ui):
        """
        :param ax: ax object encapsulates elements of a plot in a figure
        :param mean: array of weekly mean values of QALY loss across simulations
        :param ui: 2 arrays of uncertainty intervals of the weekly values of QALY loss across simulations
        :return: Standardizes the formatting of plots of weekly QALY loss
        """
        ax.plot(range(1, len(mean) + 1), mean, label=f'Average', linewidth=2,
                color='black')
        ax.fill_between(range(1, len(ui[1]) + 1), ui[0], ui[1], color='lightgrey', alpha=0.5)

        ax.set_title(f'Weekly QALY Loss for {state_name}')
        ax.set_ylabel('QALY Loss')
        ax.grid(True)
        ax.legend()


    def plot_qaly_loss_by_state_and_by_outcome(self):
        """
        Generate a bar graph of the total QALY loss per 100,000 pop for each state, with each outcome's contribution
        represented in a different color.
        """

        states_list = list(self.allStates.states.values())
        # To sort states by overall QALY loss
        sorted_states = sorted(
            states_list,
            key=lambda state_obj: (self.get_mean_ui_overall_qaly_loss_by_state(
                state_name=state_obj.name, alpha=0.05)[0]/state_obj.population)*100000)

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(8, 10))

        # Set up the positions for the bars
        y_pos = (range(len(sorted_states)))
        ax.set_ylim([-1, len(sorted_states)])


        # Iterate through each state

        for i, state_obj in enumerate(sorted_states):
            # Calculate the heights for each segment
            mean_cases, ui_cases, mean_hosps, ui_hosps, mean_deaths, ui_deaths =(
                self.get_mean_ui_overall_qaly_loss_by_outcome_and_by_state(state_name=state_obj.name, alpha=0.05))
            mean_total, ui_total = self.get_mean_ui_overall_qaly_loss_by_state(state_obj.name, alpha=0.05)
            cases_height = (mean_cases / state_obj.population) * 100000
            deaths_height = (mean_deaths / state_obj.population) * 100000
            hosps_height = (mean_hosps / state_obj.population) * 100000
            total_height = (mean_total/ state_obj.population)*100000

            #Converting UI into error bars
            cases_ui = (ui_cases / state_obj.population) * 100000
            deaths_ui = (ui_deaths / state_obj.population) * 100000
            hosps_ui = (ui_hosps / state_obj.population) * 100000
            total_ui = (ui_total/state_obj.population)*100000

            xterr_cases = [[cases_height-cases_ui[0]], [cases_ui[1]-cases_height]]
            xterr_deaths = [[deaths_height-deaths_ui[0]], [deaths_ui[1]-deaths_height]]
            xterr_hosps = [[hosps_height-hosps_ui[0]], [hosps_ui[1]-hosps_height]]
            xterr_total = [[total_height-total_ui[0]], [total_ui[1]-total_height]]


            ax.scatter(cases_height, [state_obj.name], marker='o', color='blue', label='cases')
            ax.errorbar(cases_height, [state_obj.name], xerr=xterr_cases, fmt='none', color='blue', capsize=0, alpha=0.4)
            ax.scatter(hosps_height, [state_obj.name], marker='o', color='green', label='hosps')
            ax.errorbar(hosps_height, [state_obj.name],xerr=xterr_hosps, fmt='none', color='green', capsize=0, alpha=0.4)
            ax.scatter(deaths_height, [state_obj.name], marker='o', color='red', label='deaths')
            ax.errorbar(deaths_height, [state_obj.name],xerr=xterr_deaths, fmt='none', color='red', capsize=0, alpha=0.4)
            ax.scatter(total_height, [state_obj.name], marker='o', color='black', label='total')
            ax.errorbar(total_height, [state_obj.name],xerr=xterr_total, fmt='none', color='black', capsize=0, alpha=0.4)

        # Set the labels for each state
        ax.set_yticks(y_pos)
        ax.set_yticklabels([state_obj.name for state_obj in states_list], fontsize=8, rotation=0, ha='right')

        # Set the labels and title
        ax.set_ylabel('States')
        ax.set_xlabel('Total QALY Loss per 100,000')
        ax.set_title('Total QALY Loss by State and Outcome')

        # Show the legend with unique labels
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Cases',  'Hospitalizations','Deaths', 'Total'])

        plt.tight_layout()
        output_figure(fig, filename=ROOT_DIR + '/figs/total_qaly_loss_by_state_and_outcome.png')


    def get_mean_ui_overall_qaly_loss_by_state(self, state_name, alpha=0.05):
        """
        :param state_name: Name of the state.
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss for a specific state.
        """

        state_qaly_losses = [qaly_losses[state_name] for qaly_losses in self.summaryOutcomes.overallQALYlossesByState]
        mean,ui = get_overall_mean_ui(state_qaly_losses, 0.05)
        return mean, ui

    def get_mean_ui_overall_qaly_loss_by_outcome_and_by_state(self, state_name, alpha=0.05):
        """
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss.
        """
        state_cases_qaly_losses = [qaly_loss[state_name] for qaly_loss in self.summaryOutcomes.overallQALYlossesCasesByState]
        state_hosps_qaly_losses = [qaly_losses[state_name] for qaly_losses in self.summaryOutcomes.overallQALYlossesHospsByState]
        state_deaths_qaly_losses = [qaly_losses[state_name] for qaly_losses in self.summaryOutcomes.overallQALYlossesDeathsByState]

        mean_cases, ui_cases = get_overall_mean_ui(state_cases_qaly_losses, alpha=alpha)
        mean_hosps, ui_hosps = get_overall_mean_ui(state_hosps_qaly_losses, alpha=alpha)
        mean_deaths, ui_deaths = get_overall_mean_ui(state_deaths_qaly_losses, alpha=alpha)

        return mean_cases, ui_cases, mean_hosps, ui_hosps, mean_deaths, ui_deaths

