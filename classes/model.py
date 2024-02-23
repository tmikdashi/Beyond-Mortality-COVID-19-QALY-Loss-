import os

import geopandas as gpd
import geoplot as gplt
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from classes.parameters import ParameterGenerator
from classes.support import get_mean_ui_of_a_time_series, get_overall_mean_ui
from data_preprocessing.support_functions import get_dict_of_county_data_by_type
from deampy.in_out_functions import write_csv, read_csv_rows
from deampy.format_functions import format_interval
from deampy.plots.plot_support import output_figure
from deampy.statistics import SummaryStat
from matplotlib.ticker import ScalarFormatter
from definitions import ROOT_DIR



class AnOutcome:

    def __init__(self):
        self.weeklyObs = np.array([])
        self.totalObs = 0
        self.weeklyQALYLoss = np.array([])
        self.totalQALYLoss = 0

        self.prevaxWeeklyObs = np.array([])
        self.postvaxWeeklyObs = np.array([])
        self.prevaxTotalObs = 0
        self.postvaxTotalObs = 0
        self.prevaxWeeklyQALYLoss = np.array([])
        self.postvaxWeeklyQALYLoss = np.array([])
        self.prevaxTotalQALYLoss= 0
        self.postvaxTotalQALYLoss = 0
        self.vaccination_index = 35

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

    def add_vax_traj(self, weekly_obs):
        if not isinstance(weekly_obs, np.ndarray):
            weekly_obs = np.array(weekly_obs)

        # replace missing values with 0
        weekly_obs = np.nan_to_num(weekly_obs, nan=0)

        # If self.weeklyObs is empty, initialize prevaxWeeklyObs and postvaxWeeklyObs
        if len(self.weeklyObs) == 0:
            self.prevaxWeeklyObs = weekly_obs[:self.vaccination_index]
            self.postvaxWeeklyObs = weekly_obs[self.vaccination_index:]
        else:
            # add the weekly data to the existing data
            self.prevaxWeeklyObs += weekly_obs[:self.vaccination_index]
            self.postvaxWeeklyObs += weekly_obs[self.vaccination_index:]

        self.prevaxTotalObs += sum(weekly_obs[:self.vaccination_index])
        self.postvaxTotalObs += sum(weekly_obs[self.vaccination_index:])

    def calculate_qaly_loss(self, quality_weight):
        """
        Calculates the weekly and overall QALY
        :param quality_weight: Weight to be applied to each case in calculating QALY loss.
        :return Weekly QALY loss as a numpy array or numerical values to total QALY loss.
        """
        self.weeklyQALYLoss = quality_weight * self.weeklyObs
        self.totalQALYLoss = sum(self.weeklyQALYLoss)

        self.prevaxWeeklyQALYLoss = quality_weight * self.prevaxWeeklyObs
        self.prevaxTotalQALYLoss = sum(self.prevaxWeeklyQALYLoss)

        self.postvaxWeeklyQALYLoss = quality_weight * self.postvaxWeeklyObs
        self.postvaxTotalQALYLoss = sum(self.postvaxWeeklyQALYLoss)


class PandemicOutcomes:
    def __init__(self):

        self.cases = AnOutcome()
        self.hosps = AnOutcome()
        self.deaths = AnOutcome()

        self.weeklyQALYLoss = np.array([])
        self.totalQALYLoss = 0

        self.prevaxWeeklyQALYLoss = np.array([])
        self.prevaxTotalQALYLoss = 0

        self.postvaxWeeklyQALYLoss = np.array([])
        self.postvaxTotalQALYLoss = 0

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

    def add_vax_traj(self, weekly_cases, weekly_hosp, weekly_deaths):
        """
        Add weekly cases, hospitalization, and deaths and calculate the total cases, hospitalizations, and deaths.
        :param weekly_cases: Weekly cases data as a numpy array.
        :param weekly_hosp: Weekly hospitalizations data as a numpy array.
        :param weekly_deaths: Weekly deaths data as a numpy array.
        """
        self.cases.add_vax_traj(weekly_obs=weekly_cases)
        self.hosps.add_vax_traj(weekly_obs=weekly_hosp)
        self.deaths.add_vax_traj(weekly_obs=weekly_deaths)

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

        self.prevaxWeeklyQALYLoss =self.cases.prevaxWeeklyQALYLoss + self.hosps.prevaxWeeklyQALYLoss + self.deaths.prevaxWeeklyQALYLoss
        self.prevaxTotalQALYLoss = self.cases.prevaxTotalQALYLoss + self.hosps.prevaxTotalQALYLoss + self.deaths.prevaxTotalQALYLoss

        self.postvaxWeeklyQALYLoss = self.cases.postvaxWeeklyQALYLoss + self.hosps.postvaxWeeklyQALYLoss + self.deaths.postvaxWeeklyQALYLoss
        self.postvaxTotalQALYLoss = self.cases.postvaxTotalQALYLoss + self.hosps.postvaxTotalQALYLoss + self.deaths.postvaxTotalQALYLoss



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

    def add_vax_traj(self, weekly_cases, weekly_deaths, weekly_hosp):
        """
        Add weekly data to the County object.
        :param weekly_cases: Weekly cases data as a numpy array.
        :param weekly_hosp: Weekly hospitalizations data as a numpy array.
        :param weekly_deaths: Weekly deaths data as a numpy array.
        """
        self.pandemicOutcomes.add_vax_traj(
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

    def get_vax_overall_qaly_loss(self):
        """
        Retrieves overall QALY loss before and after vaccination for the County, across outcomes.
        """
        return self.pandemicOutcomes.prevaxTotalQALYLoss, self.pandemicOutcomes.postvaxTotalQALYLoss

    def get_vax_weekly_qaly_loss(self):
        """
        Retrieves weekly QALY loss before and after vaccination for the County, across outcomes.
        """
        return self.pandemicOutcomes.prevaxWeeklyQALYLoss, self.pandemicOutcomes.postvaxWeeklyQALYLoss


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
        self.pandemicOutcomes.add_vax_traj(
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

    def get_vax_overall_qaly_loss(self):
        """
        Retrieves overall QALY loss before and after vaccination for the County, across outcomes.
        """
        return self.pandemicOutcomes.prevaxTotalQALYLoss, self.pandemicOutcomes.postvaxTotalQALYLoss

    def get_vax_weekly_qaly_loss(self):
        """
        Retrieves weekly QALY loss before and after vaccination for the County, across outcomes.
        """
        return self.pandemicOutcomes.prevaxWeeklyQALYLoss, self.pandemicOutcomes.postvaxWeeklyQALYLoss


class AllStates:

    def __init__(self):
        """
        Initialize an AllStates object.
        """

        self.states = {}  # dictionary of state objects
        self.pandemicOutcomes = PandemicOutcomes()
        self.numWeeks = 0
        self.population = 0
        self.dates = []

    def populate(self):
        """
        Populates the AllStates object with county case data.
        """

        county_case_data, dates = get_dict_of_county_data_by_type('cases')
        county_death_data, dates = get_dict_of_county_data_by_type('deaths')
        county_hosp_data, dates = get_dict_of_county_data_by_type('hospitalizations')

        self.numWeeks = len(dates)
        self.dates = dates

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

            county.add_vax_traj(
                weekly_cases=case_values, weekly_deaths=death_values, weekly_hosp=hosp_values)

            # update the nation pandemic outcomes based on the outcomes for this county
            self.pandemicOutcomes.add_vax_traj(
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
        merged_geo_data = merged_geo_data.dropna(subset=["QALY Loss"])

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
        plt.title("Cumulative County QALY Loss per 100,000 Population", fontsize=22)

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

    def get_death_QALY_loss_by_age(self, param_gen):
        deaths_by_age = (param_gen.parameters['death_age_dist'].value * self.pandemicOutcomes.deaths.totalObs)
        total_dQAlY_loss_by_age = deaths_by_age * param_gen.parameters['dQALY_loss_by_age'].value
        return total_dQAlY_loss_by_age


    def get_prevax_overall_qaly_loss_by_state_cases(self):
        overall_prevax_qaly_loss_cases_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_prevax_qaly_loss_cases_by_state[state_name] = state_obj.pandemicOutcomes.cases.prevaxTotalQALYLoss
        return  overall_prevax_qaly_loss_cases_by_state

    def get_prevax_overall_qaly_loss_by_state_hosps(self):
        overall_prevax_qaly_loss_hosps_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_prevax_qaly_loss_hosps_by_state[state_name] = state_obj.pandemicOutcomes.hosps.prevaxTotalQALYLoss
        return  overall_prevax_qaly_loss_hosps_by_state

    def get_prevax_overall_qaly_loss_by_state_deaths(self):
        overall_prevax_qaly_loss_deaths_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_prevax_qaly_loss_deaths_by_state[state_name] = state_obj.pandemicOutcomes.deaths.prevaxTotalQALYLoss
        return  overall_prevax_qaly_loss_deaths_by_state

    def get_postvax_overall_qaly_loss_by_state_cases(self):
        overall_postvax_qaly_loss_cases_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_postvax_qaly_loss_cases_by_state[state_name] = state_obj.pandemicOutcomes.cases.postvaxTotalQALYLoss
        return overall_postvax_qaly_loss_cases_by_state

    def get_postvax_overall_qaly_loss_by_state_hosps(self):
        overall_postvax_qaly_loss_hosps_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_postvax_qaly_loss_hosps_by_state[state_name] = state_obj.pandemicOutcomes.hosps.postvaxTotalQALYLoss
        return overall_postvax_qaly_loss_hosps_by_state

    def get_postvax_overall_qaly_loss_by_state_deaths(self):
        overall_postvax_qaly_loss_deaths_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_postvax_qaly_loss_deaths_by_state[state_name] = state_obj.pandemicOutcomes.deaths.postvaxTotalQALYLoss
        return overall_postvax_qaly_loss_deaths_by_state

    def get_prevax_total_qaly_loss_by_state(self):
        return {state_name: state_obj.pandemicOutcomes.prevaxTotalQALYLoss for state_name, state_obj in self.states.items()}

    def get_postvax_total_qaly_loss_by_state(self):
        return {state_name: state_obj.pandemicOutcomes.postvaxTotalQALYLoss for state_name, state_obj in self.states.items()}

    def get_prevax_weekly_qaly_loss_by_state(self):
        return {state_name: state_obj.pandemicOutcomes.prevaxWeeklyQALYLoss for state_name, state_obj in self.states.items()}

    def get_postvax_weekly_qaly_loss_by_state(self):
        return {state_name: state_obj.pandemicOutcomes.postvaxWeeklyQALYLoss for state_name, state_obj in self.states.items()}




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

        self.prevaxOverallQALYLossesCasesByState = []
        self.prevaxOverallQALYLossesHospsByState = []
        self.prevaxOverallQALYLossesDeathsByState = []

        self.postvaxOverallQALYLossesCasesByState = []
        self.postvaxOverallQALYLossesHospsByState = []
        self.postvaxOverallQALYLossesDeathsByState = []


        self.statOverallQALYLoss = None
        self.statOverallQALYLossCases = None
        self.statOverallQALYLossHosps = None
        self.statOverallQALYLossDeaths = None

        self.deathQALYLossByAge = []
        self.age_group = []



    def extract_outcomes(self, simulated_model,param_gen):

        self.overallQALYlosses.append(simulated_model.get_overall_qaly_loss())
        self.overallQALYlossesByState.append(simulated_model.get_overall_qaly_loss_by_state())
        self.overallQALYlossesByCounty.append(simulated_model.get_overall_qaly_loss_by_county())

        self.weeklyQALYlosses.append(simulated_model.get_weekly_qaly_loss())
        self.weeklyQALYlossesByState.append(simulated_model.get_weekly_qaly_loss_by_state())

        self.weeklyQALYlossesCases.append(simulated_model.pandemicOutcomes.cases.weeklyQALYLoss)
        self.weeklyQALYlossesHosps.append(simulated_model.pandemicOutcomes.hosps.weeklyQALYLoss)
        self.weeklyQALYlossesDeaths.append(simulated_model.pandemicOutcomes.deaths.weeklyQALYLoss)

        self.overallQALYlossesCases.append(simulated_model.pandemicOutcomes.cases.totalQALYLoss)
        self.overallQALYlossesHosps.append(simulated_model.pandemicOutcomes.hosps.totalQALYLoss)
        self.overallQALYlossesDeaths.append(simulated_model.pandemicOutcomes.deaths.totalQALYLoss)

        self.overallQALYlossesCasesByState.append(simulated_model.get_overall_qaly_loss_by_state_cases())
        self.overallQALYlossesHospsByState.append(simulated_model.get_overall_qaly_loss_by_state_hosps())
        self.overallQALYlossesDeathsByState.append(simulated_model.get_overall_qaly_loss_by_state_deaths())

        self.prevaxOverallQALYLossesCasesByState.append(simulated_model.get_prevax_overall_qaly_loss_by_state_cases())
        self.prevaxOverallQALYLossesHospsByState.append(simulated_model.get_prevax_overall_qaly_loss_by_state_hosps())
        self.prevaxOverallQALYLossesDeathsByState.append(simulated_model.get_prevax_overall_qaly_loss_by_state_deaths())

        self.postvaxOverallQALYLossesCasesByState.append(simulated_model.get_postvax_overall_qaly_loss_by_state_cases())
        self.postvaxOverallQALYLossesHospsByState.append(simulated_model.get_postvax_overall_qaly_loss_by_state_hosps())
        self.postvaxOverallQALYLossesDeathsByState.append(simulated_model.get_postvax_overall_qaly_loss_by_state_deaths())

        self.deathQALYLossByAge.append(simulated_model.get_death_QALY_loss_by_age(param_gen))


    def summarize(self):

        self.statOverallQALYLoss = SummaryStat(data=self.overallQALYlosses)
        self.statOverallQALYLossCases = SummaryStat(data=self.overallQALYlossesCases)
        self.statOverallQALYLossHosps = SummaryStat(data=self.overallQALYlossesHosps)
        self.statOverallQALYLossDeaths = SummaryStat(data=self.overallQALYlossesDeaths)

    def get_mean_ci_ui_overall_qaly_loss(self):
        """
        :return: Mean, confidence interval, and uncertainty interval for overall QALY loss summed over all states.
        """
        return (self.statOverallQALYLoss.get_mean(),
                self.statOverallQALYLoss.get_t_CI(alpha=0.05),
                self.statOverallQALYLoss.get_PI(alpha=0.05),
                self.statOverallQALYLossCases.get_mean(),
                self.statOverallQALYLossCases.get_t_CI(alpha=0.05),
                self.statOverallQALYLossCases.get_PI(alpha=0.05),
                self.statOverallQALYLossHosps.get_mean(),
                self.statOverallQALYLossHosps.get_t_CI(alpha=0.05),
                self.statOverallQALYLossHosps.get_PI(alpha=0.05),
                self.statOverallQALYLossDeaths.get_mean(),
                self.statOverallQALYLossDeaths.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths.get_PI(alpha=0.05),
                )


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
        self.age_group = param_gen.parameters['Age Group'].value


        for i in range(n):

            # Generate a new set of parameters
            params = param_gen.generate(rng=rng)

            # Calculate the QALY loss for this set of parameters
            self.allStates.calculate_qaly_loss(param_values=params)
            self.allStates.get_death_QALY_loss_by_age(param_gen=param_gen)

            # extract outcomes from the simulated all states
            self.summaryOutcomes.extract_outcomes(simulated_model=self.allStates, param_gen=param_gen)

        self.summaryOutcomes.summarize()

    def plot_qaly_loss_from_deaths_by_age(self):

        mean, ui = get_mean_ui_of_a_time_series(self.summaryOutcomes.deathQALYLossByAge, alpha=0.05)
        print('mean', mean)
        print('ui', ui)
        print('lower error', ui[1]-mean)
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.bar(self.age_group, mean)
        ax.errorbar(self.age_group, mean,yerr=[mean-ui[0],ui[1]- mean],fmt='none', color='black', capsize=0, alpha=0.8)

        ax.set_title('QALY Loss from Deaths by Age', size=20)
        ax.set_xlabel('Age Groups', size=16)
        ax.set_ylabel('QALY Loss', size=16)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))


        output_figure(fig, filename=ROOT_DIR + '/figs/death_qaly_loss_by_age.png')


    def print_overall_outcomes_and_qaly_loss(self):
        """
        :return: Prints the mean, confidence interval, and the uncertainty interval for the overall QALY loss .
        """

        mean_cases = self.allStates.pandemicOutcomes.cases.totalObs
        mean_hosps = self.allStates.pandemicOutcomes.hosps.totalObs
        mean_deaths = self.allStates.pandemicOutcomes.deaths.totalObs

        print("weekly national hosps", self.allStates.pandemicOutcomes.deaths.totalObs)
        print('Overall Outcomes:')
        print('  Mean Cases: {:,.0f}'.format(mean_cases))
        print('  Mean Hospitalizations: {:,.0f}'.format(mean_hosps))
        print('  Mean Deaths: {:,.0f}'.format(mean_deaths))

        mean, ci, ui,  mean_cases, ci_c, ui_c,mean_hosps, ci_h, ui_h,mean_deaths, ci_d, ui_d = self.summaryOutcomes.get_mean_ci_ui_overall_qaly_loss()

        print('Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean))
        print('  95% Confidence Interval:', format_interval(ci, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui, deci=0, format=','))

        #mean_cases, ci_c, ui_c = self.summaryOutcomes.get_mean_ci_ui_overall_qaly_loss()

        print('Overall QALY loss:')
        print('  Mean cases: {:,.0f}'.format(mean_cases))
        print('  95% Confidence Interval:', format_interval(ci_c, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_c, deci=0, format=','))

        #mean_hosps, ci_h, ui_h = self.summaryOutcomes.get_mean_ci_ui_overall_qaly_loss()

        print('Overall QALY loss:')
        print('  Mean hosps: {:,.0f}'.format(mean_hosps))
        print('  95% Confidence Interval:', format_interval(ci_h, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_h, deci=0, format=','))

        #mean_deaths, ci_d, ui_d = self.summaryOutcomes.get_mean_ci_ui_overall_qaly_loss()

        print('Overall QALY loss:')
        print('  Mean deaths: {:,.0f}'.format(mean_deaths))
        print('  95% Confidence Interval:', format_interval(ci_d, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_d, deci=0, format=','))


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
                label='Cases', linewidth=2, color='blue')
        ax.fill_between(self.allStates.dates, ui_cases[0], ui_cases[1], color='lightblue', alpha=0.25)

        ax.plot(self.allStates.dates, mean_hosps,
                label='Hospital admissions', linewidth=2, color='green')
        ax.fill_between(self.allStates.dates, ui_hosps[0], ui_hosps[1], color='lightgreen', alpha=0.25)

        ax.plot(self.allStates.dates, mean_deaths,
                label='Deaths', linewidth=2, color='red')
        ax.fill_between(self.allStates.dates, ui_deaths[0], ui_deaths[1], color='orange', alpha=0.25)


        [mean, ui] = self.get_mean_ui_weekly_qaly_loss(alpha=0.05)

        ax.plot(self.allStates.dates, mean,
                label='All health states', linewidth=2, color='black')
        ax.fill_between(self.allStates.dates, ui[0], ui[1], color='grey', alpha=0.25)
        ax.axvspan("2021-06-30","2021-10-27",alpha=0.2,color="lightblue") #delta variant
        ax.axvspan("2021-10-27", "2022-12-28", alpha=0.2, color="grey") # omicron variant

        ax.axvline("2021-03-24", linestyle='--', color='grey', label='70% of 65+ years population vaccinated')
        ax.axvline ("2021-08-11", linestyle= '--', color='black', label='70% of population vaccinated')

        ax.set_title('National Weekly QALY Loss by Health State')
        ax.set_xlabel('Date')
        ax.set_ylabel('QALY Loss')
        ax.legend()

        vals_y = ax.get_yticks()
        ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_y])
        # To label every other tick on the x-axis
        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
        plt.xticks(rotation=45)
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
            "QALY Loss per 100K": [],
            "QALY Loss": []
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
                county_qaly_loss_data["QALY Loss"].append(qaly_loss)

        # Create a DataFrame from the county data
        county_qaly_loss_df = pd.DataFrame(county_qaly_loss_data)
        # Print top 10 highest QALY loss
        top_10_highest_loss = county_qaly_loss_df.nlargest(10, "QALY Loss")
        print("Top 10 Counties with Highest QALY Loss:")
        print(top_10_highest_loss[["COUNTY", "QALY Loss"]])

        # Print top 10 highest QALY loss per 100K
        top_10_highest_loss_per_100k = county_qaly_loss_df.nlargest(10, "QALY Loss per 100K")
        print("\nTop 10 Counties with Highest QALY Loss per 100K:")
        print(top_10_highest_loss_per_100k[["COUNTY", "QALY Loss per 100K"]])

        # Print top 10 lowest QALY loss
        top_10_lowest_loss = county_qaly_loss_df.nsmallest(10, "QALY Loss")
        print("\nTop 10 Counties with Lowest QALY Loss:")
        print(top_10_lowest_loss[["COUNTY", "QALY Loss"]])

        # Print top 10 lowest QALY loss per 100K
        top_10_lowest_loss_per_100k = county_qaly_loss_df.nsmallest(10, "QALY Loss per 100K")
        print("\nTop 10 Counties with Lowest QALY Loss per 100K:")
        print(top_10_lowest_loss_per_100k[["COUNTY", "QALY Loss per 100K"]])

        county_qaly_loss_df.to_csv(ROOT_DIR + '/csv_files/county_qaly_loss.csv', index=False)

        # Merge the county QALY loss data with the geometry data
        geoData = gpd.read_file(
            "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
        )
        geoData['STATE'] = geoData['STATE'].str.lstrip('0')
        geoData['FIPS'] = geoData['STATE'] + geoData['COUNTY']
        merged_geo_data = geoData.merge(county_qaly_loss_df, left_on='FIPS', right_on='FIPS', how='left')

        # Remove counties where there is no data
        merged_geo_data = merged_geo_data.dropna(subset=["QALY Loss"])


        # Remove Alaska, HI, Puerto Rico (to be plotted later)
        stateToRemove = ["2", "15", "72"]
        merged_geo_data_mainland = merged_geo_data[~merged_geo_data.STATE.isin(stateToRemove)]

        # Explode the MultiPolygon geometries into individual polygons
        merged_geo_data_mainland = merged_geo_data_mainland.explode()

        # Plot the map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4),subplot_kw={'aspect': 'equal'})

        ax1.axis('off')
        ax1.set_title('Cumulative County QALY Loss', fontsize=15)

        scheme = mc.Quantiles(merged_geo_data_mainland["QALY Loss"], k=10)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="QALY Loss",
            linewidth=0.1,
            scheme=scheme,
            cmap="viridis",
            legend=True,
            #legend_kwds=dict(fmt='{:.0f}', interval=True),
            legend_kwargs={'title': 'Absolute QALY Loss', 'bbox_to_anchor': (1, 0.68)},
            edgecolor="black",
            ax=ax1
        )

        #ax1.legend_.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

        plt.tight_layout()

        ## Alaska ##
        stateToInclude= ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax1 = fig.add_axes([0.01, -0.2, 0.3, 0.5])
        akax1.axis('off')
        polygon_AK = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
        scheme_AK = mc.Quantiles(merged_geo_data_AK_exploded["QALY Loss"], k=2)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="QALY Loss",
            linewidth=0.1,
            scheme=scheme_AK,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=akax1,
            extent=(-180, -90, 50, 75)
        )

        akax1.get_legend().remove()

        ## Hawai'i ##
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax1 = fig.add_axes([.07, 0.20, 0.1, 0.15])
        hiax1.axis('off')
        hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
        scheme_HI = mc.Quantiles(merged_geo_data_HI_exploded["QALY Loss"], k=2)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="QALY Loss",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=hiax1,
        )

        hiax1.get_legend().remove()

        ax2.axis('off')
        ax2.set_title('Cumalative County QALY Loss per 100,000 Population', fontsize=15)

        scheme = mc.Quantiles(merged_geo_data_mainland["QALY Loss per 100K"], k=10)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme,
            cmap="viridis",
            legend=True,
            #legend_labels=dict(fmt='{:.0f}', interval=True),
            legend_kwargs={'title': 'QALY Loss per 100K', 'bbox_to_anchor': (1, 0.68)},
            edgecolor="black",
            ax=ax2
        )

        #ax2.legend_.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

        plt.tight_layout()

        ## Alaska ##
        stateToInclude = ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax2 = fig.add_axes([0.15, -0.2, 1.0, 0.5])
        akax2.axis('off')
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
            ax=akax2,
            extent=(-180, -90, 50, 75)
        )

        akax2.get_legend().remove()

        ## Hawai'i ##
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax2 = fig.add_axes([.28, 0.20, 0.7, 0.15])
        hiax2.axis('off')
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
            ax=hiax2,
        )

        hiax2.get_legend().remove()

        output_figure(fig, filename=ROOT_DIR + '/figs/map_avg_county_qaly_loss_all_simulations.png')

        return fig


    def plot_map_of_avg_qaly_loss_by_county_alt(self):
        """
        Vertically plots a map of the QALY loss per 100,000 population for each county, considering cases, deaths, and hospitalizations.
        """

        # TODO: is it possible to format the legends so that the numbers in the legend are whole numbers?

        county_qaly_loss_data = {
            "COUNTY": [],
            "FIPS": [],
            "QALY Loss per 100K": [],
            "QALY Loss": []
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
                county_qaly_loss_data["QALY Loss"].append(qaly_loss)

        # Create a DataFrame from the county data
        county_qaly_loss_df = pd.DataFrame(county_qaly_loss_data)


        county_qaly_loss_df.to_csv(ROOT_DIR + '/csv_files/county_qaly_loss.csv', index=False)

        # Merge the county QALY loss data with the geometry data
        geoData = gpd.read_file(
            "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
        )
        geoData['STATE'] = geoData['STATE'].str.lstrip('0')
        geoData['FIPS'] = geoData['STATE'] + geoData['COUNTY']
        merged_geo_data = geoData.merge(county_qaly_loss_df, left_on='FIPS', right_on='FIPS', how='left')

        # Remove counties where there is no data
        merged_geo_data = merged_geo_data.dropna(subset=["QALY Loss"])


        # Remove Alaska, HI (to be plotted later)
        stateToRemove = ["2", "15"]
        merged_geo_data_mainland = merged_geo_data[~merged_geo_data.STATE.isin(stateToRemove)]

        # Explode the MultiPolygon geometries into individual polygons
        merged_geo_data_mainland = merged_geo_data_mainland.explode()

        # Plot the map
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),subplot_kw={'aspect': 'equal'})

        ax1.axis('off')
        ax1.set_title('Cumulative County QALY Loss', fontsize=15)

        scheme = mc.Quantiles(merged_geo_data_mainland["QALY Loss"], k=10)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="QALY Loss",
            linewidth=0.1,
            scheme=scheme,
            cmap="viridis",
            legend=True,
            #legend_kwds=dict(fmt='{:.0f}', interval=True),
            legend_kwargs={'title': 'Absolute QALY Loss', 'bbox_to_anchor': (0.9, 0.68)},
            edgecolor="black",
            ax=ax1
        )

        plt.tight_layout()

        ## Alaska ##
        stateToInclude= ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax1 = fig.add_axes([-0.2, 0.15, 1, 0.5])
        akax1.axis('off')
        polygon_AK = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
        scheme_AK = mc.Quantiles(merged_geo_data_AK_exploded["QALY Loss"], k=2)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="QALY Loss",
            linewidth=0.1,
            scheme=scheme_AK,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=akax1,
            extent=(-180, -90, 50, 75)
        )

        akax1.get_legend().remove()

        ## Hawai'i ##
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax1 = fig.add_axes([0.1, 0.25, 0.2, 0.15])
        hiax1.axis('off')
        hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
        scheme_HI = mc.Quantiles(merged_geo_data_HI_exploded["QALY Loss"], k=2)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="QALY Loss",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=hiax1,
        )

        hiax1.get_legend().remove()

        ax2.axis('off')
        ax2.set_title('Cumalative County QALY Loss per 100,000 Population', fontsize=15)

        scheme = mc.Quantiles(merged_geo_data_mainland["QALY Loss per 100K"], k=10)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme,
            cmap="viridis",
            legend=True,
            #legend_labels=dict(fmt='{:.0f}', interval=True),
            legend_kwargs={'title': 'QALY Loss per 100K', 'bbox_to_anchor': (0.9, 0.68)},
            edgecolor="black",
            ax=ax2
        )

        plt.tight_layout()

        ## Alaska ##
        stateToInclude = ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax2 = fig.add_axes([-0.1, -0.40, 1, 0.5])
        akax2.axis('off')
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
            ax=akax2,
            extent=(-180, -90, 50, 75)
        )

        akax2.get_legend().remove()

        ## Hawai'i ##
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax2 = fig.add_axes([.28, -0.20, 0.3, 0.15])
        hiax2.axis('off')
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
            ax=hiax2,
        )

        hiax2.get_legend().remove()

        output_figure(fig, filename=ROOT_DIR + '/figs/map_avg_county_qaly_loss_all_simulations_alt.png')

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
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(8, 10))


        states_list = list(self.allStates.states.values())
        # To sort states by overall QALY loss
        sorted_states = sorted(
            states_list,
            key=lambda state_obj: (self.get_mean_ui_overall_qaly_loss_by_state(
                state_name=state_obj.name, alpha=0.05)[0] / state_obj.population) * 100000)

        # Set up the positions for the bars


        y_pos = (range(len(sorted_states)))

        democratic_states = ['AZ','CA','CO','CT','DE','HI','IL','KS','KY','ME','MD','MA','MI','MN','NJ','NM','NY','NC','OR','PA','RI','WA','WI']
        republican_states = ['AL', 'AK', 'AR', 'FL', 'GA', 'ID', 'IN','IA', 'LA','MI','MS','MO','MT','NE','NH','NV','ND','OH','OK','SC','SD','TN','TX',
                             'UT','VT','VA','WV','WY']

        # Iterate through each state

        for i, state_obj in enumerate(sorted_states):
            # Calculate the heights for each segment
            mean_cases, ui_cases, mean_hosps, ui_hosps, mean_deaths, ui_deaths =(
                self.get_mean_ui_overall_qaly_loss_by_outcome_and_by_state(state_name=state_obj.name, alpha=0.05))
            mean_total, ui_total = self.get_mean_ui_overall_qaly_loss_by_state(state_obj.name, alpha=0.05)
            cases_height = (mean_cases / state_obj.population) * 100000
            deaths_height_test = (mean_deaths / state_obj.population) * 100000
            hosps_height = (mean_hosps / state_obj.population) * 100000
            total_height = (mean_total/ state_obj.population)*100000


            #Converting UI into error bars
            cases_ui = (ui_cases / state_obj.population) * 100000
            deaths_ui = (ui_deaths / state_obj.population) * 100000
            hosps_ui = (ui_hosps / state_obj.population) * 100000
            total_ui = (ui_total/state_obj.population)*100000

            xterr_cases = [[cases_height-cases_ui[0]], [cases_ui[1]-cases_height]]
            xterr_deaths = [[deaths_height_test-deaths_ui[0]], [deaths_ui[1]-deaths_height_test]]
            xterr_hosps = [[hosps_height-hosps_ui[0]], [hosps_ui[1]-hosps_height]]
            xterr_total = [[total_height-total_ui[0]], [total_ui[1]-total_height]]


            ax.scatter(cases_height, [state_obj.name], marker='o', color='blue', label='cases')
            ax.errorbar(cases_height, [state_obj.name], xerr=xterr_cases, fmt='none', color='blue', capsize=0, alpha=0.4)
            ax.scatter(hosps_height, [state_obj.name], marker='o', color='green', label='hospital admissions')
            ax.errorbar(hosps_height, [state_obj.name],xerr=xterr_hosps, fmt='none', color='green', capsize=0, alpha=0.4)
            ax.scatter(deaths_height_test, [state_obj.name], marker='o', color='red', label='deaths')
            ax.errorbar(deaths_height_test, [state_obj.name],xerr=xterr_deaths, fmt='none', color='red', capsize=0, alpha=0.4)
            ax.scatter(total_height, [state_obj.name], marker='o', color='black', label='total')
            ax.errorbar(total_height, [state_obj.name],xerr=xterr_total, fmt='none', color='black', capsize=0, alpha=0.4)

        # Set the labels for each state
        ax.set_yticks(y_pos)
        y_tick_colors = ['blue' if state_obj.name in democratic_states else 'red' for state_obj in sorted_states]
        ax.set_yticklabels([state_obj.name for state_obj in sorted_states], fontsize=12, rotation=0)

        # Set the colors for ticks
        for tick, color in zip(ax.yaxis.get_major_ticks(), y_tick_colors):
            tick.label1.set_color(color)

        # Set the labels and title
        ax.set_ylabel('States', fontsize=14)
        ax.set_xlabel('Total QALY Loss per 100,000', fontsize=14)
        ax.set_title('State-level QALY Loss by Health State')

        # Show the legend with unique labels
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Cases', 'Hospital Admissions','Deaths', "All Health States "])

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

    def plot_map_of_hsa_outcomes_by_county_per_100K(self):
        """
        Generates sub-plotted maps of the number of cases, hospital admissions, and deaths per 100,000 population for each county.
        Values are computed per HSA (aggregate of county values for all counties within an HSA), but plotted by county.
        """

        # Load HSA data
        hsa_data = read_csv_rows(file_name='C:/Users/fm478/Downloads/county_names_HSA_number.csv',
                                 if_ignore_first_row=True)

        # Create a dictionary for FIPS to HSA mapping
        fips_to_hsa_mapping = {str(entry[4]): (entry[6], float(entry[8].replace(',', ''))) for entry in hsa_data}

        # List to store counties without corresponding HSA
        counties_without_hsa = []

        # Dictionary to store HSA totals
        hsa_totals_dict = {}

        # Dictionary to store aggregated values for each HSA
        hsa_aggregated_data = {}

        county_outcomes_data = {
            "COUNTY": [],
            "FIPS": [],
            "County Population": [],
            "HSA Number": [],
            "Cases": [],
            "Hosps": [],
            "Deaths": [],
            "HSA Total Cases per 100K": [],
            "HSA Total Hospitalizations per 100K": [],
            "HSA Total Deaths per 100K": [],
            "HSA Population": []
        }

        # Iterate over all states and counties
        for state in self.allStates.states.values():
            for county in state.counties.values():
                fips_code = county.fips
                hsa_info = fips_to_hsa_mapping.get(fips_code, (None, None))

                hsa_number, hsa_population = hsa_info

                if hsa_number is not None:
                    hsa_number = int(hsa_number)
                    hsa_population = int(hsa_population)

                    # Check if HSA entry exists in the dictionary
                    if hsa_number not in hsa_totals_dict:
                        hsa_totals_dict[hsa_number] = {
                            "Total Cases": 0,
                            "Total Hospitalizations": 0,
                            "Total Deaths": 0,
                            "Population": hsa_population
                        }

                    # Append county data to the list
                    county_outcomes_data["COUNTY"].append(county.name)
                    county_outcomes_data["FIPS"].append(county.fips)
                    county_outcomes_data["County Population"].append(county.population)
                    county_outcomes_data["HSA Number"].append(hsa_number)
                    county_outcomes_data["Cases"].append(county.pandemicOutcomes.cases.totalObs)
                    county_outcomes_data["Hosps"].append(county.pandemicOutcomes.hosps.totalObs)
                    county_outcomes_data["Deaths"].append(county.pandemicOutcomes.deaths.totalObs)
                    county_outcomes_data["HSA Population"].append(hsa_population)

                    # Update aggregated values for HSA
                    if hsa_number not in hsa_aggregated_data:
                        hsa_aggregated_data[hsa_number] = {
                            "Total Cases": 0,
                            "Total Hospitalizations": 0,
                            "Total Deaths": 0,
                            "Population": hsa_population
                        }
                    hsa_aggregated_data[hsa_number]["Total Cases"] += county.pandemicOutcomes.cases.totalObs
                    #hsa_aggregated_data[hsa_number]["Total Hospitalizations"] += county.pandemicOutcomes.hosps.totalObs
                    hsa_aggregated_data[hsa_number]["Total Deaths"] += county.pandemicOutcomes.deaths.totalObs

                else:
                    hsa_number = None
                    hsa_population = None

                    # Append county name, state, and FIPS to the list of counties without HSA
                    counties_without_hsa.append({
                        "County Name": county.name,
                        "State": state.name,  # Change this according to your structure
                        "FIPS": county.fips
                    })


        # Update the HSA Total values in county_outcomes_data
        for i in range(len(county_outcomes_data["HSA Number"])):
            hsa_number = county_outcomes_data["HSA Number"][i]
            if hsa_number is not None:
                hsa_number = int(hsa_number)
                hsa_total_cases_per_100K = (hsa_aggregated_data[hsa_number]["Total Cases"] / float(
                    hsa_aggregated_data[hsa_number]["Population"])) * 100000
                #hsa_total_hospitalizations_per_100K = (hsa_aggregated_data[hsa_number][
                                                           #"Total Hospitalizations"] / float(
                    #hsa_aggregated_data[hsa_number]["Population"])) * 100000
                hsa_total_hospitalizations_per_100K = county_outcomes_data["Hosps"][i] / float (
                    hsa_aggregated_data[hsa_number]["Population"]) * 100000
                hsa_total_deaths_per_100K = (hsa_aggregated_data[hsa_number]["Total Deaths"] / float(
                    hsa_aggregated_data[hsa_number]["Population"])) * 100000

                county_outcomes_data["HSA Total Cases per 100K"].append(hsa_total_cases_per_100K)
                county_outcomes_data["HSA Total Hospitalizations per 100K"].append(hsa_total_hospitalizations_per_100K)
                county_outcomes_data["HSA Total Deaths per 100K"].append(hsa_total_deaths_per_100K)
            else:
                # If HSA Number is None, set corresponding HSA Total values to None
                county_outcomes_data["HSA Total Cases per 100K"].append(None)
                county_outcomes_data["HSA Total Hospitalizations per 100K"].append(None)
                county_outcomes_data["HSA Total Deaths per 100K"].append(None)

        # Create a DataFrame from the county data
        county_outcomes_df = pd.DataFrame(county_outcomes_data)

        county_outcomes_df.to_csv(ROOT_DIR + '/csv_files/county_outcomes_by_hsa.csv', index=False)

        # Print the count of counties without HSA
        print(f"Number of counties without HSA: {len(counties_without_hsa)}")

        # Print counties without HSA
        print("Counties without HSA:")
        for county_info in counties_without_hsa:
            print(f"{county_info['County Name']} ({county_info['State']}, FIPS: {county_info['FIPS']})")

        # Merge the county QALY loss data with the geometry data
        geoData = gpd.read_file(
            "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
        )
        geoData['STATE'] = geoData['STATE'].str.lstrip('0')
        geoData['FIPS'] = geoData['STATE'] + geoData['COUNTY']
        merged_geo_data = geoData.merge(county_outcomes_df, left_on='FIPS', right_on='FIPS', how='left')

        # Remove counties where there is no data
        merged_geo_data = merged_geo_data.dropna(subset=["HSA Total Deaths per 100K"])

        # Remove Alaska, HI, Puerto Rico (to be plotted later)
        stateToRemove = ["2", "15", "72"]
        merged_geo_data_mainland = merged_geo_data[~merged_geo_data.STATE.isin(stateToRemove)]

        # Explode the MultiPolygon geometries into individual polygons
        merged_geo_data_mainland = merged_geo_data_mainland.explode()

        # Plot the map
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), subplot_kw={'aspect': 'equal'})

        ax1.axis('off')
        ax1.set_title('Cases per 100K', fontsize=15)

        scheme_cases = mc.Quantiles(merged_geo_data_mainland["HSA Total Cases per 100K"], k=10)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="HSA Total Cases per 100K",
            linewidth=0.1,
            scheme=scheme_cases,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'Cases per 100K', 'fontsize': 10, 'bbox_to_anchor': (0.95, 0.5),
                           'loc': 'center left'},
            legend_labels=None,
            edgecolor="black",
            ax=ax1
        )

        stateToInclude = ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax1 = fig.add_axes([0.15, 0.39, 0.3, 0.5])
        akax1.axis('off')
        polygon_AK = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
        scheme_AK = mc.Quantiles(merged_geo_data_AK_exploded["HSA Total Cases per 100K"], k=2)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="HSA Total Cases per 100K",
            linewidth=0.1,
            scheme=scheme_AK,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=akax1,
            extent=(-180, -90, 50, 75)
        )

        akax1.get_legend().remove()

        ## Hawai'i ##
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax1 = fig.add_axes([0.2, 0.65, 0.1, 0.15])
        hiax1.axis('off')
        hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
        scheme_HI = mc.Quantiles(merged_geo_data_HI_exploded["HSA Total Cases per 100K"], k=2)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="HSA Total Cases per 100K",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=hiax1,
        )

        hiax1.get_legend().remove()

        ax2.axis('off')
        ax2.set_title('Hospital Admissions per 100K', fontsize=15)

        scheme_hosps = mc.Quantiles(merged_geo_data_mainland["HSA Total Hospitalizations per 100K"], k=10)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="HSA Total Hospitalizations per 100K",
            linewidth=0.1,
            scheme=scheme_hosps,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'Hospital Admissions per 100K', 'fontsize': 10, 'bbox_to_anchor': (0.95, 0.5),
                           'loc': 'center left'},
            legend_labels=None,
            edgecolor="black",
            ax=ax2
        )

        stateToInclude = ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax2 = fig.add_axes([0.15, 0.06, 0.3, 0.5])
        akax1.axis('off')
        polygon_AK = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
        scheme_AK = mc.Quantiles(merged_geo_data_AK_exploded["HSA Total Hospitalizations per 100K"], k=2)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="HSA Total Hospitalizations per 100K",
            linewidth=0.1,
            scheme=scheme_AK,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=akax2,
            extent=(-180, -90, 50, 75)
        )

        akax2.get_legend().remove()

        ## Hawai'i ##
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax2 = fig.add_axes([0.2, 0.32, 0.1, 0.15])
        hiax2.axis('off')
        hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
        scheme_HI = mc.Quantiles(merged_geo_data_HI_exploded["HSA Total Hospitalizations per 100K"], k=2)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="HSA Total Hospitalizations per 100K",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=hiax2,
        )

        hiax2.get_legend().remove()

        ax3.axis('off')
        ax3.set_title('Deaths per 100K', fontsize=15)

        scheme = mc.Quantiles(merged_geo_data_mainland["HSA Total Deaths per 100K"], k=10)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="HSA Total Deaths per 100K",
            linewidth=0.1,
            scheme=scheme,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'Deaths per 100K', 'fontsize': 10, 'bbox_to_anchor': (0.95, 0.5),
                           'loc': 'center left'},
            legend_labels=None,
            edgecolor="black",
            ax=ax3
        )

        stateToInclude = ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax3 = fig.add_axes([0.15, -0.25, 0.3, 0.5])
        akax3.axis('off')
        polygon_AK = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
        scheme_AK = mc.Quantiles(merged_geo_data_AK_exploded["HSA Total Deaths per 100K"], k=2)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="HSA Total Deaths per 100K",
            linewidth=0.1,
            scheme=scheme_AK,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=akax3,
            extent=(-180, -90, 50, 75)
        )

        akax3.get_legend().remove()

        ## Hawai'i ##
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax3 = fig.add_axes([0.2, 0.01, 0.1, 0.15])
        hiax1.axis('off')
        hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
        scheme_HI = mc.Quantiles(merged_geo_data_HI_exploded["HSA Total Deaths per 100K"], k=2)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="HSA Total Deaths per 100K",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=hiax3,
        )

        hiax3.get_legend().remove()

        plt.subplots_adjust(hspace=0.01)

        plt.tight_layout()

        output_figure(fig, filename=ROOT_DIR + '/figs/map_county_outcomes_per_100K.png')

    def plot_weekly_outcomes(self):
        """
        :return: Plots National Weekly QALY Loss from Cases, Hospitalizations and Deaths across all states
        """
        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 6))

        cases = self.allStates.pandemicOutcomes.cases.weeklyObs
        hosps = self.allStates.pandemicOutcomes.hosps.weeklyObs
        deaths = self.allStates.pandemicOutcomes.deaths.weeklyObs

        ax.plot(self.allStates.dates, cases, label='Cases', linewidth=2, color='blue')
        ax.plot(self.allStates.dates, hosps, label='Hospital Admissions', linewidth=2, color='green')

        # Create a secondary y-axis for deaths
        ax2 = ax.twinx()
        ax2.plot(self.allStates.dates, deaths, label='Deaths', linewidth=2, color='red')

        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 300000])  # Set the secondary y-axis limit for deaths

        ax.axvspan("2021-06-30", "2021-10-27", alpha=0.2, color="lightblue")
        ax.axvspan("2021-10-27", "2022-12-28", alpha=0.2, color="grey")

        ax.set_title('Number of Weekly Cases, Hospital Admissions, and Deaths in the U.S.')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Cases and Hospital Admissions')

        # Move the number of deaths further to the right
        ax2.set_ylabel('Number of Deaths', rotation=270, labelpad=15, color= 'red')

        # Combine legend for deaths, cases, and hospital admissions
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        vals_y = ax.get_yticks()
        vals_y2 = ax2.get_yticks()

        ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_y])
        ax2.set_yticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_y2])

        # To label every other tick on the x-axis

        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % 8 != 0]
        ax.tick_params(axis='x', labelsize=6.5)
        plt.xticks(rotation=45)


        output_figure(fig, filename=ROOT_DIR + '/figs/national_outcomes.png')



    def subplot_weekly_cases_by_state_100K_pop(self):
        """
        :return: Plot composed of 52 state subplots of weekly QALY loss per 100K population
        """

        fig, axes =plt.subplots(nrows = 17, ncols = 3,figsize=(18,25), sharey=True)

        axes = np.ravel(axes)

        fig.suptitle('Weekly cases per 100K population', fontsize=16)


        for i, (state_name, state_obj) in enumerate(self.allStates.states.items()):

            # Calculate the weekly QALY loss per 100,000 population
            cases_per_100K = (state_obj.pandemicOutcomes.cases.weeklyObs/state_obj.population) *100000
            dates = self.allStates.dates

            axes[i].plot(dates, cases_per_100K, linewidth=2,
                  color='black')
            axes[i].set_title(f'{state_name}')
            axes[i].set_xlabel('Week', fontsize=10)
            axes[i].tick_params(axis='x', labelsize=6.5)

        plt.tight_layout(rect=[0, 0, 1, 0.94])


        # Save the plot with the state name in the filename
        filename = ROOT_DIR + f"/figs/subplots_all_states_weekly_cases.png"
        output_figure(fig, filename)


    def subplot_weekly_hosps_by_state_100K_pop(self):
        """
        :return: Plot composed of 52 state subplots of weekly QALY loss per 100K population
        """

        fig, axes =plt.subplots(nrows = 17, ncols = 3,figsize=(18,25), sharey=True)

        axes = np.ravel(axes)

        fig.suptitle('Weekly hospitalizations per 100K population', fontsize=16)


        for i, (state_name, state_obj) in enumerate(self.allStates.states.items()):

            # Calculate the weekly QALY loss per 100,000 population
            hosps_per_100K = (state_obj.pandemicOutcomes.hosps.weeklyObs/state_obj.population) *100000
            dates = self.allStates.dates

            axes[i].plot(dates, hosps_per_100K, linewidth=2,
                  color='black')
            axes[i].set_title(f'{state_name}')
            axes[i].set_xlabel('Week')
            axes[i].tick_params(axis='x', labelsize=6.5)

        plt.tight_layout(rect=[0, 0, 1, 0.94])


        # Save the plot with the state name in the filename
        filename = ROOT_DIR + f"/figs/subplots_all_states_weekly_hosps.png"
        output_figure(fig, filename)



    def subplot_weekly_deaths_by_state_100K_pop(self):
        """
        :return: Plot composed of 52 state subplots of weekly QALY loss per 100K population
        """

        fig, axes =plt.subplots(nrows = 17, ncols = 3,figsize=(18,25), sharey=True)

        axes = np.ravel(axes)

        fig.suptitle('Weekly deaths per 100K population', fontsize=16)


        for i, (state_name, state_obj) in enumerate(self.allStates.states.items()):

            # Calculate the weekly QALY loss per 100,000 population
            deaths_per_100K = (state_obj.pandemicOutcomes.deaths.weeklyObs/state_obj.population) *100000
            dates = self.allStates.dates

            axes[i].plot(dates, deaths_per_100K, linewidth=2,
                  color='black')
            axes[i].set_title(f' {state_name}')
            axes[i].set_xlabel('Week')
            axes[i].tick_params(axis='x', labelsize=6.5)

        plt.tight_layout(rect=[0, 0, 1, 0.94])


        # Save the plot with the state name in the filename
        filename = ROOT_DIR + f"/figs/subplots_all_states_weekly_deaths.png"
        output_figure(fig, filename)

    def plot_map_of_pop_over_65_by_county(self):
        """
        Plots a map of the percentage of the county's population that is over 65.
        """

        age_data = pd.read_csv('/Users/fm478/Downloads/cc-est2022-agesex-all.csv',
                               converters={'COUNTY': str, 'STATE': str})
        age_data['FIPS'] = age_data['STATE'] + age_data['COUNTY']

        # Filter rows where the year is 3
        age_data = age_data[age_data['YEAR'] == 3]

        age_data['Percentage pop over 65'] = (age_data['AGE65PLUS_TOT'] / age_data['POPESTIMATE']) * 100
        print(age_data)

        # Merge the county QALY loss data with the geometry data
        geoData = gpd.read_file(
            "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
        )

        geoData['FIPS'] = geoData['STATE'] + geoData['COUNTY']
        print(geoData)
        merged_geo_data = geoData.merge(age_data, left_on='FIPS', right_on='FIPS', how='left')

        # Remove counties where there is no data
        merged_geo_data = merged_geo_data.dropna(subset=["Percentage pop over 65"])
        print(merged_geo_data)

        # Remove Alaska, HI, Puerto Rico (to be plotted later)
        stateToRemove = ["02", "15", "72"]
        merged_geo_data_mainland = merged_geo_data[~merged_geo_data.STATE_x.isin(stateToRemove)]

        # Explode the MultiPolygon geometries into individual polygons
        merged_geo_data_mainland = merged_geo_data_mainland.explode()

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))

        ax.axis = ('off')

        ax.set_title('Percentage of Population over 65 by County', fontsize=42)

        scheme = mc.Quantiles(merged_geo_data_mainland["Percentage pop over 65"], k=10)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="Percentage pop over 65",
            linewidth=0.1,
            scheme=scheme,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'Percentage pop over 65', 'bbox_to_anchor': (1, 0.5)},
            edgecolor="black",
            ax=ax
        )

        plt.tight_layout()

        ## Alaska ##
        stateToInclude = ["02"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE_x.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])
        akax.axis('off')
        polygon_AK = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
        scheme_AK = mc.Quantiles(merged_geo_data_AK_exploded["Percentage pop over 65"], k=2)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="Percentage pop over 65",
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
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE_x.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])
        hiax.axis('off')
        hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
        scheme_HI = mc.Quantiles(merged_geo_data_HI_exploded["Percentage pop over 65"], k=2)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="Percentage pop over 65",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=hiax,
        )

        hiax.get_legend().remove()

        output_figure(fig, filename=ROOT_DIR + '/figs/map_pop_over_65.png')

        return fig

    def plot_map_of_median_age_by_county(self):
        """
        Plots a map of the median of the county's population.
        """


        age_data = pd.read_csv('/Users/fm478/Downloads/cc-est2022-agesex-all.csv', converters={'COUNTY': str,'STATE': str})
        # Filter rows where the year is 3
        age_data = age_data[age_data['YEAR'] == 3]
        age_data['FIPS'] =  age_data['STATE'] + age_data['COUNTY']


        # Merge the county QALY loss data with the geometry data
        geoData = gpd.read_file(
            "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
        )


        geoData['FIPS'] = geoData['STATE'] + geoData['COUNTY']

        merged_geo_data = geoData.merge(age_data, left_on='FIPS', right_on='FIPS', how='left')
        # Remove counties where there is no data
        merged_geo_data = merged_geo_data.dropna(subset=["MEDIAN_AGE_TOT"])

        # Remove Alaska, HI, Puerto Rico (to be plotted later)
        stateToRemove = ["02", "15", "72"]
        merged_geo_data_mainland = merged_geo_data[~merged_geo_data.STATE_x.isin(stateToRemove)]

        # Explode the MultiPolygon geometries into individual polygons
        merged_geo_data_mainland = merged_geo_data_mainland.explode()

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))

        ax.axis = ('off')


        ax.set_title('Median Age by County', fontsize=42)

        scheme = mc.Quantiles(merged_geo_data_mainland["MEDIAN_AGE_TOT"], k=10)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="MEDIAN_AGE_TOT",
            linewidth=0.1,
            scheme=scheme,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'Median Age', 'bbox_to_anchor': (1, 0.5)},
            edgecolor="black",
            ax=ax
        )


        plt.tight_layout()

        ## Alaska ##
        stateToInclude = ["02"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE_x.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])
        akax.axis('off')
        polygon_AK = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
        scheme_AK = mc.Quantiles(merged_geo_data_AK_exploded["MEDIAN_AGE_TOT"], k=2)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="MEDIAN_AGE_TOT",
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
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE_x.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()

        hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])
        hiax.axis('off')
        hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
        scheme_HI = mc.Quantiles(merged_geo_data_HI_exploded["MEDIAN_AGE_TOT"], k=2)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="MEDIAN_AGE_TOT",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=True,
            edgecolor="black",
            ax=hiax,
        )

        hiax.get_legend().remove()

        output_figure(fig, filename=ROOT_DIR + '/figs/map_median_age.png')

        return fig


    def plot_date_70pct_vaccinated_by_state(self):
        """
        Generate a bar graph of the date when 70% vaccinated for each state, with colors based on political affiliation.
        """
        vax_df = pd.read_csv(ROOT_DIR + '/csv_files/vaccinated_percentage_by_state.csv')
        vax_df['Date_Dose1_Over_70Pct_Vaccinated'] = pd.to_datetime(vax_df['Date_Dose1_Over_70Pct_Vaccinated'],
                                                                    errors='coerce')

        # Filter locations based on whether they appear in the 'states' data
        valid_locations = [state.name for state in self.allStates.states.values()] + ['US']

        # Create a dictionary mapping states to political affiliations
        state_to_affiliation = {
            'AZ': 'Republican', 'CA': 'Democratic', 'CO': 'Democratic', 'CT': 'Democratic', 'DE': 'Democratic',
            'HI': 'Democratic', 'IL': 'Democratic', 'KS': 'Republican', 'KY': 'Republican', 'ME': 'Democratic',
            'MD': 'Democratic', 'MA': 'Democratic', 'MI': 'Democratic', 'MN': 'Democratic', 'NJ': 'Democratic',
            'NM': 'Democratic', 'NY': 'Democratic', 'NC': 'Republican', 'OR': 'Democratic', 'PA': 'Democratic',
            'RI': 'Democratic', 'WA': 'Democratic', 'WI': 'Democratic', 'US': 'Unknown'
            # Set US to 'Unknown' or any other value
            # Add more states as needed
        }

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set the x-axis limits to July 2020 - November 2022 with weekly frequency
        date_range = pd.date_range(start='2021-07-01', end='2023-04-30', freq='W')
        ax.set_xlim(date_range.min(), date_range.max())

        # Create a DataFrame for states without a date
        no_date_df = vax_df[vax_df['Date_Dose1_Over_70Pct_Vaccinated'].isna()]

        # Sort DataFrame by date (US at the top, earlier dates next, then no dates)
        sorted_df = vax_df.sort_values(by=['Location', 'Date_Dose1_Over_70Pct_Vaccinated'], ascending=[False, True])

        # Keep track of labeled states to avoid overlapping labels
        labeled_states = set()

        # Iterate through each state with a date
        for _, row in sorted_df.dropna(subset=['Date_Dose1_Over_70Pct_Vaccinated']).iterrows():
            state = row['Location']

            # Skip if the state is not in the 'states' data or if the date is before 2021
            if state not in valid_locations:
                continue

            date_dose1_over_70 = row['Date_Dose1_Over_70Pct_Vaccinated']

            # Determine political affiliation based on the mapping
            political_affiliation = state_to_affiliation.get(state, 'Unknown')

            # Plot a filled bar with the color of the political affiliation
            color = 'black' if pd.isna(
                date_dose1_over_70) else 'blue' if political_affiliation == 'Democratic' else 'red'
            ax.barh(state, pd.Timestamp('2022-11-30') if pd.isna(date_dose1_over_70) else date_dose1_over_70,
                    color=color, edgecolor='black', linewidth=1)

        # Iterate through each state without a date (hollow bars)
        for _, row in no_date_df.iterrows():
            state = row['Location']

            # Skip if the state is not in the 'states' data
            if state not in valid_locations:
                continue

            # Plot a hollow bar with black lining
            ax.barh(state, date_range.max(), color='none', edgecolor='black', linewidth=1)

        # Set the labels and title
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('State', fontsize=14)
        ax.set_title('Date When 70% of Population Vaccinated by at least 1 Dose by State and Political Affiliation')

        # Hide y-axis labels on the right side
        ax.tick_params(axis='y', right=False)

        # Rotate y-axis labels for better visibility
        plt.yticks(rotation=0)

        # Set the colors for ticks
        y_tick_colors = ['blue' if state_to_affiliation.get(state, 'Unknown') == 'Democratic' else 'red' for state in
                         valid_locations]
        for tick, color in zip(ax.yaxis.get_major_ticks(), y_tick_colors):
            tick.label1.set_color(color)

        plt.tight_layout()
        output_figure(fig, filename=ROOT_DIR + '/figs/bar_plot_date_70pct_vaccinated_by_state.png')

    def get_state_vax_index(self):

        vax_df = pd.read_csv(ROOT_DIR + '/csv_files/vaccinated_percentage_by_state.csv')
        dates = self.allStates.dates
        dates = [pd.to_datetime(date) for date in dates]

        # Convert 'Date_Dose1_Over_70Pct_Vaccinated' to datetime
        vax_df['Date_Dose1_Over_70Pct_Vaccinated'] = pd.to_datetime(vax_df['Date_Dose1_Over_70Pct_Vaccinated'],
                                                                       errors='coerce')

        # Initialize a new column to store the index of the closest date
        vax_df['Index_Closest_Date'] = np.nan

        # Iterate through each state
        for index, row in vax_df.iterrows():
            # Get the state and the date of dose1 over 70%
            state = row['Location']
            date_dose1_over_70 = row['Date_Dose1_Over_70Pct_Vaccinated']

            # Find the closest date in 'dates' for the given state
            closest_date = min(dates, key=lambda x: abs((x - date_dose1_over_70).days))

            # Get the index of the closest date
            index_closest_date = dates.index(closest_date)

            # Update the 'Index_Closest_Date' column
            vax_df.at[index, 'Index_Closest_Date'] = index_closest_date

        # Display the updated DataFrame
        print(vax_df)

        # Iterate through each state
        states_list = list(self.allStates.states.values())

        # Iterate through each state
        for state_obj in states_list:
            # Get the index from the result_df
            index_closest_date = vax_df[vax_df['Location'] == state_obj.name]['Index_Closest_Date'].values[0]
            print(state_obj.name)
            print('Index Closest Date', index_closest_date)

            # Get the weekly QALY losses for the state
            state_qaly_losses = np.array(
                [qaly_losses[state_obj.name] for qaly_losses in self.summaryOutcomes.weeklyQALYlossesByState])

            # Check if index_closest_date is a valid number
            if pd.notna(index_closest_date):
                index_closest_date = int(index_closest_date)

                # Split the array into pre-vax and post-vax based on the index
                prevax_qaly_losses = state_qaly_losses[:index_closest_date + 1]
                postvax_qaly_losses = state_qaly_losses[index_closest_date + 1:]

                # Sum values for pre-vax and post-vax
                prevax_total_qaly_loss = np.sum(prevax_qaly_losses)
                postvax_total_qaly_loss = np.sum(postvax_qaly_losses)
            else:
                # If index_closest_date is NaN or invalid, consider all values in pre-vax and post-vax
                prevax_total_qaly_loss = np.sum(state_qaly_losses)
                postvax_total_qaly_loss = 0  # Assuming postvax_total_qaly_loss is 0 in this case

            # Now you can use these total values in your plotting or analysis
            print(
                f"State: {state_obj.name}, PreVax Total QALY Loss: {prevax_total_qaly_loss}, PostVax Total QALY Loss: {postvax_total_qaly_loss}")

    # ... (rest of your class definition)

    def plot_prevax_postvax_qaly_loss_by_state(self):
        """
        Generate a bar graph of the total QALY loss per 100,000 pop for each state, with pre-vax and post-vax contributions.
        """
        vax_df = pd.read_csv(ROOT_DIR + '/csv_files/vaccinated_percentage_by_state.csv')
        dates = self.allStates.dates
        dates = [pd.to_datetime(date) for date in dates]

        # Convert 'Date_Dose1_Over_70Pct_Vaccinated' to datetime
        vax_df['Date_Dose1_Over_70Pct_Vaccinated'] = pd.to_datetime(vax_df['Date_Dose1_Over_70Pct_Vaccinated'],
                                                                       errors='coerce')

        # Initialize a new column to store the index of the closest date
        vax_df['Index_Closest_Date'] = np.nan

        # Iterate through each state
        for index, row in vax_df.iterrows():
            # Get the state and the date of dose1 over 70%
            state = row['Location']
            date_dose1_over_70 = row['Date_Dose1_Over_70Pct_Vaccinated']

            # Find the closest date in 'dates' for the given state
            closest_date = min(dates, key=lambda x: abs((x - date_dose1_over_70).days))

            # Get the index of the closest date
            index_closest_date = dates.index(closest_date)

            # Update the 'Index_Closest_Date' column
            vax_df.at[index, 'Index_Closest_Date'] = index_closest_date

        # Display the updated DataFrame
        print(vax_df)

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(8, 10))

        states_list = list(self.allStates.states.values())
        # To sort states by overall QALY loss
        sorted_states = sorted(
            states_list,
            key=lambda state_obj: (self.get_mean_ui_overall_qaly_loss_by_state(
                state_name=state_obj.name, alpha=0.05)[0] / state_obj.population) * 100000)

        # Set up the positions for the bars
        y_pos = (range(len(sorted_states)))

        democratic_states = ['AZ', 'CA', 'CO', 'CT', 'DE', 'HI', 'IL', 'KS', 'KY', 'ME', 'MD', 'MA', 'MI', 'MN', 'NJ',
                             'NM', 'NY',
                             'NC', 'OR', 'PA', 'RI', 'WA', 'WI']

        # Iterate through each state
        for i, state_obj in enumerate(sorted_states):
            # Get the index from the result_df
            index_closest_date = vax_df[vax_df['Location'] == state_obj.name]['Index_Closest_Date'].values[0]

            # Get the pre-vax and post-vax values
            prevax_value = \
            vax_df[vax_df['Location'] == state_obj.name]['Prevax_Total_QALY_Loss_per_100000'].values[0]
            postvax_value = \
            vax_df[vax_df['Location'] == state_obj.name]['Postvax_Total_QALY_Loss_per_100000'].values[0]

            # Plotting the scatter points for pre-vax and post-vax
            ax.scatter([prevax_value], [postvax_value], marker='o', color='blue', label=state_obj.name)

        # Set the labels and title
        ax.set_xlabel('Prevax Total QALY Loss per 100,000', fontsize=14)
        ax.set_ylabel('Postvax Total QALY Loss per 100,000', fontsize=14)
        ax.set_title('Scatter Plot of QALY Loss (Prevax vs. Postvax) by State')

        # Show the legend with unique labels
        ax.legend()

        plt.tight_layout()
        output_figure(fig, filename=ROOT_DIR + '/figs/prevax_postvax_scatter_qaly_loss_by_state.png')

    def get_state_vax_index(self):
        # Assuming 'dates' is a list of weekly dates and 'result_df' is the DataFrame containing vaccination data
        # Replace 'your_csv_path' with the actual path to your CSV file
        result_df = pd.read_csv(ROOT_DIR + '/csv_files/vaccinated_percentage_by_state.csv')
        dates = self.allStates.dates
        dates = [pd.to_datetime(date) for date in dates]

        # Convert 'Date_Dose1_Over_70Pct_Vaccinated' to datetime
        result_df['Date_Dose1_Over_70Pct_Vaccinated'] = pd.to_datetime(result_df['Date_Dose1_Over_70Pct_Vaccinated'],
                                                                       errors='coerce')

        # Initialize a new column to store the index of the closest date
        result_df['Index_Closest_Date'] = np.nan

        # Iterate through each state
        for index, row in result_df.iterrows():
            # Get the state and the date of dose1 over 70%
            state = row['Location']
            date_dose1_over_70 = row['Date_Dose1_Over_70Pct_Vaccinated']

            # Find the closest date in 'dates' for the given state
            closest_date = min(dates, key=lambda x: abs((x - date_dose1_over_70).days))

            # Get the index of the closest date
            index_closest_date = dates.index(closest_date)

            # Update the 'Index_Closest_Date' column
            result_df.at[index, 'Index_Closest_Date'] = index_closest_date

        # Display the updated DataFrame
        print(result_df)

        # Iterate through each state
        states_list = list(self.allStates.states.values())

        # Iterate through each state
        for state_obj in states_list:
            # Get the index from the result_df
            index_closest_date = result_df[result_df['Location'] == state_obj.name]['Index_Closest_Date'].values[0]
            print(state_obj.name)
            print('Index Closest Date', index_closest_date)

            # Get the weekly QALY losses for the state
            state_qaly_losses = np.array(
                [qaly_losses[state_obj.name] for qaly_losses in self.summaryOutcomes.weeklyQALYlossesByState])

            # Check if index_closest_date is a valid number
            if pd.notna(index_closest_date):
                index_closest_date = int(index_closest_date)

                # Split the array into pre-vax and post-vax based on the index
                prevax_qaly_losses = state_qaly_losses[:index_closest_date + 1]
                postvax_qaly_losses = state_qaly_losses[index_closest_date + 1:]

                # Sum values for pre-vax and post-vax
                prevax_total_qaly_loss = np.sum(prevax_qaly_losses)
                postvax_total_qaly_loss = np.sum(postvax_qaly_losses)
            else:
                # If index_closest_date is NaN or invalid, consider all values in pre-vax and post-vax
                prevax_total_qaly_loss = np.sum(state_qaly_losses)
                postvax_total_qaly_loss = 0  # Assuming postvax_total_qaly_loss is 0 in this case

            # Now you can use these total values in your plotting or analysis
            print(
                f"State: {state_obj.name}, PreVax Total QALY Loss: {prevax_total_qaly_loss}, PostVax Total QALY Loss: {postvax_total_qaly_loss}")

    def plot_prevax_postvax_qaly_loss_by_state(self):
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(8, 10))

        result_df = pd.read_csv(ROOT_DIR + '/csv_files/vaccinated_percentage_by_state.csv')
        dates = self.allStates.dates
        dates = [pd.to_datetime(date) for date in dates]

        # Convert 'Date_Dose1_Over_70Pct_Vaccinated' to datetime
        result_df['Date_Dose1_Over_70Pct_Vaccinated'] = pd.to_datetime(result_df['Date_Dose1_Over_70Pct_Vaccinated'],
                                                                       errors='coerce')

        # Initialize a new column to store the index of the closest date
        result_df['Index_Closest_Date'] = np.nan

        # Iterate through each state
        for index, row in result_df.iterrows():
            # Get the state and the date of dose1 over 70%
            state = row['Location']
            date_dose1_over_70 = row['Date_Dose1_Over_70Pct_Vaccinated']

            # Find the closest date in 'dates' for the given state
            closest_date = min(dates, key=lambda x: abs((x - date_dose1_over_70).days))

            # Get the index of the closest date
            index_closest_date = dates.index(closest_date)

            # Update the 'Index_Closest_Date' column
            result_df.at[index, 'Index_Closest_Date'] = index_closest_date

        states_list = list(self.allStates.states.values())
        # To sort states by overall QALY loss
        sorted_states = sorted(
            states_list,
            key=lambda state_obj: (self.get_mean_ui_overall_qaly_loss_by_state(
                state_name=state_obj.name, alpha=0.05)[0] / state_obj.population) * 100000)

        # Set up the positions for the bars
        y_pos = (range(len(sorted_states)))

        democratic_states = ['AZ', 'CA', 'CO', 'CT', 'DE', 'HI', 'IL', 'KS', 'KY', 'ME', 'MD', 'MA', 'MI', 'MN', 'NJ',
                             'NM', 'NY','NC', 'OR', 'PA', 'RI', 'WA', 'WI']

        # Iterate through each state
        for i, state_obj in enumerate(sorted_states):
            # Get the index from the result_df
            index_closest_date = result_df[result_df['Location'] == state_obj.name]['Index_Closest_Date'].values[0]
            print(state_obj.name)
            print('Index Closest Date', index_closest_date)

            # Get the weekly QALY losses for the state
            state_qaly_losses = np.array(
                [qaly_losses[state_obj.name] for qaly_losses in self.summaryOutcomes.weeklyQALYlossesByState])

            # Check if index_closest_date is a valid number
            if pd.notna(index_closest_date):
                index_closest_date = int(index_closest_date)

                # Split the array into pre-vax and post-vax based on the index
                prevax_qaly_losses = state_qaly_losses[:index_closest_date + 1]
                postvax_qaly_losses = state_qaly_losses[index_closest_date + 1:]

                # Sum values for pre-vax and post-vax
                prevax_total_qaly_loss = np.sum(prevax_qaly_losses) * 100000 / state_obj.population
                postvax_total_qaly_loss = np.sum(postvax_qaly_losses) * 100000 / state_obj.population
            else:
                # If index_closest_date is NaN or invalid, consider all values in pre-vax and post-vax
                prevax_total_qaly_loss = np.sum(state_qaly_losses)*100000/state_obj.population
                postvax_total_qaly_loss = 0  # Assuming postvax_total_qaly_loss is 0 in this case

            # Plotting the scatter points for pre-vax and post-vax
            ax.scatter([prevax_total_qaly_loss], [i], color='blue', marker='o')
            ax.scatter([postvax_total_qaly_loss], [i], color='green', marker='o')

            # Plotting the scatter point for total height
            mean_total, ui_total = self.get_mean_ui_overall_qaly_loss_by_state(state_obj.name, alpha=0.05)
            total_height = (mean_total / state_obj.population) * 100000
            ax.scatter([total_height], [i], color='black', marker='o')

        # Set the labels for each state
        ax.set_yticks(y_pos)
        y_tick_colors = ['blue' if state_obj.name in democratic_states else 'red' for state_obj in sorted_states]
        ax.set_yticklabels([state_obj.name for state_obj in sorted_states], fontsize=12, rotation=0)

        # Set the colors for ticks
        for tick, color in zip(ax.yaxis.get_major_ticks(), y_tick_colors):
            tick.label1.set_color(color)

        # Set the labels and title
        ax.set_ylabel('States', fontsize=14)
        ax.set_xlabel('Total QALY Loss per 100,000', fontsize=14)
        ax.set_title('State-level QALY Loss by Health State (Pre-Vax and Post-Vax)')

        # Show the legend with unique labels
        #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        output_figure(fig, filename=ROOT_DIR + '/figs/prevax_postvax_qaly_loss_by_state.png')

    def plot_qaly_loss_by_state_and_vax_status_subplots(self):
        """
        Generate three subplots, one for each health outcome, with bar graphs representing total QALY loss per 100,000 pop
        for each state. Each subplot has pre-vaccination and post-vaccination QALY loss values separated by state.
        """

        num_states = len(self.allStates.states)
        states_list = list(self.allStates.states.values())

        # To sort states by overall QALY loss
        sorted_states = sorted(
            states_list,
            key=lambda state_obj: (self.get_mean_ui_overall_qaly_loss_by_state(
                state_name=state_obj.name, alpha=0.05)[0] / state_obj.population) * 100000)

        # Set up the positions for the bars
        y_pos = range(len(sorted_states))

        democratic_states = ['AZ', 'CA', 'CO', 'CT', 'DE', 'HI', 'IL', 'KS', 'KY', 'ME', 'MD', 'MA', 'MI', 'MN', 'NJ',
                             'NM', 'NY', 'NC', 'OR', 'PA', 'RI', 'WA', 'WI']
        republican_states = ['AL', 'AK', 'AR', 'FL', 'GA', 'ID', 'IN', 'IA', 'LA', 'MI', 'MS', 'MO', 'MT', 'NE', 'NH',
                             'NV', 'ND', 'OH', 'OK', 'SC', 'SD', 'TN', 'TX',
                             'UT', 'VT', 'VA', 'WV', 'WY']

        # Set up the figure and axis for three subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

        # Set up the positions for the bars
        bar_positions = np.arange(2 * num_states)

        # Set up the width for each state bar
        bar_width = 0.8

        # Set up colors for each segment and vaccination status
        pre_vax_color = 'lightblue'
        post_vax_color = 'black'

        # Iterate through each state
        for i, state_obj in enumerate(sorted_states):
            # Calculate the heights for each segment
            prevax_mean_cases, prevax_ui_cases, prevax_mean_hosps, prevax_ui_hosps, prevax_mean_deaths, \
                prevax_ui_deaths, postvax_mean_cases, postvax_ui_cases, postvax_mean_hosps, postvax_ui_hosps, \
                postvax_mean_deaths, postvax_ui_deaths = (
                self.get_mean_ui_overall_vax_qaly_loss_by_outcome_and_state(state_obj.name, alpha=0.05))

            prevax_cases_height = (prevax_mean_cases / state_obj.population) * 100000
            postvax_cases_height = (postvax_mean_cases / state_obj.population) * 100000

            prevax_hosps_height = (prevax_mean_hosps / state_obj.population) * 100000
            postvax_hosps_height = (postvax_mean_hosps / state_obj.population) * 100000

            prevax_deaths_height = (prevax_mean_deaths / state_obj.population) * 100000
            postvax_deaths_height = (postvax_mean_deaths / state_obj.population) * 100000

            mean_total, ui_total = self.get_mean_ui_overall_qaly_loss_by_state(state_obj.name, alpha=0.05)
            total_height = (mean_total / state_obj.population) * 100000

            # Converting UI into error bars
            prevax_cases_ui = (prevax_ui_cases / state_obj.population) * 100000
            postvax_cases_ui = (postvax_ui_cases / state_obj.population) * 100000
            prevax_hosps_ui = (prevax_ui_hosps / state_obj.population) * 100000
            postvax_hosps_ui = (postvax_ui_hosps / state_obj.population) * 100000
            prevax_deaths_ui = (prevax_ui_deaths / state_obj.population) * 100000
            postvax_deaths_ui = (postvax_ui_deaths / state_obj.population) * 100000

            prevax_xterr_cases = [[prevax_cases_height - prevax_cases_ui[0]],
                                  [prevax_cases_ui[1] - prevax_cases_height]]
            postvax_xterr_cases = [[postvax_cases_height - postvax_cases_ui[0]],
                                   [postvax_cases_ui[1] - postvax_cases_height]]
            prevax_xterr_hosps = [[prevax_hosps_height - prevax_hosps_ui[0]],
                                  [prevax_hosps_ui[1] - prevax_hosps_height]]
            postvax_xterr_hosps = [[postvax_hosps_height - postvax_hosps_ui[0]],
                                   [postvax_hosps_ui[1] - postvax_hosps_height]]
            prevax_xterr_deaths = [[prevax_deaths_height - prevax_deaths_ui[0]],
                                   [prevax_deaths_ui[1] - prevax_deaths_height]]
            postvax_xterr_deaths = [[postvax_deaths_height - postvax_deaths_ui[0]],
                                    [postvax_deaths_ui[1] - postvax_deaths_height]]

            # Plot the segments for pre-vaccination
            axs[0].bar(2 * i, prevax_cases_height, color=pre_vax_color, width=bar_width, align='center',
                       yerr=prevax_xterr_cases, capsize=5, label='Pre-Vax' if i == 0 else "")
            axs[1].bar(2 * i, prevax_deaths_height, color=pre_vax_color, width=bar_width, align='center',
                       yerr=prevax_xterr_deaths, capsize=5, label='Pre-Vax' if i == 0 else "")
            axs[2].bar(2 * i, prevax_hosps_height, color=pre_vax_color, width=bar_width, align='center',
                       yerr=prevax_xterr_hosps, capsize=5, label='Pre-Vax' if i == 0 else "")

            # Plot the segments for post-vaccination
            axs[0].bar(2 * i + 1, postvax_cases_height, color=post_vax_color, width=bar_width, align='center',
                       yerr=postvax_xterr_cases, capsize=5, label='Post-Vax' if i == 0 else "")
            axs[1].bar(2 * i + 1, postvax_deaths_height, color=post_vax_color, width=bar_width, align='center',
                       yerr=postvax_xterr_deaths, capsize=5, label='Post-Vax' if i == 0 else "")
            axs[2].bar(2 * i + 1, postvax_hosps_height, color=post_vax_color, width=bar_width, align='center',
                       yerr=postvax_xterr_hosps, capsize=5, label='Post-Vax' if i == 0 else "")

        # Set the labels for each state and vaccination status
        axs[2].set_xticks(2 * bar_positions)
        axs[2].set_xticklabels([state_obj.name for state_obj in states_list] * 2, fontsize=8, rotation=45, ha='right')

        # Set the labels and title for each subplot
        axs[0].set_ylabel('Cases QALY Loss per 100,000')
        axs[1].set_ylabel('Deaths QALY Loss per 100,000')
        axs[2].set_ylabel('Hospitalizations QALY Loss per 100,000')
        axs[2].set_xlabel('States')
        axs[0].set_title('Total QALY Loss by State and Outcome (Pre-Post Vaccination)')

        # Show the legend with unique labels for each subplot
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Pre-Vax', 'Post-Vax'])
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Pre-Vax', 'Post-Vax'])
        axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Pre-Vax', 'Post-Vax'])

        plt.tight_layout()
        # Add the appropriate path to save the figure
        output_figure(fig, filename=ROOT_DIR + '/figs/total_qaly_loss_by_state_and_outcome_and_vax_status_subplots.png')

    def plot_qaly_loss_by_state_and_vax_status_subplots_alt(self):
        """
        Generate three subplots, one for each health outcome, with bar graphs representing total QALY loss per 100,000 pop
        for each state. Each subplot has pre-vaccination and post-vaccination QALY loss values separated by state.
        """

        num_states = len(self.allStates.states)
        states_list = list(self.allStates.states.values())

        # To sort states by overall QALY loss
        sorted_states = sorted(
            states_list,
            key=lambda state_obj: (self.get_mean_ui_overall_qaly_loss_by_state(
                state_name=state_obj.name, alpha=0.05)[0] / state_obj.population) * 100000)

            # Set up the positions for the bars
        x_pos = range(len(sorted_states))

        democratic_states = ['AZ', 'CA', 'CO', 'CT', 'DE', 'HI', 'IL', 'KS', 'KY', 'ME', 'MD', 'MA', 'MI', 'MN',
                                 'NJ', 'NM', 'NY', 'NC', 'OR', 'PA', 'RI', 'WA', 'WI']
        republican_states = ['AL', 'AK', 'AR', 'FL', 'GA', 'ID', 'IN', 'IA', 'LA', 'MI', 'MS', 'MO', 'MT', 'NE',
                                 'NH', 'NV', 'ND', 'OH', 'OK', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WV', 'WY']

        # Set up the figure and axis for three subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 8), sharey=False)

        # Set up the positions for the bars
        bar_positions = np.arange(num_states)

        # Set up the width for each state bar
        bar_width = 0.8

        # Set up colors for each segment and vaccination status
        pre_vax_color = 'lightblue'
        post_vax_color = 'orange'

        # Iterate through each state
        for i, state_obj in enumerate(sorted_states):
                # Calculate the total QALY loss for each outcome
            total_qaly_loss = (self.get_mean_ui_overall_qaly_loss_by_state(state_obj.name, alpha=0.05)[
                                    0] / state_obj.population) * 100000

                # Plot the bar for pre-vaccination
            axs[0].barh(i, total_qaly_loss, color=pre_vax_color, height=bar_width,
                        label='Pre-Vax' if i == 0 else "")

                # Plot the bar for post-vaccination
            axs[1].barh(i, total_qaly_loss, color=post_vax_color, height=bar_width,
                        label='Post-Vax' if i == 0 else "")

                # Plot the bar for total QALY loss
            axs[2].barh(i, total_qaly_loss, color=[pre_vax_color, post_vax_color], height=bar_width,
                        edgecolor='black', linewidth=0.5, label=['Pre-Vax', 'Post-Vax'] if i == 0 else "")

            # Set the labels for each state and vaccination status
        axs[2].set_yticks(bar_positions)
        axs[2].set_yticklabels([state_obj.name for state_obj in states_list], fontsize=8, ha='right')

            # Set the labels and title for each subplot
        axs[0].set_xlabel('Cases QALY Loss per 100,000')
        axs[1].set_xlabel('Deaths QALY Loss per 100,000')
        axs[2].set_xlabel('Hospitalizations QALY Loss per 100,000')
        axs[2].set_ylabel('States')
        axs[0].set_title('Total QALY Loss by State and Outcome (Pre-Vaccination)')
        axs[1].set_title('Total QALY Loss by State and Outcome (Post-Vaccination)')

            # Show the legend with unique labels for each subplot
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Pre-Vax'])
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Post-Vax'])
        axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Pre-Vax', 'Post-Vax'])

        plt.tight_layout()
            # Add the appropriate path to save the figure
        output_figure(fig,filename=ROOT_DIR + '/figs/total_qaly_loss_by_state_and_outcome_and_vax_status_subplots_alt.png')

    def plot_qaly_loss_by_state_and_by_outcome_alt_2(self):
        """
        Generate subplots for each health outcome with scatter plots of the total QALY loss per 100,000 pop for each state,
        including pre-vaccination and post-vaccination contributions.
        """

        num_states = len(self.allStates.states)
        states_list = list(self.allStates.states.values())

        # Set up the figure and axis for three subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=False)

        # Set up the positions for the y-axis
        y_pos = np.arange(num_states)

        # Set up the width for each state scatter plot
        scatter_size = 50

        # Set up colors for each segment and vaccination status
        pre_vax_color = 'lightblue'
        post_vax_color = 'orange'

        # Iterate through each state
        for i, state_obj in enumerate(states_list):
            # Calculate the heights for each segment
            (prevax_mean_cases, prevax_ui_cases, prevax_mean_hosps, prevax_ui_hosps, prevax_mean_deaths,
             prevax_ui_deaths, postvax_mean_cases, postvax_ui_cases, postvax_mean_hosps, postvax_ui_hosps,
             postvax_mean_deaths, postvax_ui_deaths) = (self.get_mean_ui_overall_vax_qaly_loss_by_outcome_and_state(state_obj, alpha=0.05))

            prevax_cases_height = (prevax_mean_cases / state_obj.population) * 100000
            postvax_cases_height = (postvax_mean_cases/ state_obj.population) * 100000

            prevax_hosps_height = (prevax_mean_hosps / state_obj.population) * 100000
            postvax_hosps_height = (postvax_mean_hosps/ state_obj.population) * 100000

            prevax_deaths_height = (prevax_mean_deaths / state_obj.population) * 100000
            postvax_deaths_height = (postvax_mean_deaths / state_obj.population) * 100000

            # Converting UI into error bars
            prevax_cases_ui = (prevax_ui_cases / state_obj.population) * 100000
            prevax_deaths_ui = (prevax_ui_deaths / state_obj.population) * 100000
            prevax_hosps_ui = (prevax_ui_hosps / state_obj.population) * 100000
            postvax_cases_ui = (postvax_ui_cases / state_obj.population) * 100000
            postvax_deaths_ui = (postvax_ui_deaths / state_obj.population) * 100000
            postvax_hosps_ui = (postvax_ui_hosps / state_obj.population) * 100000


            xterr_prevax_cases = [[prevax_cases_height - prevax_cases_ui[0]], [prevax_cases_ui[1] - prevax_cases_height]]
            xterr_prevax_deaths = [[prevax_deaths_height - prevax_deaths_ui[0]], [prevax_deaths_ui[1] - prevax_deaths_height]]
            xterr_prevax_hosps = [[prevax_hosps_height - prevax_hosps_ui[0]], [prevax_hosps_ui[1] - prevax_hosps_height]]

            xterr_postvax_cases = [[postvax_cases_height - postvax_cases_ui[0]],
                                  [postvax_cases_ui[1] - postvax_cases_height]]
            xterr_postvax_deaths = [[postvax_deaths_height - postvax_deaths_ui[0]],
                                   [postvax_deaths_ui[1] - postvax_deaths_height]]
            xterr_postvax_hosps = [[postvax_hosps_height - postvax_hosps_ui[0]],
                                  [postvax_hosps_ui[1] - postvax_hosps_height]]

            # Plot the scatter points for pre-vaccination cases
            axs[0].scatter(prevax_cases_height, i, color=pre_vax_color, s=scatter_size,
                           label='Pre-Vax' if i == 0 else "")
            # Plot the error bars for pre-vaccination cases
            axs[0].errorbar(prevax_cases_height, i, xerr=xterr_prevax_cases, fmt='none', color=pre_vax_color, capsize=0,
                            alpha=0.4)

            # Plot the scatter points for post-vaccination cases
            axs[0].scatter(postvax_cases_height, i, color=post_vax_color, s=scatter_size,
                           label='Post-Vax' if i == 0 else "")
            # Plot the error bars for post-vaccination cases
            axs[0].errorbar(postvax_cases_height, i, xerr=xterr_postvax_cases, fmt='none', color=post_vax_color,
                            capsize=0,
                            alpha=0.4)

            # Plot the scatter points for pre-vaccination cases
            axs[1].scatter(prevax_hosps_height, i, color=pre_vax_color, s=scatter_size,
                           label='Pre-Vax' if i == 0 else "")
            # Plot the error bars for pre-vaccination cases
            axs[1].errorbar(prevax_hosps_height, i, xerr=xterr_prevax_hosps, fmt='none', color=pre_vax_color, capsize=0,
                            alpha=0.4)

            # Plot the scatter points for post-vaccination cases
            axs[1].scatter(postvax_hosps_height, i, color=post_vax_color, s=scatter_size,
                           label='Post-Vax' if i == 0 else "")
            # Plot the error bars for post-vaccination cases
            axs[1].errorbar(postvax_hosps_height, i, xerr=xterr_postvax_hosps, fmt='none', color=post_vax_color,
                            capsize=0,
                            alpha=0.4)

            # Plot the scatter points for pre-vaccination cases
            axs[2].scatter(prevax_deaths_height, i, color=pre_vax_color, s=scatter_size,
                           label='Pre-Vax' if i == 0 else "")
            # Plot the error bars for pre-vaccination cases
            axs[2].errorbar(prevax_deaths_height, i, xerr=xterr_prevax_deaths, fmt='none', color=pre_vax_color,
                            capsize=0,
                            alpha=0.4)

            # Plot the scatter points for post-vaccination cases
            axs[2].scatter(postvax_deaths_height, i, color=post_vax_color, s=scatter_size,
                           label='Post-Vax' if i == 0 else "")
            # Plot the error bars for post-vaccination cases
            axs[2].errorbar(postvax_deaths_height, i, xerr=xterr_postvax_deaths, fmt='none', color=post_vax_color,
                            capsize=0,
                            alpha=0.4)

        # Set the labels for each state and vaccination status on the y-axis
        axs[0].set_yticks(y_pos)
        axs[0].set_yticklabels([state_obj.name for state_obj in states_list], fontsize=8, ha='right')
        axs[1].set_yticks(y_pos)
        axs[1].set_yticklabels([state_obj.name for state_obj in states_list], fontsize=8, ha='right')
        axs[2].set_yticks(y_pos)
        axs[2].set_yticklabels([state_obj.name for state_obj in states_list], fontsize=8, ha='right')

        # Set the labels and title for the first subplot
        axs[0].set_xlabel('QALY Loss per 100,000')
        axs[0].set_title('Cases QALY Loss by State (Pre-Post Vaccination)')
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Pre-Vax', 'Post-Vax'])

        axs[1].set_xlabel('QALY Loss per 100,000')
        axs[1].set_ylabel('States')  # Add y-axis label for the second subplot
        axs[1].set_title('Hospitalizations QALY Loss by State (Pre-Post Vaccination)')
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Pre-Vax', 'Post-Vax'])

        axs[2].set_xlabel('QALY Loss per 100,000')
        axs[2].set_ylabel('States')  # Add y-axis label for the third subplot
        axs[2].set_title('Deaths QALY Loss by State (Pre-Post Vaccination)')
        axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Pre-Vax', 'Post-Vax'])

        plt.tight_layout()
        # Add the appropriate path to save the figure
        output_figure(fig, filename=ROOT_DIR + '/figs/qaly_loss_by_state_and_outcome_scatter_plots.png')

    def get_mean_ui_overall_vax_qaly_loss_by_outcome_and_state(self, state_obj, alpha=0.05):
        """
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss
        """
        prevax_state_cases_qaly_losses = [qaly_loss[state_obj.name] for qaly_loss in
                                          self.summaryOutcomes.prevaxOverallQALYLossesCasesByState]
        prevax_state_hosps_qaly_losses = [qaly_loss[state_obj.name] for qaly_loss in
                                          self.summaryOutcomes.prevaxOverallQALYLossesHospsByState]
        prevax_state_deaths_qaly_losses = [qaly_loss[state_obj.name] for qaly_loss in
                                          self.summaryOutcomes.prevaxOverallQALYLossesDeathsByState]
        postvax_state_cases_qaly_losses = [qaly_loss[state_obj.name] for qaly_loss in
                                          self.summaryOutcomes.postvaxOverallQALYLossesCasesByState]
        postvax_state_hosps_qaly_losses = [qaly_loss[state_obj.name] for qaly_loss in
                                          self.summaryOutcomes.postvaxOverallQALYLossesHospsByState]
        postvax_state_deaths_qaly_losses = [qaly_loss[state_obj.name] for qaly_loss in
                                           self.summaryOutcomes.postvaxOverallQALYLossesDeathsByState]

        prevax_mean_cases, prevax_ui_cases = get_overall_mean_ui(prevax_state_cases_qaly_losses, alpha =alpha)
        prevax_mean_hosps, prevax_ui_hosps = get_overall_mean_ui(prevax_state_hosps_qaly_losses, alpha=alpha)
        prevax_mean_deaths, prevax_ui_deaths = get_overall_mean_ui(prevax_state_deaths_qaly_losses, alpha=alpha)

        postvax_mean_cases, postvax_ui_cases = get_overall_mean_ui(postvax_state_cases_qaly_losses, alpha=alpha)
        postvax_mean_hosps, postvax_ui_hosps = get_overall_mean_ui(postvax_state_hosps_qaly_losses, alpha=alpha)
        postvax_mean_deaths, postvax_ui_deaths = get_overall_mean_ui(postvax_state_deaths_qaly_losses, alpha=alpha)

        return (prevax_mean_cases, prevax_ui_cases,prevax_mean_hosps, prevax_ui_hosps,prevax_mean_deaths,
                prevax_ui_deaths, postvax_mean_cases, postvax_ui_cases,postvax_mean_hosps, postvax_ui_hosps,
                postvax_mean_deaths, postvax_ui_deaths)


    def print_state_prevax_values(self):
        """
        Generate pre-vaccination and post-vaccination CSV files for each county.
        """

        for state in self.allStates.states.values():
            print("prevax cases:", state.name, state.pandemicOutcomes.cases.prevaxWeeklyObs)
            print("cases:", state.name, state.pandemicOutcomes.cases.weeklyObs)
            print("QALY cases:",state.name, state.pandemicOutcomes.cases.weeklyQALYLoss)
            print("Total Cases QALY Loss per 100K:", state.name, (state.pandemicOutcomes.cases.totalQALYLoss/state.population)*100000)
            print("prevax hosps:", state.name, state.pandemicOutcomes.hosps.prevaxWeeklyObs)
            print("prevax deaths:", state.name, state.pandemicOutcomes.deaths.prevaxWeeklyObs)

            print("prevax QALY cases:", state.name, state.pandemicOutcomes.cases.prevaxWeeklyQALYLoss)
            print("prevax total cases:", state.name, state.pandemicOutcomes.cases.prevaxTotalObs)
            print("prevax total QALY cases per 100K:", state.name, (state.pandemicOutcomes.cases.prevaxTotalQALYLoss/state.population)*100000)

    def print_county_prevax_values(self):
        """
        Generate pre-vaccination and post-vaccination CSV files for each county.
        """

        for state in self.allStates.states.values():
            for county in state.counties.values():
                print("prevax cases:", county.name, county.pandemicOutcomes.cases.prevaxWeeklyObs)
                print("cases:", county.name, county.pandemicOutcomes.cases.weeklyObs)
                print("QALY cases:",county.name, county.pandemicOutcomes.cases.weeklyQALYLoss)
                print("Total Cases QALY Loss:", county.name, county.pandemicOutcomes.cases.totalQALYLoss)
                print("prevax hosps:", county.name, county.pandemicOutcomes.hosps.prevaxWeeklyObs)
                print("prevax deaths:", county.name, county.pandemicOutcomes.deaths.prevaxWeeklyObs)

                print("prevax QALY cases:", county.name, county.pandemicOutcomes.cases.prevaxWeeklyQALYLoss)
                print("prevax total cases:", county.name, county.pandemicOutcomes.cases.prevaxTotalObs)
                print("prevax total QALY cases:", county.name, county.pandemicOutcomes.cases.prevaxTotalQALYLoss)






