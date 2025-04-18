import geopandas as gpd
import geoplot as gplt
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classes.parameters import ParameterGenerator
from data_preprocessing.support_functions import get_dict_of_county_data_by_type
from deampy.plots.plot_support import output_figure
from definitions import ROOT_DIR


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
        :param case_weight:
        :param death_weight:
        :param hosp_weight:
        """

        # calculate QALY loss for each state
        for state in self.states.values():
            state.calculate_qaly_loss(
            case_weight=param_values.qWeightCase, hosp_weight=param_values.qWeightHosp, death_weight=param_values.qWeightDeath)

        # calculate QALY loss for the nation
        self.pandemicOutcomes.calculate_qaly_loss(
            case_weight=param_values.qWeightCase, hosp_weight=param_values.qWeightHosp, death_weight=param_values.qWeightDeath)

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

    def get_overall_qaly_loss_by_county(self, print_results):
        """
        Print the overall QALY loss for each county.

        :return: Overall QALY loss summed across timepoints for each county
        """
        overall_qaly_loss_by_county = {}
        for state_name, state_obj in self.states.items():
            for county_name, county_obj in state_obj.counties.items():
                overall_qaly_loss_by_county[county_name, state_name]=county_obj.pandemicOutcomes.totalQALYLoss
        if print_results:
            for state_name, state_obj in self.states.items():
                for county_name, county_obj in state_obj.counties.items():
                    print(f"Overall QALY Loss for {county_name}, {state_name}: {county_obj.pandemicOutcomes.totalQALYLoss}")
        return overall_qaly_loss_by_county


    def get_overall_qaly_loss_by_state(self,print_results=False):
        """
        Print the overall QALY loss for each county.

        :return: Overall QALY loss by states summed across all timepoints.
        """
        overall_qaly_loss_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_by_state[state_name]=state_obj.pandemicOutcomes.totalQALYLoss
        if print_results:
            for state_name, state_obj in self.states.items():
                print(f"Overall QALY Loss for {state_name}: {state_obj.pandemicOutcomes.totalQALYLoss}")
        return overall_qaly_loss_by_state

    def get_weekly_qaly_loss_by_state(self):
        """
        Calculate and return the weekly QALY loss for each state.

        :return: A dictionary where keys are state names and values are the weekly QALY losses as numpy arrays.
        """

        for state_name, state_obj in self.states.items():
            print(f"Weekly QALY Loss for {state_name}: {state_obj.pandemicOutcomes.weeklyQALYLoss}")

    def get_weekly_qaly_loss_by_county(self):
        """
        Calculate and return the weekly QALY loss for each county.

        :return: A dictionary where keys are county names, and values are the weekly QALY losses as numpy arrays.
        """
        for state_name, state_obj in self.states.items():
            for county_name, county_obj in state_obj.counties.items():
                print(f"Weekly QALY Loss for  {county_name},{state_name}: {county_obj.pandemicOutcomes.weeklyQALYLoss}")

    def get_overall_qaly_loss_for_a_county(self, county_name, state_name, ):
        """
        Get the overall QALY loss for a specific state.

        :param county_name: Name of the county.
        :param state_name: Name of the state.
        :return: Overall QALY loss for the specified county, summed over all timepoints
        """
        state_obj = self.states.get(state_name)
        if state_obj:
            county_obj = state_obj.counties.get(county_name)
            if county_obj:
                print(f"Overall QALY Loss for {county_name},{state_name} : {county_obj.pandemicOutcomes.totalQALYLoss}")

    def get_overall_qaly_loss_for_a_state(self, state_name):
        """
        Get the overall QALY loss for a specific state.

        :param state_name: Name of the state.
        :return: Overall QALY loss for the specified state, summed over all timepoints.
        """
        state_obj = self.states.get(state_name)
        print(f"Overall QALY Loss for {state_name}: {state_obj.pandemicOutcomes.totalQALYLoss}")

    def get_weekly_qaly_loss_for_a_state(self, state_name):
        """
        Get the overall QALY loss for a specific state.

        :param state_name: Name of the state.
        :return: Weekly QALY loss for the specified state.
        """
        state_obj = self.states.get(state_name)
        print(f"Weekly QALY Loss for {state_name}: {state_obj.pandemicOutcomes.weeklyQALYLoss}")

    def get_weekly_qaly_loss_for_a_county(self, county_name, state_name, ):
        """
        Get the weekly QALY loss for a specific state.

        :param county_name: Name of the county.
        :param state_name: Name of the state.
        :return: Weekly QALY loss for the specified county.
        """
        state_obj = self.states.get(state_name)
        if state_obj:
            county_obj = state_obj.counties.get(county_name)
            if county_obj:
                print(f"Overall QALY Loss for {county_name},{state_name} : {county_obj.pandemicOutcomes.weeklyQALYLoss}")

    def plot_weekly_qaly_loss_by_state(self):
        """
        Plots the weekly QALY loss per 100,000 population for each state in a single plot

        :return: Plot of weekly QALY loss per 100,000 population for each state.
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

        # Plot the total weekly QALY loss per 100,000 population
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

        if not merged_geo_data_mainland.empty:
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
        else:
            print("No data to plot")

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
        self.allStates = AllStates()
        self.allStates.populate()
        self.summaryOutcomes = SummaryOutcomes()

        self.overallQALYlosses = []
        self.weeklyQALYlosses = []

        self.overallQALYlossesByState=[]
        self.overallQALYlossesByCounty =[]

        self.weeklyQALYlossesCases =[]
        self.weeklyQALYlossesHosps = []
        self.weeklyQALYlossesDeaths = []

        self.overallCountyQALYlosses = []


    def simulate(self, n):
        """
        Simulates the model n times
        :param n: (int) number of times parameters should be sampled and the model simulated
        """

        rng = np.random.RandomState(1)
        param_gen = ParameterGenerator()
        self.age_group = param_gen.parameters['Age Group'].value


        for i in range(n):
            # generate a new set of parameters
            params = param_gen.generate(rng)

            self.allStates.calculate_qaly_loss(params)

            # extract outcomes from the simulated all states
            self.summaryOutcomes.extract_outcomes(simulated_model=self.allStates, param_gen=param_gen)

            overall_qaly_loss = self.allStates.get_overall_qaly_loss()
            self.overallQALYlosses.append(overall_qaly_loss)

            weekly_qaly_loss = self.allStates.get_weekly_qaly_loss()
            self.weeklyQALYlosses.append(weekly_qaly_loss)

            overall_qaly_loss_by_state= self.allStates.get_overall_qaly_loss_by_state(False) #False was added to prevent automatic printing
            self.overallQALYlossesByState.append(overall_qaly_loss_by_state)

            overall_qaly_loss_by_county = self.allStates.get_overall_qaly_loss_by_county(False)  # False was added to prevent automatic printing
            self.overallQALYlossesByCounty.append(overall_qaly_loss_by_county)
            print(self.overallQALYlossesByCounty)

            weekly_qaly_loss_cases = self.allStates.pandemicOutcomes.cases.weeklyQALYLoss
            self.weeklyQALYlossesCases.append(weekly_qaly_loss_cases)

            weekly_qaly_loss_hosps = self.allStates.pandemicOutcomes.hosps.weeklyQALYLoss
            self.weeklyQALYlossesHosps.append(weekly_qaly_loss_hosps)

            weekly_qaly_loss_deaths = self.allStates.pandemicOutcomes.deaths.weeklyQALYLoss
            self.weeklyQALYlossesDeaths.append(weekly_qaly_loss_deaths)

            #county_qaly_loss = self.allStates.pandemicOutcomes.county.totalQALYLoss
            #self.overallCountyQALYlosses.append(county_qaly_loss)

            for state in self.allStates.states.values():
                for county in state.counties.values():
                    # Calculate the QALY loss per 100,000 population
                    qaly_loss = county.pandemicOutcomes.totalQALYLoss
                    self.overallCountyQALYlosses.append(qaly_loss)
                    print('{county}:',self.overallCountyQALYlosses)


    def get_overall_qaly_loss(self):
        """
        :return: Overall QALY loss summed over all states.
        """
        print('Overall QALY Loss:', self.overallQALYlosses)
        print('Average QALY Loss across simulations:', np.mean(self.overallQALYlosses))


    def get_weekly_qaly_loss(self):
        """
        :return: Overall QALY loss summed over all states.
        """
        print('Weekly QALY Loss:', self.weeklyQALYlosses)
        print('Average Weekly QALY Loss across simulations:', np.mean(self.weeklyQALYlosses, axis=0))



    def get_overall_qaly_loss_by_state(self):

        state_qaly_losses = {state_name: [] for state_name in self.allStates.states.keys()}

        for i, qaly_losses_by_state in enumerate(self.overallQALYlossesByState):
            for state_name, qaly_loss in qaly_losses_by_state.items():
                state_qaly_losses[state_name].append(qaly_loss)

        for state_name, qaly_losses in state_qaly_losses.items():
            print(f" Overall QALY Loss in {state_name}: {', '.join(map(str, qaly_losses))}")

        for state_name, qaly_losses in state_qaly_losses.items():
            print(f" Average QALY Loss across simulations in {state_name}:",np.mean(qaly_losses))


    def get_overall_qaly_loss_by_county(self):

        for state in self.allStates.states.values():
            for county in state.counties.values():
                # Calculate the QALY loss per 100,000 population
                qaly_loss = county.pandemicOutcomes.totalQALYLoss
                self.overallCountyQALYlosses.append(qaly_loss)
                print('{county}:', self.overallCountyQALYlosses)

        county_qaly_losses = {state_name: [] for state_name in self.allStates.county.keys()}

        for i, qaly_losses_by_county in enumerate(self.overallQALYlossesByCounty):
            for state_name, qaly_loss in qaly_losses_by_state.items():
                couty_qaly_losses[state_name].append(qaly_loss)

        for state_name, qaly_losses in state_qaly_losses.items():
            print(f" Overall QALY Loss in {state_name}: {', '.join(map(str, qaly_losses))}")

        for state_name, qaly_losses in state_qaly_losses.items():
            print(f" Average QALY Loss across simulations in {state_name}:",np.mean(qaly_losses))





    def plot_weekly_qaly_loss(self):
        """
        Plots National Weekly QALY Loss from Cases, Hospitalizations and Deaths across all states
        """
        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the individual weekly QALY losses
        for i, weekly_qaly_loss in enumerate(self.weeklyQALYlosses):
            ax.plot(range(1, len(weekly_qaly_loss) + 1), weekly_qaly_loss, label=f'Simulation {i + 1}')

        # Plot the average weekly QALY loss in bold
        average_weekly_qaly_loss = np.mean(self.weeklyQALYlosses, axis=0)
        ax.plot(range(1, len(average_weekly_qaly_loss) + 1), average_weekly_qaly_loss, label='Average across simulations', linewidth=3,
                color='black')

        ax.set_title('National Weekly QALY Loss from Cases, Hospitalizations and Deaths')
        ax.set_xlabel('Date')
        ax.set_ylabel('QALY Loss')
        ax.grid(True)
        plt.legend()

        plt.xticks(rotation=90)
        ax.tick_params(axis='x', labelsize=6.5)


        output_figure(fig, filename=ROOT_DIR + '/figs/simulations_national_qaly_loss.png')


    def plot_weekly_qaly_loss_by_outcome(self):
        """
        Plots national weekly QALY Loss across all states broken down by cases, hospitalizations and deaths.
        """

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the lines for each outcome
        weeks = range(1, len(self.allStates.pandemicOutcomes.cases.weeklyQALYLoss) + 1)
        for i, weekly_qaly_loss in enumerate(self.weeklyQALYlossesCases):
            ax.plot(range(1, len(weekly_qaly_loss) + 1), weekly_qaly_loss,  color='blue', linewidth =0.5)
        for i, weekly_qaly_loss in enumerate(self.weeklyQALYlossesHosps):
            ax.plot(range(1, len(weekly_qaly_loss) + 1), weekly_qaly_loss,
                    color= 'green',  linewidth =0.5)
        for i, weekly_qaly_loss in enumerate(self.weeklyQALYlossesDeaths):
            ax.plot(range(1, len(weekly_qaly_loss) + 1), weekly_qaly_loss,
                    color='red',  linewidth =0.5)

        ax.plot(weeks, np.mean( self.weeklyQALYlossesCases, axis=0), label='Cases average', color='blue', linewidth =3)
        ax.plot(weeks,  np.mean(self.weeklyQALYlossesHosps, axis=0), label='Hospitalizations', color='green', linewidth =3)
        ax.plot(weeks,  np.mean(self.weeklyQALYlossesDeaths, axis=0), label='Deaths', color='red', linewidth =3)

        ax.set_title('Weekly National QALY Loss by Outcome ')
        ax.set_xlabel('Date')
        ax.set_ylabel('QALY Loss ')
        ax.grid(True)

        plt.xticks(rotation=90)
        ax.tick_params(axis='x', labelsize=6.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10)
        plt.subplots_adjust(top=0.45)

        plt.tight_layout()
        plt.show()
        #output_figure(fig, filename=ROOT_DIR + '/figs/national_weekly_qaly_loss_by_outcome.png')

    #def get_county_qaly_loss (self):

    def plot_map_of_qaly_loss_by_county(self):
        """
        Plots a map of the QALY loss per 100,000 population for each county, considering cases, deaths, and hospitalizations.
        """

        county_qaly_loss_data = {
            "COUNTY": [],
            "FIPS": [],
            "QALY Loss per 100K": []
        }

        for state in self.allStates.states.values():
            for county in state.counties.values():
                # Calculate the QALY loss per 100,000 population
                qaly_loss = county.pandemicOutcomes.totalQALYLoss
                qaly_loss_per_100k = (qaly_loss / county.population) * 100000
                # Append county data to the list
                county_qaly_loss_data["COUNTY"].append(county.name)
                county_qaly_loss_data["FIPS"].append(county.fips)
                county_qaly_loss_data["QALY Loss per 100K"].append(qaly_loss_per_100k)
                print(county_qaly_loss_data)

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

        if not merged_geo_data_mainland.empty:
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
        else:
            print("No data to plot")

        plt.tight_layout()
        output_figure(fig, filename=ROOT_DIR + '/figs/avg_map_county_qaly_loss_all_simulations.png')

        return fig

    def get_weekly_qaly_loss( self):
        # TODO: I think this function can be deleted. It seems like a something that was used in previous versions in the code
        """
        :return: Overall QALY loss summed over all states.
        """
        print('Weekly QALY Loss:', self.summaryOutcomes.weeklyQALYlosses)
        print('Average Weekly QALY Loss across simulations:', np.mean(self.summaryOutcomes.weeklyQALYlosses, axis=0))


    def print_mean_ui_overall_qaly_loss_by_state(self, alpha=0.05):
        # TODO: This functions isn't used in the rest of the code, but may be useful for examining values
        """
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the overall QALY loss by State
        """

        state_qaly_losses = {state_name: [] for state_name in self.allStates.states.keys()}

        for i, qaly_losses_by_state in enumerate(self.summaryOutcomes.overallQALYlossesByState):
            for state_name, qaly_loss in qaly_losses_by_state.items():
                state_qaly_losses[state_name].append(qaly_loss)

        for state_name, qaly_losses in state_qaly_losses.items():
            mean = np.mean(qaly_losses)
            ui = np.percentile(qaly_losses, q=[alpha / 2 * 100, 100 - alpha / 2 * 100])

            print(f" Overall QALY Loss in {state_name}: mean={mean}, ui={ui}")


    def print_mean_ui_weekly_qaly_loss_by_state(self,
                                                alpha=0.05):  # TODO: Same as above, this functions isn't used in the rest of the code
        '''
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss by State
        '''

        state_qaly_losses = {state_name: [] for state_name in self.allStates.states.keys()}

        for i, qaly_losses_by_state in enumerate(self.summaryOutcomes.weeklyQALYlossesByState):
            for state_name, qaly_loss in qaly_losses_by_state.items():
                state_qaly_losses[state_name].append(qaly_loss)

        for state_name, qaly_losses in state_qaly_losses.items():
            mean = np.mean(qaly_losses, axis=0)
            ui = np.percentile(qaly_losses, q=[alpha / 2 * 100, 100 - alpha / 2 * 100], axis=0)

            print(f" Weeklu QALY Loss in {state_name}: mean={mean}, ui={ui}")


    def print_overall_qaly_loss_by_outcome(self):
        """
        Print the mean, confidence interval, and uncertainty interval for overall QALY loss by outcome class.
        """
        outcomes = ['cases', 'hosps', 'deaths']

        for outcome in outcomes:
            mean, ci, ui = self.get_mean_ci_ui_overall_qaly_loss_by_outcome(outcome)

            print(f'Overall QALY loss for {outcome.capitalize()}:')
            print('  Mean:', mean)
            print('  95% Confidence Interval:', ci)
            print('  95% Uncertainty Interval:', ui)





