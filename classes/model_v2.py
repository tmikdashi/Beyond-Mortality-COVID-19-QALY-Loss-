import geopandas as gpd
import geoplot as gplt
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deampy.plots.plot_support import output_figure

from data_preprocessing.support_functions import get_dict_of_county_data_by_type
from definitions import ROOT_DIR


class Outcome:

    def __init__(self):
        self.weeklyObs = np.array([])
        self.totalObs = None
        self.weeklyQALYLoss = np.array([])
        self.totalQALYLoss = None

    def add_traj(self, weekly_obs):
        """
        Add weekly data to the Outcome object.
        :param weekly_obs: Weekly data as a numpy array.
        """
        if not isinstance(weekly_obs, np.ndarray):
            weekly_obs = np.array(weekly_obs)

        self.weeklyObs = np.nan_to_num(weekly_obs, nan=0)
        self.totalObs = sum(self.weeklyObs)

    def calculate_qaly_loss(self, quality_weight):
        """
        Calculates the weekly and overall QALY
        :param quality_weight: Weight to be applied to each case in calculating QALY loss.
        :return Weekly QALY loss as a numpy array or numerical values to total QALY loss.
        """
        self.weeklyQALYLoss = quality_weight * self.weeklyObs
        self.totalQALYLoss = sum(self.weeklyQALYLoss)


class Outcomes:
    def __init__(self):
        self.cases = Outcome()
        self.hosps = Outcome()
        self.deaths = Outcome()

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

    def calculate_qaly_loss(self, case_weight, death_weight, hosp_weight):
        """
        Calculates the weekly and overall QALY
        :param case_weight: cases-specific weight to be applied to each case in calculating QALY loss.
        :param death_weight: death-specific weight to be applied to each death in calculating QALY loss.
        :param hosp_weight: hosp-specific weight to be applied to each hospitalization in calculating QALY loss.
        :return Overall and QALY loss for across all outcomes.
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
        self.outcomes = Outcomes()

    def add_traj(self, weekly_cases, weekly_deaths, weekly_hosp):
        """
        Add weekly data to the County object.
        :param weekly_cases: Weekly cases data as a numpy array.
        :param weekly_hosp: Weekly hospitalizations data as a numpy array.
        :param weekly_deaths: Weekly deaths data as a numpy array.
        """
        self.outcomes.add_traj(
            weekly_cases=weekly_cases, weekly_hosp=weekly_hosp, weekly_deaths=weekly_deaths)

    def calculate_qaly_loss(self, case_weight, death_weight, hosp_weight):
        """
        Calculates the weekly and total QALY loss for the County.

        :param case_weight: cases-specific weight to be applied to each case in calculating QALY loss.
        :param death_weight: death-specific weight to be applied to each death in calculating QALY loss.
        :param hosp_weight: hosp-specific weight to be applied to each hospitalization in calculating QALY loss.
        :return QALY loss for each county.

        """

        self.outcomes.calculate_qaly_loss(
            case_weight=case_weight, hosp_weight=hosp_weight, death_weight=death_weight)

    def get_overall_qaly_loss(self):
        """
        Retrieves total QALY loss for the County, across outcomes.
        """
        return self.outcomes.totalQALYLoss

    def get_weekly_qaly_loss(self):
        """
        Retrieves weekly QALY loss for the County, across outcomes.
        """
        return self.outcomes.weeklyQALYLoss


class State:
    def __init__(self, name, num_weeks):
        """
        Initialize a State object.

        :param name: Name of the state.
        """
        self.name = name
        self.population = 0
        self.counties = {}  # Dictionary of county objects
        self.outcomes = Outcomes()
        self.numWeeks = num_weeks

    def add_county(self, county):
        """
        Add a County object to the State and calculates the population size of the state

        :param county: County object to be added to the State.
        """
        self.counties[county.name] = county
        self.population += county.population
        self.outcomes.add_traj(
            weekly_cases=county.outcomes.cases.weeklyObs,
            weekly_hosp=county.outcomes.hosps.weeklyObs,
            weekly_deaths=county.outcomes.deaths.weeklyObs)

    def calculate_qaly_loss(self, case_weight, death_weight, hosp_weight):
        """
        Calculates QALY loss for the State.
        #TODO: for this function I also explicitly defined QALY loss by outcome. These values are used later on for
        outcome-specific plots (see below). I expected outcome-specific values of totalQALYLoss and
        weeklyQALYLoss to be automatically calculated/accessible when state.outcomes.totalQALYLoss is calculated but
        that isn't the case. Is there perhaps something I'm missing in my code that is preventing this?

        :param case_weight: : cases-specific weight to be applied to each case in calculating QALY loss.
        :param death_weight: death-specific weight to be applied to each death in calculating QALY loss.
        :param hosp_weight: hosp-specific weight to be applied to each hospitalization in calculating QALY loss.
        :return: Total and Weekly QALY loss for the State.
        """
        state_qaly_loss = {outcome_name: 0 for outcome_name in ['cases', 'hosps', 'deaths']}
        state_weekly_qaly_loss = {outcome_name: np.zeros(self.numWeeks) for outcome_name in
                                  ['cases', 'hosps', 'deaths']}

        for county in self.counties.values():
            for outcome_name in state_qaly_loss.keys():
                state_qaly_loss[outcome_name] += getattr(county.outcomes, outcome_name).totalQALYLoss
                state_weekly_qaly_loss[outcome_name] += getattr(county.outcomes, outcome_name).weeklyQALYLoss

        # I realize this is very redundant and am debugging some aspects of the code to get rid of this
        self.outcomes.cases.totalQALYLoss = state_qaly_loss['cases']
        self.outcomes.hosps.totalQALYLoss = state_qaly_loss['hosps']
        self.outcomes.deaths.totalQALYLoss = state_qaly_loss['deaths']

        self.outcomes.cases.weeklyQALYLoss = state_weekly_qaly_loss['cases']
        self.outcomes.hosps.weeklyQALYLoss = state_weekly_qaly_loss['hosps']
        self.outcomes.deaths.weeklyQALYLoss = state_weekly_qaly_loss['deaths']

        # Update totalQALYLoss and weeklyQALYLoss for all outcomes
        self.outcomes.totalQALYLoss = sum(state_qaly_loss.values())
        self.outcomes.weeklyQALYLoss = (
                self.outcomes.cases.weeklyQALYLoss +
                self.outcomes.deaths.weeklyQALYLoss +
                self.outcomes.hosps.weeklyQALYLoss)



    def get_overall_qaly_loss(self):
        """
        Retrieves total QALY loss for the State, across outcomes.
        """
        return self.outcomes.totalQALYLoss

    def get_weekly_qaly_loss(self):
        """
        Retrieves weekly QALY loss for the State, across outcomes.
        """
        return self.outcomes.weeklyQALYLoss



class AllStates:
    def __init__(self, county_case_csvfile, county_death_csvfile, county_hosp_csvfile):
        """
        Initialize an AllStates object.

        :param county_case_csvfile: (string) path to the csv file containing county data

        """

        self.states = {}  # dictionary of state objects
        self.outcomes = Outcomes()
        self.countyCaseCSVfile = pd.read_csv(county_case_csvfile)
        self.countyDeathCSVfile = pd.read_csv(county_death_csvfile)
        self.countyHospCSVfile = pd.read_csv(county_hosp_csvfile)
        self.numWeeks = 0
        self.totalPopulation = 0

    def populate(self, case_weight, death_weight, hosp_weight):
        """
        Populates the AllStates object with county case data.
        """

        county_case_data, dates = get_dict_of_county_data_by_type('cases')
        county_death_data, dates = get_dict_of_county_data_by_type('deaths')
        county_hosp_data, dates = get_dict_of_county_data_by_type('hospitalizations')

        self.numWeeks = len(dates)


        # Creating a chained exception to handle situations where data is available for cases but not for deaths/hosp
        for (county, state, fips, population), case_values in county_case_data.items():
            try:
                death_values = county_death_data[(county, state, fips, population)]
                hosp_values = county_hosp_data[(county, state, fips, population)]
            except KeyError as e:
                raise KeyError(f"Data not found for {county}, {state}, {fips}, {population}.") from e

            if state not in self.states:
                self.states[state] = State(name=state, num_weeks=self.numWeeks)


            county_obj = County(
                name=county,
                state=state,
                fips=fips,
                population=int(population))

            # Add weekly data to County object and County object to the state
            county_obj.add_traj(weekly_cases=case_values, weekly_deaths=death_values, weekly_hosp=hosp_values)

            # Performing calculations such that following functions simply extract calculated values
            county_obj.calculate_qaly_loss(case_weight, death_weight, hosp_weight)

            self.states[state].add_county(county_obj)


        for state_obj in self.states.values():
            state_obj.calculate_qaly_loss(case_weight, death_weight, hosp_weight)

        for state_name, state_obj in self.states.items():
            self.totalPopulation += state_obj.population



    def get_overall_qaly_loss(self):
        """
        Returns overall QALY Loss, cumulating across all states and across all timepoints.

        :return: Overall QALY loss summed over all states and timepoints.
        """
        total_qaly_loss = 0
        for state_obj in self.states.values():
            total_qaly_loss += state_obj.outcomes.totalQALYLoss
        self.outcomes.totalQALYLoss = total_qaly_loss
        print(f"Total QALY Loss for all states: {self.outcomes.totalQALYLoss}")

    def get_weekly_qaly_loss(self):
        """
        Calculate and return the weekly QALY loss for each state.

        :return: Weekly QALY losses across all states as numpy array
        """

        weekly_qaly_losses = []

        for state_obj in self.states.values():
            weekly_qaly_losses.append(state_obj.outcomes.weeklyQALYLoss)

        self.outcomes.weeklyQALYLoss = np.sum(weekly_qaly_losses, axis=0)

        print(f"Weekly QALY Loss for all states: {self.outcomes.weeklyQALYLoss}")



    def get_weekly_qaly_loss_by_outcome(self):
        """
        Calculate and return the weekly QALY loss for each state.

        :return: Weekly QALY losses across all states as numpy array
        """

        weekly_qaly_losses_cases = {state_name: state_obj.outcomes.cases.weeklyQALYLoss for state_name, state_obj in
                                    self.states.items()}
        weekly_qaly_losses_deaths = {state_name: state_obj.outcomes.deaths.weeklyQALYLoss for state_name, state_obj in
                                     self.states.items()}
        weekly_qaly_losses_hosps = {state_name: state_obj.outcomes.hosps.weeklyQALYLoss for state_name, state_obj in
                                    self.states.items()}

        # Update the outcomes object with the overall weekly QALY loss
        self.outcomes.cases.weeklyQALYLoss = np.sum(list(weekly_qaly_losses_cases.values()), axis=0)
        self.outcomes.deaths.weeklyQALYLoss = np.sum(list(weekly_qaly_losses_deaths.values()), axis=0)
        self.outcomes.hosps.weeklyQALYLoss = np.sum(list(weekly_qaly_losses_hosps.values()), axis=0)

    def get_overall_qaly_loss_by_county(self):
        """
        Print the overall QALY loss for each county.

        :return: Overall QALY loss summed across timepoints for each county
        """
        for state_name, state_obj in self.states.items():
            for county_name, county_obj in state_obj.counties.items():
                print(f"Overall QALY Loss for {county_name}, {state_name}: {county_obj.outcomes.totalQALYLoss}")

    def get_overall_qaly_loss_by_state(self):
        """
        Print the overall QALY loss for each county.

        :return: Overall QALY loss by states summed across all timepoints.
        """
        for state_name, state_obj in self.states.items():
            print(f"Overall QALY Loss for {state_name}: {state_obj.outcomes.totalQALYLoss}")

    def get_weekly_qaly_loss_by_state(self):
        """
        Calculate and return the weekly QALY loss for each state.

        :return: A dictionary where keys are state names and values are the weekly QALY losses as numpy arrays.
        """

        for state_name, state_obj in self.states.items():
            print(f"Weekly QALY Loss for {state_name}: {state_obj.outcomes.weeklyQALYLoss}")

    def get_weekly_qaly_loss_by_county(self):
        """
        Calculate and return the weekly QALY loss for each county.

        :return: A dictionary where keys are county names, and values are the weekly QALY losses as numpy arrays.
        """
        for state_name, state_obj in self.states.items():
            for county_name, county_obj in state_obj.counties.items():
                print(f"Weekly QALY Loss for  {county_name},{state_name}: {county_obj.outcomes.weeklyQALYLoss}")

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
                print(f"Overall QALY Loss for {county_name},{state_name} : {county_obj.outcomes.totalQALYLoss}")

    def get_overall_qaly_loss_for_a_state(self, state_name):
        """
        Get the overall QALY loss for a specific state.

        :param state_name: Name of the state.
        :return: Overall QALY loss for the specified state, summed over all timepoints.
        """
        state_obj = self.states.get(state_name)
        print(f"Overall QALY Loss for {state_name}: {state_obj.outcomes.totalQALYLoss}")

    def get_weekly_qaly_loss_for_a_state(self, state_name):
        """
        Get the overall QALY loss for a specific state.

        :param state_name: Name of the state.
        :return: Weekly QALY loss for the specified state.
        """
        state_obj = self.states.get(state_name)
        print(f"Weekly QALY Loss for {state_name}: {state_obj.outcomes.weeklyQALYLoss}")

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
                print(f"Overall QALY Loss for {county_name},{state_name} : {county_obj.outcomes.weeklyQALYLoss}")


    def plot_weekly_qaly_loss_by_state(self):
        """
        Plots the weekly QALY loss per 100,000 population for each state in a single plot

        :return: Plot of weekly QALY loss per 100,000 population for each state.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for state_name, state_obj in self.states.items():
            weeks = range(1, len(state_obj.outcomes.weeklyQALYLoss) + 1)

            # Calculate the weekly QALY loss per 100,000 population
            state_qaly_loss_per_100k = [(qaly_loss / state_obj.population) * 100000 for qaly_loss in
                                        state_obj.outcomes.weeklyQALYLoss]

            ax.plot(weeks, state_qaly_loss_per_100k, label=state_name)

        ax.set_title('Weekly QALY Loss per 100,000 Population by State')
        ax.set_xlabel('Week')
        ax.set_ylabel('QALY Loss per 100,000 Population')
        ax.legend()
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
        weeks = range(1, len(self.outcomes.weeklyQALYLoss) + 1)
        ax.plot(weeks, self.outcomes.weeklyQALYLoss)

        ax.set_title('National Weekly QALY Loss from Cases, Hospitalizations and Deaths')
        ax.set_xlabel('Date')
        ax.set_ylabel('QALY Loss')
        ax.legend()
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
                qaly_loss = county.outcomes.totalQALYLoss
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
        weeks = range(1, len(self.outcomes.cases.weeklyQALYLoss) + 1)
        ax.plot(weeks, self.outcomes.cases.weeklyQALYLoss, label='Cases', color='blue')
        ax.plot(weeks, self.outcomes.hosps.weeklyQALYLoss, label='Hospitalizations', color='green')
        ax.plot(weeks, self.outcomes.deaths.weeklyQALYLoss, label='Deaths', color='red')

        ax.set_title('Weekly National QALY Loss by Outcome ')
        ax.set_xlabel('Date')
        ax.set_ylabel('QALY Loss ')
        ax.legend()
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
            cases_height = (state_obj.outcomes.cases.totalQALYLoss/ state_obj.population) * 100000
            deaths_height = (state_obj.outcomes.deaths.totalQALYLoss/ state_obj.population) * 100000
            hosps_height = (state_obj.outcomes.hosps.totalQALYLoss/ state_obj.population) * 100000

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

