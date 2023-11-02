import geopandas as gpd
import geoplot as gplt
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deampy.plots.plot_support import output_figure

from data_preprocessing.support_functions import get_dict_of_county_data_by_type
from definitions import ROOT_DIR


# TODO: All three classes that we have so far have a lot in common in terms of
#  outcomes (e.g., weekly cases, total cases, etc) they store. So we can simplify things a little
#  by creating two new classes


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
        :param quality_weight:
        """
        self.weeklyQALYLoss = quality_weight * self.weeklyObs
        self.totalQALYLoss = sum(self.weeklyQALYLoss)


class Outcomes:
    def __int__(self):
        self.cases = Outcome()
        self.hosps = Outcome()
        self.deaths = Outcome()

        self.weeklyQALYLoss = 0
        self.totalQALYLoss = np.array([])

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
        :param case_weight:
        :param death_weight:
        :param hosp_weight:
        """

        # TODO: The reason I changed "get_qaly_loss" to "calculate_qaly_loss" is because you could do the
        #   calculation once and then you could get the loss by using .weeklyQALYLoss and .totalQALYLoss
        #   whenever you need them.
        self.cases.calculate_qaly_loss(quality_weight=case_weight)
        self.hosps.calculate_qaly_loss(quality_weight=hosp_weight)
        self.deaths.calculate_qaly_loss(quality_weight=death_weight)

        self.weeklyQALYLoss = self.cases.weeklyQALYLoss + self.hosps.weeklyQALYLoss + self.deaths.weeklyQALYLoss
        self.totalQALYLoss = self.cases.totalQALYLoss + self.hosps.totalQALYLoss + self.deaths.totalQALYLoss


# TODO: I updated the County class accordingly and I wonder if you could revise everything else.
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
        Calculates the weekly anad total QALY loss for the County.
        :param case_weight: Weight to be applied to each case in calculating QALY loss.
        """

        self.outcomes.calculate_qaly_loss(
            case_weight=case_weight, hosp_weight=hosp_weight, death_weight=death_weight)


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

    def add_county(self, county):
        """
        Add a County object to the State and calculates the population size of the state

        :param county: County object to be added to the State.
        """
        self.counties[county.name] = county
        self.population += county.population
        # TODO: note how we can now access the county's outcomes by using county.outcomes
        self.outcomes.add_traj(
            weekly_cases=county.outcomes.cases.weeklyObs,
            weekly_hosp=county.outcomes.hosps.weeklyObs,
            weekly_deaths=county.outcomes.deaths.weeklyObs)

    def get_overall_qaly_loss(self, case_weight, death_weight, hosp_weight):
        """
        Calculates the overall QALY loss for the State.

        :param case_weight: Weight to be applied to each case in calculating QALY loss.
        :return: Overall QALY loss for the State.
        """
        # TODO now you can just to this:
        return self.outcomes.totalQALYLoss
        #
        # overall_case_qaly_loss = case_weight * self.totalCases
        # overall_death_qaly_loss = death_weight * self.totalDeaths
        # overall_hosp_qaly_loss = hosp_weight * self.totalHosp
        #
        # return overall_case_qaly_loss + overall_death_qaly_loss + overall_hosp_qaly_loss

    def get_weekly_qaly_loss(self, case_weight, death_weight, hosp_weight):
        """
        Calculates the weekly QALY loss for the State.

        :param case_weight: Weight to be applied to each case in calculating QALY loss.
        :return: Weekly QALY loss as a numpy array for the State.
        """
        # TODO now you can just to this:
        return self.outcomes.weeklyQALYLoss
        #
        # weekly_case_qaly_loss = np.array(case_weight * self.weeklyCases)
        # weekly_death_qaly_loss = np.array(death_weight * self.weeklyDeaths)
        # weekly_hosp_qaly_loss = np.array(hosp_weight * self.weeklyHosp)
        #
        # return np.array(weekly_case_qaly_loss + weekly_death_qaly_loss + weekly_hosp_qaly_loss)


class AllStates:
    def __init__(self, county_case_csvfile, county_death_csvfile, county_hosp_csvfile):
        """
        Initialize an AllStates object.

        :param county_case_csvfile: (string) path to the csv file containing county data

        """

        self.states = {}  # dictionary of state objects
        self.countyCaseCSVfile = pd.read_csv(county_case_csvfile)
        self.countyDeathCSVfile = pd.read_csv(county_death_csvfile)
        self.countyHospCSVfile = pd.read_csv(county_hosp_csvfile)
        self.totalCases = 0
        self.totalDeaths = 0
        self.totalHosp = 0
        self.numWeeks = 0
        #self.weeklyCases = np.array([], dtype=int)
        #self.weeklyDeaths = np.zeros(self.numWeeks, dtype=int)
        #self.weeklyHosp = np.zeros(self.numWeeks, dtype=int)

    def populate(self):
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

            # Create a new County object
            county_obj = County(
                name=county,
                state=state,
                fips=fips,
                population=int(population))

            # Add weekly cases data to County object and County object to the state
            county_obj.add_traj(weekly_cases=case_values, weekly_deaths=death_values, weekly_hosp=hosp_values)
            self.states[state].add_county(county_obj)

            # update across all states
            self.totalCases += county_obj.totalCases
            self.totalDeaths += county_obj.totalDeaths
            self.totalHosp += county_obj.totalHosp

    def get_overall_qaly_loss_by_state(self, case_weight, death_weight, hosp_weight):
        """
        Returns overall QALY Loss by state, cumulating across time.

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Dictionary with state names as keys and their respective overall QALY losses
        """
        overall_qaly_loss_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_by_state[state_name] = state_obj.get_overall_qaly_loss(
                case_weight=case_weight, death_weight=death_weight, hosp_weight=hosp_weight)

        return overall_qaly_loss_by_state

    def get_weekly_qaly_loss_by_state(self, case_weight, death_weight, hosp_weight):
        """
        Returns weekly QALY Loss by state, as a timeseries that cumulates across counties

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Dictionary with state names as keys and their respective weekly QALY losses as a timeseries
        """
        weekly_qaly_loss_by_state = {}
        for state_name, state_obj in self.states.items():
            weekly_qaly_loss_by_state[state_name] = state_obj.get_weekly_qaly_loss(
                case_weight=case_weight, death_weight=death_weight, hosp_weight=hosp_weight)

        return weekly_qaly_loss_by_state

    def get_overall_qaly_loss(self, case_weight, death_weight, hosp_weight):
        """
        Returns overall QALY Loss, cumulating across all states and across all timepoints.

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Overall QALY loss summed over all states and timepoints.
        """
        overall_case_qaly_loss = case_weight * self.totalCases
        overall_death_qaly_loss = death_weight * self.totalDeaths
        overall_hosp_qaly_loss = hosp_weight * self.totalHosp

        return overall_case_qaly_loss + overall_death_qaly_loss + overall_hosp_qaly_loss

    def get_weekly_qaly_loss(self, case_weight, death_weight, hosp_weight):
        """
        Returns weekly QALY loss across all states.

        :param case_weight: Coefficient applied to each case in calculating QALY loss
        :return: Weekly QALY loss as a numpy array summed over all states across different timepoints
        """

        weekly_qaly_loss = np.zeros(self.numWeeks, dtype=float)

        for state in self.states.values():
            state_qaly_loss = state.get_weekly_qaly_loss(
                case_weight=case_weight, death_weight=death_weight, hosp_weight=hosp_weight)
            weekly_qaly_loss += state_qaly_loss

        return np.array(weekly_qaly_loss)

    def get_overall_qaly_loss_for_a_county(self, county_name, state_name, case_weight, death_weight, hosp_weight):
        """
        For a given county, returns the overall QALY loss, cumulating over time.
        The county is identified by both its county name and state.

        :param county_name: Name of the county.
        :param state_name: Name of the state.
        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Overall QALY loss for the specified county.
        """

        return self.states[state_name].counties[county_name].get_overall_qaly_loss(
            case_weight=case_weight, death_weight=death_weight, hosp_weight=hosp_weight)

    def get_weekly_qaly_loss_for_a_county(self, county_name, state_name, case_weight, death_weight, hosp_weight):
        """
        For a given county, returns the weekly QALY loss, including cases, deaths, and hospitalizations, across timepoints.
        The county is identified by both its county name and state.

        :param county_name: Name of the county.
        :param state_name: Name of the state.
        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :param death_weight: Coefficient applied to each death in calculating QALY loss.
        :param hosp_weight: Coefficient applied to each hospitalization in calculating QALY loss.
        :return: Weekly QALY loss for the specified county as a timeseries.
        """
        county = self.states[state_name].counties[county_name]
        return county.get_weekly_qaly_loss(case_weight, death_weight, hosp_weight)
    def get_overall_qaly_loss_for_a_state(self, state_name, case_weight,death_weight, hosp_weight):
        """
        Get the overall QALY loss for a specific state, including cases, deaths, and hospitalizations.

        :param state_name: Name of the state.
        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :param death_weight: Coefficient applied to each death in calculating QALY loss.
        :param hosp_weight: Coefficient applied to each hospitalization in calculating QALY loss.
        :return: Overall QALY loss for the specified state.
        """
        state = self.states[state_name]
        return state.get_overall_qaly_loss(case_weight,death_weight, hosp_weight)

    def get_weekly_qaly_loss_for_a_state(self, state_name, case_weight, death_weight, hosp_weight):
        """
        Get the weekly QALY loss for a specific state, including cases, deaths, and hospitalizations, as a timeseries.

        :param state_name: Name of the state.
        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :param death_weight: Coefficient applied to each death in calculating QALY loss.
        :param hosp_weight: Coefficient applied to each hospitalization in calculating QALY loss.
        :return: Weekly QALY loss for the specified state as a timeseries.
        """
        state = self.states[state_name]
        return state.get_weekly_qaly_loss(case_weight,death_weight, hosp_weight)
    def plot_weekly_qaly_loss_by_state(self, case_weight, death_weight, hosp_weight):
        """
        Plots the weekly QALY loss per 100,000 population for each state in a single plot

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :param death_weight: Coefficient applied to each death in calculating QALY loss.
        :param hosp_weight: Coefficient applied to each hospitalization in calculating QALY loss.
        :return: Plot of weekly QALY loss per 100,000 population for each state in a single plot.
        """
        weekly_qaly_loss_by_state = self.get_weekly_qaly_loss_by_state(case_weight, death_weight, hosp_weight)

        # Define dates
        county_data, dates = get_dict_of_county_data_by_type('cases')

        fig, ax = plt.subplots(figsize=(12, 6))

        for state_name, weekly_qaly_loss in weekly_qaly_loss_by_state.items():
            weeks = range(1, len(weekly_qaly_loss) + 1)

            # Calculate the weekly QALY loss per 100,000 population
            state_obj = self.states[state_name]
            state_qaly_loss_per_100k = [(qaly_loss / state_obj.population) * 100000 for qaly_loss in weekly_qaly_loss]

            ax.plot(dates, state_qaly_loss_per_100k, label=state_name)

        ax.set_title('Weekly QALY Loss per 100,000 Population by State')
        ax.set_xlabel('Date')
        ax.set_ylabel('QALY Loss per 100,000 Population')
        ax.legend()
        ax.grid(True)

        plt.xticks(rotation=90)
        ax.tick_params(axis='x', labelsize=6.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10)
        plt.subplots_adjust(top=0.45)

        output_figure(fig, filename=ROOT_DIR + '/figs/weekly_qaly_loss_by_state.png')

    def plot_weekly_qaly_loss(self, case_weight, death_weight, hosp_weight):
        """
        Plots the weekly QALY loss per 100,000 population summed over all states

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Plot of weekly QALY loss per 100,000 population across all states
        """

        # Define dates
        county_data, dates = get_dict_of_county_data_by_type('cases')

        # Calculate the total weekly QALY loss and national population for all states
        total_weekly_qaly_loss = self.get_weekly_qaly_loss(case_weight, death_weight, hosp_weight)
        total_population = sum(state.population for state in self.states.values())

        # Calculate the weekly QALY loss per 100,000 population
        total_weekly_qaly_loss_per_100k = [(qaly_loss / total_population) * 100000 for qaly_loss in
                                           total_weekly_qaly_loss]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the total weekly QALY loss per 100,000 population
        weeks = range(1, len(total_weekly_qaly_loss_per_100k) + 1)
        ax.plot(dates, total_weekly_qaly_loss_per_100k, label='All States', color='blue')

        ax.set_title('National Weekly QALY Loss per 100,000 Population')
        ax.set_xlabel('Date')
        ax.set_ylabel('QALY Loss per 100,000 population')
        ax.legend()
        ax.grid(True)

        plt.xticks(rotation=90)
        ax.tick_params(axis='x', labelsize=6.5)

        output_figure(fig, filename=ROOT_DIR + '/figs/national_qaly_loss.png')

    def plot_map_of_qaly_loss_by_county(self, case_weight, death_weight, hosp_weight):
        """
        Plots a map of the QALY loss per 100,000 population for each county, considering cases, deaths, and hospitalizations.

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :param death_weight: Coefficient applied to each death in calculating QALY loss.
        :param hosp_weight: Coefficient applied to each hospitalization in calculating QALY loss.
        :return: A map of the QALY loss per 100,000 population for each county
        """

        # Create a list to store data for plotting
        county_qaly_loss_data = {
            "COUNTY": [],
            "FIPS": [],
            "QALY Loss per 100K": []
        }

        # Iterate over counties within states in the AllStates object
        for state in self.states.values():
            for county in state.counties.values():
                # Calculate the QALY loss per 100,000 population
                qaly_loss = (
                            case_weight * county.totalCases + death_weight * county.totalDeaths + hosp_weight * county.totalHosp)
                qaly_loss_per_100k = (qaly_loss / county.population) * 100000
                # Append county data to the list
                county_qaly_loss_data["COUNTY"].append(county.name)
                county_qaly_loss_data["FIPS"].append(county.fips)
                county_qaly_loss_data["QALY Loss per 100K"].append(qaly_loss_per_100k)

        # Create a DataFrame from the county data
        county_qaly_loss_df = pd.DataFrame(county_qaly_loss_data)

        # Load the json file with county coordinates
        geoData = gpd.read_file(
            "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
        )

        # Adjusting geoData's identification of a county by creating consistent FIPS format
        geoData['STATE'] = geoData['STATE'].str.lstrip('0')
        geoData['FIPS'] = geoData['STATE'] + geoData['COUNTY']

        # Merge the county QALY loss data with the geometry data
        merged_geo_data = geoData.merge(county_qaly_loss_df, left_on='FIPS', right_on='FIPS', how='left')

        # Remove counties where there is no data
        merged_geo_data = merged_geo_data.dropna(subset=["QALY Loss per 100K"])

        # Remove Alaska, HI, Puerto Rico (to be plotted later)
        stateToRemove = ["02", "15", "72"]
        merged_geo_data_mainland = merged_geo_data[~merged_geo_data.STATE.isin(stateToRemove)]

        # Explode the MultiPolygon geometries into individual polygons
        merged_geo_data_mainland = merged_geo_data_mainland.explode()

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.set_aspect('equal')  # Set aspect ratio to 'equal' for a square map

        # SAFETY CHECK: Check if merged_geo_data_mainland DataFrame is empty --
        if not merged_geo_data_mainland.empty:
            # Set up the color scheme:
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

            # Set the xlim and ylim to zoom in on a specific region
            ax.set_xlim([-170.0, 60])
            ax.set_ylim([25, 76])

            plt.title("Cumulative County QALY Loss per 100K", fontsize=24)
        else:
            # Handle the case where the DataFrame is empty (e.g., no data to plot)
            print("No data to plot")

        plt.tight_layout()

        output_figure(fig, filename=ROOT_DIR + '/figs/map_county_qaly_loss.png')

