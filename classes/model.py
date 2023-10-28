import geopandas as gpd
import geoplot as gplt
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deampy.plots.plot_support import output_figure

from data_preprocessing.support_functions import get_dict_of_county_data_by_type
from definitions import ROOT_DIR


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

        self.weeklyCases = np.array([])
        self.totalCases = None

    def add_traj(self, weekly_cases):
        """
        Add weekly case data to the County object.

        :param weekly_cases: Weekly cases data as a numpy array.
        """
        # TODO: you could modify this function to also get weekly deaths and hospitalizations as arguments
        #  and then update self.totalHospitalizations and self.totalDeaths accordingly.

        if not isinstance(weekly_cases, np.ndarray):
            weekly_cases = np.array(weekly_cases)

        self.weeklyCases = np.nan_to_num(weekly_cases, nan=0)
        self.totalCases = sum(self.weeklyCases)

    def get_weekly_qaly_loss(self, case_weight):
        """
        Calculates  weekly QALY loss for the County.

        :param case_weight: Weight to be applied to each case in calculating QALY loss.
        :return: Weekly QALY loss as a numpy array.
        """
        # TODO: you could modify this function to also get hosp_weight and death_weight as arguments and then
        #  update the formula below accordingly. Remember that death_weight is a little complicated
        #  since that dependents on the age of the person who died. For now, just use some made up numbers
        #  (say 20) and we will sort this out.
        return case_weight * self.weeklyCases

    def get_overall_qaly_loss(self, case_weight):
        """
        Calculates overall QALY loss for the County, across all timepoints.

        :param case_weight: Weight to be applied to each case in calculating QALY loss.
        :return: Overall QALY loss for the County.
         """
        # TODO: the same comment as above.

        return case_weight * self.totalCases


class State:
    def __init__(self,name, num_weeks):
        """
        Initialize a State object.

        :param name: Name of the state.
        """
        self.name = name
        self.population = 0
        self.counties = {}  # Dictionary of county objects
        self.weeklyCases = np.zeros(num_weeks, dtype=int)
        self.weeklyQALYLoss = []
        self.totalCases = 0

    def add_county(self, county):
        """
        Add a County object to the State and calculates the population size of the state

        :param county: County object to be added to the State.
        """
        self.counties[county.name] = county
        self.population += county.population
        # TODO: like this, you could also define self.totalHospitalizations and self.totalDeaths and update them here
        self.totalCases += county.totalCases
        # TODO: like this, you could also define self.weeklyHospitalizations and self.weeklyDeaths and update them here
        self.weeklyCases = np.add(self.weeklyCases, county.weeklyCases)  # Aggregate the weekly cases

    def get_overall_qaly_loss(self, case_weight):
        """
        Calculates the overall QALY loss for the State.

        :param case_weight: Weight to be applied to each case in calculating QALY loss.
        :return: Overall QALY loss for the State.
        """
        # TODO: you could modify this function to also get hosp_weight and death_weight as arguments and then
        #  update the formula below accordingly.
        return case_weight * self.totalCases

    def get_weekly_qaly_loss(self, case_weight):
        """
        Calculates the weekly QALY loss for the State.

        :param case_weight: Weight to be applied to each case in calculating QALY loss.
        :return: Weekly QALY loss as a numpy array for the State.
        """
        # TODO: same as above
        return np.array(case_weight * self.weeklyCases)


class AllStates:
    def __init__(self, county_data_csvfile):
        """
        Initialize an AllStates object.

        :param county_data_csvfile: (string) path to the csv file containing county data
        """

        self.states = {}  # dictionary of state objects
        self.countyDataCSVfile = pd.read_csv(county_data_csvfile)
        self.totalCases = 0
        self.numWeeks = 0

    def populate(self, data_type):
        """
        Populates the AllStates object with county case data.
        """

        # TODO: if you made above changes, then we probably don't need data_type here as an argument
        #  and you could read the data on cases, hospitalizations, and deaths all here...

        county_data, dates = get_dict_of_county_data_by_type(data_type)

        self.numWeeks = len(dates)

        for (county, state, fips, population), data_values in county_data.items():
            # If the state does not already exist in self.states, add it to the dictionary
            if state not in self.states:
                self.states[state] = State(name=state, num_weeks=self.numWeeks)  # Pass num_weeks parameter

            # Create a new County object
            county_obj = County(
                name=county,
                state=state,
                fips=fips,
                population=int(population))

            # Add weekly cases data to County object and County object to the state
            county_obj.add_traj(weekly_cases=data_values)
            self.states[state].add_county(county_obj)

            # update total cases across all states
            self.totalCases += county_obj.totalCases

    def get_overall_qaly_loss_by_state(self, case_weight):
        """
        Returns overall QALY Loss by state, cumulating across time.

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Dictionary with state names as keys and their respective overall QALY losses
        """
        overall_qaly_loss_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_by_state[state_name] = state_obj.get_overall_qaly_loss(case_weight)

        return overall_qaly_loss_by_state

    def get_weekly_qaly_loss_by_state(self, case_weight):
        """
        Returns weekly QALY Loss by state, as a timeseries that cumulates across counties

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Dictionary with state names as keys and their respective weekly QALY losses as a timeseries
        """
        weekly_qaly_loss_by_state = {}

        for state_name, state_obj in self.states.items():
            weekly_qaly_loss_by_state[state_name] = state_obj.get_weekly_qaly_loss(case_weight)

        return weekly_qaly_loss_by_state

    def get_overall_qaly_loss(self, case_weight):
        """
        Returns overall QALY Loss, cumulating across all states and across all timepoints.

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Overall QALY loss summed over all states and timepoints.
        """
        return case_weight * self.totalCases

    def get_weekly_qaly_loss(self, case_weight):
        """
        Returns weekly QALY loss across all states.

        :param case_weight: Coefficient applied to each case in calculating QALY loss
        :return: Weekly QALY loss as a numpy array summed over all states across different timepoints
        """

        weekly_qaly_loss = np.zeros(self.numWeeks, dtype=float)
        for state in self.states.values():
            weekly_qaly_loss += state.get_weekly_qaly_loss(case_weight)
        return np.array(weekly_qaly_loss)

    def get_overall_qaly_loss_for_a_county(self, county_name, state_name, case_weight):
        """
        For a given county, returns the overall QALY loss, cumulating over time.
        The county is identified by both its county name and state.

        :param county_name: Name of the county.
        :param state_name: Name of the state.
        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Overall QALY loss for the specified county.
        """

        return self.states[state_name].counties[county_name].get_overall_qaly_loss(case_weight)

    def get_weekly_qaly_loss_for_a_county(self, county_name, state_name, case_weight):
        """
        For a given county, returns the weekly QALY loss, across timepoints.
        The county is identified by both its county name and state.

        :param county_name: Name of the county.
        :param state_name: Name of the state.
        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Weekly QALY loss for the specified county as a timeseries.
        """
        return case_weight * self.states[state_name].counties[county_name].weeklyCases

    def get_overall_qaly_loss_for_a_state(self, state_name, case_weight):
        """
        Get the overall QALY loss for a specific state.

        :param state_name: Name of the state.
        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Overall QALY loss for the specified state.
        """
        return self.states[state_name].get_overall_qaly_loss(case_weight)

    def get_weekly_qaly_loss_for_a_state(self, state_name, case_weight):
        """
        Get the weekly QALY loss for a specific state, as a timeseries.

        :param state_name: Name of the state.
        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Weekly QALY loss for the specified state as a timeseries
        """
        return self.states[state_name].get_weekly_qaly_loss(case_weight)

    def plot_weekly_qaly_loss_by_state(self, case_weight):
        """
        Plots the weekly QALY loss per 100,000 population for each state in a single plot

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Plot of weekly QALY loss per 100,000 population for each state in a single plot
        """
        weekly_qaly_loss_by_state = self.get_weekly_qaly_loss_by_state(case_weight)

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

    def plot_weekly_qaly_loss(self, case_weight):
        """
        Plots the weekly QALY loss per 100,000 population summed over all states

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
        :return: Plot of weekly QALY loss per 100,000 population across all states
        """

        # Define dates
        county_data, dates = get_dict_of_county_data_by_type('cases')

        # Calculate the total weekly QALY loss and national population for all states
        total_weekly_qaly_loss = self.get_weekly_qaly_loss(case_weight)
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

    def plot_map_of_qaly_loss_by_county(self, case_weight):
        """
        Plots a map of the QALY loss per 100,000 population for each county

        :param case_weight: Coefficient applied to each case in calculating QALY loss.
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
                # Append county data to the list
                county_qaly_loss_data["COUNTY"].append(county.name)
                county_qaly_loss_data["FIPS"].append(county.fips)
                qaly_loss_per_100k = (county.get_overall_qaly_loss(case_weight) / county.population) * 100000
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


class AllDataTypes:
    def __init__(self, cases_csvfile, hospitalizations_csvfile, deaths_csvfile):
        """
        Initialize an AllDataTypes object.

        :param cases_csvfile: Path to the CSV file containing cases data.
        :param hospitalizations_csvfile: Path to the CSV file containing hospitalizations data.
        :param deaths_csvfile: Path to the CSV file containing deaths data.
        """

        self.all_states_cases = AllStates(county_data_csvfile=cases_csvfile)
        self.all_states_hospitalizations = AllStates(county_data_csvfile=hospitalizations_csvfile)
        self.all_states_deaths = AllStates(county_data_csvfile=deaths_csvfile)
        #defining colors for plots
        self.colors = {
            'cases': 'blue',
            'deaths': 'red',
            'hospitalizations': 'green'
        }

        # Populate data
        self.all_states_cases.populate('cases')
        self.all_states_hospitalizations.populate('hospitalizations')
        self.all_states_deaths.populate('deaths')

    def get_total_qaly_loss(self, case_weight, hospitalizations_weight, deaths_weight):
        """
        Calculate the total QALY loss over time for all states, accumulating over data types.

        :param case_weight: Coefficient applied to cases in calculating QALY loss.
        :param hospitalizations_weight: Coefficient applied to hospitalizations in calculating QALY loss.
        :param deaths_weight: Coefficient applied to deaths in calculating QALY loss.
        :return: Total QALY loss for all states.
        """
        total_cases_qaly_loss = self.all_states_cases.get_overall_qaly_loss(case_weight)
        total_hospitalizations_qaly_loss = self.all_states_hospitalizations.get_overall_qaly_loss(hospitalizations_weight)
        total_deaths_qaly_loss = self.all_states_deaths.get_overall_qaly_loss(deaths_weight)

        total_qaly_loss = total_cases_qaly_loss + total_hospitalizations_qaly_loss + total_deaths_qaly_loss
        return total_qaly_loss

    def get_total_weekly_qaly_loss(self, case_weight, hospitalizations_weight, deaths_weight):
        """
        Get the total weekly QALY loss for all states.

        :param case_weight: Coefficient applied to cases in calculating QALY loss.
        :param hospitalizations_weight: Coefficient applied to hospitalizations in calculating QALY loss.
        :param deaths_weight: Coefficient applied to deaths in calculating QALY loss.
        :return: Total weekly QALY loss for all states.
        """
        total_weekly_cases = self.all_states_cases.get_weekly_qaly_loss(case_weight)
        total_weekly_hospitalizations = self.all_states_hospitalizations.get_weekly_qaly_loss(hospitalizations_weight)
        total_weekly_deaths = self.all_states_deaths.get_weekly_qaly_loss(deaths_weight)

        # Sum the weekly QALY loss for each data type
        total_weekly_qaly_loss = total_weekly_cases + total_weekly_hospitalizations + total_weekly_deaths

        return total_weekly_qaly_loss

    def get_total_overall_qaly_loss_by_state(self, case_weight, hospitalizations_weight, deaths_weight):
        """
        Get the total overall QALY loss by state for all states.

        :param case_weight: Coefficient applied to cases in calculating QALY loss.
        :param hospitalizations_weight: Coefficient applied to hospitalizations in calculating QALY loss.
        :param deaths_weight: Coefficient applied to deaths in calculating QALY loss.
        :return: Dictionary with state names as keys and their respective total overall QALY losses.
        """
        total_overall_qaly_loss_by_state_cases = self.all_states_cases.get_overall_qaly_loss_by_state(case_weight)
        total_overall_qaly_loss_by_state_hospitalizations = self.all_states_hospitalizations.get_overall_qaly_loss_by_state(hospitalizations_weight)
        total_overall_qaly_loss_by_state_deaths = self.all_states_deaths.get_overall_qaly_loss_by_state(deaths_weight)

        total_overall_qaly_loss_by_state = {}

        for state_name in total_overall_qaly_loss_by_state_cases:
            total_overall_qaly_loss_by_state[state_name] = (
                total_overall_qaly_loss_by_state_cases[state_name] +
                total_overall_qaly_loss_by_state_hospitalizations[state_name] +
                total_overall_qaly_loss_by_state_deaths[state_name]
            )

        return total_overall_qaly_loss_by_state

    def get_qaly_loss_by_state_and_type(self, case_weight, hospitalizations_weight, deaths_weight):
        """
        Get the total overall QALY loss by state broken down by data type

        :param case_weight: Coefficient applied to cases in calculating QALY loss.
        :param hospitalizations_weight: Coefficient applied to hospitalizations in calculating QALY loss.
        :param deaths_weight: Coefficient applied to deaths in calculating QALY loss.
        :return: Dictionary with state names as keys and their respective overall QALY losses by data type
        """
        qaly_loss_by_state_and_type = {}

        for state_name, state_obj in self.all_states_cases.states.items():
            # Calculate QALY loss for each data type
            cases_qaly_loss = state_obj.get_overall_qaly_loss(case_weight)
            hospitalizations_qaly_loss = self.all_states_hospitalizations.states[state_name].get_overall_qaly_loss(
                hospitalizations_weight)
            deaths_qaly_loss = self.all_states_deaths.states[state_name].get_overall_qaly_loss(deaths_weight)

            qaly_loss_by_state_and_type[state_name] = {
                'cases': cases_qaly_loss,
                'hospitalizations': hospitalizations_qaly_loss,
                'deaths': deaths_qaly_loss
            }

        return qaly_loss_by_state_and_type

    def plot_weekly_qaly_loss_by_data_type(self, case_weight, hospitalizations_weight, deaths_weight):
        """
                    Plots weekly QALY loss per 100,000 population across all states, broken down by data type

                    :param case_weight: Coefficient applied to each case in calculating QALY loss.
                    :param hospitalizations_weight: Coefficient applied to each hospitalization in calculating QALY loss.
                    :param deaths_weight: Coefficient applied to each death in calculating QALY loss.
                    :return: Plot weekly QALY loss per 100,000 population across all states, broken down by data type
            """
        # Calculate the weekly QALY loss for each data type
        total_weekly_cases = self.all_states_cases.get_weekly_qaly_loss(case_weight)
        total_weekly_hospitalizations = self.all_states_hospitalizations.get_weekly_qaly_loss(hospitalizations_weight)
        total_weekly_deaths = self.all_states_deaths.get_weekly_qaly_loss(deaths_weight)

        # Get the dates from your data
        county_data, dates = get_dict_of_county_data_by_type('cases')

        # Determine the maximum length among the data
        max_length = max(len(total_weekly_cases), len(total_weekly_hospitalizations), len(total_weekly_deaths))

        # Pad the arrays with zeros to match the maximum length
        total_weekly_cases = np.pad(total_weekly_cases, (0, max_length - len(total_weekly_cases)))
        total_weekly_hospitalizations = np.pad(total_weekly_hospitalizations,
                                               (0, max_length - len(total_weekly_hospitalizations)))
        total_weekly_deaths = np.pad(total_weekly_deaths, (0, max_length - len(total_weekly_deaths)))

        # Get the total population
        total_population = sum(state.population for state in self.all_states_cases.states.values())

        # Calculate the weekly QALY loss per 100,000 population for each data type
        total_weekly_cases_per_100k = [(qaly_loss / total_population) * 100000 for qaly_loss in total_weekly_cases]
        total_weekly_hospitalizations_per_100k = [(qaly_loss / total_population) * 100000 for qaly_loss in
                                                  total_weekly_hospitalizations]
        total_weekly_deaths_per_100k = [(qaly_loss / total_population) * 100000 for qaly_loss in total_weekly_deaths]

        # Create a line plot with dates on the x-axis
        fig, ax = plt.subplots(figsize=(12, 6))
        x = dates[:max_length]  # Take the first 'max_length' dates
        ax.plot(x, total_weekly_cases_per_100k, label="Cases per 100K")
        ax.plot(x, total_weekly_hospitalizations_per_100k, label="Hospitalizations per 100K")
        ax.plot(x, total_weekly_deaths_per_100k, label="Deaths per 100K")

        ax.set_xlabel("Date")
        ax.tick_params(axis='x', labelsize=6.5)
        plt.xticks(rotation=90)
        ax.set_ylabel("QALY Loss per 100,000 Population")
        ax.set_title("Weekly QALY Loss by Data Type (per 100,000 Population)")
        ax.legend()
        ax.grid(True)

        # Save the figure as an output file
        output_figure(fig, filename=ROOT_DIR + '/figs/total_weekly_qaly_loss_by_data_type.png')

    def plot_total_weekly_qaly_loss(self, case_weight, hospitalizations_weight, deaths_weight):
        """
                Plots the weekly QALY loss per 100,000 population summed over all states and all data types

                :param case_weight: Coefficient applied to each case in calculating QALY loss.
                :param hospitalizations_weight: Coefficient applied to each hospitalization in calculating QALY loss.
                :param deaths_weight: Coefficient applied to each death in calculating QALY loss.
                :return: Plot of total weekly QALY loss per 100,000 population summed over all states and all data types
                """
        # Calculate the total weekly QALY loss for all states
        total_weekly_qaly_loss = self.get_total_weekly_qaly_loss(case_weight, hospitalizations_weight, deaths_weight)

        # Get the dates from your data
        county_data, dates = get_dict_of_county_data_by_type('cases')

        # Get the total population
        total_population = sum(state.population for state in self.all_states_cases.states.values())

        # Calculate the total weekly QALY loss per 100,000 population
        total_weekly_qaly_loss_per_100k = [(qaly_loss / total_population) * 100000 for qaly_loss in
                                           total_weekly_qaly_loss]

        # Create a line plot with dates on the x-axis
        fig, ax = plt.subplots(figsize=(12, 6))
        x = dates[:len(total_weekly_qaly_loss)]  # Use the same number of dates as QALY loss data
        ax.plot(x, total_weekly_qaly_loss_per_100k, label="Total Weekly QALY Loss per 100K", color='blue')

        ax.set_xlabel("Date")
        ax.tick_params(axis='x', labelsize=6.5)
        plt.xticks(rotation=90)
        ax.set_ylabel("Total Weekly QALY Loss per 100,000 Population")
        ax.set_title("Total Weekly QALY Loss (per 100,000 Population) from Cases, Hospitalizations, and Deaths")
        ax.legend()
        ax.grid(True)

        # Save the figure as an output file
        output_figure(fig, filename=ROOT_DIR + '/figs/total_weekly_qaly_loss.png')

    def plot_qaly_loss_by_state_and_type(self, case_weight, hospitalizations_weight, deaths_weight):
        """
                Provides bar graph of QALY loss per 100,000 population in each state, broken down by data type

                :param case_weight: Coefficient applied to each case in calculating QALY loss.
                :param hospitalizations_weight: Coefficient applied to each hospitalization in calculating QALY loss.
                :param deaths_weight: Coefficient applied to each death in calculating QALY loss.
                :return: Bar graph of QALY loss per 100,000 population in each state, broken down by data type
        """
        # Calculate QALY loss for each data type (cases, hospitalizations, deaths)
        qaly_loss_by_state_and_type = self.get_qaly_loss_by_state_and_type(case_weight, hospitalizations_weight,
                                                                           deaths_weight)

        # Create a bar graph where each bar represents a state and shows QALY loss by data type within each bar
        data_types = ['cases', 'hospitalizations', 'deaths']
        state_names = list(qaly_loss_by_state_and_type.keys())
        num_data_types = len(data_types)
        width = 0.8

        fig, ax = plt.subplots(figsize=(12, 6))

        bottom = np.zeros(len(state_names))

        for i, data_type in enumerate(data_types):
            qaly_loss = [qaly_loss_by_state_and_type[state_name][data_type] for state_name in state_names]

            # Calculate QALY loss per 100,000 population based on the state's population
            state_populations = [self.all_states_cases.states[state_name].population for state_name in state_names]
            qaly_loss_per_100k = [(loss / pop) * 100000 for loss, pop in zip(qaly_loss, state_populations)]

            ax.bar(state_names, qaly_loss_per_100k, width, label=data_type, color=self.colors[data_type], bottom=bottom)
            bottom += qaly_loss_per_100k

        ax.set_xlabel('States')
        ax.set_ylabel('QALY Loss per 100,000 Population')
        ax.set_title('QALY Loss per 100,000 Population by State and Data Type')
        ax.legend()

        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()

        # Save the figure as an output file
        output_figure(fig, filename=ROOT_DIR + '/figs/total_qaly_loss_by_state_and_type.png')
