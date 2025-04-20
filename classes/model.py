import os

import geopandas as gpd
import geoplot as gplt
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.colors import Normalize
from matplotlib.colors import BoundaryNorm


from classes.parameters import ParameterGenerator, ParameterValues
from classes.support import get_mean_ui_of_a_time_series, get_overall_mean_ui
from data_preprocessing.support_functions import get_dict_of_county_data_by_type
from deampy.in_out_functions import write_csv, read_csv_rows
from deampy.format_functions import format_interval
from deampy.plots.plot_support import output_figure
from deampy.statistics import SummaryStat
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from definitions import ROOT_DIR
from matplotlib import rc
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression


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

        # Check if weekly_obs is empty
        if weekly_obs.size == 0:
            return  # or handle this case as needed

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
        self.infections = AnOutcome()
        self.hosps = AnOutcome()
        self.deaths = AnOutcome()
        self.hosp_non_icu = AnOutcome()
        self.hosp_icu = AnOutcome()
        self.icu = AnOutcome()
        self.icu_total = AnOutcome()
        self.total_hosp = AnOutcome()

        self.longCOVID_1 = AnOutcome()

        self.longCOVID_2 = AnOutcome()
        self.cases_lc_2 = AnOutcome()
        self.hosps_lc_2 = AnOutcome()
        self.icu_lc_2 = AnOutcome()

        self.longCOVID_vax_LB = AnOutcome()
        self.lc_v_lb = AnOutcome()
        self.lc_uv_lb = AnOutcome()

        self.longCOVID_vax_UB = AnOutcome()
        self.lc_v_ub = AnOutcome()
        self.lc_uv_ub = AnOutcome()

        self.deaths_sa_1_a = AnOutcome()
        self.deaths_sa_1_b = AnOutcome()
        self.deaths_sa_1_c = AnOutcome()

        self.deaths_sa_2_a = AnOutcome()
        self.deaths_sa_2_b = AnOutcome()
        self.deaths_sa_2_c = AnOutcome()

        self.deaths_sa_3_a = AnOutcome()
        self.deaths_sa_3_b = AnOutcome()
        self.deaths_sa_3_c = AnOutcome()

        self.total_1 = AnOutcome()
        self.total_2 = AnOutcome()
        self.total_sa_1_a = AnOutcome()
        self.total_sa_1_b = AnOutcome()
        self.total_sa_1_c = AnOutcome()
        self.total_sa_2_a = AnOutcome()
        self.total_sa_2_b = AnOutcome()
        self.total_sa_2_c = AnOutcome()
        self.total_sa_3_a = AnOutcome()
        self.total_sa_3_b = AnOutcome()
        self.total_sa_3_c = AnOutcome()

        self.total_vax_lb = AnOutcome()
        self.total_vax_ub = AnOutcome()

        self.weeklyQALYLoss = np.array([])
        self.totalQALYLoss = 0

    def add_traj(self, weekly_cases, weekly_infections, weekly_hosps, weekly_deaths, weekly_icu,
                 weekly_lc_v_lb,weekly_lc_uv_lb, weekly_lc_v_ub,weekly_lc_uv_ub):
        self.cases.add_traj(weekly_obs=weekly_cases)
        self.infections.add_traj(weekly_obs=weekly_infections)
        self.hosps.add_traj(weekly_obs=weekly_hosps)
        self.deaths.add_traj(weekly_obs=weekly_deaths)

        self.hosp_non_icu.add_traj(weekly_obs=weekly_hosps)
        self.hosp_icu.add_traj(weekly_obs=weekly_hosps)
        self.icu.add_traj(weekly_obs=weekly_icu)

        self.longCOVID_1.add_traj(weekly_obs=weekly_infections)

        self.cases_lc_2.add_traj(weekly_infections)
        self.hosps_lc_2.add_traj(weekly_obs=weekly_hosps)
        self.icu_lc_2.add_traj(weekly_obs=weekly_hosps)

        self.lc_v_lb.add_traj(weekly_obs=weekly_lc_v_lb)
        self.lc_uv_lb.add_traj(weekly_obs=weekly_lc_uv_lb)
        self.lc_v_ub.add_traj(weekly_obs=weekly_lc_v_ub)
        self.lc_uv_ub.add_traj(weekly_obs=weekly_lc_uv_ub)

        self.deaths_sa_1_a.add_traj(weekly_obs=weekly_deaths)
        self.deaths_sa_1_b.add_traj(weekly_obs=weekly_deaths)
        self.deaths_sa_1_c.add_traj(weekly_obs=weekly_deaths)

        self.deaths_sa_2_a.add_traj(weekly_obs=weekly_deaths)
        self.deaths_sa_2_b.add_traj(weekly_obs=weekly_deaths)
        self.deaths_sa_2_c.add_traj(weekly_obs=weekly_deaths)

        self.deaths_sa_3_a.add_traj(weekly_obs=weekly_deaths)
        self.deaths_sa_3_b.add_traj(weekly_obs=weekly_deaths)
        self.deaths_sa_3_c.add_traj(weekly_obs=weekly_deaths)

    def calculate_qaly_loss(self, case_weight, death_weight, icu_weight, hosp_icu_weight, hosp_ward_weight,
                            long_covid_weight_1, long_covid_weight_2_nh,
                            long_covid_weight_2_h, long_covid_weight_2_i,
                            death_sa_1a_weight,death_sa_1b_weight,death_sa_1c_weight,
                            death_sa_2a_weight,death_sa_2b_weight,death_sa_2c_weight,
                            death_sa_3a_weight,death_sa_3b_weight,death_sa_3c_weight,
                            long_covid_weight_1_v,long_covid_weight_1_uv):

        self.cases.calculate_qaly_loss(quality_weight=case_weight)
        self.deaths.calculate_qaly_loss(quality_weight=death_weight)
        self.icu.calculate_qaly_loss(quality_weight=icu_weight)
        self.infections.calculate_qaly_loss(quality_weight=case_weight)

        self.hosp_non_icu.calculate_qaly_loss(quality_weight=hosp_ward_weight)
        self.hosp_icu.calculate_qaly_loss(quality_weight=hosp_icu_weight)

        self.icu_total.weeklyQALYLoss = (self.hosp_icu.weeklyQALYLoss + self.icu.weeklyQALYLoss)
        self.icu_total.totalQALYLoss = (self.hosp_icu.totalQALYLoss + self.icu.totalQALYLoss)

        self.total_hosp.weeklyQALYLoss = self.icu.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
        self.total_hosp.totalQALYLoss = self.icu.totalQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.totalQALYLoss

        # Under LC Scemario 1:
        self.longCOVID_1.calculate_qaly_loss(quality_weight=long_covid_weight_1)
        self.total_1.weeklyQALYLoss = (self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                               + self.deaths.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_1.weeklyQALYLoss)
        self.total_1.totalQALYLoss = (self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                              + self.deaths.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_1.totalQALYLoss)


        # Under LC Scemario 2:
        self.cases_lc_2.calculate_qaly_loss(quality_weight=long_covid_weight_2_nh)
        self.hosps_lc_2.calculate_qaly_loss(quality_weight=long_covid_weight_2_h)
        self.icu_lc_2.calculate_qaly_loss(quality_weight=long_covid_weight_2_i)

        self.longCOVID_2.weeklyQALYLoss = self.cases_lc_2.weeklyQALYLoss + self.hosps_lc_2.weeklyQALYLoss + self.icu_lc_2.weeklyQALYLoss
        self.longCOVID_2.totalQALYLoss = self.cases_lc_2.totalQALYLoss + self.hosps_lc_2.totalQALYLoss + self.icu_lc_2.totalQALYLoss

        self.total_2.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_2.weeklyQALYLoss)
        self.total_2.totalQALYLoss = (
                    self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                    + self.deaths.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_2.totalQALYLoss)

        self.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_2.weeklyQALYLoss)
        self.totalQALYLoss = (
                    self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                    + self.deaths.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_2.totalQALYLoss)

        #Long COVID Vax
        self.lc_v_lb.calculate_qaly_loss(quality_weight=long_covid_weight_1_v)
        self.lc_uv_lb.calculate_qaly_loss(quality_weight=long_covid_weight_1_uv)
        self.lc_v_ub.calculate_qaly_loss(quality_weight=long_covid_weight_1_v)
        self.lc_uv_ub.calculate_qaly_loss(quality_weight=long_covid_weight_1_uv)

        self.longCOVID_vax_LB.weeklyQALYLoss=self.lc_v_lb.weeklyQALYLoss+self.lc_uv_lb.weeklyQALYLoss
        self.longCOVID_vax_LB.totalQALYLoss = self.lc_v_lb.totalQALYLoss + self.lc_uv_lb.totalQALYLoss

        self.longCOVID_vax_UB.weeklyQALYLoss = self.lc_v_ub.weeklyQALYLoss + self.lc_uv_ub.weeklyQALYLoss
        self.longCOVID_vax_UB.totalQALYLoss = self.lc_v_ub.totalQALYLoss + self.lc_uv_ub.totalQALYLoss

        # Deaths SA: SMR
        self.deaths_sa_1_a.calculate_qaly_loss(quality_weight=death_sa_1a_weight)
        self.deaths_sa_1_b.calculate_qaly_loss(quality_weight=death_sa_1b_weight)
        self.deaths_sa_1_c.calculate_qaly_loss(quality_weight=death_sa_1c_weight)
        self.deaths_sa_2_a.calculate_qaly_loss(quality_weight=death_sa_2a_weight)
        self.deaths_sa_2_b.calculate_qaly_loss(quality_weight=death_sa_2b_weight)
        self.deaths_sa_2_c.calculate_qaly_loss(quality_weight=death_sa_2c_weight)
        self.deaths_sa_3_a.calculate_qaly_loss(quality_weight=death_sa_3a_weight)
        self.deaths_sa_3_b.calculate_qaly_loss(quality_weight=death_sa_3b_weight)
        self.deaths_sa_3_c.calculate_qaly_loss(quality_weight=death_sa_3c_weight)

        self.total_sa_1_a.totalQALYLoss=(self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                              + self.deaths_sa_1_a.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_1.totalQALYLoss)
        self.total_sa_1_b.totalQALYLoss = (self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                    + self.deaths_sa_1_b.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_1.totalQALYLoss)
        self.total_sa_1_c.totalQALYLoss=(self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                              + self.deaths_sa_1_c.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_1.totalQALYLoss)
        self.total_sa_2_a.totalQALYLoss=(self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                              + self.deaths_sa_2_a.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_1.totalQALYLoss)
        self.total_sa_2_b.totalQALYLoss = (self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                    + self.deaths_sa_2_b.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_1.totalQALYLoss)
        self.total_sa_2_c.totalQALYLoss=(self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                              + self.deaths_sa_2_c.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_1.totalQALYLoss)
        self.total_sa_3_a.totalQALYLoss = (self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                    + self.deaths_sa_3_a.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_1.totalQALYLoss)
        self.total_sa_3_b.totalQALYLoss = (self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                    + self.deaths_sa_3_b.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_1.totalQALYLoss)
        self.total_sa_3_c.totalQALYLoss = (self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                    + self.deaths_sa_3_c.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_1.totalQALYLoss)

        self.total_sa_1_a.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths_sa_1_a.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_1.weeklyQALYLoss)
        self.total_sa_1_b.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths_sa_1_b.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_1.weeklyQALYLoss)
        self.total_sa_1_c.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths_sa_1_c.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_1.weeklyQALYLoss)
        self.total_sa_2_a.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths_sa_2_a.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_1.weeklyQALYLoss)
        self.total_sa_2_b.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths_sa_2_b.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_1.weeklyQALYLoss)
        self.total_sa_2_c.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths_sa_2_c.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_1.weeklyQALYLoss)
        self.total_sa_3_a.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths_sa_3_a.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_1.weeklyQALYLoss)
        self.total_sa_3_b.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths_sa_3_b.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_1.weeklyQALYLoss)
        self.total_sa_3_c.weeklyQALYLoss = (
                    self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                    + self.deaths_sa_3_c.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_1.weeklyQALYLoss)

        self.total_vax_lb.weeklyQALYLoss= (self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                            + self.deaths.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_vax_LB.weeklyQALYLoss)

        self.total_vax_ub.weeklyQALYLoss = (self.infections.weeklyQALYLoss + self.hosp_non_icu.weeklyQALYLoss + self.hosp_icu.weeklyQALYLoss
                             + self.deaths.weeklyQALYLoss + self.icu.weeklyQALYLoss + self.longCOVID_vax_UB.weeklyQALYLoss)

        self.total_vax_lb.totalQALYLoss= (self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                            + self.deaths.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_vax_LB.totalQALYLoss)

        self.total_vax_ub.totalQALYLoss = (self.infections.totalQALYLoss + self.hosp_non_icu.totalQALYLoss + self.hosp_icu.totalQALYLoss
                             + self.deaths.totalQALYLoss + self.icu.totalQALYLoss + self.longCOVID_vax_UB.totalQALYLoss)


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

    def add_traj(self, weekly_cases, weekly_infections,  weekly_hosps, weekly_deaths, weekly_icu,
                 weekly_lc_v_lb, weekly_lc_uv_lb, weekly_lc_v_ub, weekly_lc_uv_ub):

        """
        Add weekly data to the County object.
        :param weekly_cases: Weekly cases data as a numpy array.
        :param weekly_hosp: Weekly hospitalizations data as a numpy array.
        :param weekly_deaths: Weekly deaths data as a numpy array.
        """
        self.pandemicOutcomes.add_traj(
            weekly_cases=weekly_cases, weekly_infections=weekly_infections, weekly_hosps=weekly_hosps, weekly_deaths=weekly_deaths, weekly_icu=weekly_icu,
            weekly_lc_v_lb =  weekly_lc_v_lb,weekly_lc_uv_lb=weekly_lc_uv_lb, weekly_lc_v_ub=weekly_lc_v_ub,weekly_lc_uv_ub=weekly_lc_uv_ub)


    def calculate_qaly_loss(self, case_weight, death_weight, icu_weight, hosp_icu_weight, hosp_ward_weight,
                            long_covid_weight_1, long_covid_weight_2_nh,
                            long_covid_weight_2_h, long_covid_weight_2_i,
                            death_sa_1a_weight,death_sa_1b_weight,death_sa_1c_weight,
                            death_sa_2a_weight,death_sa_2b_weight,death_sa_2c_weight,
                            death_sa_3a_weight,death_sa_3b_weight,death_sa_3c_weight,
                            long_covid_weight_1_v,long_covid_weight_1_uv):
        """
        Calculates the weekly and total QALY loss for the County.

        :param case_weight: cases-specific weight to be applied to each case in calculating QALY loss.
        :param death_weight: death-specific weight to be applied to each death in calculating QALY loss.
        :param hosp_weight: hosp-specific weight to be applied to each hospitalization in calculating QALY loss.
        :return QALY loss for each county.
        """

        self.pandemicOutcomes.calculate_qaly_loss(
            case_weight=case_weight, death_weight=death_weight, icu_weight=icu_weight, hosp_icu_weight=hosp_icu_weight,
            hosp_ward_weight=hosp_ward_weight, long_covid_weight_1=long_covid_weight_1,
             long_covid_weight_2_nh=long_covid_weight_2_nh,
            long_covid_weight_2_h=long_covid_weight_2_h, long_covid_weight_2_i=long_covid_weight_2_i,
            death_sa_1a_weight=death_sa_1a_weight , death_sa_1b_weight=death_sa_1b_weight, death_sa_1c_weight=death_sa_1c_weight,
            death_sa_2a_weight=death_sa_2a_weight, death_sa_2b_weight=death_sa_2b_weight, death_sa_2c_weight=death_sa_2c_weight,
            death_sa_3a_weight=death_sa_3a_weight, death_sa_3b_weight=death_sa_3b_weight, death_sa_3c_weight=death_sa_3c_weight,
            long_covid_weight_1_v=long_covid_weight_1_v,long_covid_weight_1_uv=long_covid_weight_1_uv)

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
            weekly_infections=county.pandemicOutcomes.infections.weeklyObs,
            weekly_hosps=county.pandemicOutcomes.hosps.weeklyObs,
            weekly_deaths=county.pandemicOutcomes.deaths.weeklyObs,
            weekly_icu=county.pandemicOutcomes.icu.weeklyObs,
            weekly_lc_v_lb=county.pandemicOutcomes.lc_v_lb.weeklyObs,
            weekly_lc_uv_lb=county.pandemicOutcomes.lc_uv_lb.weeklyObs,
            weekly_lc_v_ub=county.pandemicOutcomes.lc_v_ub.weeklyObs,
            weekly_lc_uv_ub=county.pandemicOutcomes.lc_uv_ub.weeklyObs
            )



    def calculate_qaly_loss(self, case_weight, death_weight, icu_weight, hosp_icu_weight, hosp_ward_weight,
                            long_covid_weight_1, long_covid_weight_2_nh,
                            long_covid_weight_2_h, long_covid_weight_2_i,
                            death_sa_1a_weight,death_sa_1b_weight,death_sa_1c_weight,
                            death_sa_2a_weight,death_sa_2b_weight,death_sa_2c_weight,
                            death_sa_3a_weight,death_sa_3b_weight,death_sa_3c_weight,
                            long_covid_weight_1_v,long_covid_weight_1_uv):

        """
        Calculates QALY loss for the State.
        :param case_weight: cases-specific weight to be applied to each case in calculating QALY loss.
        :param hosp_weight: hosp-specific weight to be applied to each hospitalization in calculating QALY loss.
        :param death_weight: death-specific weight to be applied to each death in calculating QALY loss.
        """

        for county in self.counties.values():
            county.calculate_qaly_loss(
                case_weight=case_weight, death_weight=death_weight, icu_weight=icu_weight,
                hosp_icu_weight=hosp_icu_weight,
                hosp_ward_weight=hosp_ward_weight, long_covid_weight_1=long_covid_weight_1,
                long_covid_weight_2_nh=long_covid_weight_2_nh,
                long_covid_weight_2_h=long_covid_weight_2_h, long_covid_weight_2_i=long_covid_weight_2_i,
                death_sa_1a_weight=death_sa_1a_weight, death_sa_1b_weight=death_sa_1b_weight,death_sa_1c_weight=death_sa_1c_weight,
                death_sa_2a_weight=death_sa_2a_weight, death_sa_2b_weight=death_sa_2b_weight,death_sa_2c_weight=death_sa_2c_weight,
                death_sa_3a_weight=death_sa_3a_weight, death_sa_3b_weight=death_sa_3b_weight,death_sa_3c_weight=death_sa_3c_weight,
                long_covid_weight_1_v=long_covid_weight_1_v, long_covid_weight_1_uv=long_covid_weight_1_uv)

            # Calculate QALY loss for the state
        self.pandemicOutcomes.calculate_qaly_loss(
            case_weight=case_weight, death_weight=death_weight, icu_weight=icu_weight, hosp_icu_weight=hosp_icu_weight,
            hosp_ward_weight=hosp_ward_weight, long_covid_weight_1=long_covid_weight_1,
             long_covid_weight_2_nh=long_covid_weight_2_nh,
            long_covid_weight_2_h=long_covid_weight_2_h, long_covid_weight_2_i=long_covid_weight_2_i,
            death_sa_1a_weight=death_sa_1a_weight, death_sa_1b_weight=death_sa_1b_weight,death_sa_1c_weight=death_sa_1c_weight,
            death_sa_2a_weight=death_sa_2a_weight, death_sa_2b_weight=death_sa_2b_weight,death_sa_2c_weight=death_sa_2c_weight,
            death_sa_3a_weight=death_sa_3a_weight, death_sa_3b_weight=death_sa_3b_weight,death_sa_3c_weight=death_sa_3c_weight,
            long_covid_weight_1_v=long_covid_weight_1_v, long_covid_weight_1_uv=long_covid_weight_1_uv)

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
        self.dates = []

    def populate(self):
        """
        Populates the AllStates object with county case data.
        """

        county_case_data, dates = get_dict_of_county_data_by_type('cases')
        county_infection_data, dates = get_dict_of_county_data_by_type('infections')
        county_death_data, dates = get_dict_of_county_data_by_type('deaths')
        county_hosp_data, dates = get_dict_of_county_data_by_type('hospitalizations')
        county_icu_data, dates = get_dict_of_county_data_by_type('icu')
        county_lc_v_lb_data, dates = get_dict_of_county_data_by_type('infections_v_LB')
        county_lc_uv_lb_data, dates = get_dict_of_county_data_by_type('infections_uv_LB')
        county_lc_v_ub_data, dates = get_dict_of_county_data_by_type('infections_v_UB')
        county_lc_uv_ub_data, dates = get_dict_of_county_data_by_type('infections_uv_UB')


        self.numWeeks = len(dates)

        self.dates = dates

        for (county_name, state, fips, population), case_values in county_case_data.items():

            self.population += int(population)

            # making sure data is available for deaths and hospitalizations
            try:
                infection_values = county_infection_data[(county_name, state, fips, population)]
                death_values = county_death_data[(county_name, state, fips, population)]
                hosp_values = county_hosp_data[(county_name, state, fips, population)]
                icu_values = county_icu_data[(county_name, state, fips, population)]
                lc_v_lb_values = county_lc_v_lb_data[(county_name, state, fips, population)]
                lc_uv_lb_data_values = county_lc_uv_lb_data[(county_name, state, fips, population)]
                lc_v_ub_data_values = county_lc_v_ub_data[(county_name, state, fips, population)]
                lc_uv_ub_data_values = county_lc_uv_ub_data[(county_name, state, fips, population)]



            except KeyError as e:
                raise KeyError(f"Data not found for {county_name}, {state}, {fips}, {population}.") from e

            # create a new county
            county = County(
                name=county_name, state=state, fips=fips, population=int(population))

            # Add weekly data to county object
            county.add_traj(weekly_cases=case_values,
                weekly_infections=infection_values, weekly_hosps=hosp_values, weekly_deaths=death_values,weekly_icu=icu_values,
                            weekly_lc_v_lb=lc_v_lb_values,
                            weekly_lc_uv_lb=lc_uv_lb_data_values,
                            weekly_lc_v_ub=lc_v_ub_data_values,
                            weekly_lc_uv_ub=lc_uv_ub_data_values
                            )


            # update the nation pandemic outcomes based on the outcomes for this county
            self.pandemicOutcomes.add_traj(
                weekly_cases=county.pandemicOutcomes.cases.weeklyObs,
                weekly_infections=county.pandemicOutcomes.infections.weeklyObs,
                weekly_deaths=county.pandemicOutcomes.deaths.weeklyObs,
                weekly_hosps = county.pandemicOutcomes.hosps.weeklyObs,
                weekly_icu=county.pandemicOutcomes.icu.weeklyObs,
                weekly_lc_v_lb=county.pandemicOutcomes.lc_v_lb.weeklyObs,
                weekly_lc_uv_lb=county.pandemicOutcomes.lc_uv_lb.weeklyObs,
                weekly_lc_v_ub=county.pandemicOutcomes.lc_v_ub.weeklyObs,
                weekly_lc_uv_ub=county.pandemicOutcomes.lc_uv_ub.weeklyObs
                )

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
                death_weight=param_values.qWeightDeath,
                icu_weight=param_values.qWeightICU,
                hosp_icu_weight=param_values.qWeightICUHosp,
                hosp_ward_weight=param_values.qWeightHosp,
                long_covid_weight_1=param_values.qWeightLongCOVID_1,
                long_covid_weight_2_nh=param_values.qWeightLongCOVID_2_nh,
                long_covid_weight_2_h=param_values.qWeightLongCOVID_2_h,
                long_covid_weight_2_i=param_values.qWeightLongCOVID_2_i,
                death_sa_1a_weight=param_values.qWeightDeath_sa_1_a,
                death_sa_1b_weight=param_values.qWeightDeath_sa_1_b,
                death_sa_1c_weight=param_values.qWeightDeath_sa_1_c,
                death_sa_2a_weight=param_values.qWeightDeath_sa_2_a,
                death_sa_2b_weight=param_values.qWeightDeath_sa_2_b,
                death_sa_2c_weight=param_values.qWeightDeath_sa_2_c,
                death_sa_3a_weight=param_values.qWeightDeath_sa_3_a,
                death_sa_3b_weight=param_values.qWeightDeath_sa_3_b,
                death_sa_3c_weight=param_values.qWeightDeath_sa_3_c,
                long_covid_weight_1_v=param_values.qWeightLongCOVID_1_v,
                long_covid_weight_1_uv=param_values.qWeightLongCOVID_1_uv
            )



        # calculate QALY loss for the nation
        self.pandemicOutcomes.calculate_qaly_loss(
            case_weight=param_values.qWeightCase,
            death_weight=param_values.qWeightDeath,
            icu_weight=param_values.qWeightICU,
            hosp_icu_weight=param_values.qWeightICUHosp,
            hosp_ward_weight=param_values.qWeightHosp,
            long_covid_weight_1=param_values.qWeightLongCOVID_1,
            long_covid_weight_2_nh=param_values.qWeightLongCOVID_2_nh,
            long_covid_weight_2_h=param_values.qWeightLongCOVID_2_h,
            long_covid_weight_2_i=param_values.qWeightLongCOVID_2_i,
            death_sa_1a_weight=param_values.qWeightDeath_sa_1_a,
            death_sa_1b_weight=param_values.qWeightDeath_sa_1_b,
            death_sa_1c_weight=param_values.qWeightDeath_sa_1_c,
            death_sa_2a_weight=param_values.qWeightDeath_sa_2_a,
            death_sa_2b_weight=param_values.qWeightDeath_sa_2_b,
            death_sa_2c_weight=param_values.qWeightDeath_sa_2_c,
            death_sa_3a_weight=param_values.qWeightDeath_sa_3_a,
            death_sa_3b_weight=param_values.qWeightDeath_sa_3_b,
            death_sa_3c_weight=param_values.qWeightDeath_sa_3_c,
            long_covid_weight_1_v=param_values.qWeightLongCOVID_1_v,
            long_covid_weight_1_uv=param_values.qWeightLongCOVID_1_uv
            )

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
                        'hosp (non icu pts)': self.pandemicOutcomes.hosp_non_icu.totalQALYLoss,
                        'hosp(icu pts)': self.pandemicOutcomes.hosp_icu.totalQALYLoss,
                        'deaths': self.pandemicOutcomes.deaths.totalQALYLoss,
                        'infections': self.pandemicOutcomes.infections.totalQALYLoss,
                        'icu': self.pandemicOutcomes.icu.totalQALYLoss,
                        'icu (total pt experience)': self.pandemicOutcomes.icu_total.totalQALYLoss,
                        'total hosp (icu + general ward)': self.pandemicOutcomes.total_hosp.totalQALYLoss,
                        'longcovid_1': self.pandemicOutcomes.longCOVID_1.totalQALYLoss,
                        'longcovid_2': self.pandemicOutcomes.longCOVID_2.totalQALYLoss,
                        'deaths, smr=1.75, qCM=0.85, r=3%': self.pandemicOutcomes.deaths_sa_1_a.totalQALYLoss,
                        'deaths, smr=1.75, qCM=0.8, r= 3%': self.pandemicOutcomes.deaths_sa_1_b.totalQALYLoss,
                        'deaths, smr=1.75 , qCM=0.75, r= 3%': self.pandemicOutcomes.deaths_sa_1_c.totalQALYLoss,
                        'deaths, smr=2, qCM=0.85, r=3%': self.pandemicOutcomes.deaths_sa_2_a.totalQALYLoss,
                        'deaths, smr=2, qCM=0.8, r= 3%': self.pandemicOutcomes.deaths_sa_2_b.totalQALYLoss,
                        'deaths, smr=2, qCM=0.0.75, r= 3%': self.pandemicOutcomes.deaths_sa_2_c.totalQALYLoss,
                        'deaths, smr=2.25, qCM=0.85, r=3%': self.pandemicOutcomes.deaths_sa_3_a.totalQALYLoss,
                        'deaths, smr=2.25, qCM=0.8, r= 3%': self.pandemicOutcomes.deaths_sa_3_b.totalQALYLoss,
                        'deaths, smr=2.25 , qCM=0.75, r= 3%': self.pandemicOutcomes.deaths_sa_3_c.totalQALYLoss,
                        'total (LC1)': self.pandemicOutcomes.total_1.totalQALYLoss,
                        'total (LC2)': self.pandemicOutcomes.total_2.totalQALYLoss,
                        'total, smr=1.75, qCM=0.85, r=3%': self.pandemicOutcomes.total_sa_1_a.totalQALYLoss,
                        'total, smr=1.75, qCM=0.8, r= 3%': self.pandemicOutcomes.total_sa_1_b.totalQALYLoss,
                        'total, smr=1.75 , qCM=0.75, r= 3%': self.pandemicOutcomes.total_sa_1_c.totalQALYLoss,
                        'total, smr=2, qCM=0.85, r=3%': self.pandemicOutcomes.total_sa_2_a.totalQALYLoss,
                        'total, smr=2, qCM=0.8, r= 3%': self.pandemicOutcomes.total_sa_2_b.totalQALYLoss,
                        'total, smr=2, qCM=0.0.75, r= 3%': self.pandemicOutcomes.total_sa_2_c.totalQALYLoss,
                        'total, smr=2.25, qCM=0.85, r=3%': self.pandemicOutcomes.total_sa_3_a.totalQALYLoss,
                        'total, smr=2.25, qCM=0.8, r= 3%': self.pandemicOutcomes.total_sa_3_b.totalQALYLoss,
                        'total, smr=2.25 , qCM=0.75, r= 3%': self.pandemicOutcomes.total_sa_3_c.totalQALYLoss
                        }

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
        plt.title("Cumulative QALY Loss per 100,000 Population by County", fontsize=22)

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
        ax.plot(weeks, self.pandemicOutcomes.infections.weeklyQALYLoss, label='Infections', color='blue')
        ax.plot(weeks, self.pandemicOutcomes.total_hosp.weeklyQALYLoss,label='Hospitalizations (including ICU)', color='green')
        ax.plot(weeks, self.pandemicOutcomes.deaths.weeklyQALYLoss, label='Deaths', color='red')
        ax.plot(weeks, self.pandemicOutcomes.longCOVID_1.weeklyQALYLoss, label='Long COVID', color = 'lightblue')

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
        long_covid_color = 'light blue'

        # Iterate through each state
        for i, state_obj in enumerate(states_list):
            # Calculate the heights for each segment
            infections_height = (state_obj.pandemicOutcomes.infections.totalQALYLoss / state_obj.population) * 100000
            deaths_height = (state_obj.pandemicOutcomes.deaths.totalQALYLoss / state_obj.population) * 100000
            hosps_height = ((state_obj.pandemicOutcomes.hosp_non_icu.totalQALYLoss + state_obj.pandemicOutcomes.hosp_icu.totalQALYLoss + state_obj.pandemicOutcomes.icu.totalQALYLoss) / state_obj.population) * 100000
            long_covid_height= (state_obj.pandemicOutcomes.longCOVID_1.totalQALYLoss/state_obj.population) *100000

            # Plot the segments
            ax.bar(i, infections_height, color=cases_color, width=bar_width, align='center', label='Infections' if i == 0 else "")
            ax.bar(i, deaths_height, bottom=infections_height, color=deaths_color, width=bar_width, align='center',
                   label='Deaths' if i == 0 else "")
            ax.bar(i, hosps_height, bottom=infections_height + deaths_height, color=hosps_color, width=bar_width,
                   align='center', label='Hospitalizations (including ICU)' if i == 0 else "")
            ax.bar(i, long_covid_height, bottom=infections_height + deaths_height + hosps_height, color=long_covid_color, width=bar_width,
                   align='center' , label = 'Long Covid' if i == 0 else "")

        # Set the labels for each state
        ax.set_xticks(bar_positions)
        ax.set_xticklabels([state_obj.name for state_obj in states_list], fontsize=8, rotation=45, ha='right')

        # Set the labels and title
        ax.set_xlabel('States')
        ax.set_ylabel('Total QALY Loss per 100,000')
        ax.set_title('Total QALY Loss by State and Outcome')

        # Show the legend with unique labels
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Cases', 'Deaths', 'Hospitalizations (including ICU)', 'Long COVID'])

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

    def get_overall_qaly_loss_by_state_infections(self):
        """
        :return: (dictionary) Overall QALY loss from cases by states (as dictionary key)
        """
        overall_qaly_loss_infections_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_infections_by_state[state_name] = state_obj.pandemicOutcomes.infections.totalQALYLoss
        return overall_qaly_loss_infections_by_state

    def get_overall_qaly_loss_by_state_hosp_non_icu(self):
        """
        :return: (dictionary) Overall QALY loss from hosps (ward portion among pts never admitted to ICU) by states (as dictionary key)
        """
        overall_qaly_loss_hosp_non_icu_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_hosp_non_icu_by_state[state_name] = state_obj.pandemicOutcomes.hosp_non_icu.totalQALYLoss
        return overall_qaly_loss_hosp_non_icu_by_state

    def get_overall_qaly_loss_by_state_hosp_icu(self):
        """
        :return: (dictionary) Overall QALY loss from hosps (ward portion among ICU patients) by states (as dictionary key)
        """
        overall_qaly_loss_hosp_icu_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_hosp_icu_by_state[state_name] = state_obj.pandemicOutcomes.hosp_icu.totalQALYLoss
        return overall_qaly_loss_hosp_icu_by_state

    def get_overall_qaly_loss_by_state_deaths(self):
        """
        :return: (dictionary) Overall QALY loss from deaths by states (as dictionary key)
        """
        overall_qaly_loss_deaths_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_deaths_by_state[state_name] = state_obj.pandemicOutcomes.deaths.totalQALYLoss
        return overall_qaly_loss_deaths_by_state

    def get_overall_qaly_loss_by_state_icu(self):
        """
        :return: (dictionary) Overall QALY loss from icu by states (as dictionary key)
        """
        overall_qaly_loss_icu_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_icu_by_state[state_name] = state_obj.pandemicOutcomes.icu.totalQALYLoss
        return overall_qaly_loss_icu_by_state

    def get_overall_qaly_loss_by_state_icu_total(self):
        """
        :return: (dictionary) Overall QALY loss from icu (total patient experience) by states (as dictionary key)
        """
        overall_qaly_loss_icu_total_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_icu_total_by_state[state_name] = state_obj.pandemicOutcomes.icu_total.totalQALYLoss
        return overall_qaly_loss_icu_total_by_state

    def get_overall_qaly_loss_by_state_total_hosp(self):
        """
        :return: (dictionary) Overall QALY loss from hosps (total among ICU and non ICU) by states (as dictionary key)
        """
        overall_qaly_loss_total_hosp_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_total_hosp_by_state[state_name] = state_obj.pandemicOutcomes.total_hosp.totalQALYLoss
        return overall_qaly_loss_total_hosp_by_state

    def get_overall_qaly_loss_by_state_long_covid_1(self):
        """
        :return: (dictionary) Overall QALY loss from Long COVID (approach 1) by states (as dictionary key)
        """
        overall_qaly_loss_long_covid_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_long_covid_by_state[state_name] = state_obj.pandemicOutcomes.longCOVID_1.totalQALYLoss
        return overall_qaly_loss_long_covid_by_state

    def get_overall_qaly_loss_by_state_long_covid_2(self):
        """
        :return: (dictionary) Overall QALY loss from Long COVID 2 by states (as dictionary key)
        """
        overall_qaly_loss_long_covid_2_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_long_covid_2_by_state[state_name] = state_obj.pandemicOutcomes.longCOVID_2.totalQALYLoss
        return overall_qaly_loss_long_covid_2_by_state

    def get_overall_qaly_loss_by_state_total_1(self):
        """
        :return: (dictionary) Overall QALY loss from Long COVID (approach 1) by states (as dictionary key)
        """
        overall_qaly_loss_total_1_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_total_1_by_state[state_name] = state_obj.pandemicOutcomes.total_1.totalQALYLoss
        return overall_qaly_loss_total_1_by_state

    def get_overall_qaly_loss_by_state_total_2(self):
        """
        :return: (dictionary) Overall QALY loss from Long COVID (approach 1) by states (as dictionary key)
        """
        overall_qaly_loss_total_2_by_state = {}
        for state_name, state_obj in self.states.items():
            overall_qaly_loss_total_2_by_state[state_name] = state_obj.pandemicOutcomes.total_2.totalQALYLoss
        return overall_qaly_loss_total_2_by_state


    def get_death_QALY_loss_by_age(self, param_gen):
        deaths_by_age = (param_gen.parameters['death_age_dist'].value * self.pandemicOutcomes.deaths.totalObs)
        total_dQAlY_loss_by_age = deaths_by_age * param_gen.parameters['dQALY_loss_by_age'].value
        return total_dQAlY_loss_by_age

    def get_cases_QALY_loss_by_age(self, param_gen, param_values):
        cases_by_age = param_gen.parameters['hosps_age_dist'].value * self.pandemicOutcomes.cases.totalObs # at this stage assuming that the age distribution of cases is equivalent to the hosp age distrbution
        total_dQAlY_loss_by_age = cases_by_age * param_values.qWeightCase
        return total_dQAlY_loss_by_age



class SummaryOutcomes:

    def __init__(self):

        # Lists for the outcomes of interest to collect
        self.overallQALYlosses = []
        self.overallQALYlossesByState = []
        self.overallQALYlossesByCounty = []
        self.overallQALYlossesCases = []
        self.overallQALYlossesInfections = []
        self.overallQALYlossesDeaths = []
        self.overallQALYlossesHospNonICU = []
        self.overallQALYlossesHospICU = []
        self.overallQALYlossesICU = []
        self.overallQALYlossesICUTotal = []
        self.overallQALYlossesTotalHosp = []
        self.overallQALYlossesLongCOVID_1 = []
        self.overallQALYlossesLongCOVID_2 = []
        self.overallQALYlossesLongCOVID_vax_LB =[]
        self.overallQALYlossesLongCOVID_vax_UB =[]


        self.overallQALYlossesDeaths_SA_1a =[]
        self.overallQALYlossesDeaths_SA_1b =[]
        self.overallQALYlossesDeaths_SA_1c = []
        self.overallQALYlossesDeaths_SA_2a = []
        self.overallQALYlossesDeaths_SA_2b = []
        self.overallQALYlossesDeaths_SA_2c = []
        self.overallQALYlossesDeaths_SA_3a = []
        self.overallQALYlossesDeaths_SA_3b = []
        self.overallQALYlossesDeaths_SA_3c = []

        self.overallQALYlossesTotal_1 = []
        self.overallQALYlossesTotal_2 = []
        self.overallQALYlossesTotal_SA_1a = []
        self.overallQALYlossesTotal_SA_1b = []
        self.overallQALYlossesTotal_SA_1c = []
        self.overallQALYlossesTotal_SA_2a = []
        self.overallQALYlossesTotal_SA_2b = []
        self.overallQALYlossesTotal_SA_2c = []
        self.overallQALYlossesTotal_SA_3a = []
        self.overallQALYlossesTotal_SA_3b = []
        self.overallQALYlossesTotal_SA_3c = []
        self.overallQALYlossesTotal_vax_LB =[]
        self.overallQALYlossesTotal_vax_UB = []


        self.weeklyQALYlosses = []
        self.weeklyQALYlossesByState = []
        self.weeklyQALYlossesCases = []
        self.weeklyQALYlossesInfections = []
        self.weeklyQALYlossesHospNonICU = []
        self.weeklyQALYlossesHospICU = []
        self.weeklyQALYlossesDeaths = []
        self.weeklyQALYlossesICU= []
        self.weeklyQALYlossesICUTotal = []
        self.weeklyQALYlossesTotalHosp =[]
        self.weeklyQALYlossesLongCOVID_1 = []
        self.weeklyQALYlossesLongCOVID_2 = []
        self.weeklyQALYlossesLongCOVID_vax_LB = []
        self.weeklyQALYlossesLongCOVID_vax_UB = []


        self.weeklyQALYlossesDeaths_SA_1a = []
        self.weeklyQALYlossesDeaths_SA_1b = []
        self.weeklyQALYlossesDeaths_SA_1c = []
        self.weeklyQALYlossesDeaths_SA_2a = []
        self.weeklyQALYlossesDeaths_SA_2b = []
        self.weeklyQALYlossesDeaths_SA_2c = []
        self.weeklyQALYlossesDeaths_SA_3a = []
        self.weeklyQALYlossesDeaths_SA_3b = []
        self.weeklyQALYlossesDeaths_SA_3c = []

        self.weeklyQALYlossesTotal_1 = []
        self.weeklyQALYlossesTotal_2 = []
        self.weeklyQALYlossesTotal_SA_1a = []
        self.weeklyQALYlossesTotal_SA_1b = []
        self.weeklyQALYlossesTotal_SA_1c = []
        self.weeklyQALYlossesTotal_SA_2a = []
        self.weeklyQALYlossesTotal_SA_2b = []
        self.weeklyQALYlossesTotal_SA_2c = []
        self.weeklyQALYlossesTotal_SA_3a = []
        self.weeklyQALYlossesTotal_SA_3b = []
        self.weeklyQALYlossesTotal_SA_3c = []

        self.weeklyQALYlossesTotal_vax_LB = []
        self.weeklyQALYlossesTotal_vax_UB = []

        self.overallQALYlossesCasesByState = []
        self.overallQALYlossesInfectionsByState = []
        self.overallQALYlossesHospNonICUByState = []
        self.overallQALYlossesHospICUByState = []
        self.overallQALYlossesDeathsByState = []
        self.overallQALYlossesICUByState =[]
        self.overallQALYlossesICUTotalByState = []
        self.overallQALYlossesTotalHospByState = []
        self.overallQALYlossesLongCOVID_1_ByState = []
        self.overallQALYlossesLongCOVID_2_ByState = []
        self.overallQALYlossesTotal1ByState = []
        self.overallQALYlossesTotal2ByState = []



        self.overallQALYlossessByStateandOutcome =[]

        self.statOverallQALYLoss = None
        self.statOverallQALYLossCases = None
        self.statOverallQALYLossInfections = None
        self.statOverallQALYLossHospNonICU = None
        self.statOverallQALYLossHospICU = None
        self.statOverallQALYLossDeaths = None
        self.statOverallQALYLossICU = None
        self.statOverallQALYLossICUTotal = None
        self.statOverallQALYLossTotalHosp = None
        self.statOverallQALYLossLongCOVID_1 = None
        self.statOverallQALYLossLongCOVID_2 = None
        self.statOverallQALYLossLongCOVID_vax_LB = None
        self.statOverallQALYLossLongCOVID_vax_UB = None


        self.statOverallQALYLossDeaths_SA_1a= None
        self.statOverallQALYLossDeaths_SA_1b = None
        self.statOverallQALYLossDeaths_SA_1c= None
        self.statOverallQALYLossDeaths_SA_2a = None
        self.statOverallQALYLossDeaths_SA_2b = None
        self.statOverallQALYLossDeaths_SA_2c = None
        self.statOverallQALYLossDeaths_SA_3a = None
        self.statOverallQALYLossDeaths_SA_3b = None
        self.statOverallQALYLossDeaths_SA_3c = None

        self.statOverallQALYLossTotal_1 = None
        self.statOverallQALYLossTotal_2 = None
        self.statOverallQALYLossTotal_SA_1a = None
        self.statOverallQALYLossTotal_SA_1b = None
        self.statOverallQALYLossTotal_SA_1c = None
        self.statOverallQALYLossTotal_SA_2a = None
        self.statOverallQALYLossTotal_SA_2b = None
        self.statOverallQALYLossTotal_SA_2c = None
        self.statOverallQALYLossTotal_SA_3a = None
        self.statOverallQALYLossTotal_SA_3b = None
        self.statOverallQALYLossTotal_SA_3c = None
        self.statOverallQALYLossTotal_vax_LB = None
        self.statOverallQALYLossTotal_vax_UB = None

        self.deathQALYLossByAge = []
        self.casesQALYLossByAge = []
        self.age_group = []



    def extract_outcomes(self, simulated_model,param_gen,param_values):

        self.overallQALYlosses.append(simulated_model.get_overall_qaly_loss())
        self.overallQALYlossesByState.append(simulated_model.get_overall_qaly_loss_by_state())
        self.overallQALYlossesByCounty.append(simulated_model.get_overall_qaly_loss_by_county())

        self.weeklyQALYlosses.append(simulated_model.get_weekly_qaly_loss())
        self.weeklyQALYlossesByState.append(simulated_model.get_weekly_qaly_loss_by_state())

        self.weeklyQALYlossesCases.append(simulated_model.pandemicOutcomes.cases.weeklyQALYLoss)
        self.weeklyQALYlossesHospNonICU.append(simulated_model.pandemicOutcomes.hosp_non_icu.weeklyQALYLoss)
        self.weeklyQALYlossesHospICU.append(simulated_model.pandemicOutcomes.hosp_icu.weeklyQALYLoss)
        self.weeklyQALYlossesDeaths.append(simulated_model.pandemicOutcomes.deaths.weeklyQALYLoss)
        self.weeklyQALYlossesInfections.append(simulated_model.pandemicOutcomes.infections.weeklyQALYLoss)
        self.weeklyQALYlossesICU.append(simulated_model.pandemicOutcomes.icu.weeklyQALYLoss)
        self.weeklyQALYlossesICUTotal.append(simulated_model.pandemicOutcomes.icu_total.weeklyQALYLoss)
        self.weeklyQALYlossesTotalHosp.append(simulated_model.pandemicOutcomes.total_hosp.weeklyQALYLoss)
        self.weeklyQALYlossesLongCOVID_1.append(simulated_model.pandemicOutcomes.longCOVID_1.weeklyQALYLoss)
        self.weeklyQALYlossesLongCOVID_2.append(simulated_model.pandemicOutcomes.longCOVID_2.weeklyQALYLoss)
        self.weeklyQALYlossesLongCOVID_vax_LB.append(simulated_model.pandemicOutcomes.longCOVID_vax_LB.weeklyQALYLoss)
        self.weeklyQALYlossesLongCOVID_vax_UB.append(simulated_model.pandemicOutcomes.longCOVID_vax_UB.weeklyQALYLoss)

        self.weeklyQALYlossesDeaths_SA_1a.append(simulated_model.pandemicOutcomes.deaths_sa_1_a.weeklyQALYLoss)
        self.weeklyQALYlossesDeaths_SA_1b.append(simulated_model.pandemicOutcomes.deaths_sa_1_b.weeklyQALYLoss)
        self.weeklyQALYlossesDeaths_SA_1c.append(simulated_model.pandemicOutcomes.deaths_sa_1_c.weeklyQALYLoss)
        self.weeklyQALYlossesDeaths_SA_2a.append(simulated_model.pandemicOutcomes.deaths_sa_2_a.weeklyQALYLoss)
        self.weeklyQALYlossesDeaths_SA_2b.append(simulated_model.pandemicOutcomes.deaths_sa_2_b.weeklyQALYLoss)
        self.weeklyQALYlossesDeaths_SA_2c.append(simulated_model.pandemicOutcomes.deaths_sa_2_c.weeklyQALYLoss)
        self.weeklyQALYlossesDeaths_SA_3a.append(simulated_model.pandemicOutcomes.deaths_sa_3_a.weeklyQALYLoss)
        self.weeklyQALYlossesDeaths_SA_3b.append(simulated_model.pandemicOutcomes.deaths_sa_3_b.weeklyQALYLoss)
        self.weeklyQALYlossesDeaths_SA_3c.append(simulated_model.pandemicOutcomes.deaths_sa_3_c.weeklyQALYLoss)

        self.weeklyQALYlossesTotal_1.append(simulated_model.pandemicOutcomes.total_1.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_2.append(simulated_model.pandemicOutcomes.total_2.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_SA_1a.append(simulated_model.pandemicOutcomes.total_sa_1_a.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_SA_1b.append(simulated_model.pandemicOutcomes.total_sa_1_b.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_SA_1c.append(simulated_model.pandemicOutcomes.total_sa_1_c.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_SA_2a.append(simulated_model.pandemicOutcomes.total_sa_2_a.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_SA_2b.append(simulated_model.pandemicOutcomes.total_sa_2_b.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_SA_2c.append(simulated_model.pandemicOutcomes.total_sa_2_c.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_SA_3a.append(simulated_model.pandemicOutcomes.total_sa_3_a.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_SA_3b.append(simulated_model.pandemicOutcomes.total_sa_3_b.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_SA_3c.append(simulated_model.pandemicOutcomes.total_sa_3_c.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_vax_LB.append(simulated_model.pandemicOutcomes.total_vax_lb.weeklyQALYLoss)
        self.weeklyQALYlossesTotal_vax_UB.append(simulated_model.pandemicOutcomes.total_vax_ub.weeklyQALYLoss)

        self.overallQALYlossesCases.append(simulated_model.pandemicOutcomes.cases.totalQALYLoss)
        self.overallQALYlossesInfections.append(simulated_model.pandemicOutcomes.infections.totalQALYLoss)
        self.overallQALYlossesHospNonICU.append(simulated_model.pandemicOutcomes.hosp_non_icu.totalQALYLoss)
        self.overallQALYlossesHospICU.append(simulated_model.pandemicOutcomes.hosp_icu.totalQALYLoss)
        self.overallQALYlossesDeaths.append(simulated_model.pandemicOutcomes.deaths.totalQALYLoss)
        self.overallQALYlossesICU.append(simulated_model.pandemicOutcomes.icu.totalQALYLoss)
        self.overallQALYlossesICUTotal.append(simulated_model.pandemicOutcomes.icu_total.totalQALYLoss)
        self.overallQALYlossesTotalHosp.append(simulated_model.pandemicOutcomes.total_hosp.totalQALYLoss)
        self.overallQALYlossesLongCOVID_1.append(simulated_model.pandemicOutcomes.longCOVID_1.totalQALYLoss)
        self.overallQALYlossesLongCOVID_2.append(simulated_model.pandemicOutcomes.longCOVID_2.totalQALYLoss)
        self.overallQALYlossesLongCOVID_vax_LB.append(simulated_model.pandemicOutcomes.longCOVID_vax_LB.totalQALYLoss)
        self.overallQALYlossesLongCOVID_vax_UB.append(simulated_model.pandemicOutcomes.longCOVID_vax_UB.totalQALYLoss)

        self.overallQALYlossesDeaths_SA_1a.append(simulated_model.pandemicOutcomes.deaths_sa_1_a.totalQALYLoss)
        self.overallQALYlossesDeaths_SA_1b.append(simulated_model.pandemicOutcomes.deaths_sa_1_b.totalQALYLoss)
        self.overallQALYlossesDeaths_SA_1c.append(simulated_model.pandemicOutcomes.deaths_sa_1_c.totalQALYLoss)
        self.overallQALYlossesDeaths_SA_2a.append(simulated_model.pandemicOutcomes.deaths_sa_2_a.totalQALYLoss)
        self.overallQALYlossesDeaths_SA_2b.append(simulated_model.pandemicOutcomes.deaths_sa_2_b.totalQALYLoss)
        self.overallQALYlossesDeaths_SA_2c.append(simulated_model.pandemicOutcomes.deaths_sa_2_c.totalQALYLoss)
        self.overallQALYlossesDeaths_SA_3a.append(simulated_model.pandemicOutcomes.deaths_sa_3_a.totalQALYLoss)
        self.overallQALYlossesDeaths_SA_3b.append(simulated_model.pandemicOutcomes.deaths_sa_3_b.totalQALYLoss)
        self.overallQALYlossesDeaths_SA_3c.append(simulated_model.pandemicOutcomes.deaths_sa_3_c.totalQALYLoss)

        self.overallQALYlossesTotal_1.append(simulated_model.pandemicOutcomes.total_1.totalQALYLoss)
        self.overallQALYlossesTotal_2.append(simulated_model.pandemicOutcomes.total_2.totalQALYLoss)
        self.overallQALYlossesTotal_SA_1a.append(simulated_model.pandemicOutcomes.total_sa_1_a.totalQALYLoss)
        self.overallQALYlossesTotal_SA_1b.append(simulated_model.pandemicOutcomes.total_sa_1_b.totalQALYLoss)
        self.overallQALYlossesTotal_SA_1c.append(simulated_model.pandemicOutcomes.total_sa_1_c.totalQALYLoss)
        self.overallQALYlossesTotal_SA_2a.append(simulated_model.pandemicOutcomes.total_sa_2_a.totalQALYLoss)
        self.overallQALYlossesTotal_SA_2b.append(simulated_model.pandemicOutcomes.total_sa_2_b.totalQALYLoss)
        self.overallQALYlossesTotal_SA_2c.append(simulated_model.pandemicOutcomes.total_sa_2_c.totalQALYLoss)
        self.overallQALYlossesTotal_SA_3a.append(simulated_model.pandemicOutcomes.total_sa_3_a.totalQALYLoss)
        self.overallQALYlossesTotal_SA_3b.append(simulated_model.pandemicOutcomes.total_sa_3_b.totalQALYLoss)
        self.overallQALYlossesTotal_SA_3c.append(simulated_model.pandemicOutcomes.total_sa_3_c.totalQALYLoss)
        self.overallQALYlossesTotal_vax_LB.append(simulated_model.pandemicOutcomes.total_vax_lb.totalQALYLoss)
        self.overallQALYlossesTotal_vax_UB.append(simulated_model.pandemicOutcomes.total_vax_ub.totalQALYLoss)

        self.overallQALYlossesCasesByState.append(simulated_model.get_overall_qaly_loss_by_state_cases())
        self.overallQALYlossesInfectionsByState.append(simulated_model.get_overall_qaly_loss_by_state_infections())
        self.overallQALYlossesHospNonICUByState.append(simulated_model.get_overall_qaly_loss_by_state_hosp_non_icu())
        self.overallQALYlossesHospICUByState.append(simulated_model.get_overall_qaly_loss_by_state_hosp_icu())
        self.overallQALYlossesDeathsByState.append(simulated_model.get_overall_qaly_loss_by_state_deaths())
        self.overallQALYlossesICUByState.append(simulated_model.get_overall_qaly_loss_by_state_icu())
        self.overallQALYlossesICUTotalByState.append(simulated_model.get_overall_qaly_loss_by_state_icu_total())
        self.overallQALYlossesTotalHospByState.append(simulated_model.get_overall_qaly_loss_by_state_total_hosp())
        self.overallQALYlossesLongCOVID_1_ByState.append(simulated_model.get_overall_qaly_loss_by_state_long_covid_1())
        self.overallQALYlossesLongCOVID_2_ByState.append(simulated_model.get_overall_qaly_loss_by_state_long_covid_2())
        self.overallQALYlossesTotal1ByState.append(simulated_model.get_overall_qaly_loss_by_state_total_1())
        self.overallQALYlossesTotal2ByState.append(simulated_model.get_overall_qaly_loss_by_state_total_2())


        self.deathQALYLossByAge.append(simulated_model.get_death_QALY_loss_by_age(param_gen))

        self.casesQALYLossByAge.append(simulated_model.get_cases_QALY_loss_by_age(param_gen, param_values))


    def summarize(self):

        self.statOverallQALYLoss = SummaryStat(data=self.overallQALYlosses)
        self.statOverallQALYLossCases = SummaryStat(data=self.overallQALYlossesCases)
        self.statOverallQALYLossHospNonICU = SummaryStat(data=self.overallQALYlossesHospNonICU)
        self.statOverallQALYLossHospICU = SummaryStat(data=self.overallQALYlossesHospICU)
        self.statOverallQALYLossInfections=SummaryStat(data=self.overallQALYlossesInfections)
        self.statOverallQALYLossDeaths = SummaryStat(data=self.overallQALYlossesDeaths)
        self.statOverallQALYLossICU= SummaryStat(data=self.overallQALYlossesICU)
        self.statOverallQALYLossTotalHosp = SummaryStat(data=self.overallQALYlossesTotalHosp)
        self.statOverallQALYLossLongCOVID_1 = SummaryStat(data=self.overallQALYlossesLongCOVID_1)
        self.statOverallQALYLossLongCOVID_2 = SummaryStat(data=self.overallQALYlossesLongCOVID_2)
        self.statOverallQALYLossLongCOVID_vax_LB=SummaryStat(data=self.overallQALYlossesLongCOVID_vax_LB)
        self.statOverallQALYLossLongCOVID_vax_UB = SummaryStat(data=self.overallQALYlossesLongCOVID_vax_UB)

        self.statOverallQALYLossDeaths_SA_1a = SummaryStat(data=self.overallQALYlossesDeaths_SA_1a)
        self.statOverallQALYLossDeaths_SA_1b = SummaryStat(data=self.overallQALYlossesDeaths_SA_1b)
        self.statOverallQALYLossDeaths_SA_1c = SummaryStat(data=self.overallQALYlossesDeaths_SA_1c)
        self.statOverallQALYLossDeaths_SA_2a = SummaryStat(data=self.overallQALYlossesDeaths_SA_2a)
        self.statOverallQALYLossDeaths_SA_2b = SummaryStat(data=self.overallQALYlossesDeaths_SA_2b)
        self.statOverallQALYLossDeaths_SA_2c = SummaryStat(data=self.overallQALYlossesDeaths_SA_2c)
        self.statOverallQALYLossDeaths_SA_3a = SummaryStat(data=self.overallQALYlossesDeaths_SA_3a)
        self.statOverallQALYLossDeaths_SA_3b = SummaryStat(data=self.overallQALYlossesDeaths_SA_3b)
        self.statOverallQALYLossDeaths_SA_3c = SummaryStat(data=self.overallQALYlossesDeaths_SA_3c)

        self.statOverallQALYLossTotal_1 = SummaryStat(data=self.overallQALYlossesTotal_1)
        self.statOverallQALYLossTotal_2 = SummaryStat(data=self.overallQALYlossesTotal_2)
        self.statOverallQALYLossTotal_SA_1a = SummaryStat(data=self.overallQALYlossesTotal_SA_1a)
        self.statOverallQALYLossTotal_SA_1b = SummaryStat(data=self.overallQALYlossesTotal_SA_1b)
        self.statOverallQALYLossTotal_SA_1c = SummaryStat(data=self.overallQALYlossesTotal_SA_1c)
        self.statOverallQALYLossTotal_SA_2a = SummaryStat(data=self.overallQALYlossesTotal_SA_2a)
        self.statOverallQALYLossTotal_SA_2b = SummaryStat(data=self.overallQALYlossesTotal_SA_2b)
        self.statOverallQALYLossTotal_SA_2c = SummaryStat(data=self.overallQALYlossesTotal_SA_2c)
        self.statOverallQALYLossTotal_SA_3a = SummaryStat(data=self.overallQALYlossesTotal_SA_3a)
        self.statOverallQALYLossTotal_SA_3b = SummaryStat(data=self.overallQALYlossesTotal_SA_3b)
        self.statOverallQALYLossTotal_SA_3c = SummaryStat(data=self.overallQALYlossesTotal_SA_3c)
        self.statOverallQALYLossTotal_vax_LB=SummaryStat(data=self.overallQALYlossesTotal_vax_LB)
        self.statOverallQALYLossTotal_vax_UB = SummaryStat(data=self.overallQALYlossesTotal_vax_UB)


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

                self.statOverallQALYLossInfections.get_mean(),
                self.statOverallQALYLossInfections.get_t_CI(alpha=0.05),
                self.statOverallQALYLossInfections.get_PI(alpha=0.05),


                self.statOverallQALYLossHospNonICU.get_mean(),
                self.statOverallQALYLossHospNonICU.get_t_CI(alpha=0.05),
                self.statOverallQALYLossHospNonICU.get_PI(alpha=0.05),

                self.statOverallQALYLossHospICU.get_mean(),
                self.statOverallQALYLossHospICU.get_t_CI(alpha=0.05),
                self.statOverallQALYLossHospICU.get_PI(alpha=0.05),

                self.statOverallQALYLossDeaths.get_mean(),
                self.statOverallQALYLossDeaths.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths.get_PI(alpha=0.05),

                self.statOverallQALYLossICU.get_mean(),
                self.statOverallQALYLossICU.get_t_CI(alpha=0.05),
                self.statOverallQALYLossICU.get_PI(alpha=0.05),

                self.statOverallQALYLossTotalHosp.get_mean(),
                self.statOverallQALYLossTotalHosp.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotalHosp.get_PI(alpha=0.05),

                self.statOverallQALYLossLongCOVID_1.get_mean(),
                self.statOverallQALYLossLongCOVID_1.get_t_CI(alpha=0.05),
                self.statOverallQALYLossLongCOVID_1.get_PI(alpha=0.05),

                self.statOverallQALYLossLongCOVID_2.get_mean(),
                self.statOverallQALYLossLongCOVID_2.get_t_CI(alpha=0.05),
                self.statOverallQALYLossLongCOVID_2.get_PI(alpha=0.05),

                self.statOverallQALYLossLongCOVID_vax_LB.get_mean(),
                self.statOverallQALYLossLongCOVID_vax_LB.get_t_CI(alpha=0.05),
                self.statOverallQALYLossLongCOVID_vax_LB.get_PI(alpha=0.05),

                self.statOverallQALYLossLongCOVID_vax_UB.get_mean(),
                self.statOverallQALYLossLongCOVID_vax_UB.get_t_CI(alpha=0.05),
                self.statOverallQALYLossLongCOVID_vax_UB.get_PI(alpha=0.05),

                self.statOverallQALYLossDeaths_SA_1a.get_mean(),
                self.statOverallQALYLossDeaths_SA_1a.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_1a.get_PI(alpha=0.05),

                self.statOverallQALYLossDeaths_SA_1b.get_mean(),
                self.statOverallQALYLossDeaths_SA_1b.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_1b.get_PI(alpha=0.05),

                self.statOverallQALYLossDeaths_SA_1c.get_mean(),
                self.statOverallQALYLossDeaths_SA_1c.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_1c.get_PI(alpha=0.05),

                self.statOverallQALYLossDeaths_SA_2a.get_mean(),
                self.statOverallQALYLossDeaths_SA_2a.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_2a.get_PI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_2b.get_mean(),
                self.statOverallQALYLossDeaths_SA_2b.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_2b.get_PI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_2c.get_mean(),
                self.statOverallQALYLossDeaths_SA_2c.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_2c.get_PI(alpha=0.05),

                self.statOverallQALYLossDeaths_SA_3a.get_mean(),
                self.statOverallQALYLossDeaths_SA_3a.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_3a.get_PI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_3b.get_mean(),
                self.statOverallQALYLossDeaths_SA_3b.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_3b.get_PI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_3c.get_mean(),
                self.statOverallQALYLossDeaths_SA_3c.get_t_CI(alpha=0.05),
                self.statOverallQALYLossDeaths_SA_3c.get_PI(alpha=0.05),

                self.statOverallQALYLossTotal_1.get_mean(),
                self.statOverallQALYLossTotal_1.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_1.get_PI(alpha=0.05),
                self.statOverallQALYLossTotal_2.get_mean(),
                self.statOverallQALYLossTotal_2.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_2.get_PI(alpha=0.05),

                self.statOverallQALYLossTotal_SA_1a.get_mean(),
                self.statOverallQALYLossTotal_SA_1a.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_1a.get_PI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_1b.get_mean(),
                self.statOverallQALYLossTotal_SA_1b.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_1b.get_PI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_1c.get_mean(),
                self.statOverallQALYLossTotal_SA_1c.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_1c.get_PI(alpha=0.05),

                self.statOverallQALYLossTotal_SA_2a.get_mean(),
                self.statOverallQALYLossTotal_SA_2a.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_2a.get_PI(alpha=0.05),

                self.statOverallQALYLossTotal_SA_2b.get_mean(),
                self.statOverallQALYLossTotal_SA_2b.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_2b.get_PI(alpha=0.05),

                self.statOverallQALYLossTotal_SA_2c.get_mean(),
                self.statOverallQALYLossTotal_SA_2c.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_2c.get_PI(alpha=0.05),

                self.statOverallQALYLossTotal_SA_3a.get_mean(),
                self.statOverallQALYLossTotal_SA_3a.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_3a.get_PI(alpha=0.05),

                self.statOverallQALYLossTotal_SA_3b.get_mean(),
                self.statOverallQALYLossTotal_SA_3b.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_3b.get_PI(alpha=0.05),

                self.statOverallQALYLossTotal_SA_3c.get_mean(),
                self.statOverallQALYLossTotal_SA_3c.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_SA_3c.get_PI(alpha=0.05),

                self.statOverallQALYLossTotal_vax_LB.get_mean(),
                self.statOverallQALYLossTotal_vax_LB.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_vax_LB.get_PI(alpha=0.05),

                self.statOverallQALYLossTotal_vax_UB.get_mean(),
                self.statOverallQALYLossTotal_vax_UB.get_t_CI(alpha=0.05),
                self.statOverallQALYLossTotal_vax_UB.get_PI(alpha=0.05),

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


        for i in range(n):

            # Generate a new set of parameters
            params = param_gen.generate(rng=rng)

            # Calculate the QALY loss for this set of parameters
            self.allStates.calculate_qaly_loss(param_values=params)
            self.allStates.get_death_QALY_loss_by_age(param_gen=param_gen)
            self.allStates.get_cases_QALY_loss_by_age(param_gen=param_gen, param_values=params)

            # extract outcomes from the simulated all states
            self.summaryOutcomes.extract_outcomes(simulated_model=self.allStates, param_gen=param_gen, param_values=params)


        self.summaryOutcomes.summarize()


    def calculate_death_qaly_loss_proportion(self):
        """
        Calculate death QALY loss by age group as a proportion of the total death QALY loss.

        Returns:
            A list containing the death QALY loss proportion for each age group.
        """
        # Get mean QALY loss for deaths by age group
        deaths_mean, _ = get_mean_ui_of_a_time_series(self.summaryOutcomes.deathQALYLossByAge, alpha=0.05)

        # Calculate total death QALY loss
        total_death_qaly_loss = np.sum(deaths_mean)

        # Calculate death QALY loss proportion for each age group
        death_qaly_loss_proportion = [death_qaly_loss / total_death_qaly_loss for death_qaly_loss in deaths_mean]

        print(death_qaly_loss_proportion)

        return death_qaly_loss_proportion

    def print_outcomes_proportion_of_qaly_loss(self):
        (mean, ci, ui, mean_cases, ci_c, ui_c,  mean_infections, ci_inf, ui_inf,mean_hosps_non_icu, ci_hosps_non_icu, ui_hosps_non_icu,
         mean_hosps_icu, ci_hosps_icu, ui_hosps_icu, mean_deaths, ci_d, ui_d,
         mean_icu, ci_icu, ui_icu, mean_total_hosps, ci_total_hosps, ui_total_hosps,
         mean_lc_1, ci_lc_1, ui_lc_1,
         mean_lc_2, ci_lc_2, ui_lc_2,
         mean_lc_vax_lb, ci_lc_vax_lb, ui_lc_vax_lb,
         mean_lc_vax_ub, ci_lc_vax_ub, ui_lc_vax_ub,
         mean_deaths_sa_1a, ci_deaths_sa_1a, ui_deaths_sa_1a,
         mean_deaths_sa_1b, ci_deaths_sa_1b, ui_deaths_sa_1b,
         mean_deaths_sa_1c, ci_deaths_sa_1c, ui_deaths_sa_1c,
         mean_deaths_sa_2a, ci_deaths_sa_2a, ui_deaths_sa_2a,
         mean_deaths_sa_2b, ci_deaths_sa_2b, ui_deaths_sa_2b,
         mean_deaths_sa_2c, ci_deaths_sa_2c, ui_deaths_sa_2c,
         mean_deaths_sa_3a, ci_deaths_sa_3a, ui_deaths_sa_3a,
         mean_deaths_sa_3b, ci_deaths_sa_3b, ui_deaths_sa_3b,
         mean_deaths_sa_3c, ci_deaths_sa_3c, ui_deaths_sa_3c,
         mean_total_1, ci_total_1, ui_total_1,
         mean_total_2, ci_total_2, ui_total_2,
         mean_total_sa_1a, ci_total_sa_1a, ui_total_sa_1a,
         mean_total_sa_1b, ci_total_sa_1b, ui_total_sa_1b,
         mean_total_sa_1c, ci_total_sa_1c, ui_total_sa_1c,
         mean_total_sa_2a, ci_total_sa_2a, ui_total_sa_2a,
         mean_total_sa_2b, ci_total_sa_2b, ui_total_sa_2b,
         mean_total_sa_2c, ci_total_sa_2c, ui_total_sa_2c,
         mean_total_sa_3a, ci_total_sa_3a, ui_total_sa_3a,
         mean_total_sa_3b, ci_total_sa_3b, ui_total_sa_3b,
         mean_total_sa_3c, ci_total_sa_3c, ui_total_sa_3c,
         mean_total_vax_lb, ci_total_vax_lb, ui_total_vax_lb,
         mean_total_vax_ub, ci_total_vax_ub, ui_total_vax_ub,
         ) = self.summaryOutcomes.get_mean_ci_ui_overall_qaly_loss()

        print('Proportions (LC1):')
        print('infections:', mean_infections/(mean_infections+mean_deaths+mean_total_hosps+mean_lc_1)*100)
        print('total hosps:', mean_total_hosps / (mean_infections + mean_deaths + mean_total_hosps + mean_lc_1) * 100)
        print('ICU:', mean_icu / (mean_infections + mean_deaths + mean_total_hosps + mean_lc_1) * 100)
        print('ICU pts experience:', (mean_icu+mean_hosps_icu) / (mean_infections + mean_deaths + mean_total_hosps + mean_lc_1) * 100)
        print('deaths:', mean_deaths / (mean_infections + mean_deaths + mean_total_hosps + mean_lc_1) * 100)
        print('LC1:', mean_lc_1 / (mean_infections + mean_deaths + mean_total_hosps + mean_lc_1) * 100)

    def print_overall_outcomes_and_qaly_loss(self):
        """
        :return: Prints the mean, confidence interval, and the uncertainty interval for the overall QALY loss .
        """

        mean_cases = self.allStates.pandemicOutcomes.cases.totalObs
        mean_infections =self.allStates.pandemicOutcomes.infections.totalObs
        mean_hosps = self.allStates.pandemicOutcomes.hosp_non_icu.totalObs
        mean_icu = self.allStates.pandemicOutcomes.icu.totalObs
        mean_deaths = self.allStates.pandemicOutcomes.deaths.totalObs


        print('Overall Outcomes:')
        print('  Number of Infections: {:,.0f}'.format(mean_infections))
        print('  Number of Hospital Admissions: {:,.0f}'.format(mean_hosps))
        print('  Mean Deaths: {:,.0f}'.format(mean_deaths))
        print('  Mean ICU: {:,.0f}'.format(mean_icu))


        (mean, ci, ui, mean_cases, ci_c, ui_c, mean_infections, ci_inf, ui_inf,mean_hosps_non_icu, ci_hosps_non_icu, ui_hosps_non_icu,
         mean_hosps_icu, ci_hosps_icu, ui_hosps_icu, mean_deaths, ci_d, ui_d,
         mean_icu, ci_icu, ui_icu, mean_total_hosps, ci_total_hosps, ui_total_hosps,
         mean_lc_1, ci_lc_1, ui_lc_1,
         mean_lc_2, ci_lc_2, ui_lc_2,
         mean_lc_vax_lb, ci_lc_vax_lb, ui_lc_vax_lb,
         mean_lc_vax_ub, ci_lc_vax_ub, ui_lc_vax_ub,
         mean_deaths_sa_1a, ci_deaths_sa_1a, ui_deaths_sa_1a,
         mean_deaths_sa_1b, ci_deaths_sa_1b, ui_deaths_sa_1b,
         mean_deaths_sa_1c, ci_deaths_sa_1c, ui_deaths_sa_1c,
         mean_deaths_sa_2a, ci_deaths_sa_2a, ui_deaths_sa_2a,
         mean_deaths_sa_2b, ci_deaths_sa_2b, ui_deaths_sa_2b,
         mean_deaths_sa_2c, ci_deaths_sa_2c, ui_deaths_sa_2c,
         mean_deaths_sa_3a, ci_deaths_sa_3a, ui_deaths_sa_3a,
         mean_deaths_sa_3b, ci_deaths_sa_3b, ui_deaths_sa_3b,
         mean_deaths_sa_3c, ci_deaths_sa_3c, ui_deaths_sa_3c,
         mean_total_1, ci_total_1, ui_total_1,
         mean_total_2, ci_total_2, ui_total_2,
         mean_total_sa_1a, ci_total_sa_1a, ui_total_sa_1a,
         mean_total_sa_1b, ci_total_sa_1b, ui_total_sa_1b,
         mean_total_sa_1c, ci_total_sa_1c, ui_total_sa_1c,
         mean_total_sa_2a, ci_total_sa_2a, ui_total_sa_2a,
         mean_total_sa_2b, ci_total_sa_2b, ui_total_sa_2b,
         mean_total_sa_2c, ci_total_sa_2c, ui_total_sa_2c,
         mean_total_sa_3a, ci_total_sa_3a, ui_total_sa_3a,
         mean_total_sa_3b, ci_total_sa_3b, ui_total_sa_3b,
         mean_total_sa_3c, ci_total_sa_3c, ui_total_sa_3c,
         mean_total_vax_lb, ci_total_vax_lb, ui_total_vax_lb,
         mean_total_vax_ub, ci_total_vax_ub, ui_total_vax_ub,
         ) = self.summaryOutcomes.get_mean_ci_ui_overall_qaly_loss()

        print('Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean))
        print('  95% Confidence Interval:', format_interval(ci, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui, deci=0, format=','))


        print('Cases Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_cases))
        print('  95% Confidence Interval:', format_interval(ci_c, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_c, deci=0, format=','))

        print('Symptomatic Infections Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_infections))
        print('  95% Confidence Interval:', format_interval(ci_inf, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_inf, deci=0, format=','))

        print(' Hosps (non ICU pts, ward care) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_hosps_non_icu))
        print('  95% Confidence Interval:', format_interval(ci_hosps_non_icu, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_hosps_non_icu, deci=0, format=','))

        print('Hosps (ICU pts, ward care) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_hosps_icu))
        print('  95% Confidence Interval:', format_interval(ci_hosps_icu, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_hosps_icu, deci=0, format=','))

        print('Deaths Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_deaths))
        print('  95% Confidence Interval:', format_interval(ci_d, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_d, deci=0, format=','))

        print('ICU Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_icu))
        print('  95% Confidence Interval:', format_interval(ci_icu, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_icu, deci=0, format=','))

        print('Total Hops (ICU + ward care) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_total_hosps))
        print('  95% Confidence Interval:', format_interval(ci_total_hosps, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_total_hosps, deci=0, format=','))

        print('Long COVID (1) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_lc_1))
        print('  95% Confidence Interval:', format_interval(ci_lc_1, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_lc_1, deci=0, format=','))


        print('Long COVID (2) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_lc_2))
        print('  95% Confidence Interval:', format_interval(ci_lc_2, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_lc_2, deci=0, format=','))

        print('Long COVID (vax LB) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_lc_vax_lb))
        print('  95% Confidence Interval:', format_interval(ci_lc_vax_lb, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_lc_vax_lb, deci=0, format=','))

        print('Long COVID (vax UB) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_lc_vax_ub))
        print('  95% Confidence Interval:', format_interval(ci_lc_vax_ub, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_lc_vax_ub, deci=0, format=','))

        print('Overall QALY loss (LC1):')
        print('  Mean: {:,.0f}'.format(mean_total_1))
        print('  95% Confidence Interval:', format_interval(ci_total_1, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_total_1, deci=0, format=','))

        print('Overall QALY loss (LC 2):')
        print('  Mean: {:,.0f}'.format(mean_total_2))
        print('  95% Confidence Interval:', format_interval(ci_total_2, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_total_2, deci=0, format=','))

        print('Overall QALY loss (Total LB):')
        print('  Mean: {:,.0f}'.format(mean_total_vax_lb))
        print('  95% Confidence Interval:', format_interval(ci_total_vax_lb, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_total_vax_lb, deci=0, format=','))

        print('Overall QALY loss (Total UB):')
        print('  Mean: {:,.0f}'.format(mean_total_vax_ub))
        print('  95% Confidence Interval:', format_interval(ci_total_vax_ub, deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval(ui_total_vax_ub, deci=0, format=','))



    def print_qaly_loss_prorated(self):
        """
        :return: Prints the mean, confidence interval, and the uncertainty interval for the overall QALY loss .
        """

        (mean, ci, ui, mean_cases, ci_c, ui_c, mean_infections, ci_inf, ui_inf, mean_hosps_non_icu,
         ci_hosps_non_icu, ui_hosps_non_icu,
         mean_hosps_icu, ci_hosps_icu, ui_hosps_icu, mean_deaths, ci_d, ui_d,
         mean_icu, ci_icu, ui_icu, mean_total_hosps, ci_total_hosps, ui_total_hosps,
         mean_lc_1, ci_lc_1, ui_lc_1,
         mean_lc_2, ci_lc_2, ui_lc_2,
         mean_lc_vax_lb, ci_lc_vax_lb, ui_lc_vax_lb,
         mean_lc_vax_ub, ci_lc_vax_ub, ui_lc_vax_ub,
         mean_deaths_sa_1a, ci_deaths_sa_1a, ui_deaths_sa_1a,
         mean_deaths_sa_1b, ci_deaths_sa_1b, ui_deaths_sa_1b,
         mean_deaths_sa_1c, ci_deaths_sa_1c, ui_deaths_sa_1c,
         mean_deaths_sa_2a, ci_deaths_sa_2a, ui_deaths_sa_2a,
         mean_deaths_sa_2b, ci_deaths_sa_2b, ui_deaths_sa_2b,
         mean_deaths_sa_2c, ci_deaths_sa_2c, ui_deaths_sa_2c,
         mean_deaths_sa_3a, ci_deaths_sa_3a, ui_deaths_sa_3a,
         mean_deaths_sa_3b, ci_deaths_sa_3b, ui_deaths_sa_3b,
         mean_deaths_sa_3c, ci_deaths_sa_3c, ui_deaths_sa_3c,
         mean_total_1, ci_total_1, ui_total_1,
         mean_total_2, ci_total_2, ui_total_2,
         mean_total_sa_1a, ci_total_sa_1a, ui_total_sa_1a,
         mean_total_sa_1b, ci_total_sa_1b, ui_total_sa_1b,
         mean_total_sa_1c, ci_total_sa_1c, ui_total_sa_1c,
         mean_total_sa_2a, ci_total_sa_2a, ui_total_sa_2a,
         mean_total_sa_2b, ci_total_sa_2b, ui_total_sa_2b,
         mean_total_sa_2c, ci_total_sa_2c, ui_total_sa_2c,
         mean_total_sa_3a, ci_total_sa_3a, ui_total_sa_3a,
         mean_total_sa_3b, ci_total_sa_3b, ui_total_sa_3b,
         mean_total_sa_3c, ci_total_sa_3c, ui_total_sa_3c,
         mean_total_vax_lb, ci_total_vax_lb, ui_total_vax_lb,
         mean_total_vax_ub, ci_total_vax_ub, ui_total_vax_ub,
         ) = self.summaryOutcomes.get_mean_ci_ui_overall_qaly_loss()

        print('Prorated Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean*52/129))
        print('  95% Confidence Interval:', format_interval((ci[0]*52/129,ci[1]*52/129), deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval((ui[0]*52/129,ui[1]*52/129), deci=0, format=','))


        print('Prorated Cases Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_cases*52/129))
        print('  95% Confidence Interval:', format_interval((ci_c[0]*52/129,ci_c[1]*52/129), deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval((ui_c[0]*52/129,ui_c[1]*52/129), deci=0, format=','))

        print('Prorated Symptomatic Infections Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_infections * 52 / 129))
        print('  95% Confidence Interval:',
              format_interval((ci_inf[0] * 52 / 129, ci_inf[1] * 52 / 129), deci=0, format=','))
        print('  95% Uncertainty Interval:',
              format_interval((ui_inf[0] * 52 / 129, ui_inf[1] * 52 / 129), deci=0, format=','))


        print('Prorated Hosps (non ICU pts, ward care) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_hosps_non_icu*52/129))
        print('  95% Confidence Interval:', format_interval((ci_hosps_non_icu[0]*52/129,ci_hosps_non_icu[1]*52/129),deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval((ui_hosps_non_icu[0]*52/129,ui_hosps_non_icu[1]*52/129), deci=0, format=','))

        print('Prorated Hosps (ICU pts, ward care) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_hosps_icu*52/129))
        print('  95% Confidence Interval:', format_interval((ci_hosps_icu[0]*52/129,ci_hosps_icu[1]*52/129), deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval((ui_hosps_icu[0]*52/129,ui_hosps_icu[1]*52/129),deci=0, format=','))

        print('Prorated ICU Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_icu*52/129))
        print('  95% Confidence Interval:', format_interval((ci_icu[0]*52/129,ci_icu[1]*52/129), deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval((ui_icu[0]*52/129,ui_icu[1]*52/129),deci=0, format=','))

        print('Prorated Total Hops (ICU + ward care) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_total_hosps*52/129))
        print('  95% Confidence Interval:', format_interval((ci_total_hosps[0]*52/129,ci_total_hosps[1]*52/129), deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval((ui_total_hosps[0]*52/129,ui_total_hosps[1]*152/129), deci=0, format=','))

        print('Prorated Long COVID (1) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_lc_1*52/129))
        print('  95% Confidence Interval:', format_interval((ci_lc_1[0]*52/129,ci_lc_1[1]*52/129), deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval((ui_lc_1[0]*52/129,ui_lc_1[1]*52/129), deci=0, format=','))


        print('Prorated Long COVID (2) Overall QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_lc_2*52/129))
        print('  95% Confidence Interval:', format_interval((ci_lc_2[0]*52/129,ci_lc_2[1]*52/129), deci=0, format=','))
        print('  95% Uncertainty Interval:', format_interval((ui_lc_2[0]*52/129,ui_lc_2[1]*52/129), deci=0, format=','))

        print('Prorated Deaths QALY loss:')
        print('  Mean: {:,.0f}'.format(mean_deaths * 52 / 129))
        print('  95% Confidence Interval:', format_interval((ci_d[0] * 52 / 129, ci_d[1] * 52 / 129), deci=0, format=','))
        print('  95% Uncertainty Interval:',format_interval((ui_d[0] * 52 / 129, ui_d[1] * 52 / 129), deci=0, format=','))

        print('Prorated Total QALY loss (LC1):')
        print('  Mean: {:,.0f}'.format(mean_total_1 * 52 / 129))
        print('  95% Confidence Interval:',
              format_interval((ci_total_1[0] * 52 / 129, ci_total_1[1] * 52 / 129), deci=0, format=','))
        print('  95% Uncertainty Interval:',
              format_interval((ui_total_1[0] * 52 / 129, ui_total_1[1] * 52 / 129), deci=0, format=','))

        print('Prorated Total QALY loss (LC2):')
        print('  Mean: {:,.0f}'.format(mean_total_2 * 52 / 129))
        print('  95% Confidence Interval:',
              format_interval((ci_total_2[0] * 52 / 129, ci_total_2[1] * 52 / 129), deci=0, format=','))
        print('  95% Uncertainty Interval:',
              format_interval((ui_total_2[0] * 52 / 129, ui_total_2[1] * 52 / 129), deci=0, format=','))

    def print_qaly_loss_per_outcome(self):


        (mean, ci, ui, mean_cases, ci_c, ui_c, mean_infections, ci_inf, ui_inf,mean_hosps_non_icu, ci_hosps_non_icu, ui_hosps_non_icu,
         mean_hosps_icu, ci_hosps_icu, ui_hosps_icu, mean_deaths, ci_d, ui_d,
         mean_icu, ci_icu, ui_icu, mean_total_hosps, ci_total_hosps, ui_total_hosps,
         mean_lc_1, ci_lc_1, ui_lc_1,
         mean_lc_2, ci_lc_2, ui_lc_2,
         mean_deaths_sa_1a, ci_deaths_sa_1a, ui_deaths_sa_1a,
         mean_deaths_sa_1b, ci_deaths_sa_1b, ui_deaths_sa_1b,
         mean_deaths_sa_1c, ci_deaths_sa_1c, ui_deaths_sa_1c,
         mean_deaths_sa_2a, ci_deaths_sa_2a, ui_deaths_sa_2a,
         mean_deaths_sa_2b, ci_deaths_sa_2b, ui_deaths_sa_2b,
         mean_deaths_sa_2c, ci_deaths_sa_2c, ui_deaths_sa_2c,
         mean_deaths_sa_3a, ci_deaths_sa_3a, ui_deaths_sa_3a,
         mean_deaths_sa_3b, ci_deaths_sa_3b, ui_deaths_sa_3b,
         mean_deaths_sa_3c, ci_deaths_sa_3c, ui_deaths_sa_3c,
         mean_total_1, ci_total_1, ui_total_1,
         mean_total_2, ci_total_2, ui_total_2,
         mean_total_sa_1a, ci_total_sa_1a, ui_total_sa_1a,
         mean_total_sa_1b, ci_total_sa_1b, ui_total_sa_1b,
         mean_total_sa_1c, ci_total_sa_1c, ui_total_sa_1c,
         mean_total_sa_2a, ci_total_sa_2a, ui_total_sa_2a,
         mean_total_sa_2b, ci_total_sa_2b, ui_total_sa_2b,
         mean_total_sa_2c, ci_total_sa_2c, ui_total_sa_2c,
         mean_total_sa_3a, ci_total_sa_3a, ui_total_sa_3a,
         mean_total_sa_3b, ci_total_sa_3b, ui_total_sa_3b,
         mean_total_sa_3c, ci_total_sa_3c, ui_total_sa_3c) = self.summaryOutcomes.get_mean_ci_ui_overall_qaly_loss()


        print('Cases :')
        print('  Mean QALY Loss per case: ', (mean_cases/self.allStates.pandemicOutcomes.cases.totalObs))
        print('Symptomatic Infections :')
        print('  Mean QALY Loss per symptomatic infections', (mean_infections / (self.allStates.pandemicOutcomes.infections.totalObs)))
        print('  95% Confidence Interval:', (ci_inf/ self.allStates.pandemicOutcomes.infections.totalObs))
        print('  95% Uncertainty Interval:', format_interval(ui_inf/ self.allStates.pandemicOutcomes.infections.totalObs, deci=0, format=','))

        print(' Hosps:')
        print('  Mean QALY loss per hospital admission:', (mean_total_hosps/self.allStates.pandemicOutcomes.hosps.totalObs))
        print('  95% Confidence Interval:', (ci_total_hosps/self.allStates.pandemicOutcomes.hosps.totalObs))
        print('  95% Uncertainty Interval:', (ui_total_hosps/self.allStates.pandemicOutcomes.hosps.totalObs))

        print('Deaths:')
        print('  Mean QALY loss per Deaths:', (mean_deaths/self.allStates.pandemicOutcomes.deaths.totalObs))
        print('  95% Confidence Interval:', (ci_d/self.allStates.pandemicOutcomes.cases.totalObs))
        print('  95% Uncertainty Interval:', (ui_d/self.allStates.pandemicOutcomes.cases.totalObs))


    def get_mean_ui_weekly_qaly_loss(self, alpha=0.05):
        """
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss.
        """

        mean, ui = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlosses, alpha=alpha)
        return mean, ui

    def Fig_1_plot_weekly_qaly_loss_by_outcome(self):

        """
        :return: Plots National Weekly QALY Loss from Cases, Hospitalizations and Deaths across all states
        """
        # Create a plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,11))
        #fig, ax = plt.subplots(figsize=(10, 6))



        [mean_cases, ui_cases, mean_infections, ui_infections, mean_deaths, ui_deaths, mean_hosps_non_icu, ui_hosps_non_icu,
                mean_hosps_icu, ui_hosps_icu,mean_icu, ui_icu, mean_total_hosps, ui_total_hosps,
                mean_lc_1, ui_lc_1, mean_lc_2, ui_lc_2,
                mean_lc_vax_lb,ui_lc_vax_lb, mean_lc_vax_ub, ui_lc_vax_ub,
                mean_deaths_sa_1a, ui_deaths_sa_1a, mean_deaths_sa_1b, ui_deaths_sa_1b, mean_deaths_sa_1c, ui_deaths_sa_1c,
                mean_deaths_sa_2a, ui_deaths_sa_2a, mean_deaths_sa_2b, ui_deaths_sa_2b, mean_deaths_sa_2c, ui_deaths_sa_2c,
                mean_deaths_sa_3a, ui_deaths_sa_3a, mean_deaths_sa_3b, ui_deaths_sa_3b, mean_deaths_sa_3c, ui_deaths_sa_3c,
                mean_total_1, ui_total_1, mean_total_2, ui_total_2,
                mean_total_sa_1a, ui_total_sa_1a, mean_total_sa_1b, ui_total_sa_1b, mean_total_sa_1c,ui_total_sa_1c,
                mean_total_sa_2a, ui_total_sa_2a, mean_total_sa_2b, ui_total_sa_2b, mean_total_sa_2c,ui_total_sa_2c,
                mean_total_sa_3a, ui_total_sa_3a, mean_total_sa_3b, ui_total_sa_3b, mean_total_sa_3c, ui_total_sa_3c,
                mean_total_vax_lb,ui_total_vax_lb,mean_total_vax_ub,ui_total_vax_ub] = (
            self.get_mean_ui_weekly_qaly_loss_by_outcome(alpha=0.05))

        ax1.plot(self.allStates.dates, mean_cases,
                label='Detected Cases', linewidth=2, color='blue', linestyle='dashed')
        ax1.fill_between(self.allStates.dates, ui_cases[0], ui_cases[1], color='lightblue', alpha=0.25)

        ax1.plot(self.allStates.dates, mean_infections,
                 label='Symptomatic Infections', linewidth=2, color='blue')
        ax1.fill_between(self.allStates.dates, ui_infections[0], ui_infections[1], color='lightblue', alpha=0.25)

        ax1.plot(self.allStates.dates, mean_total_hosps,
                label='Hospital admissions (including ICU)', linewidth=2, color='green')
        ax1.fill_between(self.allStates.dates,ui_total_hosps[0], ui_total_hosps[1], color='grey', alpha=0.25)

        ax1.plot(self.allStates.dates, mean_deaths,
                label='Deaths', linewidth=2, color='red')
        ax1.fill_between(self.allStates.dates, ui_deaths[0], ui_deaths[1], color='red', alpha=0.25)

        ax1.plot(self.allStates.dates, mean_lc_1,
                label='Long COVID (Simplified Approach)', linewidth=2, color='purple')
        ax1.fill_between(self.allStates.dates, ui_lc_1[0], ui_lc_1[1], color='purple', alpha=0.25)

        ax1.plot(self.allStates.dates, mean_total_1,
                 label='Total', linewidth=2, color='black')
        ax1.fill_between(self.allStates.dates, ui_total_1[0], ui_total_1[1], color='grey', alpha=0.25)

        # Subplot 2
        ax2.plot(self.allStates.dates, mean_cases,
                 label='Detected Cases', linewidth=2, color='blue', linestyle='dashed')
        ax2.fill_between(self.allStates.dates, ui_cases[0], ui_cases[1], color='lightblue', alpha=0.25)

        ax2.plot(self.allStates.dates, mean_infections,
                 label='Symptomatic Infections', linewidth=2, color='blue')
        ax2.fill_between(self.allStates.dates, ui_infections[0], ui_infections[1], color='lightblue', alpha=0.25)

        ax2.plot(self.allStates.dates, mean_total_hosps,
                 label='Hospital admissions (including ICU)', linewidth=2, color='green')
        ax2.fill_between(self.allStates.dates, ui_total_hosps[0], ui_total_hosps[1], color='grey', alpha=0.25)

        ax2.plot(self.allStates.dates, mean_deaths,
                 label='Deaths', linewidth=2, color='red')
        ax2.fill_between(self.allStates.dates, ui_deaths[0], ui_deaths[1], color='red', alpha=0.25)

        ax2.plot(self.allStates.dates, mean_lc_2,
                 label='Long COVID (Health State-Dependent Appraoch)', linewidth=2, color='purple')
        ax2.fill_between(self.allStates.dates, ui_lc_2[0], ui_lc_2[1], color='purple', alpha=0.25)

        ax2.plot(self.allStates.dates, mean_total_2,
                 label='Total', linewidth=2, color='black')
        ax2.fill_between(self.allStates.dates, ui_total_2[0], ui_total_2[1], color='grey', alpha=0.25)

        ax1.axvspan("2021-06-30", "2021-10-27", alpha=0.2, color="lightblue")  # delta variant
        ax1.axvspan("2021-10-27", "2022-12-28", alpha=0.2, color="grey")  # omicron variant
        ax1.axvline(x="2021-08-04", color='black', linestyle='--')

        ax1.set_title('National Weekly QALY Loss', fontsize=16)
        ax1.set_xlabel('Date', fontsize=14)
        ax1.set_ylabel('QALY Loss', fontsize=14)
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3)
        ax1.set_ylim(0,400000)
        #ax.legend()

        date_range = self.allStates.dates
        tick_positions = range(0, len(date_range))
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(
            [date_range[i] if i % 4 == 0 else '' for i in tick_positions],  # Label every 4th tick mark
            fontsize=10, rotation=45 )

        # Make the labeled tick marks slightly longer and bold
        for i, tick in enumerate(ax1.xaxis.get_major_ticks()):
            if i % 4 == 0:  # Every 4th tick mark
                tick.label1.set_fontsize(10)
                tick.label1.set_rotation(45)
                tick.label1.set_horizontalalignment('right')
                tick.label1.set_weight('normal')
                tick.tick1line.set_markersize(6)
                tick.tick1line.set_linewidth(2)
                tick.tick2line.set_markersize(6)
                tick.tick2line.set_linewidth(2)

            else:
                tick.label1.set_fontsize(10)
                tick.label1.set_weight('normal')
        ax1.text(0.01, 0.98, "A", transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')


        ax2.axvspan("2021-06-30", "2021-10-27", alpha=0.2, color="lightblue")  # delta variant
        ax2.axvspan("2021-10-27", "2022-12-28", alpha=0.2, color="grey")  # omicron variant
        ax2.axvline(x="2021-08-04", color='black', linestyle='--')

        ax2.set_title('National Weekly QALY Loss', fontsize=16)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.set_ylabel('QALY Loss', fontsize=14)
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3)
        ax2.set_ylim(0, 400000)
        # ax.legend()

        date_range = self.allStates.dates
        tick_positions = range(0, len(date_range))
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(
            [date_range[i] if i % 4 == 0 else '' for i in tick_positions],  # Label every 4th tick mark
            fontsize=10, rotation=45)

        # Make the labeled tick marks slightly longer and bold
        for i, tick in enumerate(ax2.xaxis.get_major_ticks()):
            if i % 4 == 0:  # Every 4th tick mark
                tick.label1.set_fontsize(10)
                tick.label1.set_rotation(45)
                tick.label1.set_horizontalalignment('right')
                tick.label1.set_weight('normal')
                tick.tick1line.set_markersize(6)
                tick.tick1line.set_linewidth(2)
                tick.tick2line.set_markersize(6)
                tick.tick2line.set_linewidth(2)

            else:
                tick.label1.set_fontsize(10)
                tick.label1.set_weight('normal')
        ax2.text(0.01, 0.98, "B", transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')


        output_figure(fig, filename=ROOT_DIR + '/figs/Fig_1_national_qaly_loss_by_outcome.png')

    def Fig_S5_plot_weekly_deaths_sensitivity_analysis(self):
        # Create a plot
        fig, axes = plt.subplots(3, 3, figsize=(21, 18))

        # Get data using your method
        [mean_cases, ui_cases, mean_infections, ui_infections, mean_deaths, ui_deaths, mean_hosps_non_icu, ui_hosps_non_icu,
        mean_hosps_icu, ui_hosps_icu, mean_icu, ui_icu, mean_total_hosps, ui_total_hosps,
        mean_lc_1, ui_lc_1, mean_lc_2, ui_lc_2,
        mean_lc_vax_lb, ui_lc_vax_lb, mean_lc_vax_ub, ui_lc_vax_ub,
        mean_deaths_sa_1a, ui_deaths_sa_1a, mean_deaths_sa_1b, ui_deaths_sa_1b, mean_deaths_sa_1c, ui_deaths_sa_1c,
        mean_deaths_sa_2a, ui_deaths_sa_2a, mean_deaths_sa_2b, ui_deaths_sa_2b, mean_deaths_sa_2c, ui_deaths_sa_2c,
        mean_deaths_sa_3a, ui_deaths_sa_3a, mean_deaths_sa_3b, ui_deaths_sa_3b, mean_deaths_sa_3c, ui_deaths_sa_3c,
        mean_total_1, ui_total_1, mean_total_2, ui_total_2,
        mean_total_sa_1a, ui_total_sa_1a, mean_total_sa_1b, ui_total_sa_1b, mean_total_sa_1c, ui_total_sa_1c,
        mean_total_sa_2a, ui_total_sa_2a, mean_total_sa_2b, ui_total_sa_2b, mean_total_sa_2c, ui_total_sa_2c,
        mean_total_sa_3a, ui_total_sa_3a, mean_total_sa_3b, ui_total_sa_3b, mean_total_sa_3c, ui_total_sa_3c,
        mean_total_vax_lb, ui_total_vax_lb, mean_total_vax_ub, ui_total_vax_ub] = self.get_mean_ui_weekly_qaly_loss_by_outcome(
            alpha=0.05)

        # Unpack the data into variables for each subplot
        variables = [
            ('Deaths 1a', 'red', mean_deaths_sa_1a, ui_deaths_sa_1a),
            ('Deaths 1b', 'red', mean_deaths_sa_1b, ui_deaths_sa_1b),
            ('Deaths 1c', 'red', mean_deaths_sa_1c, ui_deaths_sa_1c),
            ('Deaths 2a', 'red', mean_deaths_sa_2a, ui_deaths_sa_2a),
            ('Deaths 2b', 'red', mean_deaths_sa_2b, ui_deaths_sa_2b),
            ('Deaths 2c', 'red', mean_deaths_sa_2c, ui_deaths_sa_2c),
            ('Deaths 3a', 'red', mean_deaths_sa_3a, ui_deaths_sa_3a),
            ('Deaths 3b', 'red', mean_deaths_sa_3b, ui_deaths_sa_3b),
            ('Deaths 3c', 'red', mean_deaths_sa_3c, ui_deaths_sa_3c)
        ]

        # Total data for each scenario
        total_data = [
            (mean_total_sa_1a, ui_total_sa_1a),
            (mean_total_sa_1b, ui_total_sa_1b),
            (mean_total_sa_1c, ui_total_sa_1c),
            (mean_total_sa_2a, ui_total_sa_2a),
            (mean_total_sa_2b, ui_total_sa_2b),
            (mean_total_sa_2c, ui_total_sa_2c),
            (mean_total_sa_3a, ui_total_sa_3a),
            (mean_total_sa_3b, ui_total_sa_3b),
            (mean_total_sa_3c, ui_total_sa_3c)
        ]

        # Data for Deaths 2b (Base analysis) to be added to all subplots
        deaths_2b_mean = mean_deaths_sa_2b
        deaths_2b_ui = ui_deaths_sa_2b

        # Plotting each subplot
        for i, (label, color, mean, ui) in enumerate(variables):
            row = i // 3
            col = i % 3
            ax = axes[row, col]

            ax.plot(self.allStates.dates, mean, label=label, linewidth=2, color=color)
            ax.fill_between(self.allStates.dates, ui[0], ui[1], color=color, alpha=0.25)

            # Plot total QALY loss in black for each subplot
            mean_total, ui_total = total_data[i]
            ax.plot(self.allStates.dates, mean_total, label='Total QALY Loss', linewidth=2, color='black')
            ax.fill_between(self.allStates.dates, ui_total[0], ui_total[1], color='grey', alpha=0.25)

            # Add Deaths 2b line to all subplots
            ax.plot(self.allStates.dates, deaths_2b_mean, label='Deaths in base analysis', linestyle='--', linewidth=2,
                    color='red')
            ax.fill_between(self.allStates.dates, deaths_2b_ui[0], deaths_2b_ui[1], color='red', alpha=0.15)

            ax.axvspan("2021-06-30", "2021-10-27", alpha=0.2, color="lightblue")
            ax.axvspan("2021-10-27", "2022-12-28", alpha=0.2, color="grey")
            ax.axvline(x="2021-08-04", color='black', linestyle='--')

            # Set title only on the top row
            if row == 0:
                if col == 0:
                    ax.set_title('qCM=15%', fontsize=18)
                elif col == 1:
                    ax.set_title('qCM=20%', fontsize=18)
                elif col == 2:
                    ax.set_title('qCM=25%', fontsize=18)

            # Set ylabel only on the left column
            if col == 0:
                if row == 0:
                    ax.set_ylabel('SMR=1.75', fontsize=18, labelpad=20)
                elif row == 1:
                    ax.set_ylabel('SMR=2', fontsize=18, labelpad=20)
                elif row == 2:
                    ax.set_ylabel('SMR=2.25', fontsize=18, labelpad=20)

            ax.set_xlabel('Date', fontsize=14)

            # Adjust the QALY Loss label position
            if col == 0:
                ax.text(-0.15, 0.5, 'QALY Loss', va='center', rotation='vertical', fontsize=14, transform=ax.transAxes)

            # Set y-axis range to be consistent
            ax.set_ylim(0, 250000)

        # Common settings for all subplots
        date_range = self.allStates.dates
        for ax in axes.flatten():
            ax.set_xticks(np.arange(0, len(date_range), step=8))
            ax.set_xticklabels([date_range[i] if i % 8 == 0 else '' for i in np.arange(0, len(date_range), step=8)],
                               fontsize=10, rotation=45)

            # Adjust tick parameters
            ax.tick_params(axis='x', which='major', labelsize=10, rotation=45)

        # Adding text labels (A, B, C, ...)
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        for ax, label in zip(axes.flatten(), labels):
            ax.text(0.01, 0.98, label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

        # Set document title
        fig.suptitle('National Weekly QALY Loss', fontsize=24, y=1.02)

        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # Adding a simplified legend with horizontal orientation
        legend_lines = [Line2D([0], [0], color='red', lw=2),
                        Line2D([0], [0], linestyle='--', color='red', lw=2),
                        Line2D([0], [0], color='black', lw=2)]

        fig.legend(legend_lines, ['Deaths', 'Deaths in base analysis', 'Total QALY Loss'], loc='lower center',
                   bbox_to_anchor=(0.5, -0.02), fontsize=14, ncol=3)

        # Save the figure (replace with your output_figure function)
        output_figure(fig, filename=ROOT_DIR + '/figs/Fig_S5_deaths_sensitivity_analysis_sa_1.png')

        return fig

    def Fig_S6_plot_weekly_qaly_loss_by_outcome_vax(self):

        """
        :return: Plots National Weekly QALY Loss from Cases, Hospitalizations and Deaths across all states
        """
        # Create a plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 13))
        #fig, ax = plt.subplots(figsize=(10, 6))



        [mean_cases, ui_cases, mean_infections, ui_inf, mean_deaths, ui_deaths, mean_hosps_non_icu, ui_hosps_non_icu,
                mean_hosps_icu, ui_hosps_icu,mean_icu, ui_icu, mean_total_hosps, ui_total_hosps,
                mean_lc_1, ui_lc_1, mean_lc_2, ui_lc_2,
                mean_lc_vax_lb,ui_lc_vax_lb, mean_lc_vax_ub, ui_lc_vax_ub,
                mean_deaths_sa_1a, ui_deaths_sa_1a, mean_deaths_sa_1b, ui_deaths_sa_1b, mean_deaths_sa_1c, ui_deaths_sa_1c,
                mean_deaths_sa_2a, ui_deaths_sa_2a, mean_deaths_sa_2b, ui_deaths_sa_2b, mean_deaths_sa_2c, ui_deaths_sa_2c,
                mean_deaths_sa_3a, ui_deaths_sa_3a, mean_deaths_sa_3b, ui_deaths_sa_3b, mean_deaths_sa_3c, ui_deaths_sa_3c,
                mean_total_1, ui_total_1, mean_total_2, ui_total_2,
                mean_total_sa_1a, ui_total_sa_1a, mean_total_sa_1b, ui_total_sa_1b, mean_total_sa_1c,ui_total_sa_1c,
                mean_total_sa_2a, ui_total_sa_2a, mean_total_sa_2b, ui_total_sa_2b, mean_total_sa_2c,ui_total_sa_2c,
                mean_total_sa_3a, ui_total_sa_3a, mean_total_sa_3b, ui_total_sa_3b, mean_total_sa_3c, ui_total_sa_3c,
                mean_total_vax_lb,ui_total_vax_lb,mean_total_vax_ub,ui_total_vax_ub] = (
            self.get_mean_ui_weekly_qaly_loss_by_outcome(alpha=0.05))

        ax1.plot(self.allStates.dates, mean_infections,
                 label='Symptomatic Infections', linewidth=2, color='blue')
        ax1.fill_between(self.allStates.dates, ui_inf[0], ui_inf[1], color='lightblue', alpha=0.25)

        ax1.plot(self.allStates.dates, mean_total_hosps,
                 label='Hospital admissions (including ICU)', linewidth=2, color='green')
        ax1.fill_between(self.allStates.dates, ui_total_hosps[0], ui_total_hosps[1], color='grey', alpha=0.25)

        ax1.plot(self.allStates.dates, mean_deaths,
                 label='Deaths', linewidth=2, color='red')
        ax1.fill_between(self.allStates.dates, ui_deaths[0], ui_deaths[1], color='red', alpha=0.25)

        ax1.plot(self.allStates.dates, mean_lc_1,
                 label='Long COVID (Simplified Approach)', linewidth=2, color='purple')
        ax1.fill_between(self.allStates.dates, ui_lc_1[0], ui_lc_1[1], color='purple', alpha=0.25)

        ax1.plot(self.allStates.dates, mean_total_1,
                 label='Total', linewidth=2, color='black')
        ax1.fill_between(self.allStates.dates, ui_total_1[0], ui_total_1[1], color='grey', alpha=0.25)

        # Subplot 2
        ax2.plot(self.allStates.dates, mean_infections,
                 label='Symptomatic Infections', linewidth=2, color='blue')
        ax2.fill_between(self.allStates.dates, ui_inf[0], ui_inf[1], color='lightblue', alpha=0.25)

        ax2.plot(self.allStates.dates, mean_total_hosps,
                label='Hospital admissions (including ICU)', linewidth=2, color='green')
        ax2.fill_between(self.allStates.dates,ui_total_hosps[0], ui_total_hosps[1], color='grey', alpha=0.25)

        ax2.plot(self.allStates.dates, mean_deaths,
                label='Deaths', linewidth=2, color='red')
        ax2.fill_between(self.allStates.dates, ui_deaths[0], ui_deaths[1], color='red', alpha=0.25)


        ax2.plot(self.allStates.dates, mean_lc_vax_lb,
                 label='Long COVID (BTI: 50%)', linewidth=2, color='purple')
        ax2.fill_between(self.allStates.dates, ui_lc_vax_lb[0], ui_lc_vax_lb[1], color='purple', alpha=0.25)

        ax2.plot(self.allStates.dates, mean_total_1,
                 label='Total', linewidth=2, color='black')
        ax2.fill_between(self.allStates.dates, ui_total_vax_lb[0], ui_total_vax_lb[1], color='grey', alpha=0.25)


        # Subplot 3
        ax3.plot(self.allStates.dates, mean_infections,
                 label='Symptomatic Infections', linewidth=2, color='blue')
        ax3.fill_between(self.allStates.dates, ui_inf[0], ui_inf[1], color='lightblue', alpha=0.25)

        ax3.plot(self.allStates.dates, mean_total_hosps,
                 label='Hospital admissions (including ICU)', linewidth=2, color='green')
        ax3.fill_between(self.allStates.dates, ui_total_hosps[0], ui_total_hosps[1], color='grey', alpha=0.25)

        ax3.plot(self.allStates.dates, mean_deaths,
                 label='Deaths', linewidth=2, color='red')
        ax3.fill_between(self.allStates.dates, ui_deaths[0], ui_deaths[1], color='red', alpha=0.25)

        ax3.plot(self.allStates.dates, mean_lc_vax_ub,
                 label='Long COVID (BTI: 80%)', linewidth=2, color='purple')
        ax3.fill_between(self.allStates.dates, ui_lc_vax_ub[0], ui_lc_vax_ub[1], color='purple', alpha=0.25)

        #ax2.plot(self.allStates.dates, mean_lc_vax_ub,
                 #label='Long COVID (vaccination UB)', linewidth=2, color='purple')
        #ax2.fill_between(self.allStates.dates, ui_lc_2[0], ui_lc_2[1], color='purple', alpha=0.25)

        ax3.plot(self.allStates.dates, mean_total_vax_ub,
                 label='Total', linewidth=2, color='black')
        ax3.fill_between(self.allStates.dates, ui_total_vax_ub[0], ui_total_vax_ub[1], color='grey', alpha=0.25)

        y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0])
        y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1])
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        ax3.set_ylim(y_min, y_max)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3)
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3)


        ax1.axvspan("2021-06-30", "2021-10-27", alpha=0.2, color="lightblue")  # delta variant
        ax1.axvspan("2021-10-27", "2022-12-28", alpha=0.2, color="grey")  # omicron variant
        ax1.axvline(x="2021-08-04", color='black', linestyle='--')

        ax1.set_title('National Weekly QALY Loss', fontsize=16)
        ax1.set_xlabel('Date', fontsize=14)
        ax1.set_ylabel('QALY Loss', fontsize=14)
        #ax.legend()

        date_range = self.allStates.dates
        tick_positions = range(0, len(date_range))
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(
            [date_range[i] if i % 4 == 0 else '' for i in tick_positions],  # Label every 4th tick mark
            fontsize=10, rotation=45 )

        # Make the labeled tick marks slightly longer and bold
        for i, tick in enumerate(ax1.xaxis.get_major_ticks()):
            if i % 4 == 0:  # Every 4th tick mark
                tick.label1.set_fontsize(10)
                tick.label1.set_rotation(45)
                tick.label1.set_horizontalalignment('right')
                tick.label1.set_weight('normal')
                tick.tick1line.set_markersize(6)
                tick.tick1line.set_linewidth(2)
                tick.tick2line.set_markersize(6)
                tick.tick2line.set_linewidth(2)

            else:
                tick.label1.set_fontsize(10)
                tick.label1.set_weight('normal')
        ax1.text(0.01, 0.98, "A", transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')


        ax2.axvspan("2021-06-30", "2021-10-27", alpha=0.2, color="lightblue")  # delta variant
        ax2.axvspan("2021-10-27", "2022-12-28", alpha=0.2, color="grey")  # omicron variant
        ax2.axvline(x="2021-08-04", color='black', linestyle='--')

        ax2.set_title('National Weekly QALY Loss', fontsize=16)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.set_ylabel('QALY Loss', fontsize=14)
        # ax.legend()

        date_range = self.allStates.dates
        tick_positions = range(0, len(date_range))
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(
            [date_range[i] if i % 4 == 0 else '' for i in tick_positions],  # Label every 4th tick mark
            fontsize=10, rotation=45)

        # Make the labeled tick marks slightly longer and bold
        for i, tick in enumerate(ax2.xaxis.get_major_ticks()):
            if i % 4 == 0:  # Every 4th tick mark
                tick.label1.set_fontsize(10)
                tick.label1.set_rotation(45)
                tick.label1.set_horizontalalignment('right')
                tick.label1.set_weight('normal')
                tick.tick1line.set_markersize(6)
                tick.tick1line.set_linewidth(2)
                tick.tick2line.set_markersize(6)
                tick.tick2line.set_linewidth(2)

            else:
                tick.label1.set_fontsize(10)
                tick.label1.set_weight('normal')
        ax2.text(0.01, 0.98, "B", transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')

        ax3.axvspan("2021-06-30", "2021-10-27", alpha=0.2, color="lightblue")  # delta variant
        ax3.axvspan("2021-10-27", "2022-12-28", alpha=0.2, color="grey")  # omicron variant
        ax3.axvline(x="2021-08-04", color='black', linestyle='--')

        ax3.set_title('National Weekly QALY Loss', fontsize=16)
        ax3.set_xlabel('Date', fontsize=14)
        ax3.set_ylabel('QALY Loss', fontsize=14)

        date_range = self.allStates.dates
        tick_positions = range(0, len(date_range))
        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels(
            [date_range[i] if i % 4 == 0 else '' for i in tick_positions],  # Label every 4th tick mark
            fontsize=10, rotation=45)

        # Make the labeled tick marks slightly longer and bold
        for i, tick in enumerate(ax3.xaxis.get_major_ticks()):
            if i % 4 == 0:  # Every 4th tick mark
                tick.label1.set_fontsize(10)
                tick.label1.set_rotation(45)
                tick.label1.set_horizontalalignment('right')
                tick.label1.set_weight('normal')
                tick.tick1line.set_markersize(6)
                tick.tick1line.set_linewidth(2)
                tick.tick2line.set_markersize(6)
                tick.tick2line.set_linewidth(2)

            else:
                tick.label1.set_fontsize(10)
                tick.label1.set_weight('normal')
        ax3.text(0.01, 0.98, "C", transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top')
        #plt.subplots_adjust(hspace=0.15)

        output_figure(fig, filename=ROOT_DIR + '/figs/Fig_S6_vax_national_qaly_loss_by_outcome.png')

    def get_mean_ui_weekly_qaly_loss_by_outcome(self, alpha=0.05):
        """
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss.
        """

        mean_cases, ui_cases = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesCases, alpha=alpha)
        mean_infections, ui_infections = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesInfections, alpha=alpha)
        mean_deaths, ui_deaths = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths, alpha=alpha)
        mean_hosps_non_icu, ui_hosps_non_icu = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesHospNonICU, alpha=alpha)
        mean_hosps_icu, ui_hosps_icu = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesHospICU, alpha=alpha)
        mean_icu, ui_icu = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesICU, alpha=alpha)
        mean_total_hosps, ui_total_hosps = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotalHosp, alpha=alpha)
        mean_lc_1, ui_lc_1 = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesLongCOVID_1, alpha=alpha)
        mean_lc_2, ui_lc_2 = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesLongCOVID_2,alpha=alpha)
        mean_lc_vax_lb, ui_lc_vax_lb = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesLongCOVID_vax_LB,alpha=alpha)
        mean_lc_vax_ub, ui_lc_vax_ub = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesLongCOVID_vax_UB, alpha=alpha)
        mean_deaths_sa_1a, ui_deaths_sa_1a = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths_SA_1a, alpha=alpha)
        mean_deaths_sa_1b, ui_deaths_sa_1b = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths_SA_1b, alpha=alpha)
        mean_deaths_sa_1c, ui_deaths_sa_1c = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths_SA_1c, alpha=alpha)
        mean_deaths_sa_2a, ui_deaths_sa_2a = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths_SA_2a, alpha=alpha)
        mean_deaths_sa_2b, ui_deaths_sa_2b = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths_SA_2b, alpha=alpha)
        mean_deaths_sa_2c, ui_deaths_sa_2c = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths_SA_2c, alpha=alpha)
        mean_deaths_sa_3a, ui_deaths_sa_3a = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths_SA_3a, alpha=alpha)
        mean_deaths_sa_3b, ui_deaths_sa_3b = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths_SA_3b, alpha=alpha)
        mean_deaths_sa_3c, ui_deaths_sa_3c = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesDeaths_SA_3c, alpha=alpha)
        mean_total_1, ui_total_1 = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_1, alpha=alpha)
        mean_total_2, ui_total_2 = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_2, alpha=alpha)
        mean_total_sa_1a, ui_total_sa_1a = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_SA_1a, alpha=alpha)
        mean_total_sa_1b, ui_total_sa_1b = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_SA_1b, alpha=alpha)
        mean_total_sa_1c, ui_total_sa_1c = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_SA_1c, alpha=alpha)
        mean_total_sa_2a, ui_total_sa_2a = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_SA_2a, alpha=alpha)
        mean_total_sa_2b, ui_total_sa_2b = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_SA_2b, alpha=alpha)
        mean_total_sa_2c, ui_total_sa_2c = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_SA_2c, alpha=alpha)
        mean_total_sa_3a, ui_total_sa_3a = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_SA_3a, alpha=alpha)
        mean_total_sa_3b, ui_total_sa_3b = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_SA_3b, alpha=alpha)
        mean_total_sa_3c, ui_total_sa_3c = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_SA_3c, alpha=alpha)
        mean_total_vax_lb, ui_total_vax_lb = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_vax_LB, alpha=alpha)
        mean_total_vax_ub, ui_total_vax_ub = get_mean_ui_of_a_time_series(self.summaryOutcomes.weeklyQALYlossesTotal_vax_UB, alpha=alpha)

        return (mean_cases, ui_cases, mean_infections, ui_infections, mean_deaths, ui_deaths, mean_hosps_non_icu, ui_hosps_non_icu,
                mean_hosps_icu, ui_hosps_icu,mean_icu, ui_icu, mean_total_hosps, ui_total_hosps,
                mean_lc_1, ui_lc_1, mean_lc_2, ui_lc_2,
                mean_lc_vax_lb,ui_lc_vax_lb, mean_lc_vax_ub, ui_lc_vax_ub,
                mean_deaths_sa_1a, ui_deaths_sa_1a, mean_deaths_sa_1b, ui_deaths_sa_1b, mean_deaths_sa_1c, ui_deaths_sa_1c,
                mean_deaths_sa_2a, ui_deaths_sa_2a, mean_deaths_sa_2b, ui_deaths_sa_2b, mean_deaths_sa_2c, ui_deaths_sa_2c,
                mean_deaths_sa_3a, ui_deaths_sa_3a, mean_deaths_sa_3b, ui_deaths_sa_3b, mean_deaths_sa_3c, ui_deaths_sa_3c,
                mean_total_1, ui_total_1, mean_total_2, ui_total_2,
                mean_total_sa_1a, ui_total_sa_1a, mean_total_sa_1b, ui_total_sa_1b, mean_total_sa_1c,ui_total_sa_1c,
                mean_total_sa_2a, ui_total_sa_2a, mean_total_sa_2b, ui_total_sa_2b, mean_total_sa_2c,ui_total_sa_2c,
                mean_total_sa_3a, ui_total_sa_3a, mean_total_sa_3b, ui_total_sa_3b, mean_total_sa_3c, ui_total_sa_3c,
                mean_total_vax_lb,ui_total_vax_lb,mean_total_vax_ub,ui_total_vax_ub)



    def Fig_2_plot_map_of_avg_qaly_loss_by_county_4(self):
        """
        Vertically plots a map of the QALY loss per 100,000 population for each county, considering cases, deaths, and hospitalizations.
        """

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

        # Combine mainland, Alaska, and Hawaii for consistent binning
        merged_geo_data_all = merged_geo_data.copy()

        # Remove Alaska and Hawaii from mainland plot
        stateToRemove = ["2", "15"]
        merged_geo_data_mainland = merged_geo_data[~merged_geo_data.STATE.isin(stateToRemove)]

        # Explode the MultiPolygon geometries into individual polygons
        merged_geo_data_mainland = merged_geo_data_mainland.explode()

        # Plot the map
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        ax1.axis('off')
        ax1.set_title('Cumulative QALY Loss by County', fontsize=18)
        ax1.text(0.01, 0.98, "A", transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')

        # Exclude the extreme value for binning
        max_value = merged_geo_data_all["QALY Loss"].max()
        data_for_binning = merged_geo_data_all[merged_geo_data_all["QALY Loss"] != max_value]["QALY Loss"]

        # Calculate quantiles based on data excluding the extreme value
        scheme_all = mc.Quantiles(data_for_binning, k=6)
        rounded_bins_all = np.round(scheme_all.bins)

        # Add the extreme value to the last bin
        rounded_bins_all[-2] = max(rounded_bins_all[-2], max_value)
        rounded_bins_all = rounded_bins_all[:-1]  # Remove the last value

        # Apply the bins to the mainland
        scheme = mc.UserDefined(merged_geo_data_mainland["QALY Loss"], bins=rounded_bins_all)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="QALY Loss",
            linewidth=0.1,
            scheme=scheme,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'QALY Loss', 'loc': 'center right', 'fontsize': 12, 'bbox_to_anchor': (1.08, 0.4)},
            edgecolor="black",
            ax=ax1
        )

        # Alaska 1
        stateToInclude = ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax1 = fig.add_axes([0.07, 0.45, 0.25, 0.25])
        akax1.axis('off')
        polygon_AK = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
        scheme_AK = mc.UserDefined(merged_geo_data_AK_exploded["QALY Loss"], bins=rounded_bins_all)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="QALY Loss",
            linewidth=0.1,
            scheme=scheme_AK,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=akax1,
            extent=(-180, -90, 50, 75)
        )

        # Hawaii 1
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()
        hiax1 = fig.add_axes([0.07, 0.55, 0.15, 0.25])
        hiax1.axis('off')
        hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
        scheme_HI = mc.UserDefined(merged_geo_data_HI_exploded["QALY Loss"], bins=rounded_bins_all)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="QALY Loss",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=hiax1,
        )

        ax2.axis('off')
        ax2.set_title('Cumulative QALY Loss per 100,000 Population by County', fontsize=18)
        ax2.text(0.01, 0.98, "B", transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')

        # Repeat the process for the second plot using "QALY Loss per 100K"
        data_for_binning_100k = merged_geo_data_all[merged_geo_data_all["QALY Loss per 100K"] != max_value][
            "QALY Loss per 100K"]
        scheme_all_100k = mc.Quantiles(data_for_binning_100k, k=6)
        rounded_bins_all_100k = np.round(scheme_all_100k.bins)

        # Add the extreme value to the last bin for "per 100K"
        rounded_bins_all_100k[-2] = max(rounded_bins_all_100k[-2], max_value)
        rounded_bins_all_100k = rounded_bins_all_100k[:-1]  # Remove the last value

        # Apply the bins to the mainland
        scheme_100k = mc.UserDefined(merged_geo_data_mainland["QALY Loss per 100K"], bins=rounded_bins_all_100k)

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme_100k,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'QALY Loss per 100K', 'loc': 'center right', 'fontsize': 12,
                           'bbox_to_anchor': (1.08, 0.4)},
            edgecolor="black",
            ax=ax2
        )

        # Alaska 2
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax2 = fig.add_axes([0.06, -0.05, 0.25, 0.25])
        akax2.axis('off')
        scheme_AK = mc.UserDefined(merged_geo_data_AK_exploded["QALY Loss per 100K"], bins=rounded_bins_all_100k)

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme_AK,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=akax2,
            extent=(-180, -90, 50, 75)
        )

        # Hawaii 2
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()
        hiax2 = fig.add_axes([0.05, 0.05, 0.15, 0.25])
        hiax2.axis('off')
        scheme_HI = mc.UserDefined(merged_geo_data_HI_exploded["QALY Loss per 100K"], bins=rounded_bins_all_100k)

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="QALY Loss per 100K",
            linewidth=0.1,
            scheme=scheme_HI,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=hiax2,
        )

        output_figure(fig, filename=ROOT_DIR + '/figs/Fig_2_map_avg_county_qaly_loss_all_simulations_4.png')

        return fig

    def get_mean_ui_overall_qaly_loss_by_county(self, state_name, county_name, alpha=0.05):
        """
        :param state_name: Name of the state.
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss for a specific state.
        """

        county_qaly_losses = [qaly_losses[state_name, county_name] for qaly_losses in
                              self.summaryOutcomes.overallQALYlossesByCounty]
        mean, ui = get_overall_mean_ui(county_qaly_losses, 0.05)
        return mean, ui



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

    def Fig_3_plot_qaly_loss_by_state_and_by_outcome(self):
        """
        Generate a bar graph of the total QALY loss per 100,000 pop for each state, with each outcome's contribution
        represented in a different color based on governors political affiliation November 2018-November 2022
        (see 2018 gubernatorial election results)
        """
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))

        states_list = list(self.allStates.states.values())

        # Sort states by overall QALY loss
        sorted_states = sorted(
            states_list,
            key=lambda state_obj: (self.get_mean_ui_overall_qaly_loss_by_state(
                state_name=state_obj.name, alpha=0.05)[0] / state_obj.population) * 100000
        )

        # Set up the positions for the bars
        y_pos = range(len(sorted_states))

        democratic_states = ['CA', 'CO', 'CT', 'DC', 'DE', 'HI', 'IL', 'KS', 'KY', 'ME', 'MI', 'MN', 'NC', 'NJ', 'NV',
                             'NM', 'NY', 'OR', 'PA', 'RI', 'WA', 'WI']
        republican_states = ['AL', 'AK', 'AR', 'AZ', 'FL', 'GA', 'ID', 'IN', 'IA', 'LA', 'MA', 'MD', 'MS', 'MO', 'MT',
                             'NE', 'NH', 'ND', 'OH', 'OK', 'SC', 'SD', 'TN', 'TX',
                             'UT', 'VT', 'WV', 'WY']
        switch_states = ['VA']

        # Iterate through each state
        for i, state_obj in enumerate(sorted_states):
            # Calculate the heights for each segment
            (mean_cases, ui_cases, mean_infections, ui_infections, mean_hosps_non_icu, ui_hosps_non_icu, mean_hosps_icu, ui_hosps_icu,
                mean_deaths, ui_deaths, mean_icu, ui_icu, mean_total_hosps,
             ui_total_hosps, mean_lc_1, ui_lc_1,mean_lc_2, ui_lc_2, mean_total_1, ui_total_1, mean_total_2, ui_total_2)= self.get_mean_ui_overall_qaly_loss_by_outcome_and_by_state(
                state_name=state_obj.name, alpha=0.05)

            mean_total, ui_total = self.get_mean_ui_overall_qaly_loss_by_state(state_obj.name, alpha=0.05)

            cases_height = (mean_cases / state_obj.population) * 100000
            infections_height = (mean_infections / state_obj.population) * 100000
            deaths_height = (mean_deaths / state_obj.population) * 100000
            hosps_height = ((mean_hosps_icu + mean_hosps_non_icu + mean_icu) / state_obj.population) * 100000
            lc_1_height = (mean_lc_1 / state_obj.population) * 100000
            lc_2_height = (mean_lc_2 / state_obj.population) * 100000
            total_height = (mean_total / state_obj.population) * 100000
            total_1_height = (mean_total_1 / state_obj.population) * 100000
            total_2_height = (mean_total_2 / state_obj.population) * 100000

            # Convert UI into error bars
            cases_ui = (ui_cases / state_obj.population) * 100000
            infections_ui = (ui_infections / state_obj.population) * 100000
            deaths_ui = (ui_deaths / state_obj.population) * 100000
            hosps_ui = ((ui_hosps_non_icu + ui_hosps_icu + ui_icu) / state_obj.population) * 100000
            lc_1_ui = (ui_lc_1 / state_obj.population) * 100000
            lc_2_ui = (ui_lc_2 / state_obj.population) * 100000
            total_ui = (ui_total / state_obj.population) * 100000
            total_1_ui = (ui_total_1 / state_obj.population) * 100000
            total_2_ui = (ui_total_2 / state_obj.population) * 100000

            xterr_si = [[infections_height - infections_ui[0]], [infections_ui[1] - infections_height]]
            xterr_cases = [[cases_height - cases_ui[0]], [cases_ui[1] - cases_height]]
            xterr_deaths = [[deaths_height - deaths_ui[0]], [deaths_ui[1] - deaths_height]]
            xterr_hosps = [[hosps_height - hosps_ui[0]], [hosps_ui[1] - hosps_height]]
            xterr_lc_1 = [[lc_1_height - lc_1_ui[0]], [lc_1_ui[1] - lc_1_height]]
            xterr_lc_2 = [[lc_2_height - lc_2_ui[0]], [lc_2_ui[1] - lc_2_height]]
            xterr_total = [[total_height - total_ui[0]], [total_ui[1] - total_height]]
            xterr_total_1 = [[total_1_height - total_1_ui[0]], [total_1_ui[1] - total_1_height]]
            xterr_total_2 = [[total_2_height - total_2_ui[0]], [total_2_ui[1] - total_2_height]]

            ax.scatter(infections_height, [state_obj.name], marker='o', color='blue', label='Symptomatic Infections')
            ax.errorbar(infections_height, [state_obj.name], xerr=xterr_si, fmt='none', color='blue', capsize=0,
                        alpha=0.4)


            ax.scatter(hosps_height, [state_obj.name], marker='o', color='green', label='Hospital Admissions (including ICU)')
            ax.errorbar(hosps_height, [state_obj.name], xerr=xterr_hosps, fmt='none', color='green', capsize=0,
                        alpha=0.4)

            ax.scatter(deaths_height, [state_obj.name], marker='o', color='red', label='Deaths')
            ax.errorbar(deaths_height, [state_obj.name], xerr=xterr_deaths, fmt='none', color='red', capsize=0,
                        alpha=0.4)

            ax.scatter(lc_1_height, [state_obj.name], marker='o', color='purple', label='Long COVID')
            ax.errorbar(lc_1_height, [state_obj.name], xerr=xterr_lc_1, fmt='none', color='purple', capsize=0,
                        alpha=0.4)

            ax.scatter(total_1_height, [state_obj.name], marker='o', color='black', label='Total')
            ax.errorbar(total_1_height, [state_obj.name], xerr=xterr_total_1, fmt='none', color='grey', capsize=0,
                        alpha=0.4)

            #if state_obj.name == "VT" or state_obj.name == "AZ":
                #print(f"Total QALY loss for {state_obj.name}: {total_1_height},{ total_1_ui} per 100,000 population")

        # Set the labels for each state
        ax.set_yticks(y_pos)
        y_tick_colors = [
            'blue' if state_obj in democratic_states else 'red' if state_obj in republican_states else 'purple'
            for state_obj in [state_obj.name for state_obj in sorted_states]]
        ax.set_yticklabels([state_obj.name for state_obj in sorted_states], fontsize=12, rotation=0)

        # Set the colors for ticks
        for tick, color in zip(ax.yaxis.get_major_ticks(), y_tick_colors):
            tick.label1.set_color(color)

        # Set the labels and title
        ax.set_ylabel('States', fontsize=14)
        ax.set_xlabel('QALY Loss per 100,000 Population', fontsize=14)
        ax.set_title('State-level QALY Loss by Contributor', fontsize=16)

        # Show the legend with unique labels
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.12),
                  ncol=5, fancybox=True, shadow=True, fontsize=12)

        # Adjust layout to make the plot take up more space
        plt.tight_layout()
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)


        output_figure(fig, filename=ROOT_DIR + '/figs/Fig_3_total_qaly_loss_by_state_and_outcome.png')

    def get_mean_ui_overall_qaly_loss_by_state(self, state_name, alpha=0.05):
        """
        :param state_name: Name of the state.
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss for a specific state.
        """

        state_qaly_losses = [qaly_losses[state_name] for qaly_losses in self.summaryOutcomes.overallQALYlossesByState]
        mean, ui = get_overall_mean_ui(state_qaly_losses, 0.05)
        return mean, ui

    def get_mean_ui_overall_qaly_loss_by_outcome_and_by_state(self, state_name, alpha=0.05):
        """
        :param alpha: (float) significance value for calculating uncertainty intervals
        :return: mean and uncertainty interval for the weekly QALY loss.

        """
        state_cases_qaly_losses = [qaly_loss[state_name] for qaly_loss in
                                   self.summaryOutcomes.overallQALYlossesCasesByState]
        state_infections_qaly_losses = [qaly_loss[state_name] for qaly_loss in
                                   self.summaryOutcomes.overallQALYlossesInfectionsByState]
        state_hosps_non_icu_qaly_losses = [qaly_losses[state_name] for qaly_losses in
                                   self.summaryOutcomes.overallQALYlossesHospNonICUByState]
        state_hosps_icu_qaly_losses = [qaly_losses[state_name] for qaly_losses in
                                           self.summaryOutcomes.overallQALYlossesHospICUByState]
        state_total_hosps_qaly_losses=[qaly_losses[state_name] for qaly_losses in
                                            self.summaryOutcomes.overallQALYlossesTotalHospByState]
        state_deaths_qaly_losses = [qaly_losses[state_name] for qaly_losses in
                                    self.summaryOutcomes.overallQALYlossesDeathsByState]
        state_icu_qaly_losses = [qaly_losses[state_name] for qaly_losses in
                                 self.summaryOutcomes.overallQALYlossesICUByState]
        state_long_covid_1_qaly_losses = [qaly_losses[state_name] for qaly_losses in
                                        self.summaryOutcomes.overallQALYlossesLongCOVID_1_ByState]
        state_long_covid_2_qaly_losses = [qaly_losses[state_name] for qaly_losses in
                                          self.summaryOutcomes.overallQALYlossesLongCOVID_2_ByState]
        state_total_1_qaly_losses = [qaly_losses[state_name] for qaly_losses in
                                         self.summaryOutcomes.overallQALYlossesTotal1ByState]
        state_total_2_qaly_losses = [qaly_losses[state_name] for qaly_losses in
                                         self.summaryOutcomes.overallQALYlossesTotal2ByState]

        mean_cases, ui_cases = get_overall_mean_ui(state_cases_qaly_losses, alpha=alpha)
        mean_infections, ui_infections= get_overall_mean_ui(state_infections_qaly_losses, alpha=alpha)
        mean_hosps_non_icu, ui_hosps_non_icu = get_overall_mean_ui(state_hosps_non_icu_qaly_losses, alpha=alpha)
        mean_hosps_icu, ui_hosps_icu = get_overall_mean_ui(state_hosps_icu_qaly_losses, alpha=alpha)
        mean_deaths, ui_deaths = get_overall_mean_ui(state_deaths_qaly_losses, alpha=alpha)
        mean_icu, ui_icu = get_overall_mean_ui(state_icu_qaly_losses, alpha=alpha)
        mean_total_hosps, ui_total_hosps = get_overall_mean_ui(state_total_hosps_qaly_losses, alpha=alpha)
        mean_lc_1, ui_lc_1 = get_overall_mean_ui(state_long_covid_1_qaly_losses, alpha=alpha)
        mean_lc_2, ui_lc_2 = get_overall_mean_ui(state_long_covid_2_qaly_losses, alpha=alpha)
        mean_total_1, ui_total_1 = get_overall_mean_ui(state_total_1_qaly_losses, alpha=alpha)
        mean_total_2, ui_total_2 = get_overall_mean_ui(state_total_2_qaly_losses, alpha=alpha)

        return (mean_cases, ui_cases, mean_infections, ui_infections, mean_hosps_non_icu, ui_hosps_non_icu, mean_hosps_icu, ui_hosps_icu,
                mean_deaths, ui_deaths, mean_icu, ui_icu, mean_total_hosps,
             ui_total_hosps, mean_lc_1, ui_lc_1,mean_lc_2, ui_lc_2, mean_total_1, ui_total_1, mean_total_2, ui_total_2)


    def Fig_S2_plot_map_of_county_outcomes(self):
        """
        Generates sub-plotted maps of the number of cases, hospital admissions, and deaths per 100,000 population for each county.
        Values are computed per county and plotted by county.
        """

        # Prepare the data for plotting
        county_outcomes_data = {
            "COUNTY": [],
            "FIPS": [],
            "Symptomatic Infections": [],
            "Hosps": [],
            "Deaths": []
        }

        for state in self.allStates.states.values():
            for county in state.counties.values():
                county_outcomes_data["COUNTY"].append(county.name)
                county_outcomes_data["FIPS"].append(county.fips)
                county_outcomes_data["Symptomatic Infections"].append(county.pandemicOutcomes.infections.totalObs)
                county_outcomes_data["Hosps"].append(county.pandemicOutcomes.hosps.totalObs)
                county_outcomes_data["Deaths"].append(county.pandemicOutcomes.deaths.totalObs)

        # Create a DataFrame from the county data
        county_outcomes_df = pd.DataFrame(county_outcomes_data)
        county_outcomes_df.to_csv(ROOT_DIR + '/csv_files/county_outcomes.csv', index=False)

        # Load geometry data and merge with outcomes data
        geoData = gpd.read_file(
            "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
        )
        geoData['STATE'] = geoData['STATE'].str.lstrip('0')
        geoData['FIPS'] = geoData['STATE'] + geoData['COUNTY']
        merged_geo_data = geoData.merge(county_outcomes_df, left_on='FIPS', right_on='FIPS', how='left')

        # Filter data
        merged_geo_data = merged_geo_data.dropna(subset=["Deaths"])  # Adjust according to data availability
        stateToRemove = ["2", "15", "72"]
        merged_geo_data_mainland = merged_geo_data[~merged_geo_data.STATE.isin(stateToRemove)]

        # Explode geometries
        merged_geo_data_mainland = merged_geo_data_mainland.explode()

        # Helper function to format legend labels
        def format_legend_labels(breaks):
            labels = []
            for i in range(len(breaks) - 1):
                labels.append(f'{int(round(breaks[i], 0)):,} - {int(round(breaks[i + 1], 0)):,}')
            return labels

        # Plotting the maps
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), subplot_kw={'aspect': 'equal'})

        # Cases per 100K
        ax1.axis('off')
        ax1.set_title('Infections', fontsize=15)
        ax1.text(0.01, 0.98, "A", transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')

        scheme_cases = mc.Quantiles(merged_geo_data_mainland["Symptomatic Infections"], k=4)
        legend_labels_cases = format_legend_labels([scheme_cases.yb.min(), *scheme_cases.bins])

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="Symptomatic Infections",
            linewidth=0.1,
            scheme=scheme_cases,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'Infections', 'fontsize': 10, 'bbox_to_anchor': (0.95, 0.5),
                           'loc': 'center left'},
            legend_labels=legend_labels_cases,
            edgecolor="black",
            ax=ax1
        )

        # Alaska Cases
        stateToInclude = ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax1 = fig.add_axes([0.15, 0.39, 0.3, 0.5])
        akax1.axis('off')

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="Symptomatic Infections",
            linewidth=0.1,
            scheme=scheme_cases,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=akax1,
            extent=(-180, -90, 50, 75)
        )

        # Hawaii Cases
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()
        hiax1 = fig.add_axes([0.2, 0.65, 0.1, 0.15])
        hiax1.axis('off')

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="Symptomatic Infections",
            linewidth=0.1,
            scheme=scheme_cases,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=hiax1,
        )

        # Hospital Admissions per 100K
        ax2.axis('off')
        ax2.set_title('Hospital Admissions', fontsize=15)
        ax2.text(0.01, 0.98, "B", transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')

        scheme_hosps = mc.Quantiles(merged_geo_data_mainland["Hosps"], k=4)
        legend_labels_hosps = format_legend_labels([scheme_hosps.yb.min(), *scheme_hosps.bins])

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="Hosps",
            linewidth=0.1,
            scheme=scheme_hosps,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'Hospital Admissions', 'fontsize': 10, 'bbox_to_anchor': (0.95, 0.5),
                           'loc': 'center left'},
            legend_labels=legend_labels_hosps,
            edgecolor="black",
            ax=ax2
        )

        # Alaska Hospital Admissions
        akax2 = fig.add_axes([0.15, 0.06, 0.3, 0.5])
        akax2.axis('off')

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="Hosps",
            linewidth=0.1,
            scheme=scheme_hosps,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=akax2,
            extent=(-180, -90, 50, 75)
        )

        # Hawaii Hospital Admissions
        hiax2 = fig.add_axes([0.2, 0.32, 0.1, 0.15])
        hiax2.axis('off')

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="Hosps",
            linewidth=0.1,
            scheme=scheme_hosps,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=hiax2,
        )

        # Deaths per 100K
        ax3.axis('off')
        ax3.set_title('Deaths', fontsize=15)
        ax3.text(0.01, 0.98, "C", transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top')

        scheme_deaths = mc.Quantiles(merged_geo_data_mainland["Deaths"], k=4)
        legend_labels_deaths = format_legend_labels([scheme_deaths.yb.min(), *scheme_deaths.bins])

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="Deaths",
            linewidth=0.1,
            scheme=scheme_deaths,
            cmap="viridis",
            legend=True,
            legend_kwargs={'title': 'Deaths', 'fontsize': 10, 'bbox_to_anchor': (0.95, 0.5),
                           'loc': 'center left'},
            legend_labels=legend_labels_deaths,
            edgecolor="black",
            ax=ax3
        )

        # Alaska Deaths
        akax3 = fig.add_axes([0.15, -0.25, 0.3, 0.5])
        akax3.axis('off')

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="Deaths",
            linewidth=0.1,
            scheme=scheme_deaths,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=akax3,
            extent=(-180, -90, 50, 75)
        )

        # Hawaii Deaths
        hiax3 = fig.add_axes([0.2, 0.01, 0.1, 0.15])
        hiax3.axis('off')

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="Deaths",
            linewidth=0.1,
            scheme=scheme_deaths,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=hiax3,
        )

        # Save the plot to a file
        plt.tight_layout()
        output_figure(fig, filename=ROOT_DIR + '/figs/Fig_S2_map_county_outcomes.png')
        plt.show()

    def Fig_S3_plot_map_of_hsa_outcomes_by_county_per_100K(self):
        """
        Generates sub-plotted maps of the number of cases, hospital admissions, and deaths per 100,000 population for each county.
        Values are computed per HSA (aggregate of county values for all counties within an HSA), but plotted by county.
        """

        # Load HSA data
        hsa_data = read_csv_rows(ROOT_DIR + '/Data/county_names_HSA_number.csv',
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
            "Symptomatic Infections": [],
            "Hosps": [],
            "Deaths": [],
            "HSA Total Symptomatic Infections per 100K": [],
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
                            "Total Symptomatic Infections": 0,
                            "Total Hospitalizations": 0,
                            "Total Deaths": 0,
                            "Population": hsa_population
                        }

                    # Append county data to the list
                    county_outcomes_data["COUNTY"].append(county.name)
                    county_outcomes_data["FIPS"].append(county.fips)
                    county_outcomes_data["County Population"].append(county.population)
                    county_outcomes_data["HSA Number"].append(hsa_number)
                    county_outcomes_data["Symptomatic Infections"].append(county.pandemicOutcomes.infections.totalObs)
                    county_outcomes_data["Hosps"].append(county.pandemicOutcomes.hosps.totalObs)
                    county_outcomes_data["Deaths"].append(county.pandemicOutcomes.deaths.totalObs)
                    county_outcomes_data["HSA Population"].append(hsa_population)

                    # Update aggregated values for HSA
                    if hsa_number not in hsa_aggregated_data:
                        hsa_aggregated_data[hsa_number] = {
                            "Total Symptomatic Infections": 0,
                            "Total Hospitalizations": 0,
                            "Total Deaths": 0,
                            "Population": hsa_population
                        }
                    hsa_aggregated_data[hsa_number]["Total Symptomatic Infections"] += county.pandemicOutcomes.infections.totalObs
                    hsa_aggregated_data[hsa_number]["Total Hospitalizations"] += county.pandemicOutcomes.hosps.totalObs
                    hsa_aggregated_data[hsa_number]["Total Deaths"] += county.pandemicOutcomes.deaths.totalObs

                else:
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
                hsa_total_cases_per_100K = (hsa_aggregated_data[hsa_number]["Total Symptomatic Infections"] / float(
                    hsa_aggregated_data[hsa_number]["Population"])) * 100000
                hsa_total_hospitalizations_per_100K = (hsa_aggregated_data[hsa_number][
                                                           "Total Hospitalizations"] / float(
                    hsa_aggregated_data[hsa_number]["Population"])) * 100000
                hsa_total_deaths_per_100K = (hsa_aggregated_data[hsa_number]["Total Deaths"] / float(
                    hsa_aggregated_data[hsa_number]["Population"])) * 100000

                county_outcomes_data["HSA Total Symptomatic Infections per 100K"].append(hsa_total_cases_per_100K)
                county_outcomes_data["HSA Total Hospitalizations per 100K"].append(hsa_total_hospitalizations_per_100K)
                county_outcomes_data["HSA Total Deaths per 100K"].append(hsa_total_deaths_per_100K)
            else:
                # If HSA Number is None, set corresponding HSA Total values to None
                county_outcomes_data["HSA Total Symptomatic Infections per 100K"].append(None)
                county_outcomes_data["HSA Total Hospitalizations per 100K"].append(None)
                county_outcomes_data["HSA Total Deaths per 100K"].append(None)

        # Create a DataFrame from the county data
        county_outcomes_df = pd.DataFrame(county_outcomes_data)
        county_outcomes_df.to_csv(ROOT_DIR + '/csv_files/county_outcomes_by_hsa.csv', index=False)


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

        # Define color schemes for different outcomes
        def get_color_scheme(data):
            return mc.Quantiles(data, k=4)

        # Plot the map
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), subplot_kw={'aspect': 'equal'})

        def format_legend_labels(breaks):
            labels = []
            for i in range(len(breaks) - 1):
                labels.append(f'{int(round(breaks[i], 0)):,} - {int(round(breaks[i + 1], 0)):,}')
            return labels

        # Cases per 100K
        ax1.axis('off')
        ax1.set_title('Infections per 100,000 Population', fontsize=15)
        ax1.text(0.01, 0.98, "A", transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')

        scheme_cases = get_color_scheme(merged_geo_data_mainland["HSA Total Symptomatic Infections per 100K"])
        legend_labels_cases = format_legend_labels([scheme_cases.yb.min(), *scheme_cases.bins])

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="HSA Total Symptomatic Infections per 100K",
            linewidth=0.1,
            scheme=scheme_cases,
            cmap="viridis",
            legend=legend_labels_cases,
            legend_kwargs={'title': 'Infections per 100K', 'fontsize': 10, 'bbox_to_anchor': (0.95, 0.5),
                           'loc': 'center left'},
            edgecolor="black",
            ax=ax1
        )

        # Alaska
        stateToInclude = ["2"]
        merged_geo_data_AK = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude)]
        merged_geo_data_AK_exploded = merged_geo_data_AK.explode()
        akax1 = fig.add_axes([0.15, 0.39, 0.3, 0.5])
        akax1.axis('off')
        scheme_AK_cases = get_color_scheme(merged_geo_data_AK_exploded["HSA Total Symptomatic Infections per 100K"])


        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="HSA Total Symptomatic Infections per 100K",
            linewidth=0.1,
            scheme=scheme_cases,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=akax1,
            extent=(-180, -90, 50, 75)
        )


        # Hawaii
        stateToInclude_HI = ["15"]
        merged_geo_data_HI = merged_geo_data[merged_geo_data.STATE.isin(stateToInclude_HI)]
        merged_geo_data_HI_exploded = merged_geo_data_HI.explode()
        hiax1 = fig.add_axes([0.2, 0.65, 0.1, 0.15])
        hiax1.axis('off')
        scheme_HI_cases = get_color_scheme(merged_geo_data_HI_exploded["HSA Total Symptomatic Infections per 100K"])

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="HSA Total Symptomatic Infections per 100K",
            linewidth=0.1,
            scheme=scheme_cases,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=hiax1,
        )


        # Hospital Admissions per 100K
        ax2.axis('off')
        ax2.set_title('Hospital Admissions per 100,000 Population', fontsize=15)
        ax2.text(0.01, 0.98, "B", transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')

        scheme_hosps = get_color_scheme(merged_geo_data_mainland["HSA Total Hospitalizations per 100K"])
        legend_labels_hosps = format_legend_labels([scheme_hosps.yb.min(), *scheme_hosps.bins])

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="HSA Total Hospitalizations per 100K",
            linewidth=0.1,
            scheme=scheme_hosps,
            cmap="viridis",
            legend=legend_labels_hosps,
            legend_kwargs={'title': 'Hospital Admissions per 100K', 'fontsize': 10, 'bbox_to_anchor': (0.95, 0.5),
                           'loc': 'center left'},
            edgecolor="black",
            ax=ax2
        )

        # Alaska
        akax2 = fig.add_axes([0.15, 0.06, 0.3, 0.5])
        akax2.axis('off')
        scheme_AK_hosps = get_color_scheme(merged_geo_data_AK_exploded["HSA Total Hospitalizations per 100K"])

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="HSA Total Hospitalizations per 100K",
            linewidth=0.1,
            scheme=scheme_hosps,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=akax2,
            extent=(-180, -90, 50, 75)
        )


        # Hawaii
        hiax2 = fig.add_axes([0.2, 0.32, 0.1, 0.15])
        hiax2.axis('off')
        scheme_HI_hosps = get_color_scheme(merged_geo_data_HI_exploded["HSA Total Hospitalizations per 100K"])

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="HSA Total Hospitalizations per 100K",
            linewidth=0.1,
            scheme=scheme_hosps,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=hiax2,
        )


        # Deaths per 100K
        ax3.axis('off')
        ax3.set_title('Deaths per 100,000 Population', fontsize=15)
        ax3.text(0.01, 0.98, "C", transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top')

        scheme_deaths = get_color_scheme(merged_geo_data_mainland["HSA Total Deaths per 100K"])
        legend_labels_deaths = format_legend_labels([scheme_deaths.yb.min(), *scheme_deaths.bins])

        gplt.choropleth(
            merged_geo_data_mainland,
            hue="HSA Total Deaths per 100K",
            linewidth=0.1,
            scheme=scheme_deaths,
            cmap="viridis",
            legend=legend_labels_deaths,
            legend_kwargs={'title': 'Deaths per 100K', 'fontsize': 10, 'bbox_to_anchor': (0.95, 0.5),
                           'loc': 'center left'},
            edgecolor="black",
            ax=ax3
        )

        # Alaska
        akax3 = fig.add_axes([0.15, -0.25, 0.3, 0.5])
        akax3.axis('off')
        scheme_AK_deaths = get_color_scheme(merged_geo_data_AK_exploded["HSA Total Deaths per 100K"])

        gplt.choropleth(
            merged_geo_data_AK_exploded,
            hue="HSA Total Deaths per 100K",
            linewidth=0.1,
            scheme=scheme_deaths,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=akax3,
            extent=(-180, -90, 50, 75)
        )


        # Hawaii
        hiax3 = fig.add_axes([0.2, 0.01, 0.1, 0.15])
        hiax3.axis('off')
        scheme_HI_deaths = get_color_scheme(merged_geo_data_HI_exploded["HSA Total Deaths per 100K"])

        gplt.choropleth(
            merged_geo_data_HI_exploded,
            hue="HSA Total Deaths per 100K",
            linewidth=0.1,
            scheme=scheme_deaths,
            cmap="viridis",
            legend=False,
            edgecolor="black",
            ax=hiax3,
        )


        plt.subplots_adjust(hspace=0.01)
        plt.tight_layout()

        output_figure(fig, filename=ROOT_DIR + '/figs/Fig_S3_map_county_hsa_outcomes_per_100K.png')

    def Fig_S1_plot_weekly_outcomes(self):
        """
        :return: Plots National Weekly QALY Loss from Cases, Hospitalizations and Deaths across all states
        """
        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 6))

        infections = self.allStates.pandemicOutcomes.infections.weeklyObs
        cases = self.allStates.pandemicOutcomes.cases.weeklyObs
        hosps = self.allStates.pandemicOutcomes.hosps.weeklyObs
        deaths = self.allStates.pandemicOutcomes.deaths.weeklyObs

        # Plot outcomes on the primary axis
        ax.plot(self.allStates.dates, infections, label='Infections', linewidth=2, color='blue')
        ax.plot(self.allStates.dates, cases, label='Detected Cases', linewidth=2, color='blue', linestyle='dashed')
        ax.plot(self.allStates.dates, hosps, label='Hospital Admissions', linewidth=2, color='green')

        # Create a secondary axis for deaths
        ax2 = ax.twinx()
        ax2.plot(self.allStates.dates, deaths, label='Deaths', linewidth=2, color='red')

        # Set the limits for the primary axis (cases, hospital admissions)
        ax.set_ylim([0, 25000000])

        # Set the limits for the secondary y-axis (deaths) to 30,000
        ax2.set_ylim([0, 300000])  # Set the secondary y-axis limit for deaths to 30,000

        # Set the ticks for the secondary axis based on the data range
        ax2.set_yticks(ax2.get_yticks())

        # Update the tick labels for the secondary axis
        ax2.tick_params(axis='y', labelcolor='red')

        # Color the y-axis for deaths red
        ax2.spines['right'].set_color('red')  # Set the right spine color to red
        ax2.yaxis.label.set_color('red')  # Set the label color to red
        ax2.tick_params(axis='y', labelcolor='red')  # Set the tick label color to red

        ax.axvspan("2021-06-30", "2021-10-27", alpha=0.2, color="lightblue")
        ax.axvspan("2021-10-27", "2022-12-28", alpha=0.2, color="grey")

        ax.set_title('Number of Weekly Infections, Detected Cases, Hospital Admissions, and Deaths in the U.S.')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Cases and Hospital Admissions')

        # Move the number of deaths further to the right
        ax2.set_ylabel('Number of Deaths', rotation=270, labelpad=15, color='red')

        # Combine legend for deaths, cases, and hospital admissions
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        # Set y-tick labels with commas for better readability
        vals_y = ax.get_yticks()
        vals_y2 = ax2.get_yticks()

        ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_y])
        ax2.set_yticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_y2])

        date_range = self.allStates.dates
        tick_positions = range(0, len(date_range))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [date_range[i] if i % 4 == 0 else '' for i in tick_positions],  # Label every 4th tick mark
            fontsize=10, rotation=45  # Rotate labels at 45 degrees
        )

        # Make the labeled tick marks slightly longer and bold
        for i, tick in enumerate(ax.xaxis.get_major_ticks()):
            if i % 4 == 0:  # Every 4th tick mark
                tick.label1.set_fontsize(10)  # Adjust font size for the labeled tick mark
                tick.label1.set_rotation(45)  # Rotate the label if needed
                tick.label1.set_horizontalalignment('right')
                tick.label1.set_weight('normal')
                tick.tick1line.set_markersize(6)
                tick.tick1line.set_linewidth(2)
                tick.tick2line.set_markersize(6)
                tick.tick2line.set_linewidth(2)

            else:
                tick.label1.set_fontsize(10)  # Adjust font size for the non-labeled tick marks
                tick.label1.set_weight('normal')  # Set the label to normal weight

        output_figure(fig, filename=ROOT_DIR + '/figs/Fig_S1_national_outcomes.png')


    def plot_weekly_outcomes_vax(self):
        """
        :return: Plots National Weekly QALY Loss from Cases, Hospitalizations and Deaths across all states
        """
        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 6))

        cases = self.allStates.pandemicOutcomes.cases.weeklyObs
        infections = self.allStates.pandemicOutcomes.infections.weeklyObs
        vax_lb = (self.allStates.pandemicOutcomes.lc_v_lb.weeklyObs)
        #vax_ub = self.allStates.pandemicOutcomes.longCOVID_vax_UB.weeklyObs
        hosps = self.allStates.pandemicOutcomes.hosps.weeklyObs
        deaths = self.allStates.pandemicOutcomes.deaths.weeklyObs

        ax.plot(self.allStates.dates, cases, label='Cases', linewidth=2, color='blue')
        ax.plot(self.allStates.dates, infections,  label='Symptomatic Infections', linewidth=2, color='black')
        ax.plot(self.allStates.dates, hosps, label='Hospital Admissions', linewidth=2, color='green')
        #ax.plot(self.allStates.dates, vax_ub, label='Vax Cases UB', linewidth=2, color='purple')
        ax.plot(self.allStates.dates, vax_lb, label='Vax Cases LB', linewidth=2, color='purple')

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
        ax2.set_ylabel('Number of Deaths', rotation=270, labelpad=15, color='red')

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
        plt.show()




