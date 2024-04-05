import numpy as np
import pandas as pd

from deampy.parameters import Beta, Gamma, Dirichlet, ConstantArray
from definitions import ROOT_DIR
import random


class ParameterValues:

    def __init__(self):

        self.qWeightCase = None
        self.qWeightHosp = None
        self.qWeightDeath = None
        self.qWeightICU = None
        self.qWeightLongCOVID = None

    def __str__(self):
        return "qWeightCase: {:.4f}, qWeightHosp: {:.4f}, qWeightDeath: {:.4f}, qWeightICU: {:.4f}, qWeightLongCOVID{:.4f}".format(
            self.qWeightCase, self.qWeightHosp, self.qWeightDeath, self.qWeightICU, self.qWeightLongCOVID)


class ParameterGenerator:

    def __init__(self):

        self.parameters = dict()

        # parameters to calculate the QALY loss due to a case
        self.parameters['case_prob_symp'] = Beta(mean=0.692, st_dev=0.115)  # SD based on 95% CI-- may need to be revised
        self.parameters['case_weight_symp'] = Beta(mean=0.43, st_dev=0.015)
        self.parameters['case_dur_symp'] = Gamma(mean=10/365.25, st_dev=1/365.25) # Based on CDC isolation guidelines, can be replaced by exp decay function

        # parameters to calculate the QALY loss due to a hospitalizations
        hosp_data = pd.read_csv(ROOT_DIR + '/csv_files/hosps_by_age.csv')
        self.parameters['hosp_dur_stay'] = Gamma(mean=6/365.25, st_dev=3.704/365.25) # assuming SD = IQR/1.35
        self.parameters['hosp_weight'] = Beta(mean=0.5, st_dev=0.05)
        self.parameters['hosps_age_dist'] = Dirichlet(par_ns=hosp_data['COVID-19 Hosps'])
        self.parameters['hosps_prob_non_icu'] = Beta(mean=0.8, st_dev=0.115) #TODO

        # parameters to calculate the QALY loss due to a death
        data = pd.read_csv(ROOT_DIR + '/csv_files/deaths_by_age.csv')
        self.parameters['death_age_dist'] = Dirichlet(par_ns=data['COVID-19 Deaths'])
        self.parameters['dQALY_loss_by_age'] = ConstantArray(values=[22.53, 20.89, 19.08, 16.96, 14.30, 11.52, 8.61, 5.50, 3.00, 1.46])
        self.parameters['Age Group']= ConstantArray(values=data['Age Group'])

        #parameters to calculate the QALY loss due to a ICU hopsitalization
        self.parameters['icu_prob'] = Beta(mean=0.174, st_dev = 0.02) #TODO: this should likely be complementary to hosps_prob_non_icu
        self.parameters['icu_weight']= Beta(mean=0.60, st_dev = 0.1) #TODO
        #self.parameters['occupancy_dur'] = Gamma(mean=)


        # parameters to calculate the QALY loss due to long COVID
        self.parameters['long_covid_dur'] = Beta(mean=28/365.25, st_dev=1/365.25) #TODO
        self.parameters['long_covid_prob'] = Beta(mean=0.332, st_dev=0.011) #TODO
        self.parameters['long_covid_weight'] = Beta(mean=0.29, st_dev =0.015) #TODO


    def generate(self, rng):
        """
        :param rng:
        :return: (Parameters) a set of parameter values sampled from probability distributions
        """

        # sample all parameters
        self._sample_parameters(rng)

        # create a parameter object
        param = ParameterValues()

        # calculate QALY loss due to a case
        self._update_param_values(param=param)

        return param


    def _sample_parameters(self, rng):
        """
        samples all parameters
        """

        for par in self.parameters.values():
            par.sample(rng)

    def _update_param_values(self, param):

        param.qWeightCase = (self.parameters['case_weight_symp'].value
                             * self.parameters['case_prob_symp'].value
                             * self.parameters['case_dur_symp'].value)

        param.qWeightHosp = (self.parameters['hosp_dur_stay'].value
                             * self.parameters['hosp_weight'].value)

        param.qWeightDeath = np.dot(
            self.parameters['death_age_dist'].value,
            self.parameters['dQALY_loss_by_age'].value)

        param.qWeightICU = (self.parameters['icu_prob'].value
                            * self.parameters['icu_weight'].value)
                            #* self.parameters['occupancy_dur'].value)

        param.qWeightLongCOVID = (self.parameters['long_covid_dur'].value
                                  * self.parameters['long_covid_prob'].value
                                  * self.parameters['long_covid_weight'].value)

