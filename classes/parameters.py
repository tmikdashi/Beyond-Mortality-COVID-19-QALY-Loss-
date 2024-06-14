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
        self.qWeightICUHosp=None
        self.qWeightLongCOVID = None

    def __str__(self):
        return "qWeightCase: {:.4f}, qWeightHosp: {:.4f}, qWeightDeath: {:.4f}, qWeightICU: {:.4f}, qWeightLongCOVID{:.4f}".format(
            self.qWeightCase, self.qWeightHosp, self.qWeightDeath, self.qWeightICU, self.qWeightLongCOVID)



class ParameterGenerator:

    def __init__(self):

        self.parameters = dict()

        # parameters to calculate the QALY loss due to a case
        self.parameters['case_prob_symp'] = Beta(mean=0.692, st_dev=0.115)
        self.parameters['case_dur_symp'] = Gamma(mean=7/365.25, st_dev=0.5/365.25)
        self.parameters['case_weight_symp'] = Beta(mean=0.43, st_dev=0.015)

        self.parameters['prob_surv'] =ConstantArray(values=0.01)

        # parameters to calculate the QALY loss due to a hospitalizations + ICU
        hosp_data = pd.read_csv(ROOT_DIR + '/csv_files/hosps_by_age.csv')
        self.parameters['hosp_dur_stay_ICU'] = Gamma(mean=8.3/365.25, st_dev=1.25/365.25) #TODO: A REVOIR LES VALEURS SD
        self.parameters['hosp_dur_stay_ward'] = Gamma(mean=6/365.25, st_dev=1.25/365.25)
        self.parameters['hosp_weight'] = Beta(mean=0.5, st_dev=0.05)
        self.parameters['hosps_age_dist'] = Dirichlet(par_ns=hosp_data['COVID-19 Hosps'])
        self.parameters['hosps_prob_surv'] = Beta(mean=0.82, st_dev=0.02)

        self.parameters['icu_prob'] = ConstantArray(0.174)
        self.parameters['hosps_prob_non_icu'] = 1 - (self.parameters['icu_prob'].value)
        self.parameters['icu_weight'] = Beta(mean=0.60, st_dev=0.1)
        self.parameters['occupancy_dur'] = ConstantArray(values=(7 / 365))

        # parameters to calculate the QALY loss due to a death
        data = pd.read_csv(ROOT_DIR + '/csv_files/deaths_by_age.csv')
        self.parameters['death_age_dist'] = Dirichlet(par_ns=data['COVID-19 Deaths'])
        self.parameters['dQALY_loss_by_age'] = ConstantArray(values=[22.53, 20.89, 19.08, 16.96, 14.30, 11.52, 8.61, 5.50, 3.00, 1.46])
        self.parameters['Age Group']= ConstantArray(values=data['Age Group'])

        # LONG COVID 1: parameters to calculate the QALY loss due to long COVID
        self.parameters['long_covid_dur'] = Beta(mean=4/12, st_dev=(1/48))
        #self.parameters['case_prob_symp'] = Beta(mean=0.692, st_dev=0.115)
        self.parameters['long_covid_prob'] = Beta(mean=0.062, st_dev=0.0273)
        self.parameters['long_covid_weight'] = Beta(mean=0.29, st_dev =0.0275)

        # LONG COVID  2: parameters to calculate the QALY loss due to long COVID
        self.parameters['cases_prob_hosp'] = ConstantArray(values=0.058)
        # self.parameters['case_prob_symp'] = Beta(mean=0.692, st_dev=0.115)
        self.parameters['long_covid_nonhosp_prob_surv'] = ConstantArray(values=1)
        self.parameters['long_covid_nonhosp_prob_symp'] = Beta(mean=0.057, st_dev=0.028)
        self.parameters['long_covid_nonhosp_dur'] = Beta(mean=4/12, st_dev=(1/48))

        self.parameters['long_covid_hosp_surv'] = Beta(mean=0.82, st_dev=0.02)
        self.parameters['long_covid_hosp_prob_symp'] = Beta(mean=0.275, st_dev=0.089)
        self.parameters['long_covid_hosp_dur'] = Beta(mean=9/12, st_dev=1.25/12)

        self.parameters['long_covid_icu_surv'] = Beta(mean=0.61, st_dev=0.09)
        self.parameters['long_covid_icu_prob_symp'] =Beta(mean=0.431, st_dev=0.107)


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
                             * self.parameters['hosp_weight'].value
                             * (1-self.parameters['icu_prob'].value))

        param.qWeightDeath = np.dot(
            self.parameters['death_age_dist'].value,
            self.parameters['dQALY_loss_by_age'].value)

        param.qWeightICU = (self.parameters['icu_weight'].value
                            * self.parameters['occupancy_dur'].value)

        param.qWeightICUHosp = (self.parameters['hosp_dur_stay'].value
                                * self.parameters['hosp_weight'].value
                                * self.parameters['icu_prob'].value)

        param.qWeightLongCOVID_1 = ( self.parameters['cases_prob_symp'].value
                                  * self.parameters['long_covid_prob'].value
                                  * self.parameters['prob_surv'].value
                                  * self.parameters['long_covid_dur'].value
                                  * self.parameters['long_covid_weight'].value)


        param.qWeightLongCOVID_2_nh = ( (1-self.parameters['cases_prob_hosp'].value)
                                      * self.parameters['case_prob_symp'].value
                                      * self.parameters['long_covid_nonhosp_prob_surv'].value
                                      * self.parameters['long_covid_nonhosp_prob_symp'].value
                                      * self.parameters['long_covid_nonhosp_dur'].value
                                      * self.paramerts['long_covid_weight'].value)


        param.qWeightLongCOVID_2_h = ( (1-self.parameters['icu_prob'].value)
                                      * self.parameters['long_covid_hosp_prob_surv'].value
                                      * self.parameters['long_covid_hosp_prob_symp'].value
                                      * self.parameters['long_covid_hosp_dur'].value
                                      * self.paramerts['long_covid_weight'].value)

        param.qWeightLongCOVID_2_i = ( self.parameters['icu_prob'].value
                                      * self.parameters['long_covid_icu_prob_surv'].value
                                      * self.parameters['long_covid_icu_prob_symp'].value
                                      * self.parameters['long_covid_hosp_dur'].value
                                      * self.paramerts['long_covid_weight'].value)



