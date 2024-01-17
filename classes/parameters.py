import numpy as np
import pandas as pd

from deampy.parameters import Beta, Gamma, Dirichlet
from definitions import ROOT_DIR


class ParameterValues:

    def __init__(self):

        self.qWeightCase = None
        self.qWeightHosp = None
        self.qWeightDeath = None

    def __str__(self):
        return "qWeightCase: {:.4f}, qWeightHosp: {:.4f}, qWeightDeath: {:.4f}".format(
            self.qWeightCase, self.qWeightHosp, self.qWeightDeath)


class ParameterGenerator:

    def __init__(self):

        self.parameters = dict()

        # parameters to calculate the QALY loss due to a case
        self.parameters['case_prob_symp'] = Beta(mean=0.62, st_dev=0.07)  # SD based on 95% CI-- may need to be revised
        self.parameters['case_weight_symp'] = Beta(mean=0.43, st_dev=0.03)
        self.parameters['case_dur_symp'] = Gamma(mean=10/365, st_dev=2/365) # Based on CDC isolation guidelines, can be replaced by exp decay function

        # parameters to calculate the QALY loss due to a hospitalizations
        self.parameters['hosp_dur_stay'] = Gamma(mean=6/365, st_dev=1.5/365) # assuming SD = IQR/1.35
        self.parameters['hosp_weight'] = Beta(mean=0.5, st_dev=0.1)

        # parameters to calculate the QALY loss due to a death
        data = pd.read_csv(ROOT_DIR + '/csv_files/deaths_by_age.csv')

        self.parameters['death_age_dist'] = Dirichlet(par_ns=data['COVID-19 Deaths'])
        self.parameters['death_weight_by_age'] = Dirichlet(par_ns=[24.52, 22.38, 20.24, 17.93, 15.21, 12.25, 9.22, 5.93, 3.30, 1.54])

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
            self.parameters['death_weight_by_age'].value)
