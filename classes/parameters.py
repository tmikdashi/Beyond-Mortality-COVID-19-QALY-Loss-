import numpy as np

from deampy.parameters import Beta, Gamma, Dirichlet, ConstantArray


class ParameterValues:

    def __init__(self):

        self.qWeightCase = 0.1
        self.qWeightHosp = 0.3
        self.qWeightDeath = 10

    def __str__(self):
        return "qWeightCase: {:.4f}, qWeightHosp: {:.4f}, qWeightDeath: {:.4f}".format(
            self.qWeightCase, self.qWeightHosp, self.qWeightDeath)


class ParameterGenerator:

    def __init__(self, life_expectancy_array, nb_deaths_array):

        self.parameters = dict()

        # parameters to calculate the QALY loss due to a case
        self.parameters['case_weight_symp'] = Beta(mean=0.43/365, st_dev=0.03/365)
        self.parameters['case_prob_symp'] = Beta(mean=0.62, st_dev=0.07) # SD based on 95% CI-- may need to be revised
        self.parameters['case_dur_symp'] = Gamma(mean=10, st_dev=3) # Based on CDC isolation guidelines, can be replaced by exp decay function

        # parameters to calculate the QALY loss due to a hospitalizations
        self.parameters['hosp_dur_stay'] = Gamma(mean=6, st_dev=4) # assuming SD = IQR/1.35
        self.parameters['hosp_weight'] = Beta(mean=0.5/365, st_dev=0.1/365)

        # parameters to calculate the QALY loss due to a death
        # TODO: to be consistent with how we have set up the other parameter above,
        #  we should read the data to characterise the Dirichlet distribution here from the csv files
        #  (instead of taking as an input).
        self.parameters['death_age_dist'] = Dirichlet(par_ns=nb_deaths_array)

        self.parameters['death_weight_by_age'] = ConstantArray(values=life_expectancy_array)

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
        self._calculate_qaly_loss_due_to_case(param=param)
        self._calculate_qaly_loss_due_to_hosp(param=param)
        self._calculate_qaly_loss_due_to_death(param=param)

        return param

    def _sample_parameters(self, rng):
        """
        samples all parameters
        """

        for par in self.parameters.values():
            par.sample(rng)

    def _calculate_qaly_loss_due_to_case(self, param):

        param.qWeightCase = (self.parameters['case_weight_symp'].value
                             * self.parameters['case_prob_symp'].value
                             * self.parameters['case_dur_symp'].value)

    def _calculate_qaly_loss_due_to_hosp(self, param):

        param.qWeightHosp = (self.parameters['hosp_dur_stay'].value
                              * self.parameters['hosp_weight'].value)

    def _calculate_qaly_loss_due_to_death(self, param):
        param.qWeightDeath = np.dot(
            self.parameters['death_age_dist'].value,
            self.parameters['death_weight_by_age'].value)
