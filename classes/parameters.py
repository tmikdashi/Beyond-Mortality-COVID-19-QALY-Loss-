
from math import sumprod

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

    def __init__(self):

        self.parameters = dict()

        # parameters to calculate the QALY loss due to a case
        self.parameters['case_weight_symp'] = Beta(mean=0.43, st_dev=0.03)
        self.parameters['case_prob_symp'] = Beta(mean=0.62, st_dev=0.07)
        self.parameters['case_dur_symp'] = Gamma(mean=14/364, st_dev=2/364)

        # parameters to calculate the QALY loss due to a hospitalizations
        self.parameters['hosp_dur_stay'] = Gamma(mean=6., st_dev=1.25) #updated value
        self.parameters['hosp_weight'] = Beta(mean=0.5, st_dev=0.1)

        # parameters to calculate the QALY loss due to a death
        self.parameters['death_age_dist'] = Dirichlet(
            par_ns=[10, 20, 30]) # this is the number of deaths in each age group
        self.parameters['death_weight_by_age'] = ConstantArray(
            values=[80, 60, 40]) # this is the life-expectancy in each age group

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
        param.qWeightDeath = sumprod(
            self.parameters['death_age_dist'].value,
            self.parameters['death_weight_by_age'].value)
