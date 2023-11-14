from deampy.parameters import Beta, Gamma


class ParameterValues:

    def __init__(self):

        self.qWeightCase = 0.1
        self.qWeightHosp = 0.3
        self.qWeightDeath = 10

    def __str__(self):
        return "qWeightCase: {:.4f}, qWeightHosp: {}, qWeightDeath: {}".format(
            self.qWeightCase, self.qWeightHosp, self.qWeightDeath)


class ParameterGenerator:

    def __init__(self):

        self.parameters = dict()

        # parameters to calculate the QALY loss due to a case
        self.parameters['case_weight_symp'] = Beta(mean=0.8, st_dev=0.1)
        self.parameters['case_prob_symp'] = Beta(mean=0.5, st_dev=0.1)
        self.parameters['case_dur_symp'] = Gamma(mean=14/364, st_dev=2/364)

        # parameters to calculate the QALY loss due to a hospitalizations
        # to be completed ...

        # parameters to calculate the QALY loss due to a death
        # to be completed ...

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
