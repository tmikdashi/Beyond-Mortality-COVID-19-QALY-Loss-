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
        self.parameters['case_prob_symp'] = Beta(mean=0.692, st_dev=0.115)  # SD based on 95% CI-- may need to be revised
        self.parameters['case_weight_symp'] = Beta(mean=0.43, st_dev=0.015)
        self.parameters['case_dur_symp'] = Gamma(mean=10/365.25, st_dev=1/365.25) # Based on CDC isolation guidelines, can be replaced by exp decay function

        # parameters to calculate the QALY loss due to a hospitalizations
        hosp_data = pd.read_csv(ROOT_DIR + '/csv_files/hosps_by_age.csv')
        self.parameters['hosp_dur_stay'] = Gamma(mean=6/365.25, st_dev=3.704/365.25) # assuming SD = IQR/1.35
        self.parameters['hosp_weight'] = Beta(mean=0.5, st_dev=0.05)
        self.parameters['hosps_age_dist'] = Dirichlet(par_ns=hosp_data['COVID-19 Hosps'])
        #self.parameters['hosps_prob_non_icu'] = 1 - (self.parameters['icu_prob'].value)
        #self.parameters['hosps_prob_non_icu'] = Beta(mean=0.8, st_dev=0.115) #TODO

        # parameters to calculate the QALY loss due to a death
        data = pd.read_csv(ROOT_DIR + '/csv_files/deaths_by_age.csv')
        self.parameters['death_age_dist'] = Dirichlet(par_ns=data['COVID-19 Deaths'])
        self.parameters['dQALY_loss_by_age'] = ConstantArray(values=[22.53, 20.89, 19.08, 16.96, 14.30, 11.52, 8.61, 5.50, 3.00, 1.46])
        self.parameters['Age Group']= ConstantArray(values=data['Age Group'])

        #parameters to calculate the QALY loss due to a ICU hopsitalization
        self.parameters['icu_prob'] = Beta(mean=0.174, st_dev = 0.02) #TODO: A revoir
        self.parameters['icu_weight']= Beta(mean=0.60, st_dev = 0.1)
        self.parameters['occupancy_dur'] = ConstantArray(values=(7/365))


        # parameters to calculate the QALY loss due to long COVID
        #self.parameters['long_covid_dur'] = Beta(mean=30/365.25, st_dev=1/365.25)
        self.parameters['long_covid_dur'] = Beta(mean=3.99/12, st_dev=(0.4/12)/1.35)
        #self.parameters['long_covid_prob'] = Beta(mean=0.343, st_dev=0.0065)
        self.parameters['long_covid_prob'] = Beta(mean=0.037, st_dev=(0.066/1.35))
        self.parameters['long_covid_weight'] = Beta(mean=0.29, st_dev =0.033)

        #Scenario 1: Long COVID parameters
        self.parameters['long_covid_prob_1'] = Beta(mean=0.062, st_dev = ((0.133-0.024)/1.35))
        self.parameters['long_covid_dur_1_nh'] = Beta(mean=4/12, st_dev=1/12)
        self.parameters['long_covid_weight_1'] = Beta (mean=0.22, st_dev=0.16/1.35)
        self.parameters['long_covid_weight_low_1'] = Beta(mean=0.019, st_dev=(0.039-0.011)/ 1.35)
        self.parameters['long_covid_weight_up_1'] = Beta(mean=0.408, st_dev=(0.556-0.273)/ 1.35)
        self.parameters['long_covid_dur_1_L'] = Beta(mean=(((4/12)*(0.95)) +((9/12)*0.05)), st_dev=1 / 12)

        # Scenario 2: Long COVID parameters
        self.parameters['long_covid_prob_pf_2'] = Beta(mean=0.032, st_dev=((0.1 - 0.006) / 1.35))
        self.parameters['long_covid_prob_resp_2'] = Beta(mean=0.037, st_dev=((0.096 - 0.009) / 1.35))
        self.parameters['long_covid_prob_cogn_2'] = Beta(mean=0.022, st_dev=((0.076 - 0.003) / 1.35))
        self.parameters['long_covid_dur_2_nh'] = Beta(mean=4 / 12, st_dev=(1 / 12))
        self.parameters['long_covid_weight_pf_2'] = Beta(mean=0.22, st_dev=(0.16 / 1.35))
        self.parameters['long_covid_weight_cogn_low_2'] = Beta(mean=0.07, st_dev=(0.05/ 1.35))
        self.parameters['long_covid_weight_resp_low_2'] = Beta(mean=0.02, st_dev=(0.03 / 1.35))
        self.parameters['long_covid_weight_cogn_up_2'] = Beta(mean=0.38, st_dev=(0.26 / 1.35))
        self.parameters['long_covid_weight_resp_up_2'] = Beta(mean=0.41, st_dev=(0.29 / 1.35))

        # Scenario 3: Long COVID parameters
        self.parameters['long_covid_prob_surv_nonhosp_3'] = ConstantArray(values=(1))
        self.parameters['long_covid_prob_surv_hosp_3'] = ConstantArray(values=(0.768))
        self.parameters['long_covid_prob_surv_icu_3'] = ConstantArray(values=(0.62))

        self.parameters['long_covid_prob_lc_nonhosp_3_low'] = Beta(mean=0.057, st_dev=((0.131 - 0.019) / 1.35))
        self.parameters['long_covid_prob_lc_hosp_3_low'] = Beta(mean=0.275, st_dev=((0.478 - 0.121) / 1.35))
        self.parameters['long_covid_prob_lc_icu_3_low'] = Beta(mean=0.431, st_dev=((0.652 - 0.226) / 1.35))

        self.parameters['long_covid_dur_nonhosp_3'] = Beta(mean=4 / 12, st_dev=(1 / 12))
        self.parameters['long_covid_dur_hosp_3'] = Beta(mean=9 / 12, st_dev=(5/ 12))
        self.parameters['long_covid_weight_pf_3'] = Beta(mean=0.22, st_dev=(0.16 / 1.35))

        self.parameters['long_covid_prob_lc_nonhosp_3_up'] = Beta(mean=0.16, st_dev=(0.016/1.35))
        self.parameters['long_covid_prob_lc_hosp_3_up'] = Beta(mean=0.60, st_dev=(0.06/1.35))
        self.parameters['long_covid_prob_lc_icu_3_up'] = Beta(mean=0.72, st_dev=(0.072/1.35))


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

        param.qWeightLongCOVID = ( self.parameters['long_covid_dur'].value
                                  * self.parameters['long_covid_prob'].value
                                  * self.parameters['long_covid_weight'].value)


        param.qWeightLongCOVID_1_c = (self.parameters['case_prob_symp'].value
                                      *self.parameters['long_covid_dur_1_nh'].value
                                      * self.parameters['long_covid_prob_1'].value
                                      * self.parameters['long_covid_weight_1'].value)

        param.qWeightLongCOVID_1_d = (self.parameters['long_covid_dur_1_nh'].value
                                      * self.parameters['long_covid_prob_1'].value
                                      * self.parameters['long_covid_weight_1'].value)

        param.qWeightLongCOVID_1_c_L = (self.parameters['case_prob_symp'].value
                                      * self.parameters['long_covid_dur_1_L'].value
                                      * self.parameters['long_covid_prob_1'].value
                                      * self.parameters['long_covid_weight_1'].value)

        param.qWeightLongCOVID_1_d_L = (self.parameters['long_covid_dur_1_L'].value
                                      * self.parameters['long_covid_prob_1'].value
                                      * self.parameters['long_covid_weight_1'].value)

        param.qWeightLongCOVID_1_c_low = (self.parameters['case_prob_symp'].value
                                     * self.parameters['long_covid_dur_1_nh'].value
                                     * self.parameters['long_covid_prob_1'].value
                                     * self.parameters['long_covid_weight_low_1'].value)

        param.qWeightLongCOVID_1_d_low = (self.parameters['long_covid_dur_1_nh'].value
                                      * self.parameters['long_covid_prob_1'].value
                                      * self.parameters['long_covid_weight_low_1'].value)

        param.qWeightLongCOVID_1_c_up = (self.parameters['case_prob_symp'].value
                                          * self.parameters['long_covid_dur_1_nh'].value
                                          * self.parameters['long_covid_prob_1'].value
                                          * self.parameters['long_covid_weight_up_1'].value)

        param.qWeightLongCOVID_1_d_up = (self.parameters['long_covid_dur_1_nh'].value
                                          * self.parameters['long_covid_prob_1'].value
                                          * self.parameters['long_covid_weight_up_1'].value)

        param.qWeightLongCOVID_2_c_low = (self.parameters['case_prob_symp'].value
                                      * ((self.parameters['long_covid_prob_pf_2'].value*self.parameters['long_covid_dur_1_nh'].value *self.parameters['long_covid_weight_pf_2'].value)
                                        +  (self.parameters['long_covid_prob_resp_2'].value*self.parameters['long_covid_dur_1_nh'].value *self.parameters['long_covid_weight_resp_low_2'].value)
                                         +(self.parameters['long_covid_prob_cogn_2'].value*self.parameters['long_covid_dur_1_nh'].value *self.parameters['long_covid_weight_cogn_low_2'].value)))


        param.qWeightLongCOVID_2_d_low = ((self.parameters['long_covid_prob_pf_2'].value*self.parameters['long_covid_dur_1_nh'].value *self.parameters['long_covid_weight_pf_2'].value)
                                        +  (self.parameters['long_covid_prob_resp_2'].value*self.parameters['long_covid_dur_1_nh'].value *self.parameters['long_covid_weight_resp_low_2'].value)
                                         +(self.parameters['long_covid_prob_cogn_2'].value*self.parameters['long_covid_dur_1_nh'].value *self.parameters['long_covid_weight_cogn_low_2'].value))

        param.qWeightLongCOVID_2_c_up = (self.parameters['case_prob_symp'].value
                                          * ((self.parameters['long_covid_prob_pf_2'].value * self.parameters['long_covid_dur_1_nh'].value * self.parameters['long_covid_weight_pf_2'].value)
                                             + (self.parameters['long_covid_prob_resp_2'].value * self.parameters['long_covid_dur_1_nh'].value * self.parameters['long_covid_weight_resp_up_2'].value)
                                             + (self.parameters['long_covid_prob_cogn_2'].value * self.parameters['long_covid_dur_1_nh'].value * self.parameters['long_covid_weight_cogn_up_2'].value)))

        param.qWeightLongCOVID_2_d_up = ((self.parameters['long_covid_prob_pf_2'].value * self.parameters['long_covid_dur_1_nh'].value * self.parameters['long_covid_weight_pf_2'].value)
                                          + (self.parameters['long_covid_prob_resp_2'].value * self.parameters['long_covid_dur_1_nh'].value * self.parameters['long_covid_weight_resp_up_2'].value)
                                          + (self.parameters['long_covid_prob_cogn_2'].value * self.parameters['long_covid_dur_1_nh'].value * self.parameters['long_covid_weight_cogn_up_2'].value))


        param.qWeightLongCOVID_3_c_low = (self.parameters['case_prob_symp'].value
                                      *  self.parameters['long_covid_dur_nonhosp_3'].value
                                      * self.parameters['long_covid_prob_surv_nonhosp_3'].value
                                      *  self.parameters['long_covid_prob_lc_nonhosp_3_low'].value
                                      * self.parameters['long_covid_weight_pf_3'].value)

        param.qWeightLongCOVID_3_ch_low = ( self.parameters['long_covid_dur_nonhosp_3'].value
                                      * self.parameters['long_covid_prob_surv_nonhosp_3'].value
                                      * self.parameters['long_covid_prob_lc_nonhosp_3_low'].value
                                      * self.parameters['long_covid_weight_pf_3'].value)

        param.qWeightLongCOVID_3_h_low = ((1.8/10.8)*self.parameters['long_covid_dur_hosp_3'].value
                                       * self.parameters['long_covid_prob_surv_hosp_3'].value
                                       * self.parameters['long_covid_prob_lc_hosp_3_low'].value
                                       *self.parameters['long_covid_weight_pf_3'].value)

        param.qWeightLongCOVID_3_icu_low = ((9.0/10.8)*self.parameters['long_covid_dur_hosp_3'].value
                                       * self.parameters['long_covid_prob_surv_icu_3'].value
                                       * self.parameters['long_covid_prob_lc_icu_3_up'].value
                                       * self.parameters['long_covid_weight_pf_3'].value)


        param.qWeightLongCOVID_3_c_up = (self.parameters['case_prob_symp'].value
                                      *  self.parameters['long_covid_dur_nonhosp_3'].value
                                      * self.parameters['long_covid_prob_surv_nonhosp_3'].value
                                      *  self.parameters['long_covid_prob_lc_nonhosp_3_up'].value
                                      * self.parameters['long_covid_weight_pf_3'].value)

        param.qWeightLongCOVID_3_ch_up = (self.parameters['long_covid_dur_nonhosp_3'].value
                                      * self.parameters['long_covid_prob_surv_nonhosp_3'].value
                                      *  self.parameters['long_covid_prob_lc_nonhosp_3_up'].value
                                      * self.parameters['long_covid_weight_pf_3'].value)

        param.qWeightLongCOVID_3_h_up = ((1.8/10.8)*self.parameters['long_covid_dur_hosp_3'].value
                                       * self.parameters['long_covid_prob_surv_hosp_3'].value
                                       * self.parameters['long_covid_prob_lc_hosp_3_up'].value
                                       *self.parameters['long_covid_weight_pf_3'].value)

        param.qWeightLongCOVID_3_icu_up = ((9.0/10.8)*self.parameters['long_covid_dur_hosp_3'].value
                                       * self.parameters['long_covid_prob_surv_icu_3'].value
                                       * self.parameters['long_covid_prob_lc_icu_3_up'].value
                                       * self.parameters['long_covid_weight_pf_3'].value)

