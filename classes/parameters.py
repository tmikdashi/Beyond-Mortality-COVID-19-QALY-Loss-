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
        self.parameters['cases_prob_symp'] = Beta(mean=0.692, st_dev=0.115)
        self.parameters['case_dur_symp'] = Gamma(mean=7/365.25, st_dev=0.5/365.25)
        self.parameters['case_weight_symp'] = Beta(mean=0.13, st_dev=0.02) #updated value

        self.parameters['prob_surv'] =ConstantArray(values=0.983)

        # parameters to calculate the QALY loss due to a hospitalizations + ICU
        hosp_data = pd.read_csv(ROOT_DIR + '/csv_files/hosps_by_age.csv')
        self.parameters['hosp_dur_stay_ICU'] = Gamma(mean=8.3/365.25, st_dev=0.875/365.25) #TODO: A REVOIR LES VALEURS SD
        self.parameters['hosp_dur_stay_ward'] = Gamma(mean=6/365.25, st_dev=3.7/365.25)
        self.parameters['hosp_weight'] = Beta(mean=0.5, st_dev=0.05)
        self.parameters['hosps_age_dist'] = Dirichlet(par_ns=hosp_data['COVID-19 Hosps'])
        self.parameters['hosps_prob_surv'] = Beta(mean=0.82, st_dev=0.02)

        self.parameters['icu_prob'] = ConstantArray(0.174)
        #self.parameters['icu_weight'] = Beta(mean=0.60, st_dev=0.1)
        self.parameters['icu_weight'] = Beta(mean=0.70, st_dev= 0.10) #updated value
        self.parameters['occupancy_dur'] = ConstantArray(values=(7 / 365))

        # parameters to calculate the QALY loss due to a death
        data = pd.read_csv(ROOT_DIR + '/csv_files/deaths_by_age.csv')
        self.parameters['death_age_dist'] = Dirichlet(par_ns=data['COVID-19 Deaths'])
        self.parameters['dQALY_loss_by_age'] = ConstantArray(values=[22.53, 20.89, 19.08, 16.96, 14.30, 11.52, 8.61, 5.50, 3.00, 1.46]) # TODO
        self.parameters['dQALY_loss_by_age_smr_1.75_qcm_0.85_r_3'] = ConstantArray(values=[22.94,20.85,18.78,16.55,13.91,11.09,8.22,5.17,2.82,1.27])
        self.parameters['dQALY_loss_by_age_smr_1.75_qcm_0.8_r_3'] = ConstantArray(values=[21.59,19.63,17.67,15.58,13.10,10.43,7.74,4.87,2.65,1.20])
        self.parameters['dQALY_loss_by_age_smr_1.75_qcm_0.75_r_3'] = ConstantArray(values=[20.24, 18.40, 16.57, 14.60, 12.28, 9.78, 7.26, 4.57, 2.49,1.12])

        self.parameters['dQALY_loss_by_age_smr_2_qcm_0.85_r_3'] = ConstantArray(values=[22.76,20.62,18.50,16.24,13.56,10.71,7.87,4.88,2.61,1.15])
        self.parameters['dQALY_loss_by_age_smr_2_qcm_0.8_r_3'] = ConstantArray(values=[21.42, 19.41, 17.41, 15.28, 12.76, 10.08, 7.41, 4.59, 2.45, 1.08])
        self.parameters['dQALY_loss_by_age_smr_2_qcm_0.75_r_3'] = ConstantArray(values=[20.08,18.20,16.33,14.33,11.96,9.45,6.95,4.30,2.30,1.02])

        self.parameters['dQALY_loss_by_age_smr_2.25_qcm_0.85_r_3'] = ConstantArray(values=[22.60,20.41,18.25,15.95,13.24,10.37,7.56,4.62,2.43,1.05])
        self.parameters['dQALY_loss_by_age_smr_2.25_qcm_0.8_r_3'] = ConstantArray(values=[21.27,19.21,17.17,15.01,12.46,9.76,7.12,4.35,2.28,0.99])
        self.parameters['dQALY_loss_by_age_smr_2.25_qcm_0.75_r_3'] = ConstantArray(values=[19.94,18.01,16.10,14.07,11.68,9.15,6.67,4.08,2.14,0.93])
        self.parameters['Age Group']= ConstantArray(values=data['Age Group'])

        # LONG COVID 1: parameters to calculate the QALY loss due to long COVID
        self.parameters['long_covid_dur'] = Beta(mean=4/12, st_dev=(1/48))
        #self.parameters['case_prob_symp'] = Beta(mean=0.692, st_dev=0.115)
        self.parameters['long_covid_prob'] = Beta(mean=0.062, st_dev=0.0273)
        self.parameters['long_covid_weight'] = Beta(mean=0.29, st_dev =0.0275)
        self.parameters['long_covid_prob_v_red'] = Beta(mean=0.872, st_dev=0.0306)

        # LONG COVID  2: parameters to calculate the QALY loss due to long COVID
        self.parameters['cases_prob_hosp'] = ConstantArray(values=0.058)
        # self.parameters['case_prob_symp'] = Beta(mean=0.692, st_dev=0.115)
        self.parameters['long_covid_nonhosp_prob_surv'] = ConstantArray(values=1)
        self.parameters['long_covid_nonhosp_prob_symp'] = Beta(mean=0.057, st_dev=0.022)
        self.parameters['long_covid_nonhosp_dur'] = Beta(mean=4/12, st_dev=(1/48))

        self.parameters['long_covid_hosp_prob_surv'] = Beta(mean=0.82, st_dev=0.02)
        self.parameters['long_covid_hosp_prob_symp'] = Beta(mean=0.275, st_dev=0.089)
        self.parameters['long_covid_hosp_dur'] = Beta(mean=9/12, st_dev=1.25/12)

        self.parameters['long_covid_icu_prob_surv'] = Beta(mean=0.61, st_dev=0.09)
        self.parameters['long_covid_icu_prob_symp'] =Beta(mean=0.431, st_dev=0.107)

        # Vax modification: parameters to calculate impact of vax on QALY loss due to long COVID
        self.parameters['reduction_long_prob'] = Beta(mean=0.872, st_dev=0.006)







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
                              *self.parameters['cases_prob_symp'].value
                             * self.parameters['case_dur_symp'].value)

        param.qWeightHosp = ((1-self.parameters['icu_prob'].value)
                             * self.parameters['hosp_dur_stay_ward'].value
                             * self.parameters['hosp_weight'].value)

        param.qWeightICUHosp = (self.parameters['icu_prob'].value
                                * self.parameters['hosp_dur_stay_ICU'].value
                                * self.parameters['hosp_weight'].value)

        param.qWeightICU = (self.parameters['icu_weight'].value
                            * self.parameters['occupancy_dur'].value)

        param.qWeightDeath = np.dot(self.parameters['death_age_dist'].value, self.parameters['dQALY_loss_by_age'].value)

        param.qWeightDeath_sa_1_a = np.dot(self.parameters['death_age_dist'].value, self.parameters['dQALY_loss_by_age_smr_1.75_qcm_0.85_r_3'].value)
        param.qWeightDeath_sa_1_b = np.dot(self.parameters['death_age_dist'].value, self.parameters['dQALY_loss_by_age_smr_1.75_qcm_0.8_r_3'].value)
        param.qWeightDeath_sa_1_c = np.dot(self.parameters['death_age_dist'].value, self.parameters['dQALY_loss_by_age_smr_1.75_qcm_0.75_r_3'].value)

        param.qWeightDeath_sa_2_a = np.dot(self.parameters['death_age_dist'].value,self.parameters['dQALY_loss_by_age_smr_2_qcm_0.85_r_3'].value)
        param.qWeightDeath_sa_2_b = np.dot(self.parameters['death_age_dist'].value,self.parameters['dQALY_loss_by_age_smr_2_qcm_0.8_r_3'].value)
        param.qWeightDeath_sa_2_c = np.dot(self.parameters['death_age_dist'].value,self.parameters['dQALY_loss_by_age_smr_2_qcm_0.75_r_3'].value)

        param.qWeightDeath_sa_3_a = np.dot(self.parameters['death_age_dist'].value,self.parameters['dQALY_loss_by_age_smr_2.25_qcm_0.85_r_3'].value)
        param.qWeightDeath_sa_3_b = np.dot(self.parameters['death_age_dist'].value,self.parameters['dQALY_loss_by_age_smr_2.25_qcm_0.8_r_3'].value)
        param.qWeightDeath_sa_3_c = np.dot(self.parameters['death_age_dist'].value,self.parameters['dQALY_loss_by_age_smr_2.25_qcm_0.75_r_3'].value)

        param.qWeightLongCOVID_1 = (self.parameters['cases_prob_symp'].value
                                      * self.parameters['long_covid_prob'].value
                                      * self.parameters['prob_surv'].value
                                      * self.parameters['long_covid_dur'].value
                                      * self.parameters['long_covid_weight'].value)

        param.qWeightLongCOVID_1_v = (self.parameters['cases_prob_symp'].value
                                  *  self.parameters['long_covid_prob'].value
                                  * self.parameters['prob_surv'].value
                                  * self.parameters['long_covid_dur'].value
                                  * self.parameters['long_covid_weight'].value
                                 * self.parameters['long_covid_prob_v_red'].value)

        param.qWeightLongCOVID_1_uv = (self.parameters['cases_prob_symp'].value
                                    * self.parameters['long_covid_prob'].value
                                    * self.parameters['prob_surv'].value
                                    * self.parameters['long_covid_dur'].value
                                    * self.parameters['long_covid_weight'].value)


        param.qWeightLongCOVID_2_nh = ( (1-self.parameters['cases_prob_hosp'].value)
                                      * self.parameters['cases_prob_symp'].value
                                      * self.parameters['long_covid_nonhosp_prob_surv'].value
                                      * self.parameters['long_covid_nonhosp_prob_symp'].value
                                      * self.parameters['long_covid_nonhosp_dur'].value
                                      * self.parameters['long_covid_weight'].value)


        param.qWeightLongCOVID_2_h = ( (1-self.parameters['icu_prob'].value)
                                      * self.parameters['long_covid_hosp_prob_surv'].value
                                      * self.parameters['long_covid_hosp_prob_symp'].value
                                      * self.parameters['long_covid_hosp_dur'].value
                                      * self.parameters['long_covid_weight'].value)

        param.qWeightLongCOVID_2_i = ( self.parameters['icu_prob'].value
                                      * self.parameters['long_covid_icu_prob_surv'].value
                                      * self.parameters['long_covid_icu_prob_symp'].value
                                      * self.parameters['long_covid_hosp_dur'].value
                                      * self.parameters['long_covid_weight'].value)

        param.qWeightLongCOVID_1_c = (self.parameters['cases_prob_symp'].value
                                    * self.parameters['long_covid_prob'].value
                                    * self.parameters['long_covid_dur'].value
                                    * self.parameters['long_covid_weight'].value)

        param.qWeightLongCOVID_1_d = (self.parameters['long_covid_prob'].value
                                      * self.parameters['long_covid_dur'].value
                                      * self.parameters['long_covid_weight'].value)


        param.qWeightLongCOVID_1_uv = (self.parameters['cases_prob_symp'].value
                                       *self.parameters['long_covid_prob'].value
                                       * self.parameters['prob_surv'].value
                                       * self.parameters['long_covid_dur'].value
                                       * self.parameters['long_covid_weight'].value)


        param.qWeightLongCOVID_1_v = (  self.parameters['cases_prob_symp'].value
               * self.parameters['long_covid_prob'].value
                * self.parameters['prob_surv'].value
                * self.parameters['long_covid_dur'].value
                * self.parameters['long_covid_weight'].value
                * self.parameters['reduction_long_prob'].value)


        param.qWeightCase_symp = (self.parameters['cases_prob_symp'].value)





