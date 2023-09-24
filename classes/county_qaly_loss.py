import csv
import numpy as np

class CountyQALYLoss:
    def __init__(self, state, county):
        """
        :param state: (string) state name
        :param county: (string) county name
        """
        self.state = state
        self.county = county

        self.weeklyCases = []  # list of weekly cases
        self.weeklyQALYLoss = []  # list of weekly QALY loss
        self.qalyLoss = 0  # total QALY loss

    def add_traj(self, weekly_cases):
        """
        :param weekly_cases: (list or numpy.array) list of weekly cases
        """

        # convert weekly_cases to numpy.array
        if not isinstance(weekly_cases, np.ndarray):
            weekly_cases = np.array(weekly_cases)

        self.weeklyCases = weekly_cases

    def calculate_weekly_qaly_loss(self, case_weigh):
        """
        calculates the weekly and accumulated QALYs lost
        :param case_weigh: (float) quality weight of a COVID-19 case
        """

        # weekly QUALY loss
        self.weeklyQALYLoss = case_weigh * self.weeklyCases

        # accumulated QALY loss
        self.qalyLoss = sum(self.weeklyQALYLoss)