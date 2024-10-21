
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import random
import time

import open3d as o3d

import chart_studio.plotly as py
import plotly.graph_objs as go

import numpy as np
import open3d as o3d
import copy
import time
import pandas as pd

import os
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree as KDTree
import numpy as np
from scipy.spatial.transform import Rotation as R




class ARExperiment_ThreeTypicalScenarios:
    def __init__(self):
        # initialize the experimental parameters of the three scenarios
        self.test_areas = ['slope', 'highway', 'building_occlusion']
        self.initialization_errors = {
            'slope': [0.122, 0.299],
            'highway': [0.087, 0.274],
            'building_occlusion': [0.095, 0.241]
        }
        self.gravity_errors = {
            'slope': [1.55, 5.11],
            'highway': [1.41, 1.85],
            'building_occlusion': [1.49, 1.75]
        }
        self.north_vector_errors = {
            'slope': [0.9, 3.11],
            'highway': [0.8, 1.87],
            'building_occlusion': [3.2, 2.00]
        }

    def evaluate_initialization_error(self):
        print("Initialization Error Results:")
        for area in self.test_areas:
            proposed, literature = self.initialization_errors[area]
            ratio = proposed / literature
            print(f"{area.capitalize()} area: Proposed: {proposed} m, Literature: {literature} m, Ratio: {ratio:.2f}")

    def run_tests(self):
        for scenario in self.scenarios:
            # Recording the continuous measurements of the outdoor test, assuming true values plus normally distributed noise, and analyzing the results
            measurements = np.random.normal(self.true_values[scenario], self.std_devs[scenario], self.num_trials)
            self.results[scenario] = measurements

    def evaluate_precision(self):
        print("Precision Evaluation Results:")
        for scenario in self.scenarios:
            mean = np.mean(self.results[scenario])
            std_dev = np.std(self.results[scenario])
            print(f"{scenario.capitalize()} area: Mean: {mean:.3f} m, Std Dev: {std_dev:.3f} m")

    def evaluate_stability(self):
        print("\nStability Evaluation Results:")
        for scenario in self.scenarios:
            stability_index = np.mean(np.abs(np.diff(self.results[scenario])))
            print(f"{scenario.capitalize()} area: Stability Index: {stability_index:.3f} m")

    def evaluate_gravity_vector_error(self):
        print(" Gravity Direction Vector Estimation Results:")
        for area in self.test_areas:
            proposed, literature = self.gravity_errors[area]
            ratio = proposed / literature
            print(f"{area.capitalize()} area: Proposed: {proposed}°, Literature: {literature}°, Ratio: {ratio:.2f}")

    def evaluate_north_vector_error(self):
        print(" True North Vector Estimation Results:")
        for area in self.test_areas:
            proposed, literature = self.north_vector_errors[area]
            ratio = proposed / literature
            print(f"{area.capitalize()} area: Proposed: {proposed}°, Literature: {literature}°, Ratio: {ratio:.2f}")



# execute three typical scenarios test, analyze the stability and accuracy results produced by the device,
experiment = ARExperiment_ThreeTypicalScenarios()

experiment.evaluate_initialization_error()

precision_stability_test.run_tests()

precision_stability_test.evaluate_precision()

precision_stability_test.evaluate_stability()










