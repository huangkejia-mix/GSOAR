
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import random
import time

import open3d as o3d

# pip
# install
# open3d == 0.12
# .0
#
# !pip
# install
# chart_studio

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
import cv2

class OutdoorARModeling_SEPerception:
    
    def __init__(self, rtk_positions, camera_frames, gravity_values):
        self.rtk_positions = rtk_positions  # RTK GPS positions
        self.camera_frames = camera_frames    # Camera frames for SLAM
        self.gravity_values = gravity_values   # Gravity measurements
    
    def calculate_lat_lon(gps_data):
        # to calculate the average latitude and longitude based on real-time differential GNSS data
        lat_sum = sum([point[0] for point in gps_data])
        lon_sum = sum([point[1] for point in gps_data])
        return lat_sum / len(gps_data), lon_sum / len(gps_data)

    def calculate_camera_trajectory(slam_data):
	# to calculate the average camera motion trajectory of the SLAM data
        return np.mean(slam_data, axis=0)

    def estimate_true_north_vector(self):
        # 1. Estimate True North Vector
        latitude, longitude = self.get_rtk_position()
        camera_trajectory = self.compute_camera_trajectory()
        
        # Assuming a formula to estimate True North Vector
        true_north_vector = self.calculate_true_north(latitude, longitude, camera_trajectory)
        return true_north_vector
    
    def estimate_gravity_normal_vector(self):
        # 2. Estimate Gravity Normal Vector
        latitude, longitude = self.get_rtk_position()
        gravity_vector = self.measure_gravity(latitude, longitude)
        
        # Fit plane using least squares
        gravity_normal_vector = self.fit_gravity_normal_vector(gravity_vector)
        return gravity_normal_vector
    
    def estimate_coordinate_anchor(self):
        # 3. Estimate Coordinate Anchor
        ground_coordinates, height = self.get_ground_info()
        coordinate_anchor = self.calculate_coordinate_anchor(ground_coordinates, height)
        return coordinate_anchor
    
    def get_rtk_position(self):
        # Placeholder for RTK position retrieval
        return self.rtk_positions[-1]
    
    def compute_camera_trajectory(self):
        # Placeholder for SLAM camera trajectory computation
        return np.array([0, 0, 0])  # Dummy trajectory
    
    def calculate_true_north(self, latitude, longitude, camera_trajectory):
        #  calculation the real-time value of true_north
        return np.array([1, 0, 0])  # Simplified representation
    
    def measure_gravity(self, latitude, longitude):
        #  gravity measurement
        return np.array([0, 0, -9.81])  # Simplified representation
    
    def fit_gravity_normal_vector(self, gravity_vector):
        #  fitting process
        return gravity_vector / np.linalg.norm(gravity_vector)
    
    def get_ground_info(self):
        # ground information retrieval
        return np.array([0, 0]), 0  # Simplified representation
    
    def calculate_coordinate_anchor(self, ground_coordinates, height):
        #  coordinate anchor calculation
        return np.array([ground_coordinates[0], ground_coordinates[1], height])


