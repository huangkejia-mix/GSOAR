
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




def estimate_coordinate_anchor(current_position, height_info, point_cloud_data):
    """
    Estimate the coordinates of the virtual geospatial anchor.
    
    Parameters:
    current_position (array): Current ground coordinates.
    height_info (float): Current height information.
    point_cloud_data (list): Three-dimensional point cloud data.
    
    Returns:
    coordinate_anchor (array): Estimated coordinate anchor point.
    """

    # Simplified method to calculate the ground equation

    # Assume point_cloud_data is a (n, 3) numpy array
    z_values = point_cloud_data[:, 2]

    ground_level = np.mean(z_values)
    
    coordinate_anchor = np.array(current_position)

    coordinate_anchor[2] = ground_level + height_info  # Update z coordinate

    return coordinate_anchor


def calculate_slam_global_error(estimated_poses, ground_truth_poses):
    """
    Calculates the global error between the local pose and the ground truth pose estimated by a batch of monocular SLAM.
    
    Parameters:
    estimated_poses (list or array): SLAM estimated poses, shape (n, 3) or (n, 4), where n is the number of data points.
    ground_truth_poses (list or array): Ground truth poses, shape (n, 3) or (n, 4).
    
    Returns:
    global_errors (array): Global errors for each position point.
    """

    estimated_poses = np.array(estimated_poses)

    ground_truth_poses = np.array(ground_truth_poses)
    
    # Calculate the error
    global_errors = np.linalg.norm(estimated_poses - ground_truth_poses, axis=1)

    return global_errors


def coordinate_transform(p_world, R_LTP, p_0):
    """
    A global coordinate transformation model based on SE(3).
    
    Parameters:
    p_world (array): Position in the world coordinate system.
    R_LTP (array): Rotation matrix from the world coordinate system to the local tangent plane coordinate system.
    p_0 (array): Origin of the local tangent plane coordinate system.
    
    Returns:
    p_LTP (array): Position in the local tangent plane coordinate system.
    """

    p_LTP = np.dot(R_LTP.T, (p_world - p_0))

    return p_LTP


def calculate_RTK_GPS_real_time_diff_std(rtk_positions):
    """
    Calculate the standard deviation of a batch of RTK-GPS real-time differential position data.
    
    Parameters:
    rtk_positions (list or array): RTK-GPS position data, shape (n, 3), where n is the number of data points.
    
    Returns:
    std_deviation (array): Standard deviation for each coordinate axis.
    """

    rtk_positions = np.array(rtk_positions)

    std_deviation = np.std(rtk_positions, axis=0)

    return std_deviation


def estimate_gravity_normal(rtk_positions, gravity_measurements):
    """
    Based on the numerical fitting of multiple high-quality RTK-GPS positioning, estimate gravity normal vectors.
    
    Parameters:
    rtk_positions (list): RTK-GPS position data.
    gravity_measurements (list): Gravity acceleration measurements.
    
    Returns:
    gravity_normal (array): Estimated gravity normal vector.
    """

    # Assume gravity acceleration values are normalized
    normals = []

    for i in range(len(rtk_positions)):
        p_i = np.array(rtk_positions[i])
        g_i = np.array(gravity_measurements[i])

        normalized_vector = (p_i - np.mean(rtk_positions, axis=0)) / np.linalg.norm(p_i - np.mean(rtk_positions, axis=0))

        normals.append(normalized_vector * g_i)
    
    gravity_normal = np.mean(normals, axis=0)

    return gravity_normal / np.linalg.norm(gravity_normal)


def estimate_true_north(rtk_positions, slam_trajectory):
    """
    Based on the fixed solution RTK-GPS values and SLAM to estimate the true north vector.
    
    Parameters:
    rtk_positions (list): RTK-GPS position data.
    slam_trajectory (list): SLAM motion trajectory data.
    
    Returns:
    north_vector (array): Estimated true north vector.
    """

    # Simple example: Calculate average direction as the true north vector
    average_direction = np.mean(slam_trajectory, axis=0)

    north_vector = average_direction / np.linalg.norm(average_direction)

    return north_vector


def estimate_global_pose(sensor_data, global_scale_model):
    """
    Estimate the global attitude based on multiple heterogeneous sensor data and global-scale measurement models.
    
    Parameters:
    sensor_data (dict): Position information provided by sensors.
    global_scale_model (object): Global scale model.
    
    Returns:
    global_pose (array): Estimated global pose.
    """

    position = sensor_data['position']

    orientation = sensor_data['orientation']
    
    # Assume global_scale_model has a method to update position and attitude
    global_pose = global_scale_model.update_pose(position, orientation)

    return global_pose











