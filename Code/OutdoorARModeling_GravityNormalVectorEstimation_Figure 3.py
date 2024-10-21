
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import random
import time

import open3d as o3d
import numpy as np
import ransac
from scipy.spatial.transform import Rotation as R


def is_surface_flat(error, tolerance):
    """
    determine whether the surface environment conforms to the approximate geoid level. if the error is less than the tolerance, return True; Otherwise, False is returned
    :param error: 
    :param tolerance: 
    :return: 
    """
    return error < tolerance

def fit_plane_with_ransac(points):
    """
    extract and fit the plane using the RANSAC method， return the fitted plane normal
    :param points: 
    :return: 
    """
    model, inliers = ransac.fit_plane(points)
    normal_vector = model[:3]  
    return normal_vector

def estimate_gravity_vector(rtkgps_measurements, tolerance):
    """
    estimate the gravity normal vector based on RTK-GPS measurements
    :param rtkgps_measurements
    :param tolerance: 
    :return: 
    """
    gravity_vector = np.mean(rtkgps_measurements, axis=0)  # calculate the average gravity vector
    error = np.linalg.norm(gravity_vector)  # calculation error
    return gravity_vector if error < tolerance else None

def compute_gravity_normal_vector(points, rtkgps_measurements, tolerance):
    """
    the gravity normal vector is calculated according to the RTK-GPS high-precision multi-point horizontal positioning results
    :param points: points collection
    :param rtkgps_measurements: RTK-GPS measurement results
    :param tolerance: 
    :return: 
    """
     # Step 1: Determine the surface environment
     # hypothetical measurement error
     error = np.random.uniform(0, 0.5) 
    if is_surface_flat(error, tolerance):
        # Step 2: Initialize the surface environment based on SLAM motion tracking
        normal_vector = fit_plane_with_ransac(points)
        
       	Step 3: Combine the RTK-GPS measurement normals to calculate the gravity normals
        gravity_vector = normal_vector  # Suppose the gravity normal is normal
    else:
        # Step 4: Calculate gravity using a gravity sensor combined with RTK-GPS location
        gravity_vector = estimate_gravity_vector(rtkgps_measurements, tolerance)
        
        # Check if the gravity normal vector can be calculated
        if gravity_vector is None:
            print("Gravity normal vector estimation failed, please re-measure.")
            return None
        
    # Step 5: Construct a quaternion with a belt direction
    quat = R.from_rotvec(gravity_vector).as_quat()  # Convert the gravity normal vector to a quaternion
    return quat


# test cases
# randomly generate 100 3D points
points = np.random.rand(300, 3)  
# randomly generated 10 RTK-GPS measurements
rtkgps_measurements = np.random.rand(20, 3)  
# Set the tolerance for accuracy judgment
tolerance = 0.3

# Calculate the gravity normal
gravity_normal_vector = compute_gravity_normal_vector(points, rtkgps_measurements, tolerance)

# output the result
print("Quaternion representation of the gravity normal: ", gravity_normal_vector)



















