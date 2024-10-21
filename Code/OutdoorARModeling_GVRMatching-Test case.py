
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



def test_se3_coordinate_transform():
    ar_model = OutdoorARModeling_GVRMatching()
    p_W = np.array([1, 2, 3])
    p_0 = np.array([0, 0, 0])
    R_LTP = np.eye(3)  # Set the identity matrix as an example

    p_LTP = ar_model.se3_coordinate_transform(p_W, p_0, R_LTP)
    assert np.array_equal(p_LTP, p_W), "SE(3) Coordinate Transform failed."

def test_compute_geodesic_scale():
    ar_model = OutdoorARModeling_GVRMatching()
    R = 1.0  # the  center angle unit
    delta_R = 0.1  #  the set unit of surface distance

    scale = ar_model.compute_geodesic_scale(R, delta_R)
    assert scale == 10.0, "Geodesic scale calculation failed."

def test_virtual_geospatial_anchor():
    ar_model = OutdoorARModeling_GVRMatching()
    p_W = np.array([1, 2, 3])
    p_V0 = np.array([0, 0, 0])
    R_VW = np.eye(3)  # Set the identity matrix as an example
    p_0 = np.array([0, 0, 0])

    p_VW = ar_model.virtual_geospatial_anchor(p_W, p_V0, R_VW, p_0)
    assert np.array_equal(p_VW, p_W), "Virtual geospatial anchor calculation failed."

# Running test for every function 
test_se3_coordinate_transform()

test_compute_geodesic_scale()

test_virtual_geospatial_anchor()


print("All test cases running passed!")


















