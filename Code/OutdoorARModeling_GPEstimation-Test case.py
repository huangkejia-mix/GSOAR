
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


class OutdoorARModeling_GPEstimation:
    def test_estimate_incremental_displacement():
    ar_model = OutdoorARModeling_GPEstimation(
        rtk_positions=[(35.6601, -10.5903)],
        camera_frames=[],
        gravity_values=[]
    )
    world_transform = ar_model.estimate_incremental_displacement()
    assert world_transform is not None, "Incremental Displacement estimation failed."

def test_fuse_sensor_data():
    ar_model = OutdoorARModeling_GPEstimation(
        rtk_positions=np.array([[35.6601, -10.5903], [35.6675, -10.5895]]),
        camera_frames=[],
        gravity_values=[]
    )
    fused_position = ar_model.fuse_sensor_data()
    assert fused_position is not None, "Sensor fusion failed."

def test_estimate_global_pose():
    ar_model = OutdoorARModeling_GPEstimation(
        rtk_positions=[],
        camera_frames=[],
        gravity_values=[]
    )
    quaternion_pose = ar_model.estimate_global_pose()
    assert quaternion_pose is not None, "Global pose estimation failed."

# Run tests
test_estimate_incremental_displacement()

test_fuse_sensor_data()

test_estimate_global_pose()


print("All test cases running passed!")
















