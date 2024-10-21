
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import random
import time

import open3d as o3d
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
from scipy.spatial.transform import Rotation as R


class OutdoorARModeling_GVRMatching:
    
    def __init__(self):
        pass

    def se3_coordinate_transform(self, p_W, p_0, R_LTP):
        """
        the local tangent plane coordinate transformation model of SE(3) of the device in real-time motion is calculated
        """
        p_LTP = R_LTP.T @ (p_W - p_0)
        return p_LTP


    def compute_geodesic_scale(self, R, delta_R):
        """
        Calculate the global geodesic scale for the area where the device is located
        """
        if delta_R == 0:
            raise ValueError("Delta R cannot be zero.")
        s = R / delta_R
        return s


    def virtual_geospatial_anchor(self, p_W, p_V0, R_VW, p_0):
        """
        Compute the virtual geospatial anchor after the initialization of the device,
        """
        p_VW = R_VW.T @ (p_W - p_0) + p_V0
        return p_VW























