
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import random
import time

import open3d as o3d


def test_estimate_true_north_vector():
    ar_model = OutdoorARModeling_SEPerception(
        rtk_positions=[(35.6601, -10.5903)],  # Example RTK positions
        camera_frames=[],  # Dummy data
        gravity_values=[]  # Dummy data
    )
    true_north = ar_model.estimate_true_north_vector()

    assert true_north is not None, "True North Vector estimation failed."


def test_estimate_gravity_normal_vector():
    ar_model = OutdoorARModeling_SEPerception(
        rtk_positions=[(35.6601, -10.5903)],
        camera_frames=[],
        gravity_values=[9.81]  # Example gravity values
    )
    gravity_normal = ar_model.estimate_gravity_normal_vector()

    assert np.allclose(gravity_normal, np.array([0, 0, -1])), "Gravity Normal Vector estimation failed."


def test_estimate_coordinate_anchor():
    ar_model = OutdoorARModeling_SEPerception(
        rtk_positions=[(35.6601, -10.5903)],
        camera_frames=[],
        gravity_values=[]
    )
    coordinate_anchor = ar_model.estimate_coordinate_anchor()

    assert coordinate_anchor is not None, "Coordinate Anchor estimation failed."


# Run tests
test_estimate_true_north_vector()

test_estimate_gravity_normal_vector()

test_estimate_coordinate_anchor()

print("All tests result passed.")























