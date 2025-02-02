
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import random
import time


import numpy as np

# pip install geopandas pyproj PyOpenGL

import geopandas as gpd
from pyproj import Proj, transform
import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *




# Step 1: Read the Shapefile
def read_shp_file(shp_path):
    """
    Reads a shp file and returns a GeoDataFrame.
    
    Parameters:
    shp_path (str): The path to the shp file.
    
    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing building information.
    """
    gdf = gpd.read_file(shp_path)
    return gdf

# Step 2: Transform coordinates
def transform_coordinates(gdf, src_crs, dst_crs):
    """
    Transforms coordinates from the source coordinate system to the destination coordinate system.
    
    Parameters:
    gdf (geopandas.GeoDataFrame): A GeoDataFrame containing building information.
    src_crs (str): The EPSG code of the source coordinate system.
    dst_crs (str): The EPSG code of the destination coordinate system.
    
    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame with transformed coordinates.
    """
    gdf = gdf.to_crs(dst_crs)
    return gdf

# Step 3: Update building coordinates
def update_building_coordinates(gdf, slam_coordinates):
    """
    Updates the 3D display coordinates of each building based on the SLAM real-time estimated local coordinates.
    
    Parameters:
    gdf (geopandas.GeoDataFrame): A GeoDataFrame containing building information.
    slam_coordinates (dict): A dictionary of SLAM real-time estimated local coordinates.
    
    Returns:
    list: A list of updated 3D display coordinates for each building.
    """
    updated_coordinates = []
    for idx, row in gdf.iterrows():
        building_id = row['id']  # Assuming each building has a unique ID
        local_coords = slam_coordinates.get(building_id)
        if local_coords:
            # Convert local coordinates to latitude and longitude
            lat, lon = transform(src_proj, dst_proj, local_coords[0], local_coords[1])
            altitude = row['geometry'].coords[0][2]  # Assuming elevation is stored in the Z coordinate
            updated_coordinates.append((lat, lon, altitude))
    return updated_coordinates

# Step 4: Display each building's name and bounding box using OpenGL
def display_buildings(coordinates):
    """
    Displays the name and bounding box of each building using OpenGL.
    
    Parameters:
    coordinates (list): A list containing the 3D display coordinates of each building.
    """
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"Buildings Visualization")
    
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    for (lat, lon, altitude) in coordinates:
        # Convert latitude and longitude to OpenGL coordinates
        x = lon
        y = lat
        z = altitude
        
        # Draw the building's name
        glPushMatrix()
        glTranslatef(x, y, z)
        glutStrokeCharacter(font, ord('A'))  # Example: Draw the letter 'A'
        glPopMatrix()
        
        # Draw the building's bounding box
        glPushMatrix()
        glTranslatef(x, y, z)
        glBegin(GL_QUADS)
        glVertex3f(-1.0, -1.0, -1.0)
        glVertex3f(1.0, -1.0, -1.0)
        glVertex3f(1.0, 1.0, -1.0)
        glVertex3f(-1.0, 1.0, -1.0)
        glEnd()
        glPopMatrix()
    
    glFlush()
    glutMainLoop()


# Main program
if __name__ == "__main__":
    # Define source and destination coordinate systems
    src_proj = Proj(init='epsg:4326')  # WGS84
    dst_proj = Proj(init='epsg:3857')  # Web Mercator
    
    # Read shp file
    shp_path = 'industrial_park_from_Sichuan.shp'  # Path to the shp file
    gdf = read_shp_file(shp_path)
    
    # Transform coordinates
    gdf = transform_coordinates(gdf, src_crs='epsg:4326', dst_crs='epsg:3857')
    
    # Assume SLAM real-time estimated local coordinates
    slam_coordinates = {
        'building_1': (100, 200, 30),
        'building_2': (150, 250, 35),
        'building_2': (230, 405, 27),
        # ...
    }
    
    # Update building coordinates
    updated_coordinates = update_building_coordinates(gdf, slam_coordinates)
    
    # Display each building's name and bounding box using OpenGL
    display_buildings(updated_coordinates)




















