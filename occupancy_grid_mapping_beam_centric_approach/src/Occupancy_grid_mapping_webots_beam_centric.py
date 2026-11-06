"This file is meant to implement the grid based mapping algorithm on beam centric based model"
"The paradigm is adapted "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval 
# from rclpy import Node

class Map :
    def __init__(self, map_length, map_width, cell_length, cell_width, lo, l_free, l_occ) :
        # Initialization

        # Map Dimensions
        self.__map_length = map_length
        self.__map_width = map_width

        # Cell Dimensions
        self.__cell_length = cell_length
        self.__cell_width = cell_width

        # Log log_oddss
        self.lo = lo
        self.l_free = l_free
        self.l_occ = l_occ


        # Map Rows and Columns
        if not(self.get_row_and_col()) :
            self.__N = 0    
            self.__M = 0

        else :
            self.__N, self.__M = self.get_row_and_col()

        # Set Initial log_odds
        self.set_initial_log_odds()


    def set_initial_log_odds(self) :
        "Sets 1/2 as the Initial log_odds to each cell"

        self.log_odds = np.full((self.__N, self.__M), self.lo, dtype = float)
        

    def rows(self) :
        "RETURNS THE PRIVATED FIELD VARIABLE N"
        return self.__N

    def cols(self) :
        "RETURNS THE PRIVATE FIELD VARIABLE M"
        return self.__M

    def get_row_and_col(self) :
        "Returns the number of Rows and Columns of the map"

        N = 0
        M = 0
        # Check if the parameters are valid
        if np.min((self.__map_length, self.__map_width, self.__cell_length, self.__cell_width)) <= 0 :
            return None

        # Calculate the number of Rows and Columns
        
        N = self.__map_length/self.__cell_length
        M = self.__map_width/self.__cell_width

        if not (np.isclose(N, round(N)) and np.isclose(M, round(M))):
            raise ValueError("Map dimensions must be divisible by cell dimensions")
    
        return int(round(N)), int(round(M))


    def get_matrix_indices(self, i) :
        """ Returns the row number and column number of the matrix of size NxM"
        n  {0, 1, 2, 3, 4, ............, N - 1}
        m  {0, 1, 2, 3, 4, ............, M - 1}
            """
    
        m = i%self.__M
        n = (i - m)//self.__M

        return n, m 

    def get_cell_center(self, i) :
        "Returns the center points pf ith cell"
        n, m = self.get_matrix_indices(i)

        x = -self.__map_length/2 + (n + 0.5)*self.__cell_length
        y = -self.__map_width/2 + (m + 0.5)*self.__cell_width

        return x, y

    def get_cell_matrix_indices_from_cell_position(self, x, y) :
        "Returns the cell matrix index based on the cells COM"

        # Get The Column and Row 
        m = int(np.floor((y + self.__map_width/2)/self.__cell_width))
        n = int(np.floor((x + self.__map_length/2)/self.__cell_length))

        if not (0 <= n < self.__N and 0 <= m < self.__M)  :
            return None

        return n, m

    def get_beam_end_points(self, x_t, range, bearing) :
        "Returns the coordinates of beam in World frame"
        z_x = x_t[0] + range*np.cos(bearing + x_t[2])
        z_y = x_t[1] +  range*np.sin(bearing + x_t[2])    
        return [z_x, z_y]

    def get_nearest_cell_position(self, x, y) :
        "Returns cell position to the passed points"
        indices = self.get_cell_matrix_indices_from_cell_position(x, y)

        if indices is None : 
            return None
        n, m = indices
        return self.get_cell_center(n*self.__M + m)

    
    def update_log_odds(self, cell, x_t, range, z_max, obstacle_detected) :
        "Returns the log_odds of cell been occpuied based on the Robot State measurement and the maximum range a lidar could measure"
        
        # Cells Position 
        c_x = cell[0]
        c_y = cell[1]

        # Robot Position
        R_x = x_t[0]
        R_y = x_t[1]

        # Range and Bearing
        r = np.sqrt((c_x - R_x)**2 + (c_y - R_y)**2)
        

        # Assign log_odds

        # Return Lo if the cell if beyound max Range a Lidar could measure
        if (r >= z_max) :
            return self.lo

        # Return L_free if there were no Obstacle detected
        if not obstacle_detected :
            return self.l_free

        # Cell Tolerance 
        cell_tolerance = np.sqrt(self.__cell_length**2 + self.__cell_width**2)/2.

        # Return L_occ if the Lidar endpoin fallson the cell
        if abs(range - r) <= cell_tolerance:
            return self.l_occ

        # Return L_free for cells whoe Ranges are less than the range measured by Lidar
        elif r < range - cell_tolerance :
            return self.l_free

        
        return self.lo
    

    def get_cells_under_beam_line(self, x_t, range, bearing) :
        "Returns the cells positons which falls under the lidar beam"

        # Robot's Pose
        x = x_t[0]
        y = x_t[1]
        theta = x_t[2]

        # Global Bearing
        global_ber = theta + bearing

        # Cells
        cells_position = []

        s = 0.0
        step = min(self.__cell_length, self.__cell_width) / 2

        while s < range:

            c_x = x + s * np.cos(global_ber)
            c_y = y + s * np.sin(global_ber)

            cell_position = self.get_nearest_cell_position(c_x, c_y)

            if cell_position is None:
                break

            if cell_position not in cells_position:
                cells_position.append(cell_position)

            s += step

        return cells_position

    def binary_occupancy_grid_mapping(self, x_t, z_t, z_max) :

        # Assumption : The Lidar is fixed right on top of the Robot with same heading direction and thhe Lidar scans from 180 deg to -180 deg covering 360

        # Itertate throught each measurement 
        for z in z_t :
            
            # Extract Range and Bearing from Lidar Measurement 
            range, bearing = z
            # Note in the webots if there is no obstacles detected the returned values will be 'inf' which I had to set to 0 for a few reasons even though we will use z_max as it is a good Practise to do so 

            if np.isnan(range) or range <= 0. :
                continue

            if np.isposinf(range) :
                traversal_range = z_max
                obstacle_detected = False

            elif np.isfinite(range) :
                traversal_range = min(range, z_max)
                obstacle_detected = range < z_max

            else :
                continue

            # Polar conversion
            # z_x, z_y = self.get_beam_end_points(x_t = x_t, range = range, bearing = bearing)

            # Ray Traversal
            cells = self.get_cells_under_beam_line(x_t = x_t, range = traversal_range, bearing = bearing)
            
            # Update log_odds 

            # Iterate through each cell and Update the log_odds
            for cell in cells :

                # Skip the cells which are not valid
                indices = self.get_cell_matrix_indices_from_cell_position(cell[0], cell[1])

                if indices is None :
                    continue

                # Get Cell index
                n, m = indices

                # Set log_odds    
                old_log_odds = self.log_odds[n, m]

                self.log_odds[n, m] = old_log_odds + self.update_log_odds(cell = cell, x_t = x_t, range = range, z_max = z_max, obstacle_detected = obstacle_detected) - old_log_odds


        return self.log_odds


# IMPLEMENTATION 

# CSV File handling
data = pd.read_csv("Webots_worlds\controllers\point_cloud_data_occupancy_grid_mapping")
lidar_measurements = data['lidar_measurements']
lidar_measurement = literal_eval(lidar_measurements[1])

# LiDAR scan configuration
angle_min = np.radians(180)
angle_increment = -0.5*np.radians(1)

z_t = np.array([[lidar_measurement[i], (angle_min + i*angle_increment)] for i in range(len(lidar_measurement))])
print(z_t[0:5, :])
# Robot State
x_R = 0.
y_R = 0.
theta_R = 0.

# Map Class Object
map = Map(map_length = 20., map_width = 20., cell_length = 0.20, cell_width = 0.20, lo = 0., l_free = -0.8473, l_occ = 0.8473)

x_t = np.array([0., 0., 0.])

log_odds = map.binary_occupancy_grid_mapping(x_t = x_t, z_t = z_t, z_max = 15.)
print("Exe Complete")
# Convert log odds into occupancy probabilities
occupancy_probability = 1 / (1 + np.exp(-log_odds))

# Visualize
plt.figure(figsize=(15, 15))

plt.imshow(
    occupancy_probability.T,
    cmap='gray_r',
    origin='lower',
    extent=[
        -20./2,
        20./2,
        -20./2,
        20./2
    ],
    vmin=0,
    vmax=1,
    interpolation='nearest'
)

plt.colorbar(label='Occupancy Probability')

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Binary Occupancy Grid Map')

plt.axis('equal')
plt.grid(False)



# Convert ranges into a NumPy array
ranges = z_t[:, 0]

# Calculate the angle of every LiDAR beam
angles = z_t[:, 1]

# Remove invalid and out-of-range measurements
valid = (
    np.isfinite(ranges) &
    (ranges > 0) &
    (ranges < 15.)
)

valid_ranges = ranges[valid]
valid_angles = angles[valid]

# Convert polar measurements into Cartesian coordinates
x_points = x_R + valid_ranges * np.cos(theta_R + valid_angles)
y_points = y_R + valid_ranges * np.sin(theta_R + valid_angles)

# Overlay LiDAR point cloud
plt.scatter(
    x_points,
    y_points,
    s=5,
    c='red',
    marker='.',
    label='LiDAR Points'
)
plt.legend()
plt.tight_layout()
plt.show()





