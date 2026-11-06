"This file is meant to implement the binary occupancy grid mapping "

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast


# lidar data 
df = pd.read_csv("/home/rathod/Webots_worlds/controllers/DriveMyRobot/point_cloud_data_occupancy_grid_mapping")
data = df["lidar_measurements"]
lidar_measurement = ast.literal_eval(data[1])


class Map :

    def __init__(self, map_length, map_width, cell_length, cell_width, lo = 0.) :

        # It is assumed that the cell_length is one of the positive factors of map_length and cell_width is one of the factors of map_width
        # So for now it does not handle the cases other than the specified one

        
        # Map's length and width in meters
        self.map_length = map_length
        self.map_width = map_width

        # cell's length and width in meters
        self.cell_length = cell_length
        self.cell_width = cell_width

        # Row and Column of Map if provided parameters are valid
        if not (self.get_row_and_col()) : 
            # Throw and error
            self.__N = 0
            self.__M = 0

        # Assign the number of Rows and columns
        else :  
            self.__N, self.__M = self.get_row_and_col()


        # Set Initial Likelihoods
        self.likelihood = self.set_initial_likelihood(lo)

    def rows(self) :
        "RETURNS THE PRIVATED FIELD VARIABLE N"
        return self.__N

    def cols(self) :
        "RETURNS THE PRIVATE FIELD VARIABLE M"
        return self.__M


    def set_initial_likelihood(self, lo) :
        "Sets 1/2 as the Initial likelihood to each cell"

        likelihood = [[lo for j in range(self.__M)] for i in range(self.__N)]
        return np.array(likelihood)

    def get_row_and_col(self) :
        "Returns the number of Rows and Columns of the map"

        N = 0
        M = 0
        # Check if the parameters are valid
        if np.min((self.map_length, self.map_width, self.cell_length, self.cell_width)) <= 0 :
            return 0

        # Calculate the number of Rows and Columns
        else :
            N = self.map_width/self.cell_width
            M = self.map_length/self.cell_length

        return int(N), int(M)


    def get_matrix_indices(self, i) :
        """ Returns the row number and column number of the matrix of size NxM"
        n  {0, 1, 2, 3, 4, ............, N - 1}
        m  {0, 1, 2, 3, 4, ............, M - 1}
            """
    
        m = i%self.__M
        n = (i - m)/self.__M

        return int(n), int(m) 

    def get_cell_center(self, i) :
        "Returns the center points pf ith cell"
        n, m = self.get_matrix_indices(i)

        x = -self.map_width/2 + (m + 0.5)*self.cell_width
        y = -self.map_length/2 + (n + 0.5)*self.cell_length

        return x, y

    def is_in_perceptual_range(
    self,
    i,
    x_t,
    ranges,
    angle_min,
    angle_increment,
    z_max,
    beta
        ):

        # 1. Get cell centre in world coordinates
        m_x, m_y = self.get_cell_center(i)

        # 2. Robot pose
        x_R, y_R, theta_R = x_t

        # Assumption: LiDAR is at robot origin
        # and aligned with robot heading

        # 3. Relative position of cell
        dx = m_x - x_R
        dy = m_y - y_R

        # 4. Distance from LiDAR to cell
        r = np.sqrt(dx**2 + dy**2)

        # 5. Bearing of cell relative to LiDAR
        phi = wrap_angle(
            np.arctan2(dy, dx) - theta_R
        )

        # Find nearest beam
        K = len(ranges)

        k = int(round(
            (phi - angle_min) / angle_increment
        ))

        # Handle full-circle LiDAR
        if np.isclose(
            K * abs(angle_increment),
            2 * np.pi,
            atol=1.5 * abs(angle_increment)
        ):
            k %= K

        # Reject invalid beam indices
        elif not (0 <= k < K):
            return False, None, r, phi

        # 7. Calculate angular difference
        theta_k = angle_min + k * angle_increment

        angular_error = abs(
            wrap_angle(phi - theta_k)
        )

        # 8. Check beam angular width and sensing range
        if angular_error <= beta / 2 and r <= z_max:

            return True, k, r, phi

        return False, None, r, phi


def wrap_angle(angle) :
    return (angle + np.pi)%(2*np.pi) - np.pi




def inverse_sensor_model(cell, lo, l_occ, l_free, r, phi, z_t, z_max, alpha, beta) :
    "Returns the likelihood of the map cell"

   
    r = r
    phi = phi

    if r > min(z_max, z_t + alpha/2) :
        return lo

    elif (z_t < z_max) and np.abs(z_t - r) <= alpha/2 :
        return l_occ

    elif r < z_t :
        return l_free

    return lo


def binary_occupancy_grid_mapping(map, likelihood, lo, l_occ, l_free, x_t, z_t, z_max, alpha, beta) :
    """Returns the approximated likelihood of map m given the state and the measurement at time t"
        It has been assumed the Initial position or the state of the Robot is known

        m - map of size NxM
        N - number of Rows in the map
        M - number of columns in the map
        x_t - State(pose) of the robot at time t
        z_t - measurement at time t
        likelihood (NxM)- holds the likelihood of each map Cell 

        """

    # Iterate over map cells
    for i in range(map.rows()*map.cols()) :

        # Get the row and column of the ith map cell
        n, m = map.get_matrix_indices(i = i)        

        # Likelihood of the cell
        lkd = likelihood[n, m]
        lkd_initial = lkd
        # Cell
        cell = map.get_cell_center(i)

        angle_min = np.radians(180)

        angle_increment = -np.radians(1)
    
        # Check if the map cell is in the perceptual range of the robot or specifically the sensor
        inside, k, r, phi = map.is_in_perceptual_range(i = i, x_t = x_t, ranges = z_t, angle_min = angle_min, angle_increment = angle_increment, z_max = z_max, beta = beta)

        

        if inside :
            measured_range = z_t[k]
            #Update the likelihood of the cell   
            lkd = lkd + inverse_sensor_model(cell = cell, lo = lo, l_occ = l_occ, l_free = l_free, r = r, phi = phi, z_t = measured_range, z_max = z_max, alpha = alpha, beta = beta) - lo

        else :
            lkd = lkd

        likelihood[n, m] = lkd

        # Print the cell details before and after considering the position and measurement
        if n == map.rows() - 1 :

            print(f"Cell : ROw {n} Col {m}")
            print(f"Initial Likelihood : {lkd_initial} ")

            print(f'Cell location in world frame {cell} {"\n"}Measurement associated {measured_range} at a heading of {np.degrees(angle_min + k*angle_increment)}')
            print(f"Computed Range and Bearing of the cell from the lidar {r} m  {phi} deg ")
            print(f"Likelihood after : {lkd}")
            print("----------------------------------------------------------------------------------------------------------------------------------------------")


    return likelihood

# Parameters
arena_length = 20
arena_width = 20

cell_length = 0.20
cell_width = 0.20

# Likelihoods
lo = 0.
l_occ = 0.8473
l_free = -0.8473

# Map Oblject
map = Map(map_length = arena_length, map_width = arena_width, cell_length = cell_length, cell_width = cell_width, lo = lo)

# Alpha and Beta
alpha = 0.4
beta = np.radians(1)

# Lidar's maximum Range
z_max = 15

likelihood = binary_occupancy_grid_mapping(map = map, likelihood = map.likelihood, lo = lo, l_occ = l_occ, l_free = l_free, x_t = np.array([0.0, 0., 0.]), z_t = lidar_measurement, z_max = z_max, alpha = alpha, beta = beta)

# print(f"LIKE {"\n"}{likelihood[0, 0:10]}{"\n"}{likelihood[1, 0:10]}{"\n"}{likelihood[2, 0:10]}{"\n"}{likelihood[3, 0:10]}{"\n"}{likelihood[4, 0:10]}")

# Convert log odds into occupancy probabilities
occupancy_probability = 1 / (1 + np.exp(-likelihood))

# Visualize
plt.figure(figsize=(15, 15))

plt.imshow(
    occupancy_probability,
    cmap='gray_r',
    origin='lower',
    extent=[
        -arena_length/2,
        arena_length/2,
        -arena_width/2,
        arena_width/2
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


# Robot pose (same pose used in occupancy grid mapping)
x_R, y_R, theta_R = 0.0, 0.0, 0.0

# LiDAR scan configuration
angle_min = np.radians(180)
angle_increment = -np.radians(1)

# Convert ranges into a NumPy array
ranges = np.asarray(lidar_measurement, dtype=float)

# Calculate the angle of every LiDAR beam
angles = angle_min + np.arange(len(ranges)) * angle_increment

# Remove invalid and out-of-range measurements
valid = (
    np.isfinite(ranges) &
    (ranges > 0) &
    (ranges < z_max)
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