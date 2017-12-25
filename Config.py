"""
Config file containing all static parameters and functions used in the PI-GMRF algorithm.

author: Andreas Rene GEist
email: andreas.geist@tuhh.de
website: https://github.com/AndReGeist
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import scipy
from scipy import exp, sin, cos, sqrt, pi, interpolate

"""Configure the simulation parameters"""
# AUV starting state
x_auv = np.array([0.1, 0.1, 0.785]).T  # Initial AUV state
v_auv = 0.5  # AUV velocity in meter/second (constant)

# Field dimensions
field_dim = [0, 10, 0, 5]  # x_min , x_max, y_min, y_max

# Simulation variables
sigma_w_squ = 0.2 ** 2  # Measurement variance
sample_time_gmrf = 100  # Sample/Calculation time in ms of GMRF algorithm
simulation_end_time = 20000  # Run time of simulation in ms


"""Choose GMRF parameters"""
gmrf_dim = [50, 25, 15, 15]  # lxf, lyf, dvx, dvy
set_Q_init = True  # Re-Calculate precision matrix at Initialization? False: Load stored precision matrix
set_Q_check = False  # Plots Q matrix entries inside GMRF algorithm
set_gmrf_torus = False  # True -w> GMRF uses torus boundary condition, False -> GMRF uses Neumann-BC
set_GMRF_cartype = True  # Use car(1)? <-> True, Default is car(2) from Choi et al
set_prior = 1  # Choose prior case from below
if set_GMRF_cartype == False:
    if set_prior == 1:
        # Choi Parameter (size 1)
        kappa_prior = np.array([0.0625 * (2 ** 4)]).astype(float)
        alpha_prior = np.array([0.000625 * (4 ** 2)]).astype(float)
    elif set_prior == 2:
        # Choi paper
        kappa_prior = np.array([0.0625 * (2 ** 0), 0.0625 * (2 ** 2), 0.0625 * (2 ** 4), 0.0625 * (2 ** 6), 0.0625 * (2 ** 8)]).astype(float)
        alpha_prior = np.array([0.000625 * (1 ** 2), 0.000625 * (2 ** 2), 0.000625 * (4 ** 2), 0.000625 * (8 ** 2), 0.000625 * (16 ** 2)]).astype(float)
    elif set_prior == 3:
        # Choi Parameter (size 2)
        kappa_prior = np.array([0.0625 * (2 ** 2), 0.0625 * (2 ** 4)]).astype(float)
        alpha_prior = np.array([0.000625 * (2 ** 2), 0.000625 * (4 ** 2)]).astype(float)
    elif set_prior == 4:
        # Choi Parameter (size 3)
        kappa_prior = np.array([0.0625 * (2 ** 2), 0.0625 * (2 ** 4), 0.0625 * (2 ** 6)]).astype(float)
        alpha_prior = np.array([0.000625 * (2 ** 2), 0.000625 * (4 ** 2), 0.000625 * (8 ** 2)]).astype(float)
    elif set_prior == 5:
        # Extended Choi paper
        kappa_prior = np.array([1000, 100, 10, 0.0625 * (2 ** 0), 0.0625 * (2 ** 2), 0.0625 * (2 ** 4), 0.0625 * (2 ** 6), 0.0625 * (2 ** 8), 0.0625 * (2 ** 9), 0.0625 * (2 ** 10)]).astype(float)
        alpha_prior = np.array([0.000625 * (1 ** -1), 0.000625 * (1 ** 0), 0.000625 * (1 ** 1), 0.000625 * (1 ** 2), 0.000625 * (2 ** 2), 0.000625 * (4 ** 2), 0.000625 * (8 ** 2), 0.000625 * (16 ** 2), 0.000625 * (32 ** 2), 0.000625 * (64 ** 2), 0.000625 * (128 ** 2)]).astype(float)
elif set_GMRF_cartype == True:
    if set_prior == 1:
        # Solowjow Parameter for CAR(1) (size 1)
        kappa_prior = np.array([1]).astype(float)
        alpha_prior = np.array([0.001]).astype(float)
    elif set_prior == 2:
        # Same theta values (size 3)
        kappa_prior = np.array([0.5, 1, 2]).astype(float)
        alpha_prior = np.array([0.01]).astype(float)
    elif set_prior == 3:
        kappa_prior = np.array([1, 1.2]).astype(float)
        alpha_prior = np.array([0.0001, 0.001, 0.01]).astype(float)
    elif set_prior == 4:
        kappa_prior = np.array([0.8, 1, 1.2]).astype(float)
        alpha_prior = np.array([0.001, 0.01, 0.1]).astype(float)
    elif set_prior == 5:
        kappa_prior = np.array([1]).astype(float)
        alpha_prior = np.array([3, 1, 0.5, 0.3, 0.1, 0.01]).astype(float)

"""Choose control parameters"""
set_sanity_check = True  # Calculates cost for the optimal path and plots the optimal path
n_updates = 10  # Control loop updates
n_k = 15  # Number of virtual roll-out pathes
n_horizon = 10  # Control horizon length in s
N_horizon = 10  # Number of discrete rollout points
t_cstep = n_horizon / N_horizon  # Control horizon step size in s
sigma_epsilon = scipy.pi / 16  # Exploration noise in radians, 90 grad = 1,57
R_cost = 5 * np.ones(shape=(1, 1))  # Immediate control cost
border_variance_penalty = 5
pi_parameters = (n_updates, n_k, n_horizon, N_horizon, t_cstep, sigma_epsilon, R_cost)

#################################################################################################
"""DEFINE GENERAL FUNCTIONS"""

# AUV model
def auv_dynamics(x_auv, u_auv, epsilon_a, delta_t, field_limits, set_border = True):
    x_auv_out = np.zeros(shape=(3))

    x_auv_out[2] = x_auv[2] + u_auv * delta_t + epsilon_a * sqrt(delta_t)
    x_auv_out[0] = x_auv[0] + v_auv * cos(x_auv_out[2]) * delta_t
    x_auv_out[1] = x_auv[1] + v_auv * sin(x_auv_out[2]) * delta_t

    if set_border == True:
        # Prevent AUV from leaving the true field
        if x_auv_out[0] < 0:
            x_auv_out[0] = 0

        if x_auv_out[1] < 0:
            x_auv_out[1] = 0

        if x_auv_out[0] > field_limits[0]:
            x_auv_out[0] = field_limits[0]

        if x_auv_out[1] > field_limits[1]:
            x_auv_out[1] = field_limits[1]
        # if x_auv_out[0] < 0:
        #     x_auv_out[0] = 0
        #     if pi / 2 < x_auv_out[2] < pi:
        #         x_auv_out[2] = 0.49 * pi
        #     if pi < x_auv_out[2] < 1.5 * pi:
        #         x_auv_out[2] = 1.51 * pi
        #
        # if x_auv_out[1] < 0:
        #     x_auv_out[1] = 0
        #     if pi < x_auv_out[2] < 1.49 * pi:
        #         x_auv_out[2] = pi
        #     if 1.5 * pi < x_auv_out[2] <= 2 * pi:
        #         x_auv_out[2] = 0.01
        #
        # if x_auv_out[0] > field_limits[0]:
        #     x_auv_out[0] = field_limits[0]
        #     if 0 < x_auv_out[2] < pi / 2:
        #         x_auv_out[2] = 0.51 * pi
        #     if 1.5 * pi < x_auv_out[2] <= 2 * pi:
        #         x_auv_out[2] = 1.49 * pi
        #
        # if x_auv_out[1] > field_limits[1]:
        #     x_auv_out[1] = field_limits[1]
        #     if 0 < x_auv_out[2] < pi / 2:
        #         x_auv_out[2] = 1.99 * pi
        #     if pi / 2 < x_auv_out[2] < pi:
        #         x_auv_out[2] = 1.01 * pi

    if x_auv_out[2] > 2 * pi:
        a_pi = int(x_auv_out[2] / 2 * pi)
        x_auv_out[2] = x_auv_out[2] - a_pi * 2 * pi

    return x_auv_out

# Calculate new observation vector through shape function interpolation
def interpolation_matrix(x_local2, n, p, lx, xg_min, yg_min, de):
    """INTERPOLATION MATRIX:
    Define shape function matrix that maps grid vertices to
    continuous measurement locations"""
    u1 = np.zeros(shape=(n + p, 1)).astype(float)
    nx = int((x_local2[0] - xg_min) / de[0])  # Calculates the vertice column x-number at which the shape element starts.
    ny = int((x_local2[1] - yg_min) / de[1])  # Calculates the vertice row y-number at which the shape element starts.

    # Calculate position value in element coord-sys in meters
    x_el = float(0.1 * (x_local2[0] / 0.1 - int(x_local2[0] / 0.1))) - de[0] / 2
    y_el = float(0.1 * (x_local2[1] / 0.1 - int(x_local2[1] / 0.1))) - de[1] / 2

    # Define shape functions, "a" is element width in x-direction
    u1[(ny * lx) + nx] = (1 / (de[0] * de[1])) * ((x_el - de[0] / 2) * (y_el - de[1] / 2))  # u for lower left corner
    u1[(ny * lx) + nx + 1] = (-1 / (de[0] * de[1])) * ((x_el + de[0] / 2) * (y_el - de[1] / 2))  # u for lower right corner
    u1[((ny + 1) * lx) + nx] = (-1 / (de[0] * de[1])) * ((x_el - de[0] / 2) * (y_el + de[1] / 2))  # u for upper left corner
    u1[((ny + 1) * lx) + nx + 1] = (1 / (de[0] * de[1])) * ((x_el + de[0] / 2) * (y_el + de[1] / 2))  # u for upper right corner
    return u1