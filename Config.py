"""
Config file containing all static parameters and functions used in the
author: Andreas Rene Geist
email: andreas.geist@tuhh.de
website: https://github.com/AndReGeist
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
"""
addition and modification of file by Patrick Phillips summer 2019
email: pphill10@u.rochester.edu
website: https://github.com/peweetheman
"""

import numpy as np
from control_algorithms.PRM_star_control import PRM_star
from control_algorithms.RRT_star_control import RRT_star
from scipy import sin, cos, sqrt, pi
from control_algorithms.PRM_control import PRM
from control_algorithms.RRT_control import RRT

"""Configure the simulation parameters"""
# AUV starting state
x_auv = np.array([0.1, 0.1, 0.785]).T  # Initial AUV state
v_auv = 1.0  # AUV velocity in meter/second (constant)

# Field dimensions
field_dim = [0, 9.9, 0, 4.9]  # x_min , x_max, y_min, y_max

# Simulation variables
plot = False      # whether or not to plot while running
collect_data = True
sigma_w_squ = 0.2 ** 2  # Measurement variance
sample_time_gmrf = 100  # Sample/Calculation time in ms of GMRF algorithm, not used right now
simulation_end_time = 2000000 # Run time of simulation in ms, not used right now

"""Choose GMRF parameters"""
gmrf_dim = [50, 25, 15, 15]  # lxf, lyf, dvx, dvy
set_Q_init = False  # Re-Calculate precision matrix at Initialization? False: Load stored precision matrix
set_Q_check = False  # Plots Q matrix entries inside GMRF algorithm
set_gmrf_torus = True  # True -w> GMRF uses torus boundary condition, False -> GMRF uses Neumann-BC
set_GMRF_cartype = False  # Use car(1)? <-> True, Default is car(2) from Choi et al
set_prior = 3  # Choose prior case from below
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


"""CONTROL PARAMETERS"""
simulation_max_dist = 40.0   			# max distance of path for simulation tests
iterations = 100
control_algo = 'RRT_star'      # choose either 'PI', 'RRT_star', 'PRM_star', 'RRT', 'PRM'
sigma_epsilon = pi / 16         # Exploration noise in radians, 90 grad = 1,57
R_cost = 5 * np.ones(shape=(1, 1))  # Immediate control cost. This is not incorporated in sampling algorithms, though could be in the cost function.
border_variance_penalty = 1000

"""Choose control parameters for PI algorithm"""
set_sanity_check = True  # Calculates cost for the optimal path and plots the optimal path
n_updates = 10  		# Control loop updates
n_k = 10  				# Number of virtual roll-out pathes
n_horizon = 10			 # Control horizon length in s
N_horizon = 10  		# Number of discrete rollout points
t_cstep = n_horizon / N_horizon  # Control horizon step size in s
pi_parameters = (n_updates, n_k, n_horizon, N_horizon, t_cstep, sigma_epsilon, R_cost)

"""Choose control parameters for sampling control algorithms"""
max_runtime = 0.25					# Runtime for the sampling algorithm to end after. Typically takes .05 seconds more than this runtime.
max_curvature = 1.0     			# maximum curvature of a path allowed for the robot
growth = 2.0       					# distance that RRT algorithms will steer nearest node to new node
min_dist = 2.0                      # minimum distance of paths that the control algorithm will consider. needed to be >0 as we don't want to consider not moving (will get error if set to <=0). also good to not be super small, to discourage taking greedily very short informative paths that get stuck
obstacles = None  					# any obstacles in the field. currently only handles squares specified by x,y location and side length
RRT_params = (field_dim, max_runtime, max_curvature, growth, min_dist, obstacles)
PRM_params = (field_dim, max_runtime, max_curvature, min_dist, obstacles)


#################################################################################################
"""DEFINE GENERAL FUNCTIONS"""
# Run the selected control algorithm
def control_algorithm(start, u_optimal, gmrf_params, var_x, max_dist, plot):
	if control_algo == 'RRT_star':
		return RRT_star(start, RRT_params, gmrf_params, var_x, max_dist, plot)
	elif control_algo == 'PRM_star':
		return PRM_star(start, PRM_params, gmrf_params, var_x, max_dist, plot)
	elif control_algo == 'RRT':
		return RRT(start, RRT_params, gmrf_params, var_x, max_dist, plot)
	elif control_algo == 'PRM':
		return PRM(start, PRM_params, gmrf_params, var_x, max_dist, plot)

# AUV model
def auv_dynamics(x_auv, u_auv, epsilon_a, delta_t, field_dim, x_auv_new=None, set_border=True):
	x_auv_out = np.zeros(shape=3)
	if control_algo == 'PI':
		x_auv_out[2] = x_auv[2] + u_auv * delta_t + epsilon_a * sqrt(delta_t)  # computes angle
		x_auv_out[0] = x_auv[0] + v_auv * cos(x_auv_out[2]) * delta_t
		x_auv_out[1] = x_auv[1] + v_auv * sin(x_auv_out[2]) * delta_t
	else:
		x_auv_out = x_auv_new
	if set_border == True:
		# Prevent AUV from leaving the true field
		if x_auv_out[0] < field_dim[0]:
			x_auv_out[0] = field_dim[0]

		if x_auv_out[1] < field_dim[2]:
			x_auv_out[1] = field_dim[2]

		if x_auv_out[0] > field_dim[1]:
			x_auv_out[0] = field_dim[1]

		if x_auv_out[1] > field_dim[3]:
			x_auv_out[1] = field_dim[3]

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
