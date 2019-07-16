"""
main file of PI-GMRF simulation algorithm

author: Andreas Rene Geist
email: andreas.geist@tuhh.de
website: https://github.com/AndReGeist
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import Config
import gp_scripts
import control_scripts
from RRT_star_control import RRT_star
from PRM_star_Dubins_control import PRM_star_Dubins
import plot_scripts
from true_field import true_field
import time
from scipy import sqrt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# AUV starting state
x_auv = Config.x_auv

"""Initialize Field"""
true_field1 = true_field(False)
field_limits = (true_field1.x_field[-1], true_field1.y_field[-1])
# Calculate and set plot parameters
plot_settings = {"vmin": np.amin(true_field1.z_field) - 0.1, "vmax": np.amax(true_field1.z_field) + 0.1, "var_min": 0,
				 "var_max": 3, "levels": np.linspace(np.amin(true_field1.z_field) - 0.1, np.amax(true_field1.z_field) + 0.1, 20),
				 "PlotField": False, "LabelVertices": True}

# Initialize plots
fig1, hyper_x, hyper_y, bottom, colors, trajectory_1 = plot_scripts.initialize_animation1(true_field1, x_auv, **plot_settings)

# Initialize GMRF
time_1 = time.time()
gmrf1 = gp_scripts.GMRF(Config.gmrf_dim, Config.alpha_prior, Config.kappa_prior, Config.set_Q_init)
# print(gmrf1.__dict__)
time_2 = time.time()
print("Time for GMRF init: /", "{0:.2f}".format(time_2 - time_1))
# Initialize Controller
u_optimal = np.zeros(shape=(Config.N_horizon, 1))

"""#####################################################################################"""
"""START SIMULATION"""
for time_in_ms in range(0, Config.simulation_end_time):  # 1200 ms
	if time_in_ms % Config.sample_time_gmrf < 0.0000001:
		print("shape of gmrf1", gmrf1.var_x.shape)
		print("shape of truefield.z", true_field1.z_field.shape)
		# Calculate next AUV state
		# x_auv = Config.auv_dynamics(x_auv, u_optimal[0], 0, Config.sample_time_gmrf / 100, field_limits)
		trajectory_1 = np.vstack([trajectory_1, x_auv])

		# Compute discrete observation vector and new observation
		sd_obs = [int((x_auv[1]) * 1e2), int((x_auv[0]) * 1e2)]
		y_t = np.array(true_field1.z_field[sd_obs[0], sd_obs[1]]) + np.random.normal(loc=0.0, scale=sqrt(Config.sigma_w_squ), size=1)

		# Update GMRF belief
		time_3 = time.time()
		mue_x, var_x, pi_theta = gmrf1.gmrf_bayese_update(x_auv, y_t)
		time_4 = time.time()
		print("Calc. time GMRF: /", "{0:.2f}".format(time_4 - time_3))

		# Calculate optimal control path

		# u_optimal, tau_x, tau_optimal = control_scripts.pi_controller(x_auv, u_optimal, var_x, Config.pi_parameters, gmrf1.params, field_limits, Config.set_sanity_check)
		# rrt_star = None

		# PRM_star = PRM_star_Dubins(start=x_auv, space=[0, 10, 0, 5], obstacles=None, var_x=var_x, gmrf_params=gmrf1.params)
		# path_optimal, u_optimal, tau_optimal, dubins_time = PRM_star.control_algorithm()
		# print("dubins planner time: ", dubins_time)
		# x_auv = tau_optimal[:, 3]
		# tau_x = None

		RRT_star1 = RRT_star(start=x_auv, space=[0, 10, 0, 5], obstacles=None, var_x=var_x, gmrf_params=gmrf1.params)
		path_optimal, u_optimal, tau_optimal, dubins_time, rewire_time = RRT_star1.control_algorithm()
		print("dubins planner time: ", dubins_time)
		print("steer time: ", rewire_time)
		x_auv = tau_optimal[:, 3]
		tau_x = None

		time_5 = time.time()
		print("Calc. time control script: /", "{0:.2f}".format(time_5 - time_4))

		# Plot new GMRF belief and optimal control path
		plot_scripts.update_animation1(RRT_star1, pi_theta, fig1, hyper_x, hyper_y, bottom, colors, true_field1, x_auv, mue_x, var_x, gmrf1.params, trajectory_1, tau_x, tau_optimal, **plot_settings)
		time_6 = time.time()
		print("Calc. time Plot: /", "{0:.2f}".format(time_6 - time_5))

		# terminate after trajectory length exceeds bound


# CODE FOR BENCHMARKING
(lxf, lyf, dvx, dvy, lx, ly, n, p, de, l_TH, p_THETA, xg_min, xg_max, yg_min, yg_max) = gmrf1.params
# sum of variances
total_variance_sum = np.sum(gmrf1.var_x)
field_variance_sum = 0
for nx in range(15, 65):
	for ny in range(15, 40):
		field_variance_sum += gmrf1.var_x[(ny * lx) + nx]

# RMSE of mean in field bounds
mean_RMSE = 0
# grid_x, grid_y = np.mgrid[0:10:.204, 0:5:.206]
# true_mean = true_field1.f(grid_x, grid_y)

for nx in range(15, 65):
	for ny in range(15, 40):
		print("gmrf mean x, y, z: ", de[0] * nx + xg_min, de[1] * ny + yg_min, gmrf1.mue_x[(ny * lx) + nx])
		print("true field mean x, y, z: ", de[0] * nx + xg_min, de[1] * ny + yg_min, true_field1.f(de[0] * nx + xg_min, de[1] * ny + yg_min))
		mean_RMSE += (gmrf1.mue_x[(ny * lx) + nx] - true_field1.f(de[0] * nx + xg_min, de[1] * ny + yg_min)) ** 2
mean_RMSE = sqrt(mean_RMSE)
