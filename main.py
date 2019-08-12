"""
orignal author: Andreas Rene Geist
email: andreas.geist@tuhh.de
website: https://github.com/AndReGeist
license: BSD

addition and modification of file by Patrick Phillips summer 2019
email: pphill10@u.rochester.edu
website: https://github.com/peweetheman
"""
import time
import resource
import os
import numpy as np
from scipy import sqrt
from control_algorithms import control_scripts
import Config
from gp_scripts import gp_scripts
import plot_scripts
from true_field import true_field

for iter in range(1, Config.iterations):
	for i in range(10):
		Config.max_runtime = i
		# AUV starting state
		x_auv = Config.x_auv
		trajectory_1 = np.array(x_auv).reshape(1, 3)

		# Initialize Field
		true_field1 = true_field(False)
		# Calculate and set plot parameters
		plot_settings = {"vmin": np.amin(true_field1.z_field) - 0.1, "vmax": np.amax(true_field1.z_field) + 0.1, "var_min": 0,
						 "var_max": 3, "levels": np.linspace(np.amin(true_field1.z_field) - 0.1, np.amax(true_field1.z_field) + 0.1, 20),
						 "PlotField": False, "LabelVertices": True}
		# Initialize plots
		if Config.plot is True:
			fig1, hyper_x, hyper_y, bottom, colors = plot_scripts.initialize_animation1(true_field1, **plot_settings)

		# Initialize data collection
		if Config.collect_data is True:
			filename = os.path.join('data', Config.control_algo + '_runtime' + str(Config.max_runtime) + '_pathlength' + str(Config.simulation_max_dist) + "_" + str(iter))
			print("data file: ", filename)
			data = np.zeros(shape=(5, 1))

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
		# resource.setrlimit(resource.RLIMIT_STACK, (1e10, 1e12))       # I haven't really found a great way to set resource limits
		total_calc_time_control_script = 0
		path_length = 0
		myplot = None
		for time_in_ms in range(0, Config.simulation_end_time):  # 1200 ms
			if time_in_ms % Config.sample_time_gmrf < 0.0000001:
				# Compute discrete observation vector and new observation
				sd_obs = [int((x_auv[1]) * 1e2), int((x_auv[0]) * 1e2)]
				# print("sd_obs", sd_obs, np.array(true_field1.z_field[sd_obs[0], sd_obs[1]]))
				# print("or with f", sd_obs, np.array(true_field1.f(x_auv[0], x_auv[1])))
				y_t = np.array(true_field1.f(x_auv[0], x_auv[1])) + np.random.normal(loc=0.0, scale=sqrt(Config.sigma_w_squ), size=1)

				# Update GMRF belief
				time_3 = time.time()
				mue_x, var_x, pi_theta = gmrf1.gmrf_bayese_update(x_auv, y_t)
				time_4 = time.time()
				# print("Calc. time GMRF: /", "{0:.2f}".format(time_4 - time_3))

				# Run control algorithm to calculate new path. Select which one in Config file.
				if Config.control_algo == 'PI':
					u_optimal, tau_x, tau_optimal = control_scripts.pi_controller(x_auv, u_optimal, var_x, Config.pi_parameters, gmrf1.params, Config.field_dim, Config.set_sanity_check)
					x_auv = Config.auv_dynamics(x_auv, u_optimal[0], 0, Config.sample_time_gmrf / 100, Config.field_dim)
					control = None
				else:
					control = Config.control_algorithm(start=x_auv, u_optimal=u_optimal, gmrf_params=gmrf1.params, var_x=var_x, max_dist=Config.simulation_max_dist-path_length, plot=myplot)
					path_optimal, u_optimal, tau_optimal = control.control_algorithm()
					x_auv = Config.auv_dynamics(x_auv, u_optimal[0], 0, Config.sample_time_gmrf / 100, Config.field_dim, tau_optimal[:, 4])
					tau_x = None
				trajectory_1 = np.vstack([trajectory_1, x_auv])
				time_5 = time.time()
				control_calc_time = time_5 - time_4
				print("Calc. time control script: /", "{0:.2f}".format(time_5 - time_4))

				# Plot new GMRF belief and optimal control path. Comment out this region for quick data collection
				if Config.plot is True:
					traj, myplot = plot_scripts.update_animation1(control, pi_theta, fig1, hyper_x, hyper_y, bottom, colors, true_field1, x_auv, mue_x, var_x, gmrf1.params, trajectory_1, tau_x, tau_optimal, **plot_settings)
					time_6 = time.time()
					print("Calc. time Plot: /", "{0:.2f}".format(time_6 - time_5))

				# Calculate trajectory length and terminate after trajectory length exceeds bound
				path_length = 0
				for kk in range(1, np.size(trajectory_1, axis=0)):
					path_length += ((trajectory_1[kk, 0] - trajectory_1[kk-1, 0]) ** 2 + (trajectory_1[kk, 1] - trajectory_1[kk-1, 1]) ** 2)
				# print("path length: ", path_length)
				if path_length >= Config.simulation_max_dist-1:
					print("END DUE TO MAX PATH LENGTH")
					break
				print("path_length: ", path_length)

				# CODE FOR BENCHMARKING
				if Config.collect_data is True:
					(lxf, lyf, dvx, dvy, lx, ly, n, p, de, l_TH, p_THETA, xg_min, xg_max, yg_min, yg_max) = gmrf1.params

					# sum of variances
					total_variance_sum = np.sum(gmrf1.var_x)
					field_variance_sum = 0
					for nx in range(15, 65):
						for ny in range(15, 40):
							field_variance_sum += gmrf1.var_x[(ny * lx) + nx]

					# RMSE of mean in field bounds
					mean_RMSE = 0
					for nx in range(15, 65):
						for ny in range(15, 40):
							# print("gmrf mean x, y, z: ", de[0] * nx + xg_min, de[1] * ny + yg_min, gmrf1.mue_x[(ny * lx) + nx])
							# print("true field mean x, y, z: ", de[0] * nx + xg_min, de[1] * ny + yg_min, true_field1.f(de[0] * nx + xg_min, de[1] * ny + yg_min))
							mean_RMSE += (gmrf1.mue_x[(ny * lx) + nx] - true_field1.f(de[0] * nx + xg_min, de[1] * ny + yg_min)) ** 2
					mean_RMSE = sqrt(mean_RMSE)

					# organize data to write to file
					col = np.vstack((path_length, total_variance_sum, field_variance_sum, mean_RMSE, control_calc_time))
					data = np.concatenate((data, col), axis=1)
	if Config.collect_data is True:
		np.save(filename, data)
