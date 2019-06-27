"""
Control scripts.

author: Andreas Rene Geist
email: andreas.geist@tuhh.de
website: https://github.com/AndReGeist
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import Config
import numpy as np
from scipy import exp, sin, cos, sqrt, pi, interpolate
from random import randint


def pi_controller(x_auv, u_optimal, var_x, pi_parameters, gmrf_params, field_limits, set_sanity_check):
	"""Optimal Stochastic controller in PI formulation, based on Schaal et al.
	"A generalized Path Integral Control Approach for Reinforcement Learning" (2010) """
	(lxf, lyf, dvx, dvy, lx, ly, n, p, de, l_TH, p_THETA, xg_min, xg_max, yg_min, yg_max) = gmrf_params
	(n_updates, n_k, n_horizon, N_horizon, t_cstep, sigma_epsilon, R_cost) = pi_parameters

	# Initialize PI matrices
	epsilon_auv = np.zeros(shape=(N_horizon, n_k))
	tau_x = np.zeros(shape=(len(x_auv), N_horizon, n_k))
	tau_optimal = np.zeros(shape=(len(x_auv), N_horizon))
	pre_x_tau = np.zeros(shape=(N_horizon, n_k))
	control_cost = np.zeros(shape=(N_horizon, n_k))
	exp_lambda_S = np.zeros(shape=(N_horizon, n_k))
	S_tau = np.zeros(shape=(N_horizon, n_k))
	P_tau = np.zeros(shape=(N_horizon, n_k))
	u_optimal[:-1] = u_optimal[1:]
	u_optimal[-1] = 0
	# u_optimal = np.zeros(shape=(N_horizon, 1))

	for ii in range(0, n_updates):  # Repeat PI algorithm for convergence

		for jj in range(0, n_k):  # Iterate over all trajectories
			"""Sample trajectories"""
			tau_x[:, 0, jj] = x_auv  # Set initial trajectory state
			# Calculate exploration noise (Only PI-Controller Hyperparameter)
			epsilon_auv[:, jj] = sigma_epsilon * np.random.standard_normal(N_horizon)

			for kk in range(0, N_horizon - 1):  # Iterate over length of trajectory except of last entry
				# Sample roll-out trajectory
				tau_x[:, kk + 1, jj] = Config.auv_dynamics(tau_x[:, kk, jj], u_optimal[kk], epsilon_auv[kk, jj], t_cstep, field_limits, set_border=False)

			"""Calculate cost and probability weighting"""
			for kk in range(0, N_horizon):  # Iterate over length of trajectory
				# Compute variance along sampled trajectory
				M_m = 1  # Only for p=1 and this simple state model !
				# if M_m == 1:
				#   print('Tau', tau_x[0:2, kk, jj], 'FL', field_limits)
				if not (0 <= tau_x[0, kk, jj] <= field_limits[0]) or not (0 <= tau_x[1, kk, jj] <= field_limits[1]):
					# print('Tau', tau_x[0:2, kk, jj], 'FL', field_limits)
					pre_x_tau[kk, jj] = Config.border_variance_penalty
					control_cost[kk, jj] = 0
				else:
					A_z = Config.interpolation_matrix(tau_x[:, kk, jj], n, p, lx, xg_min, yg_min, de)  #
					pre_x_tau[kk, jj] = 1 / np.dot(A_z.T, var_x)
					control_cost[kk, jj] = .5 * np.dot(np.array(u_optimal[kk] + epsilon_auv[kk, jj]).T,
														np.dot(R_cost, np.array(u_optimal[kk] + epsilon_auv[kk, jj])))

			for kk in range(0, N_horizon):  # Iterate over whole sampeld trajectory

				S_tau[kk, jj] = np.sum(pre_x_tau[kk:, jj]) + np.sum(control_cost[kk:, jj])
			for kk in range(0, N_horizon):  # Iterate over whole sampeld trajectory
				exp_lambda_S[kk, jj] = exp(-10 * (S_tau[kk, jj] - np.amin(S_tau[:, jj])) /
										   (np.amax(S_tau[:, jj]) - np.amin(S_tau[:, jj])))

		"""Update control"""
		u_correction = np.zeros(shape=(N_horizon, 1))
		for kk in range(0, N_horizon):  # Iterate over length of trajectory
			for jj in range(0, n_k):  # Iterate over all trajectories
				# Roll-Out probability
				P_tau[kk, jj] = exp_lambda_S[kk, jj] / np.sum(exp_lambda_S[kk, :])

			for jj in range(0, n_k):  # Iterate over all trajectories
				u_correction[kk] += P_tau[kk, jj] * M_m * epsilon_auv[kk, jj]
		u_optimal = u_optimal + u_correction

	"""PI Sanity check"""
	if set_sanity_check == True:
		tau_optimal[:, 0] = x_auv
		var_x_test = np.zeros(shape=(N_horizon, 1))
		control_cost_test = np.zeros(shape=(N_horizon, 1))
		for kk in range(0, N_horizon - 1):  # Iterate over length of trajectory except of last entry
			tau_optimal[:, kk + 1] = Config.auv_dynamics(tau_optimal[:, kk], u_optimal[kk], 0, t_cstep, field_limits)
		for kk in range(0, N_horizon):  # Iterate over length of trajectory except of last entry
			A_test = Config.interpolation_matrix(tau_optimal[:, kk], n, p, lx, xg_min, yg_min, de)
			var_x_test[kk] = 1 / np.dot(A_test.T, var_x)
			control_cost_test[kk] = 0.5 * np.dot(np.array(u_optimal[kk]).T,
												 np.dot(R_cost, np.array(u_optimal[kk])))

		print('optimal path variance cost: ', var_x_test, 'optimal path control cost', control_cost_test)

	return u_optimal, tau_x, tau_optimal


def random_walk(x_auv):
	"""Random Walk"""
	max_step_size = ((0.2 * pi) / 1000)
	u_auv = max_step_size * randint(-500, 500)  # Control input for random walk
	x_auv = Config.auv_dynamics(x_auv, u_auv, 0.01)
	return x_auv
