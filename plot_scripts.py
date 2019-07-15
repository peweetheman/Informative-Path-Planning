"""
Plot scripts to visualize the PI-GMRF algorithm.

author: Andreas Rene Geist
email: andreas.geist@tuhh.de
website: https://github.com/AndReGeist
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import Config
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.lines import Line2D
import matplotlib.animation as animation


def initialize_animation1(true_field, x_auv, vmin, vmax, var_min, var_max, levels, PlotField, LabelVertices):
	alpha_prior = Config.alpha_prior
	kappa_prior = Config.kappa_prior

	plt.ion()
	fig1 = plt.figure(figsize=(8, 3))

	ax0 = fig1.add_subplot(221)
	cp = plt.contourf(true_field.x_field, true_field.y_field, true_field.z_field, vmin=vmin, vmax=vmax, levels=levels)
	plt.colorbar(cp)
	plt.title('True Field')
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')

	ax1 = fig1.add_subplot(222)
	plt.colorbar(cp)
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')
	ax1.set_title('GMRF Mean')

	ax2 = fig1.add_subplot(223)
	c22 = plt.contourf(true_field.x_field, true_field.y_field,
					   np.dot(np.diag(np.linspace(var_min, var_max, len(true_field.y_field), endpoint=True)),
							  np.ones((len(true_field.y_field), len(true_field.x_field)))), 10, vmin=var_min, vmax=var_max)
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')
	plt.colorbar(c22)
	ax2.set_title('GMRF Variance')

	# Initialize subplot Nr4
	number_prior = len(alpha_prior) * len(kappa_prior)
	_alpha_v, _kappa_v = np.meshgrid(alpha_prior, kappa_prior)
	alpha_v, kappa_v = _alpha_v.ravel(), _kappa_v.ravel()
	bottom = np.zeros(shape=(number_prior, 1))
	_x = np.arange(len(alpha_prior))
	_y = np.arange(len(kappa_prior))
	_xx, _yy = np.meshgrid(_x, _y)
	hyper_x, hyper_y = _xx.ravel(), _yy.ravel()
	colors = plt.cm.jet(np.arange(len(hyper_x)) / float(np.arange(len(hyper_x)).max()))

	ax3 = fig1.add_subplot(224, projection='3d')
	print("alpha prior len: ", len(alpha_prior))
	ticksx = np.arange(0.5, len(alpha_prior) + 0.5, 1)
	plt.xticks(ticksx, alpha_prior)
	plt.yticks(ticksx, kappa_prior)
	ax3.set_xlabel('alpha')
	ax3.set_ylabel('kappa')
	ax3.set_zlabel('p(theta)')
	ax3.set_title('GMRF Hyperparameter Estimate')

	plt.draw()
	trajectory_1 = np.array(x_auv).reshape(1, 3)
	return fig1, hyper_x, hyper_y, bottom, colors, trajectory_1


def update_animation1(sampling_control, pi_theta, fig1, x, y, bottom, colors, true_field, x_auv, mue_x, var_x, params, trajectory_1, tau_x, tau_optimal, vmin, vmax, var_min, var_max, levels, PlotField, LabelVertices):
	(lxf, lyf, dvx, dvy, lx, ly, n, p, de, l_TH, p_THETA, xg_min, xg_max, yg_min, yg_max) = params
	(x_min, x_max, y_min, y_max) = Config.field_dim
	# xf_grid = np.atleast_2d(np.linspace(x_min, x_max, lxf, endpoint=True)).T  # GMRF grid inside TRUE field
	# yf_grid = np.atleast_2d(np.linspace(y_min, y_max, lyf, endpoint=True)).T
	x_grid = np.atleast_2d(np.linspace(xg_min, xg_max, lx, endpoint=True)).T  # COMPLETE GMRF grid
	y_grid = np.atleast_2d(np.linspace(yg_min, yg_max, ly, endpoint=True)).T


	# Transform mean and variance into matrix for scatter
	xv, yv = np.meshgrid(x_grid, y_grid)
	mue_x_plot = mue_x[0:(lx * ly)].reshape((ly, lx))
	var_x_plot = var_x[0:(lx * ly)].reshape((ly, lx))
	# Create vectors for enumerating the GMRF nodes
	xv_list = xv.reshape((lx * ly, 1))
	yv_list = yv.reshape((lx * ly, 1))
	labels = ['{0}'.format(i) for i in range(lx * ly)]  # Labels for annotating GMRF nodes


	"""Plot True Field"""
	ax0 = fig1.add_subplot(221)
	ax0.set_title("True Field")
	cp = plt.contourf(true_field.x_field, true_field.y_field, true_field.z_field, vmin=vmin, vmax=vmax, levels=levels)
	plt.plot(x_auv[0], x_auv[1], marker='o', markerfacecolor='none')

	if PlotField == True:
		"""Plot GMRF mean"""
		ax1 = fig1.add_subplot(222)
		ax1.set_title("GMRF mean")
		c1 = ax1.contourf(np.linspace(xg_min, xg_max, num=lx, endpoint=True),
						  np.linspace(yg_min, yg_max, num=ly, endpoint=True),
						  mue_x_plot, vmin=vmin, vmax=vmax, levels=levels)
		plt.scatter(xv, yv, marker='+', facecolors='dimgrey')
		plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], "k")
		plt.plot(x_auv[0], x_auv[1], marker='o', markerfacecolor='none')

		if LabelVertices == True:
			# Label GMRF vertices
			for label, x, y in zip(labels, xv_list, yv_list):
				plt.annotate(
					label,
					xy=(x, y), xytext=(-2, 2),
					textcoords='offset points', ha='center', va='center',
					# bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
					# arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
				)

		"""Plot GMRF variance"""
		ax2 = fig1.add_subplot(223)
		ax2.set_title("GMRF variance")
		c2 = ax2.contourf(np.linspace(xg_min, xg_max, num=lx, endpoint=True),
						  np.linspace(yg_min, yg_max, num=ly, endpoint=True),
						  var_x_plot, 10, vmin=var_min, vmax=var_max)
		plt.scatter(xv, yv, marker='+', facecolors='dimgrey')
		plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], "k")
		plt.plot(trajectory_1[:, 0], trajectory_1[:, 1], color='yellow')
		for jj in range(0, Config.n_k):  # Iterate over all trajectories
			plt.plot(tau_x[0, :, jj], tau_x[1, :, jj], color='black')
		plt.plot(x_auv[0], x_auv[1], marker='o', markerfacecolor='none')
		plt.plot(tau_optimal[0, 1:], tau_optimal[1, 1:], color='blue')

	else:
		"""Plot GMRF mean"""
		ax1 = fig1.add_subplot(222)
		ax1.set_title("GMRF mean")
		c1 = ax1.contourf(np.linspace(x_min, x_max, num=lxf, endpoint=True),
						  np.linspace(y_min, y_max, num=lyf, endpoint=True),
						  mue_x_plot[dvy:(lyf + dvy), dvx:(lxf + dvx)], vmin=vmin, vmax=vmax, levels=levels)
		# plt.scatter(xv[dvy:(lyf+dvy), dvx:(lxf+dvx)], yv[dvy:(lyf+dvy), dvx:(lxf+dvx)], marker='+', facecolors='dimgrey')
		plt.plot(x_auv[0], x_auv[1], marker='o', markerfacecolor='none')

		"""Plot GMRF variance"""
		ax2 = fig1.add_subplot(223)
		ax2.set_title("GMRF variance")
		c2 = ax2.contourf(np.linspace(x_min, x_max, num=lxf, endpoint=True),
						  np.linspace(y_min, y_max, num=lyf, endpoint=True),
						  var_x_plot[dvy:(lyf + dvy), dvx:(lxf + dvx)], 10, vmin=var_min, vmax=var_max)
		# plt.scatter(xv[dvy:(lyf+dvy), dvx:(lxf+dvx)], yv[dvy:(lyf+dvy), dvx:(lxf+dvx)], marker='+', facecolors='dimgrey')
		plt.plot(trajectory_1[:, 0], trajectory_1[:, 1], color='yellow')

		if tau_x is not None:
			for jj in range(0, Config.n_k):  # Iterate over all trajectories
				plt.plot(tau_x[0, :, jj], tau_x[1, :, jj], color='black')
			"""Plot Hyperparameter estimate"""
			ax3 = fig1.add_subplot(224, projection='3d')
			ax3.set_title("Hyperparameter estimate")
		# colors = plt.cm.jet(pi_theta.flatten() / float(pi_theta.max()))  # Color height dependent
		# ax3.bar3d(x, y, bottom, 1, 1, pi_theta, color=colors, alpha=0.5)


		if sampling_control is not None:
			time1 = time.time()
			sampling_control.draw_graph(plot=plt)
			print("calc time sampling algo plotting: ", time.time() - time1)

		plt.quiver(x_auv[0], x_auv[1], np.cos(x_auv[2]), np.sin(x_auv[2]), width=.005)
		plt.plot(tau_optimal[0, :], tau_optimal[1, :], color='blue')

	fig1.canvas.draw_idle()
	plt.pause(0.05)
	plt.clf()
	return trajectory_1
