import Config
from gp_scripts import gp_scripts
import scipy
import numpy as np

# AUV starting state
x_auv = Config.x_auv
"""
Patrick Phillips summer 2019
email: pphill10@u.rochester.edu
website: https://github.com/peweetheman
"""

class true_field:
	# Calculate TEMPERATURE FIELD (Ground truth)
	def __init__(self, set_field):
		if set_field == True:
			"""Analytic field"""
			# z = np.array([[10, 10.625, 12.5, 15.625, 20],
			#               [5.625, 6.25, 8.125, 11.25, 15.625],
			#               [3, 3.125, 4, 12, 12.5],
			#               [5, 2, 3.125, 10, 10.625],
			#               [5, 8, 11, 12, 10]])
			z = np.array([[2, 4, 6, 7, 8],
						  [2.1, 5, 7, 11.25, 9.5],
						  [3, 5.6, 8.5, 17, 14.5],
						  [2.5, 5.4, 6.9, 9, 8],
						  [2, 2.3, 4, 6, 7.5]])
			X = np.atleast_2d([0, 2, 4, 6, 10])  # Specifies column coordinates of field
			Y = np.atleast_2d([0, 1, 3, 4, 5])  # Specifies row coordinates of field
			self.x_field = np.arange(Config.field_dim[0], Config.field_dim[1], 1e-2)
			self.y_field = np.arange(Config.field_dim[2], Config.field_dim[3], 1e-2)
			self.f = scipy.interpolate.griddata(X, Y, z, method='cubic')
			self.z_field = self.f(self.x_field, self.y_field)

		if set_field == False:
			"""Field from GMRF"""
			car_var = False
			kappa_field = [1]  # Kappa
			alpha_field = [0.01]  # Alpha

			self.f = gp_scripts.sample_from_GMRF(Config.gmrf_dim, kappa_field, alpha_field, car_var, plot_gmrf=False)

			self.x_field = np.arange(Config.field_dim[0], Config.field_dim[1], 1e-2)
			self.y_field = np.arange(Config.field_dim[2], Config.field_dim[3], 1e-2)
			self.z_field = self.f(self.x_field, self.y_field)
