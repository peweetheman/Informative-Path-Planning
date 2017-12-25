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
import plot_scripts

import time
import scipy
from scipy import sqrt
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# AUV starting state
x_auv = Config.x_auv

"""Initialize Field"""
# Calculate TEMPERATURE FIELD (Ground truth)
def true_field(set_field):
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
        x_field = np.arange(Config.field_dim[0], Config.field_dim[1], 1e-2)
        y_field = np.arange(Config.field_dim[2], Config.field_dim[3], 1e-2)
        f = scipy.interpolate.interp2d(X, Y, z, kind='cubic')
        z_field = f(x_field, y_field)
        return x_field, y_field, z_field
    if set_field == False:
        """Field from GMRF"""
        car_var = False
        kappa_field = [1]  # Kappa
        alpha_field = [0.01]  # Alpha

        f = gp_scripts.sample_from_GMRF(Config.gmrf_dim, kappa_field, alpha_field, car_var, plot_gmrf=False)

        x_field = np.arange(Config.field_dim[0], Config.field_dim[1], 1e-2)
        y_field = np.arange(Config.field_dim[2], Config.field_dim[3], 1e-2)
        z_field = f(x_field, y_field)
        return x_field, y_field, z_field


x_field, y_field, z_field = true_field(False)
field_limits = (x_field[-1], y_field[-1])
print("field", np.amin(z_field), np.amax(z_field))
# Calculate and set plot parameters
plot_settings = {"vmin":np.amin(z_field) - 0.1, "vmax":np.amax(z_field) + 0.1, "var_min":0,
                 "var_max":3, "levels":np.linspace(np.amin(z_field) - 0.1, np.amax(z_field) + 0.1, 20),
                 "PlotField":False, "LabelVertices":True}

# Initialize plots
fig1, hyper_x, hyper_y, bottom, colors, trajectory_1 = plot_scripts.initialize_animation1(x_field, y_field, z_field, x_auv, **plot_settings)

# Initialize GMRF
time_1 = time.time()
gmrf1 = gp_scripts.GMRF(Config.gmrf_dim, Config.alpha_prior, Config.kappa_prior, Config.set_Q_init)
#print(gmrf1.__dict__)
time_2 = time.time()
print("Time for GMRF init: /", "{0:.2f}".format(time_2-time_1))
# Initialize Controller
u_optimal = np.zeros(shape=(Config.N_horizon, 1))

"""#####################################################################################"""
"""START SIMULATION"""
for time_in_ms in range(0, Config.simulation_end_time):  # 1200 ms sekunden
    if time_in_ms % Config.sample_time_gmrf < 0.0000001:

        # Calculate next AUV state
        x_auv = Config.auv_dynamics(x_auv, u_optimal[0], 0, Config.sample_time_gmrf/100, field_limits)
        trajectory_1 = np.vstack([trajectory_1, x_auv])

        # Compute discrete observation vector and new observation
        sd_obs = [int((x_auv[1]) * 1e2), int((x_auv[0]) * 1e2)]
        y_t = np.array(z_field[sd_obs[0], sd_obs[1]]) + np.random.normal(loc=0.0, scale=sqrt(Config.sigma_w_squ), size=1)

        # Update GMRF belief
        time_3 = time.time()
        mue_x, var_x, pi_theta = gmrf1.gmrf_bayese_update(x_auv, y_t)
        time_4 = time.time()
        print("Calc. time GMRF: /", "{0:.2f}".format(time_4-time_3))
        print("var_x: ",var_x)

        # Calculate optimal control path
        u_optimal, tau_x, tau_optimal = control_scripts.pi_controller(x_auv, u_optimal, var_x, Config.pi_parameters, gmrf1.params, field_limits, Config.set_sanity_check)
        time_5 = time.time()
        print("Calc. time PI: /", "{0:.2f}".format(time_5-time_4))

        # Plot new GMRF belief and optimal control path
        plot_scripts.update_animation1(pi_theta, fig1, hyper_x, hyper_y, bottom, colors, x_field, y_field, z_field, x_auv, mue_x, var_x, gmrf1.params, trajectory_1, tau_x, tau_optimal, **plot_settings)
        time_6 = time.time()
        print("Calc. time Plot: /", "{0:.2f}".format(time_6-time_5))

