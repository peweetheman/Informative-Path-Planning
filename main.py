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
from scipy import exp, sin, cos, sqrt, pi, interpolate
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# AUV starting state
x_auv = Config.x_auv

# Initialize Field
x_field, y_field, z_field = Config.true_field(True)
field_limits = (x_field[-1], y_field[-1])

# Calculate and set plot parameters
plot_settings = {"vmin":np.amin(z_field) - 0.5, "vmax":np.amax(z_field) + 0.5, "var_min":0,
                 "var_max":5, "levels":np.linspace(np.amin(z_field) - 0.5, np.amax(z_field) + 0.5, 20),
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

        # Calculate optimal control path
        u_optimal, tau_x, tau_optimal = control_scripts.pi_controller(x_auv, u_optimal, var_x, Config.pi_parameters, gmrf1.params, field_limits, Config.set_sanity_check)
        time_5 = time.time()
        print("Calc. time PI: /", "{0:.2f}".format(time_5-time_4))

        # Plot new GMRF belief and optimal control path
        plot_scripts.update_animation1(pi_theta, fig1, hyper_x, hyper_y, bottom, colors, x_field, y_field, z_field, x_auv, mue_x, var_x, gmrf1.params, trajectory_1, tau_x, tau_optimal, **plot_settings)
        time_6 = time.time()
        print("Calc. time Plot: /", "{0:.2f}".format(time_6-time_5))

    #     # Store plot values"""
    #     stored_mue_x[:, [n_storage]] = mue_x
    #     stored_var_x[:, [n_storage]] = var_x
    #     #store_var_median_iqr[0, [n_storage]] = np.median(var_x)
    #     #store_var_median_iqr[1, [n_storage]] = np.median(var_x)
    #     #store_var_median_iqr[2, [n_storage]] = np.median(var_x)
    #
    #     stored_phi_theta[:, [n_storage]] = pi_theta
    #     time_4 = time.time()
    #     print("--- %s seconds --- GMRF time" % (time_4 - time_3))
    #     n_storage += 1

    #     time_5 = time.time()
    #     print("--- %s seconds --- PI time" % (time_5 - time_4))

