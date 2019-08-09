import numpy as np
from scipy.interpolate import interp1d
from numpy import polyfit, poly1d
import matplotlib.pyplot as plt
import os
"""
Patrick Phillips summer 2019
email: pphill10@u.rochester.edu
website: https://github.com/peweetheman
"""

pathlength = "30.0"
runtime = "1.5"
PI, RRT, PRM, RRT_star, PRM_star = False, True, True, True, True
data_PI, data_PRM, data_PRM_star, data_RRT, data_RRT_star = None, None, None, None, None

if PI:
	PI_filename = os.path.join('data', 'PI_runtime' + '0.25' + '_pathlength' + pathlength + '_' + '0' + '.npy')
	data_PI = np.load(PI_filename)
	data_PI = np.delete(data_PI, 0, 1)
if PRM_star:
	PRM_star_filename = os.path.join('data', 'PRM_star_runtime' + runtime + '_pathlength' + pathlength + '_' + '0' + '.npy')
	data_PRM_star = np.load(PRM_star_filename)
	data_PRM_star = np.delete(data_PRM_star, 0, 1)
if RRT_star:
	RRT_star_filename = os.path.join('data', 'RRT_star_runtime' + runtime + '_pathlength' + pathlength + '_' + '0' + '.npy')
	data_RRT_star = np.load(RRT_star_filename)
	data_RRT_star = np.delete(data_RRT_star, 0, 1)
if PRM:
	PRM_filename = os.path.join('data', 'PRM_runtime' + runtime + '_pathlength' + pathlength + '_' + '0' + '.npy')
	data_PRM = np.load(PRM_filename)
	data_PRM = np.delete(data_PRM, 0, 1)
if RRT:
	RRT_filename = os.path.join('data', 'RRT_runtime' + runtime + '_pathlength' + pathlength + '_' + '0' + '.npy')
	data_RRT = np.load(RRT_filename)
	data_RRT = np.delete(data_RRT, 0, 1)


for i in range(1, 50):
	if PI:
		PI_filename = os.path.join('data', 'PI_runtime' + '0.25' + '_pathlength' + pathlength + '_' + str(i+48) + '.npy')
		new_data_PI = np.load(PI_filename)
		new_data_PI = np.delete(new_data_PI, 0, 1)
		data_PI = np.concatenate((data_PI, new_data_PI), axis=1)
	if PRM_star:
		PRM_star_filename = os.path.join('data', 'PRM_star_runtime' + runtime + '_pathlength' + pathlength + '_' + str(i) + '.npy')
		new_data_PRM_star = np.load(PRM_star_filename)
		new_data_PRM_star = np.delete(new_data_PRM_star, 0, 1)
		data_PRM_star = np.concatenate((data_PRM_star, new_data_PRM_star), axis=1)
	if RRT_star:
		RRT_star_filename = os.path.join('data', 'RRT_star_runtime' + runtime + '_pathlength' + pathlength + '_' + str(i) + '.npy')
		new_data_RRT_star = np.load(RRT_star_filename)
		new_data_RRT_star = np.delete(new_data_RRT_star, 0, 1)
		data_RRT_star = np.concatenate((data_RRT_star, new_data_RRT_star), axis=1)
	if PRM:
		PRM_filename = os.path.join('data', 'PRM_runtime' + runtime + '_pathlength' + pathlength + '_' + str(i) + '.npy')
		new_data_PRM = np.load(PRM_filename)
		new_data_PRM = np.delete(new_data_PRM, 0, 1)
		data_PRM = np.concatenate((data_PRM, new_data_PRM), axis=1)
	if RRT:
		RRT_filename = os.path.join('data', 'RRT_runtime' + runtime + '_pathlength' + pathlength + '_' + str(i) + '.npy')
		new_data_RRT = np.load(RRT_filename)
		new_data_RRT = np.delete(new_data_RRT, 0, 1)
		data_RRT = np.concatenate((data_RRT, new_data_RRT), axis=1)

if PI:
	print("PI data", data_PI.shape)
	data_PI = data_PI[:, data_PI[0].argsort()]
if PRM_star:
	print("PRM_star data", data_PRM_star.shape)
	data_PRM_star = data_PRM_star[:, data_PRM_star[0].argsort()]
if RRT_star:
	print("RRT_star data", data_RRT_star.shape)
	data_RRT_star = data_RRT_star[:, data_RRT_star[0].argsort()]
if PRM:
	print("PRM data", data_PRM.shape)
	data_PRM = data_PRM[:, data_PRM[0].argsort()]
if RRT:
	print("RRT data", data_RRT.shape)
	data_RRT = data_RRT[:, data_RRT[0].argsort()]

fig1 = plt.figure(figsize=(9, 4))
xnew = np.linspace(1.1, float(pathlength)-1.5, num=100, endpoint=True)

#### FIRST PLOT OF TOTAL VARIANCE (INCLUDES VARIANCE OF COEFFICIENTS IN GMRF MEAN FUNCTION APPROX)
ax0 = fig1.add_subplot(221)
ax0.set_title('Path length vs. Total Variance (averaged over 100 trials on randomized fields)')
plt.xlabel('Path Length (m)')
plt.ylabel('Total Variance')

if PI:
	f_PI = interp1d(data_PI[0, :], data_PI[1, :])
	plt.plot(xnew, f_PI(xnew), '-', label='PI interpolation')
if PRM_star:
	f_PRM_star = interp1d(data_PRM_star[0, :], data_PRM_star[1, :])
	plt.plot(xnew, f_PRM_star(xnew), '-', label='PRM_star interpolation')
if RRT_star:
	f_RRT_star = interp1d(data_RRT_star[0, :], data_RRT_star[1, :])
	plt.plot(xnew, f_RRT_star(xnew), '-', label='RRT_star interpolation')
	# p_RRT_star = np.poly1d(np.polyfit(data_RRT_star[0, :], data_RRT_star[1, :], 10))
	# plt.plot(xnew, p_RRT_star(xnew), '--')
if PRM:
	f_PRM = interp1d(data_PRM[0, :], data_PRM[1, :])
	plt.plot(xnew, f_PRM(xnew), '-', label='PRM interpolation')
if RRT:
	f_RRT = interp1d(data_RRT[0, :], data_RRT[1, :])
	plt.plot(xnew, f_RRT(xnew), '-', label='RRT interpolation')

handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles[::-1], labels[::-1])

#### SECOND PLOT OF JUST THE  FIELD VARIANCE
ax1 = fig1.add_subplot(222)
ax1.set_title('Path length vs. Field Variance (averaged over 100 trials on randomized fields)')
plt.xlabel('Path Length (m)')
plt.ylabel('Field Variance')

# plt.scatter(data_PI[0, :], data_PI[2, :], color='blue', label='PI data')
# plt.scatter(data_PRM_star[0, :], data_PRM_star[2, :], color='yellow', label='PRM_star data')

if PI:
	f_PI = interp1d(data_PI[0, :], data_PI[2, :])
	plt.plot(xnew, f_PI(xnew), '-', label='PI interpolation')
if PRM_star:
	f_PRM_star = interp1d(data_PRM_star[0, :], data_PRM_star[2, :])
	plt.plot(xnew, f_PRM_star(xnew), '-', label='PRM_star interpolation')
if RRT_star:
	f_RRT_star = interp1d(data_RRT_star[0, :], data_RRT_star[2, :])
	plt.plot(xnew, f_RRT_star(xnew), '-', label='RRT_star interpolation')
if PRM:
	f_PRM = interp1d(data_PRM[0, :], data_PRM[2, :])
	plt.plot(xnew, f_PRM(xnew), '-', label='PRM interpolation')
if RRT:
	f_RRT = interp1d(data_RRT[0, :], data_RRT[2, :])
	plt.plot(xnew, f_RRT(xnew), '-', label='RRT interpolation')

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1])

#### THIRD PLOT OF ROOT MEAN SQUARED ERROR BETWEEN GMRF APPROXIMATION AND TRUE FIELD
ax2 = fig1.add_subplot(223)
ax2.set_title('Path length vs. RMSE (averaged over 100 trials on randomized fields)')
plt.xlabel('Path Length (m)')
plt.ylabel('RMSE')

# plt.scatter(data_PI[0, :], data_PI[3, :], color='blue', label='PI data')
# plt.scatter(data_PRM_star[0, :], data_PRM_star[3, :], color='yellow', label='PRM_star data')

if PI:
	f_PI = interp1d(data_PI[0, :], data_PI[3, :])
	plt.plot(xnew, f_PI(xnew), '-', label='PI interpolation')
if PRM_star:
	f_PRM_star = interp1d(data_PRM_star[0, :], data_PRM_star[3, :])
	plt.plot(xnew, f_PRM_star(xnew), '-', label='PRM_star interpolation')
if RRT_star:
	f_RRT_star = interp1d(data_RRT_star[0, :], data_RRT_star[3, :])
	plt.plot(xnew, f_RRT_star(xnew), '-', label='RRT_star interpolation')
	# p_RRT_star = np.poly1d(np.polyfit(data_RRT_star[0, :], data_RRT_star[3, :], 10))
	# plt.plot(xnew, p_RRT_star(xnew), '--')
if PRM:
	f_PRM = interp1d(data_PRM[0, :], data_PRM[3, :])
	plt.plot(xnew, f_PRM(xnew), '-', label='PRM interpolation')
if RRT:
	f_RRT = interp1d(data_RRT[0, :], data_RRT[3, :])
	plt.plot(xnew, f_RRT(xnew), '-', label='RRT interpolation')

handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1])


plt.show()


