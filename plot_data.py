import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
"""
Patrick Phillips summer 2019
email: pphill10@u.rochester.edu
website: https://github.com/peweetheman
"""

PI_filename = os.path.join('data', 'PI_data' + '0' + '.npy')
data_PI = np.load(PI_filename)
data_PI = np.delete(data_PI, 0, 1)

PRM_filename = os.path.join('data', 'PRM_data' + '0' + '.npy')
data_PRM = np.load(PRM_filename)
data_PRM = np.delete(data_PRM, 0, 1)

for i in range(1, 44):
	PI_filename = os.path.join('data', 'PI_data' + str(i) + '.npy')
	new_data_PI = np.load(PI_filename)
	new_data_PI = np.delete(new_data_PI, 0, 1)
	data_PI = np.concatenate((data_PI, new_data_PI), axis=1)

	PRM_filename = os.path.join('data', 'PRM_data' + str(i) + '.npy')
	new_data_PRM = np.load(PRM_filename)
	new_data_PRM = np.delete(new_data_PRM, 0, 1)
	data_PRM = np.concatenate((data_PRM, new_data_PRM), axis=1)

data_PI = data_PI[:, data_PI[0].argsort()]
data_PRM = data_PRM[:, data_PRM[0].argsort()]
print(data_PI.shape)
print(data_PRM.shape)

fig1 = plt.figure(figsize=(9, 4))

#### FIRST PLOT OF TOTAL VARIANCE (INCLUDES VARIANCE OF COEFFICIENTS IN GMRF MEAN FUNCTION APPROX)
ax0 = fig1.add_subplot(221)
ax0.set_title('Path length vs. Total Variance (averaged over 100 trials on randomized fields)')
plt.xlabel('Path Length (m)')
plt.ylabel('Total Variance')

plt.scatter(data_PI[0, :], data_PI[1, :], color='blue')
plt.scatter(data_PRM[0, :], data_PRM[1, :], color='yellow')

f_PI = interp1d(data_PI[0, :], data_PI[1, :])
f_PRM = interp1d(data_PRM[0, :], data_PRM[1, :])

xnew = np.linspace(1.1, 29, num=100, endpoint=True)
plt.plot(xnew, f_PI(xnew), '-')
plt.plot(xnew, f_PRM(xnew), '--')

#### SECOND PLOT OF JUST THE  FIELD VARIANCE
ax1 = fig1.add_subplot(222)
ax1.set_title('Path length vs. Field Variance (averaged over 100 trials on randomized fields)')
plt.xlabel('Path Length (m)')
plt.ylabel('Field Variance')

plt.scatter(data_PI[0, :], data_PI[2, :], color='blue')
plt.scatter(data_PRM[0, :], data_PRM[2, :], color='yellow')

f_PI = interp1d(data_PI[0, :], data_PI[2, :])
f_PRM = interp1d(data_PRM[0, :], data_PRM[2, :])

xnew = np.linspace(1.1, 29, num=100, endpoint=True)
plt.plot(xnew, f_PI(xnew), '-')
plt.plot(xnew, f_PRM(xnew), '--')

#### THIRD PLOT OF ROOT MEAN SQUARED ERROR BETWEEN GMRF APPROXIMATION AND TRUE FIELD
ax2 = fig1.add_subplot(223)
ax2.set_title('Path length vs. RMSE (averaged over 100 trials on randomized fields)')
plt.xlabel('Path Length (m)')
plt.ylabel('RMSE')

plt.scatter(data_PI[0, :], data_PI[3, :], color='blue')
plt.scatter(data_PRM[0, :], data_PRM[3, :], color='yellow')

f_PI = interp1d(data_PI[0, :], data_PI[3, :])
f_PRM = interp1d(data_PRM[0, :], data_PRM[3, :])

xnew = np.linspace(1.1, 29, num=100, endpoint=True)
plt.plot(xnew, f_PI(xnew), '-')
plt.plot(xnew, f_PRM(xnew), '--')

plt.show()


