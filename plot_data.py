import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

filename = 'PI_data' + '0' + '.txt.npy'
data = np.load(filename)
data = np.delete(data, 0, 1)

for i in range(1, 100):
	filename = 'PI_data' + str(i) + '.txt.npy'
	new_data = np.load(filename)
	new_data = np.delete(new_data, 0, 1)
	data = np.concatenate((data, new_data), axis=1)

fig1 = plt.figure(figsize=(9, 4))

#### FIRST PLOT OF TOTAL VARIANCE (INCLUDES VARIANCE OF COEFFICIENTS IN GMRF MEAN FUNCTION APPROX)
ax0 = fig1.add_subplot(221)
ax0.set_title('Path length vs. Total Variance (averaged over 100 trials on randomized fields)')
plt.xlabel('Path Length (m)')
plt.ylabel('Total Variance')

data = data[:, data[0].argsort()]

plt.scatter(data[0, :], data[1, :], color='blue')

f = interp1d(data[0, :], data[1, :])
xnew = np.linspace(1.1, 19.9, num=50, endpoint=True)
plt.plot(xnew, f(xnew), '-')

#### SECOND PLOT OF JUST THE  FIELD VARIANCE
ax1 = fig1.add_subplot(222)
ax1.set_title('Path length vs. Field Variance (averaged over 100 trials on randomized fields)')
plt.xlabel('Path Length (m)')
plt.ylabel('Field Variance')
plt.scatter(data[0, :], data[2, :], color='blue')

f = interp1d(data[0, :], data[2, :])
xnew = np.linspace(1.1, 19.9, num=50, endpoint=True)
plt.plot(xnew, f(xnew), '-')


#### THIRD PLOT OF ROOT MEAN SQUARED ERROR BETWEEN GMRF APPROXIMATION AND TRUE FIELD
ax2 = fig1.add_subplot(223)
ax2.set_title('Path length vs. RMSE (averaged over 100 trials on randomized fields)')
plt.xlabel('Path Length (m)')
plt.ylabel('RMSE')
plt.scatter(data[0, :], data[3, :], color='blue')

f = interp1d(data[0, :], data[3, :])
xnew = np.linspace(1.1, 19.9, num=50, endpoint=True)
plt.plot(xnew, f(xnew), '-')

plt.show()


