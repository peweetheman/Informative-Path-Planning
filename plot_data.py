import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("PRM*_data1.txt")
fig1 = plt.figure(figsize=(9, 4))
print(data.shape)
ax0 = fig1.add_subplot(221)
ax0.set_title('Path length vs. Total Variance')
plt.xlabel('Path Length (m)')
plt.ylabel('Total Variance')
plt.plot(data[0, 1:], data[1, 1:], color='blue')

ax1 = fig1.add_subplot(222)
ax1.set_title('Path length vs. Field Variance')
plt.xlabel('Path Length (m)')
plt.ylabel('Field Variance')
plt.plot(data[0, 1:], data[2, 1:], color='blue')

ax2 = fig1.add_subplot(223)
ax2.set_title('Path length vs. RMSE')
plt.xlabel('Path Length (m)')
plt.ylabel('RMSE')
plt.plot(data[0, 1:], data[3, 1:], color='blue')

plt.show()


