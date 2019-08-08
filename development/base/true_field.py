# libraries
import numpy as np
from scipy.stats import kde
from matplotlib import pyplot as plt


class true_field():

	def __init__(self, step_size):
		self.scale = 10000.0  # random scale for graph values zi

		# create random seeded data, some correlation
		np.random.seed(2312)
		data1 = np.random.multivariate_normal(mean=[15, 15], cov=[[5, 2], [2, 3]], size=10)
		data2 = np.random.multivariate_normal(mean=[25, 5], cov=[[8, 2], [2, 7]], size=25)
		data3 = np.random.multivariate_normal(mean=[3, 15], cov=[[3, 2], [2, 4]], size=25)
		data4 = 30.0 * np.random.rand(200, 2)
		data = np.concatenate((data1, data2, data3, data4), axis=0)
		x, y = data.T

		# Evaluate a gaussian kde on a regular grid of num_points x num_points
		self.k = kde.gaussian_kde([x, y])
		self.xi, self.yi = np.mgrid[0:30:step_size, 0:30:step_size]
		self.zi = self.scale * self.k(np.vstack([self.xi.flatten(), self.yi.flatten()]))

		# Make the plot
		# plt.show()
		#
		# Change color palette
		# plt.pcolormesh(self.xi, self.yi, self.zi.reshape(self.xi.shape), cmap=plt.cm.Greens_r)
		# plt.show()

		# Add color bar
		# plt.pcolormesh(self.xi, self.yi, self.zi.reshape(self.xi.shape), cmap=plt.cm.Greens_r)

	def draw(self, plt):
		plt.contourf(self.xi, self.yi, self.zi.reshape(self.xi.shape))
		plt.colorbar()

	def get_measurement(self, locations):
		return (self.scale * self.k.evaluate(locations)).flatten() 

	def get_covariance(self):
		return self.k.covariance


# def main():
# 	field = true_field()
# 	field.draw(plt)
# 	plt.show()
#
# main()