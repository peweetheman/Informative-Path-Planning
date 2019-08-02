import Config
import numpy as np
from scipy import pi
from random import randint
import time

class random_walk:
	def random_walk(self, x_auv):
		"""Random Walk"""
		max_step_size = ((0.2 * pi) / 1000)
		u_auv = max_step_size * randint(-500, 500)  # Control input for random walk
		x_auv = Config.auv_dynamics(x_auv, u_auv, 0.01)
		return x_auv
