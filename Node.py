import numpy as np

class Node(object):
	def __init__(self, pose):
		self.parent = None
		self.cost = 0.0
		self.dist = 0.0
		self.pose = np.array(pose)           # x-coordinate, y-coordinate, angle-coordinate constitutes a pose
		self.path = []             # list of poses on the way from parent node to this node
		self.u = []                # list of controls required to get from parent node to this node

		# used for PRM implementation currently
		self.path_x = []
		self.path_y = []
		self.path_angle = []

	def __eq__(self, other):
		return self.__dict__ == other.__dict__
