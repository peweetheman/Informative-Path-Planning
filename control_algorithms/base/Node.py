import numpy as np

class Node(object):
	def __init__(self, pose):
		self.parent = None
		self.cost = 0.0
		self.path_var = 0.0
		self.total_var = 0.0
		self.path_dist = 0.0
		self.dist = 0.0
		self.pose = np.array(pose)           # x-coordinate, y-coordinate, angle-coordinate (in radians) constitutes a pose
		self.u = []                # list of controls required to get from parent node to this node

		# used for implementations currently as tracking path from parent to this node
		self.path_x = []
		self.path_y = []
		self.path_angle = []

	def __eq__(self, other):
		return np.array_equal(self.pose, other.pose) and (self.dist == other.dist)

	def __len__(self):
		return len(self.pose)

	def __getitem__(self, i):
		return self.pose[i]

	def __repr__(self):
		return 'Node Pos({}, {}, {})'.format(self.pose[0], self.pose[1], self.pose[2])