import numpy as np

class Node(object):
	def __init__(self, pose):
		self.parent = None
		self.cost = 0.0
		self.dist = 0.0
		self.pose = np.array(pose)           # x-coordinate, y-coordinate, angle-coordinate constitutes a pose
		self.u = []                # list of controls required to get from parent node to this node

		#self.path = []             # list of poses on the way from parent node to this node
		# used for implementations currently
		self.path_x = []
		self.path_y = []
		self.path_angle = []

	def __eq__(self, other):
		return self.__dict__ == other.__dict__

	def __len__(self):
		return len(self.pose)

	def __getitem__(self, i):
		return self.pose[i]

	def __repr__(self):
		return 'Item({}, {}, {})'.format(self.coords[0], self.coords[1], self.data)