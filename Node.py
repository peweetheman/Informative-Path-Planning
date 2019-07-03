class Node(object):
	def __init__(self, x, y, angle):
		self.x = x
		self.y = y
		self.parent = None
		self.cost = 0.0
		self.dist = 0.0

		self.is_near = False
		self.angle = angle
		self.path_x = []
		self.path_y = []
		self.path_angle = []
		self.u = []

	def __eq__(self, other):
		return self.__dict__ == other.__dict__
