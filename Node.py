class Node(object):
	def __init__(self, x, y, angle):
		self.x = x
		self.y = y
		self.parent = None
		self.cost = 0.0
		self.dist = 0.0

		self.angle = angle
		self.path_x = []
		self.path_y = []
		self.path_angle = []

