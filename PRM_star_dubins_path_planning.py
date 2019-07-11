from Node import Node
import dubins_path_planner as plan
from true_field import true_field
import random
import math
import copy
import numpy as np
import time
import matplotlib.pyplot as plt


class RRT_star:
	# RRT ALGORITHM USING DUBINS PATHS FOR FINDING PATH FROM START TO END LOCATION

	def __init__(self, start, end, space, obstacles, growth=0.5, max_iter=50, end_sample_percent=5):
		"""
		:param start: [x,y] starting location
		:param end: [x,y[ ending location
		:param space: [min,max] bounds on square space
		:param obstacles: list of square obstacles
		:param growth: size of growth each new sample
		:param max_iter: max number of iterations for algorithm
		:param end_sample_percent: percent chance to get sample from goal location
		"""
		self.start = Node(start)
		self.end = Node(end)
		self.space = space
		self.growth = growth
		self.end_sample_percent = end_sample_percent
		self.max_iter = max_iter
		self.obstacles = obstacles
		self.node_list = [self.start]

	def rrt_star_algorithm(self):
		for i in range(self.max_iter):
			sample_node = self.get_sample()

			if self.check_collision(sample_node):
				print("here: ", sample_node.pose)
				near_nodes = self.get_near_nodes(sample_node)
				new_node = self.set_parent(sample_node, near_nodes)
				if new_node is None:    # no possible path from any of the near nodes
					continue
				self.node_list.append(new_node)
				self.rewire(new_node, near_nodes)
			# draw added edges
			self.draw_graph()

		# generate path
		last_node = self.get_best_last_node()
		if last_node is None:
			return None
		path = self.get_path(last_node)
		return path

	def get_sample(self):

		if random.randint(0, 100) > self.end_sample_percent:
			sample = Node([random.uniform(self.space[0], self.space[1]),
						  random.uniform(self.space[0], self.space[1]),
						  random.uniform(-math.pi, math.pi)])
		else:  # end point sampling
			sample = Node(self.end.pose)
		return sample

	def steer(self, source_node, destination_node):
		# take source_node and find path to destination_node
		curvature = 1.0
		px, py, pyaw, mode, clen, u = plan.dubins_path_planning(source_node.pose[0], source_node.pose[1], source_node.pose[2], destination_node.pose[0], destination_node.pose[1], destination_node.pose[2], curvature)
		new_node = copy.deepcopy(source_node)
		new_node.pose = destination_node.pose
		new_node.path_x = px
		new_node.path_y = py
		new_node.path_yaw = pyaw
		new_node.u = u
		new_node.cost = np.sum(u)
		new_node.dist += clen
		new_node.parent = source_node
		return new_node

	def set_parent(self, sample_node, near_nodes):
		# connects new_node along a minimum dist path
		if not near_nodes:
			near_nodes.append(self.nearest_node(sample_node))
		dlist = []
		for near_node in near_nodes:
			temp_node = self.steer(near_node, sample_node)
			if self.check_collision(temp_node):
				dlist.append(temp_node.dist)
			else:
				dlist.append(float("inf"))

		mindist = min(dlist)
		min_node = near_nodes[dlist.index(mindist)]
		if mindist == float("inf"):
			return None
		new_node = self.steer(min_node, sample_node)
		return new_node

	def get_best_last_node(self):
		dist_list = [node.dist for node in self.node_list]
		best_node = self.node_list[dist_list.index(min(dist_list))]
		return best_node

	def get_path(self, last_node):
		path = [last_node]
		while last_node.parent is not None:
			path.append(last_node.parent)
			last_node = last_node.parent
		return path

	def calc_dist_to_end(self, x, y):
		return np.linalg.norm([x - self.end.pose[0], y - self.end.pose[1]])

	def get_near_nodes(self, sample_node):
		# gamma_star = 2(1+1/d) ** (1/d) volume(free)/volume(total) ** 1/d and we need gamma > gamma_star
		# for asymptotical completeness see Kalman 2011. gamma = 1 satisfies
		d = 2  # dimension of the self.space
		nnode = len(self.node_list)
		r = min(50.0 * ((math.log(nnode) / nnode)) ** (1 / d), self.growth * 20.0)
		dlist = [dist(node, sample_node) for node in self.node_list]
		near_nodes = [self.node_list[dlist.index(i)] for i in dlist if i <= r ** 2]
		return near_nodes

	def rewire(self, new_node, near_nodes):
		for near_node in near_nodes:
			temp_node = self.steer(new_node, near_node)
			if near_node.dist > temp_node.dist and self.check_collision(temp_node):
				near_node.__dict__.update(vars(temp_node))

	def check_collision(self, node):
		if self.obstacles is not None:
			for (x, y, side) in self.obstacles:
				for (nx, ny) in zip(node.path_x, node.path_y):
					if ((nx > x - .8 * side / 2) & (nx < x + .8 * side / 2) & (ny > y - side / 2) & (
								ny < y + side / 2)):
						return False  # collision

		return True  # safe

	def nearest_node(self, sample):
		dlist = [dist(node, sample) for node in self.node_list]
		min_node = self.node_list[dlist.index(min(dlist))]
		return min_node

	def cost(self, node1, node2):
		return 0

	def draw_graph(self):
		plt.clf()

		for node in self.node_list:
			plt.quiver(node.pose[0], node.pose[1], math.cos(node.pose[2]), math.sin(node.pose[2]))
			if node.parent is not None:
				plt.plot(node.path_x, node.path_y, "-g")

		if self.obstacles is not None:
			for (x, y, side) in self.obstacles:
				plt.plot(x, y, "sk", ms=8 * side)

		plt.quiver(self.start.pose[0], self.start.pose[1], math.cos(self.start.pose[2]), math.sin(self.start.pose[2]), color="b")
		plt.quiver(self.end.pose[0], self.end.pose[1], math.cos(self.end.pose[2]), math.sin(self.end.pose[2]), color="r")

		plt.axis(self.space)
		plt.grid(True)
		plt.title("RRT* (distance dist function)")

		plt.pause(.1)   # need for animation


def dist(node1, node2):
	# returns distance between two nodes
	return math.sqrt((node2.pose[0] - node1.pose[0]) ** 2 + (node2.pose[1] - node1.pose[1]) ** 2 + 2 * (node1.pose[2] - node2.pose[2]) ** 2)


def main():
	start_time = time.time()
	# squares of [x,y,side length]
	obstacles = [
		(15, 17, 7),
		(4, 10, 6),
		(7, 23, 9),
		(22, 12, 5),
		(9, 15, 4)]

	# calling RRT*
	rrt_star = RRT_star(start=[15.0, 28.0, np.deg2rad(0.0)], end=[15.0, 5.0, np.deg2rad(0.0)], space=[0, 30, 0, 30], obstacles=obstacles)
	path = rrt_star.rrt_star_algorithm()

	# plotting code
	rrt_star.draw_graph()
	plt.plot([node.pose[0] for node in path], [node.pose[1] for node in path], '-r')
	plt.grid(True)
	print("--- %s seconds ---" % (time.time() - start_time))
	plt.show()


if __name__ == '__main__':
	main()
