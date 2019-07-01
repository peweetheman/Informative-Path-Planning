from Node import Node
import dubins_path_planner
from true_field import true_field
import random
import math
import copy
import numpy as np
import time
import matplotlib.pyplot as plt


class RRT_star:
	# Basic RRT* algorithm using distance as cost function

	def __init__(self, start, end, space, obstacles, growth=0.5, max_iter=500, end_sample_percent=15):
		"""
		:param start: [x,y] starting location
		:param end: [x,y[ ending location
		:param space: [min,max] bounds on square space
		:param obstacles: list of square obstacles
		:param growth: size of growth each new sample
		:param max_iter: max number of iterations for algorithm
		:param end_sample_percent: percent chance to get sample from goal location
		"""
		self.start = Node(start[0], start[1], start[2])
		self.end = Node(end[0], end[1], end[2])
		self.space = space
		self.growth = growth
		self.end_sample_percent = end_sample_percent
		self.max_iter = max_iter
		self.obstacles = obstacles

	def rrt_star_algorithm(self):
		self.node_list = [self.start]
		for i in range(self.max_iter):
			sample = self.get_sample()
			nearest_node = self.nearest_node(sample)
			new_node = self.steer(sample, nearest_node)

			if self.check_collision(new_node):
				near_nodes = self.get_near_nodes(new_node)
				self.set_parent(new_node, near_nodes)
				self.node_list.append(new_node)
				self.rewire(new_node, near_nodes)
			# draw added edges
			# self.draw_graph()

		# generate path
		last_node = self.get_best_last_node()
		if last_node is None:
			return None
		path = self.get_path(last_node)
		return path

	def get_sample(self):

		if random.randint(0, 100) > self.end_sample_percent:
			sample = Node(random.uniform(self.space[0], self.space[1]),
						  random.uniform(self.space[0], self.space[1]))
		else:  # end point sampling
			sample = Node(self.end.x, self.end.y)

		return sample

	def steer(self, sample, nearest_node):
		# take nearest_node and expand in direction of sample
		angle = math.atan2(sample.y - nearest_node.y, sample.x - nearest_node.x)
		new_node = Node(sample.x, sample.y)
		currentDistance = dist(sample, nearest_node)
		# find a point within growth of nearest_node, and closest to sample
		if currentDistance <= self.growth:
			pass
		else:
			new_node.x = nearest_node.x + self.growth * math.cos(angle)
			new_node.y = nearest_node.y + self.growth * math.sin(angle)
		new_node.dist = float("inf")
		new_node.parent = None
		return new_node

	def set_parent(self, new_node, near_nodes):
		# connects new_node along a minimum cost(distance) path
		if not near_nodes:
			return
		dlist = []
		for near_node in near_nodes:
			dis = dist(new_node, near_node)
			if self.check_collision_path(near_node, new_node):
				dlist.append(near_node.dist + dis)
			else:
				dlist.append(float("inf"))

		mincost = min(dlist)
		min_node = near_nodes[dlist.index(mincost)]

		if mincost == float("inf"):
			print("mincost is inf")
			return
		new_node.dist = mincost
		new_node.parent = min_node

	def get_best_last_node(self):
		cost_list = [node.cost for node in self.node_list]
		best_node = self.node_list[cost_list.index(min(cost_list))]
		return best_node

	def get_path(self, last_node):
		path = [last_node]
		while last_node.parent is not None:
			path.append(last_node.parent)
			last_node = last_node.parent
		return path

	def calc_dist_to_end(self, x, y):
		return np.linalg.norm([x - self.end.x, y - self.end.y])

	def get_near_nodes(self, new_node):
		# gamma_star = 2(1+1/d) ** (1/d) volume(free)/volume(total) ** 1/d and we need gamma > gamma_star
		# for asymptotical completeness see Kalman 2011. gamma = 1 satisfies
		d = 2  # dimension of the self.space
		nnode = len(self.node_list)
		r = min(50.0 * ((math.log(nnode) / nnode)) ** (1 / d), self.growth * 20.0)
		dlist = [(node.x - new_node.x) ** 2 +
				 (node.y - new_node.y) ** 2 for node in self.node_list]
		near_nodes = [self.node_list[dlist.index(i)] for i in dlist if i <= r ** 2]
		return near_nodes

	def rewire(self, new_node, near_nodes):
		for near_node in near_nodes:
			dx = new_node.x - near_node.x
			dy = new_node.y - near_node.y
			d = math.sqrt(dx ** 2 + dy ** 2)

			scost = new_node.dist + d

			if near_node.dist > scost:
				if self.check_collision_path(near_node, new_node):
					near_node.parent = new_node
					near_node.dist = scost

	def check_collision_path(self, node1, node2):
		# check for collision on path from node1 to node2
		dis = dist(node1, node2)
		dx = node2.x - node1.x
		dy = node2.y - node1.y
		angle = math.atan2(dy, dx)
		temp_node = copy.deepcopy(node1)
		for i in range(int(dis / self.growth)):
			temp_node.x += self.growth * math.cos(angle)
			temp_node.y += self.growth * math.sin(angle)
			if not self.check_collision(temp_node):
				return False

		return True

	def check_collision(self, node):
		if self.obstacles is not None:
			for (x, y, side) in self.obstacles:
				if ((node.x > x - .8 * side / 2) & (node.x < x + .8 * side / 2) & (node.y > y - side / 2) & (
						node.y < y + side / 2)):
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
			plt.plot(node.x, node.y, "yH")
			if node.parent is not None:
				plt.plot(node.path_x, node.path_y, "-g")

		if self.obstacles is not None:
			for (x, y, side) in self.obstacles:
				plt.plot(x, y, "sk", ms=8 * side)

		# draw field
		# true_field1 = true_field(1)
		# true_field1.draw(plt)

		plt.plot(self.start.x, self.start.y, "oy")
		plt.plot(self.end.x, self.end.y, "or")
		plt.axis([0, 30, 0, 30])
		plt.grid(True)
		plt.title("RRT* (distance cost function)")
		plt.pause(0.01)   # need for animation


def dist(node1, node2):
	# returns distance between two nodes
	return math.sqrt((node2.x - node1.x) ** 2 + (node2.y - node1.y) ** 2 + (node1.angle - node2.angle) ** 2)


def main():
	start_time = time.time()
	# squares of [x,y,side length]
	obstacles = [
		(15, 17, 5),
		(4, 10, 4),
		(7, 23, 3),
		(22, 12, 5),
		(9, 15, 4)]

	# calling RRT*
	rrt_star = RRT_star(start=[15.0, 28.0, np.deg2rad(0.0)], end=[15.0, 5.0, np.deg2rad(0.0)], obstacles=None, space=[0, 30])
	path = rrt_star.rrt_star_algorithm()

	# plotting code
	rrt_star.draw_graph()
	plt.plot([node.x for node in path], [node.y for node in path], '-r')
	plt.grid(True)
	print("--- %s seconds ---" % (time.time() - start_time))
	plt.show()


if __name__ == '__main__':
	main()
