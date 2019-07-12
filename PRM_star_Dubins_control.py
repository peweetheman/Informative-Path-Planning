from Node import Node
import dubins_path_planner as plan
from true_field import true_field
import random
import math
import copy
import numpy as np
import time
import matplotlib.pyplot as plt


class PRM_star_Dubins:
	# PRM* algorithm using distance as dist function

	def __init__(self, start, space, obstacles, var_x, gmrf_params, growth=0.5, max_iter=20, max_dist=20):
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
		self.space = space
		self.growth = growth
		self.max_iter = max_iter
		self.max_dist = max_dist
		self.obstacles = obstacles
		self.node_list = [self.start]
		self.var_x = var_x
		self.gmrf_params = gmrf_params

	def control_algorithm(self):
		for i in range(self.max_iter):
			sample_node = self.get_sample()

			if self.check_collision(sample_node):
				near_nodes = self.get_near_nodes(sample_node)
				new_node = self.set_parent(sample_node, near_nodes)
				if new_node is None:  # no possible path from any of the near nodes
					continue
				self.node_list.append(new_node)
				self.rewire(new_node, near_nodes)
			# animate added edges
			# self.draw_graph()

		# generate path
		last_node = self.get_best_last_node()
		if last_node is None:
			return None
		path, u_optimal, tau_optimal = self.get_path(last_node)
		return path, u_optimal, tau_optimal

	def get_sample(self):
		sample = Node([random.uniform(self.space[0], self.space[1]),
					  random.uniform(self.space[2], self.space[3]),
					  random.uniform(-math.pi, math.pi)])
		return sample

	def steer(self, source_node, destination_node):
		# take source_node and find path to destination_node
		curvature = 1.0
		px, py, pangle, mode, plength, u = plan.dubins_path_planning(source_node.pose[0], source_node.pose[1], source_node.pose[2], destination_node.pose[0], destination_node.pose[1], destination_node.pose[2], curvature)
		new_node = copy.deepcopy(source_node)
		new_node.pose = destination_node.pose
		new_node.path_x = px
		new_node.path_y = py
		new_node.path_angle = pangle
		new_node.u = u
		new_node.dist += plength
		new_node.cost += -1.0
		new_node.parent = source_node
		return new_node

	def set_parent(self, sample_node, near_nodes):
		# connects new_node along a minimum cost path
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
		cost_list = [node.cost for node in self.node_list]
		best_node = self.node_list[cost_list.index(min(cost_list))]
		return best_node

	def get_path(self, last_node):
		path = [last_node]
		u_optimal = []
		tau_optimal = np.vstack((last_node.path_x, last_node.path_y, last_node.path_angle))
		while True:
			path.append(last_node.parent)
			u_optimal = u_optimal + last_node.u
			last_node = last_node.parent
			if last_node is None:
				break
			tau_add = np.vstack((last_node.path_x, last_node.path_y, last_node.path_angle))
			tau_optimal = np.concatenate((tau_add, tau_optimal), axis=1)
		return path, u_optimal, tau_optimal

	def calc_dist_to_end(self, x, y):
		return np.linalg.norm([x - self.end.pose[0], y - self.end.pose[1]])

	def get_near_nodes(self, new_node):
		# gamma_star = 2(1+1/d) ** (1/d) volume(free)/volume(total) ** 1/d and we need gamma > gamma_star
		# for asymptotical completeness see Kalman 2011. gamma = 1 satisfies
		d = 2  # dimension of the self.space
		nnode = len(self.node_list)
		r = min(30.0 * ((math.log(nnode) / nnode)) ** (1 / d), self.growth * 10.0)
		dlist = [dist(new_node, node) for node in self.node_list]
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

	def cost(self, gmrf, path):
		return 0

	def draw_graph(self, plot=None):
		# plt.clf()
		if plot is None:  # use built in plt
			for node in self.node_list:
				plt.quiver(node.pose[0], node.pose[1], math.cos(node.pose[2]), math.sin(node.pose[2]))
				if node.parent is not None:
					plt.plot(node.path_x, node.path_y, "-g")

			if self.obstacles is not None:
				for (x, y, side) in self.obstacles:
					plt.plot(x, y, "sk", ms=8 * side)

			plt.quiver(self.start.pose[0], self.start.pose[1], math.cos(self.start.pose[2]), math.sin(self.start.pose[2]), color="b")
			plt.axis(self.space)
			plt.grid(True)
			plt.title("PRM* (distance cost function)")
			plt.pause(.1)  # need for animation

		else:  # use plot of calling
			for node in self.node_list:
				plot.quiver(node.pose[0], node.pose[1], math.cos(node.pose[2]), math.sin(node.pose[2]))
				if node.parent is not None:
					for (x, y) in zip(node.path_x, node.path_y):
						plt.plot(x, y, "yH", markersize=2)

			if self.obstacles is not None:
				for (x, y, side) in self.obstacles:
					plot.plot(x, y, "sk", ms=8 * side)

			plot.quiver(self.start.pose[0], self.start.pose[1], math.cos(self.start.pose[2]), math.sin(self.start.pose[2]), color="b")
			plot.axis(self.space)
			plot.grid(True)
			plot.title("PRM* (distance cost function)")
			plot.pause(.1)  # need for animation


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

	# calling PRM*
	PRM_star = PRM_star_Dubins(start=[15.0, 28.0, np.deg2rad(0.0)], space=[0, 30, 0, 30], obstacles=obstacles)
	path, u_optimal, tau_optimal = PRM_star.control_algorithm()
	print(u_optimal)

	# plotting code
	PRM_star.draw_graph()
	if path[-1] is not None:
		plt.plot([node.pose[0] for node in path], [node.pose[1] for node in path], '-r')
	plt.grid(True)
	print("--- %s seconds ---" % (time.time() - start_time))
	plt.show()


if __name__ == '__main__':
	main()
