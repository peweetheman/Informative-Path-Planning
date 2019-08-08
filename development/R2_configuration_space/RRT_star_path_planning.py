import copy
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from control_algorithms.base.Node import Node



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
		self.start = Node(start)
		self.end = Node(end)
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

			if self.check_collision(new_node, self.obstacles):
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
			sample = Node([random.uniform(self.space[0], self.space[1]),
						  random.uniform(self.space[0], self.space[1])])
		else:  # end point sampling
			sample = Node([self.end.pose[0], self.end.pose[1]])
		return sample

	def steer(self, sample, nearest_node):
		# take nearest_node and expand in direction of sample
		vec = np.array((sample.pose[0] - nearest_node.pose[0], sample.pose[1] - nearest_node.pose[1]))
		unit_vec = vec / np.linalg.norm(vec)
		new_node = Node([sample.pose[0], sample.pose[1]])
		currentDistance = dist(sample, nearest_node)
		# find a point within growth of nearest_node, and closest to sample
		if currentDistance <= self.growth:
			pass
		else:
			new_node.pose[0] = nearest_node.pose[0] + self.growth * unit_vec[0]
			new_node.pose[1] = nearest_node.pose[1] + self.growth * unit_vec[1]

		new_node.cost = float("inf")
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
				dlist.append(near_node.cost + dis)
			else:
				dlist.append(float("inf"))

		mincost = min(dlist)
		min_node = near_nodes[dlist.index(mincost)]

		if mincost == float("inf"):
			print("mincost is inf")
			return
		new_node.cost = mincost
		new_node.parent = min_node

	def get_best_last_node(self):

		to_goal_distances = [self.calc_dist_to_end(node.pose[0], node.pose[1]) for node in self.node_list]
		endinds = [to_goal_distances.index(i) for i in to_goal_distances if i <= self.growth]

		if not endinds:
			return None
		mincost = min([self.node_list[i].cost for i in endinds])
		for i in endinds:
			if self.node_list[i].cost == mincost:
				return self.node_list[i]
		return None

	def get_path(self, last_node):
		path = [last_node]
		while last_node.parent is not None:
			path.append(last_node.parent)
			last_node = last_node.parent
		return path

	def calc_dist_to_end(self, x, y):
		return np.linalg.norm([x - self.end.pose[0], y - self.end.pose[1]])

	def get_near_nodes(self, new_node):
		# gamma_star = 2(1+1/d) ** (1/d) volume(free)/volume(total) ** 1/d and we need gamma > gamma_star
		# for asymptotical completeness see Kalman 2011. gamma = 1 satisfies
		d = 2  # dimension of the self.space
		nnode = len(self.node_list)
		r = min(50.0 * ((math.log(nnode) / nnode)) ** (1 / d), self.growth * 20.0)
		dlist = [(node.pose[0] - new_node.pose[0]) ** 2 +
				 (node.pose[1] - new_node.pose[1]) ** 2 for node in self.node_list]
		near_nodes = [self.node_list[dlist.index(d)] for d in dlist if d <= r]
		return near_nodes

	def rewire(self, new_node, near_nodes):
		for near_node in near_nodes:
			dx = new_node.pose[0] - near_node.pose[0]
			dy = new_node.pose[1] - near_node.pose[1]
			d = math.sqrt(dx ** 2 + dy ** 2)

			scost = new_node.cost + d

			if near_node.cost > scost:
				if self.check_collision_path(near_node, new_node):
					near_node.parent = new_node
					near_node.cost = scost

	def check_collision_path(self, node1, node2):
		# check for collision on path from node1 to node2
		dis = dist(node1, node2)
		dx = node2.pose[0] - node1.pose[0]
		dy = node2.pose[1] - node1.pose[1]
		angle = math.atan2(dy, dx)
		temp_node = copy.deepcopy(node1)
		for i in range(int(dis / self.growth)):
			temp_node.pose[0] += self.growth * math.cos(angle)
			temp_node.pose[1] += self.growth * math.sin(angle)
			if not self.check_collision(temp_node, self.obstacles):
				return False

		return True

	def check_collision(self, node, obstacles):
		for (x, y, side) in obstacles:
			if ((node.pose[0] > x - .8 * side / 2) & (node.pose[0] < x + .8 * side / 2) & (node.pose[1] > y - side / 2) & (
					node.pose[1] < y + side / 2)):
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
			plt.plot(node.pose[0], node.pose[1], "yH")
			#plt.text(node.pose[0], node.pose[1], str(node.cost), color="red", fontsize=12)

			if node.parent is not None:
				plt.plot([node.pose[0], node.parent.pose[0]], [
					node.pose[1], node.parent.pose[1]], "-k")

		for (x, y, side) in self.obstacles:
			plt.plot(x, y, "sk", ms=8 * side)

		# draw field
		#true_field1 = true_field(1)
		#true_field1.draw(plt)

		plt.plot(self.start.pose[0], self.start.pose[1], "oy")
		plt.plot(self.end.pose[0], self.end.pose[1], "or")
		plt.axis([0, 30, 0, 30])
		plt.grid(True)
		plt.title("RRT* (distance cost function)")
		plt.pause(0.001)   # need for animation


def dist(node1, node2):
	# returns distance between two nodes
	return math.sqrt((node2.pose[0] - node1.pose[0]) ** 2 + (node2.pose[1] - node1.pose[1]) ** 2)


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
	rrt_star = RRT_star(start=[15, 28], end=[15, 5], obstacles=obstacles, space=[0, 30])
	path = rrt_star.rrt_star_algorithm()

	# plotting code
	rrt_star.draw_graph()
	if path is not None:
		plt.plot([node.pose[0] for node in path], [node.pose[1] for node in path], '-r')
	plt.grid(True)
	print("--- %s seconds ---" % (time.time() - start_time))
	plt.show()


if __name__ == '__main__':
	main()
