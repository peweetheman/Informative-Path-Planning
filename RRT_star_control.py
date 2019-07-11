from Node import Node
from true_field import true_field
import random
import math
import copy
import numpy as np
import time
import matplotlib.pyplot as plt


class RRT_star:
	# Basic RRT* algorithm using distance as cost function

	def __init__(self, start, space, obstacles, growth=1, max_iter=100, max_dist=20):
		"""
		:param start: [x,y] starting location
		:param space: [min,max] bounds on square space
		:param obstacles: list of square obstacles
		:param growth: size of growth each new sample
		:param max_iter: max number of iterations for algorithm
		"""
		self.start = Node(start)
		self.node_list = [self.start]
		self.space = space
		self.growth = growth
		self.max_iter = max_iter
		self.max_dist = max_dist
		self.obstacles = obstacles

	def control_algorithm(self):
		for i in range(self.max_iter):
			sample = self.get_sample()
			nearest_node = self.nearest_node(sample)
			new_node = self.steer(nearest_node, sample)

			if self.check_collision(new_node):
				near_nodes = self.get_near_nodes(new_node)
				self.set_parent(new_node, near_nodes)
				self.node_list.append(new_node)
				self.rewire(new_node, near_nodes)
			# draw added edges
			self.draw_graph()

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
		# take source_node and steer towards destination node
		vec = destination_node.pose - source_node.pose
		unit_vec = vec / np.linalg.norm(vec)
		new_node = Node(destination_node.pose)
		currentDistance = dist(source_node, destination_node)
		# find a point within growth of nearest_node, and closest to sample
		if currentDistance <= self.growth:
			pass
		else:
			new_node.pose = source_node.pose + self.growth * unit_vec
		new_node.cost = float("inf")
		new_node.parent = None
		return new_node

	def set_parent(self, new_node, near_nodes):
		# connects new_node along a minimum cost(distance) path
		if not near_nodes:
			return
		costlist = []
		for near_node in near_nodes:
			cost = self.cost(new_node, near_node)
			if self.check_collision_path(near_node, new_node):
				costlist.append(near_node.cost + cost)
			else:
				costlist.append(float("inf"))

		mincost = min(costlist)
		min_node = near_nodes[costlist.index(mincost)]
		if mincost == float("inf"):
			print("mincost is inf")
			return
		new_node.cost = mincost
		new_node.parent = min_node

	def get_best_last_node(self):
		cost_list = [node.cost for node in self.node_list]
		best_node = self.node_list[cost_list.index(min(cost_list))]
		return best_node

	def get_path(self, last_node):
		path = [last_node]
		u_optimal = []
		tau_optimal = last_node.path     # matrix of shape of 3, # of timesteps. Represents x, y, angle as a column for each timestep
		while True:
			path.append(last_node.parent)
			u_optimal = u_optimal + last_node.u            # NOT WORKING and NOT USED!!!!
			last_node = last_node.parent
			if last_node is None:
				break
			tau_add = last_node.path
			tau_optimal = np.concatenate((tau_add, tau_optimal), axis=0)
			print("tau optimal: ", tau_optimal)
		print("transpose: ", tau_optimal.T)
		return path, u_optimal, tau_optimal.T

	def get_near_nodes(self, new_node):
		# gamma_star = 2(1+1/d) ** (1/d) volume(free)/volume(total) ** 1/d. We need a gamma > gamma_star for asymptotical completeness. See Kalman 2011. gamma = 1 satisfies
		d = 2  # dimension of the self.space
		nnode = len(self.node_list)
		r = min(50.0 * ((math.log(nnode) / nnode)) ** (1 / d), self.growth * 20.0)

		dlist = [dist(node, new_node) for node in self.node_list]
		near_nodes = [self.node_list[dlist.index(i)] for i in dlist if i <= r ** 2]
		return near_nodes

	def rewire(self, new_node, near_nodes):
		for near_node in near_nodes:
			temp_cost = new_node.cost + self.cost(new_node, near_node)

			if near_node.cost > temp_cost:
				if self.check_collision_path(near_node, new_node):
					near_node.parent = new_node
					near_node.cost = temp_cost

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
			if not self.check_collision(temp_node):
				return False
		return True

	def check_collision(self, node):
		for (x, y, side) in self.obstacles:
			if ((node.pose[0] > x - .8 * side / 2) & (node.pose[0] < x + .8 * side / 2) & (node.pose[1] > y - side / 2) & (
					node.pose[1] < y + side / 2)):
				return False  # collision

		return True  # safe

	def nearest_node(self, sample):
		dlist = [dist(node, sample) for node in self.node_list]
		min_node = self.node_list[dlist.index(min(dlist))]
		return min_node

	def cost(self, node1, node2):
		return dist(node1, node2)

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
		plt.axis([0, 30, 0, 30])
		plt.grid(True)
		plt.title("RRT* (distance cost function)")
		plt.pause(0.001)   # need for animation

def dist(node1, node2):
	# returns distance between two nodes
	return math.sqrt((node2.pose[0] - node1.pose[0]) ** 2 + (node2.pose[1] - node1.pose[1]) ** 2 + 3 * (node1.pose[2] - node2.pose[2]) ** 2)


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
	rrt_star = RRT_star(start=[15.0, 28.0, np.deg2rad(0.0)],  space=[0, 30, 0, 30], obstacles=obstacles)
	path, u_optimal, tau_optimal = rrt_star.control_algorithm()
	print(u_optimal)


	# plotting code
	rrt_star.draw_graph()
	plt.plot([node.pose[0] for node in path], [node.pose[1] for node in path], '-r')
	plt.grid(True)
	print("--- %s seconds ---" % (time.time() - start_time))
	plt.show()


if __name__ == '__main__':
	main()
