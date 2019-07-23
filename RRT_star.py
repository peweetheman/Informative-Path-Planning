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
	# Basic RRT* algorithm using distance as dist function

	def __init__(self, start, goal, space, obstacles, growth=4.0, max_iter=100, max_dist=40, max_curvature=1.0, end_sample_percent=15):
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
		self.end = Node(goal)
		self.space = space
		self.growth = growth
		self.max_iter = max_iter
		self.max_dist = max_dist
		self.obstacles = obstacles
		self.node_list = [self.start]
		self.max_curvature = max_curvature
		self.local_planner_time = 0.0
		self.end_sample_percent = end_sample_percent

	def control_algorithm(self):
		# RRT* ALGORITHM
		for i in range(self.max_iter):
			sample_node = self.get_sample()
			nearest_node = self.nearest_node(sample_node)
			new_node = self.steer(nearest_node, sample_node)

			if self.check_collision(new_node.pose[0], new_node.pose[1]):
				near_nodes = self.get_near_nodes(new_node)
				self.set_parent(new_node, near_nodes)
				if new_node.parent is None:    # no possible path from any of the near nodes
					continue
				self.node_list.append(new_node)
				self.rewire(new_node, near_nodes)
			# animate added edges
				#self.draw_near(near_nodes, new_node, new_node)
				print("portion of dubins: ", self.local_planner_time/time.time())

		# generate path
		last_node = self.get_best_last_node()
		if last_node is None:
			return None, None, None
		path, u_optimal, tau_optimal = self.get_path(last_node)
		return path, u_optimal, tau_optimal

	def get_sample(self):
		if random.randint(0, 100) > self.end_sample_percent:
			sample = Node([random.uniform(self.space[0], self.space[1]),
						  random.uniform(self.space[0], self.space[1]),
						  random.uniform(-math.pi, math.pi)])
		else:  # end point sampling
			sample = Node(self.end.pose)
		return sample

	def steer_2(self, source_node, dest_node):
		# take source_node and steer towards destination node
		dtheta = np.array([-self.max_curvature, 0, self.max_curvature])
		dx = np.cos(source_node.pose[2] + dtheta/2)
		dy = np.sin(source_node.pose[2] + dtheta/2)
		vec = np.array([dx, dy, dtheta])
		new_node_list = [Node(source_node.pose + self.growth * vec[:, 0]), Node(source_node.pose + self.growth * vec[:, 1]), Node(source_node.pose + self.growth * vec[:, 2])]
		d_list = [dist(new_node, dest_node) for new_node in new_node_list]
		mindist = min(d_list)
		new_node = new_node_list[d_list.index(mindist)]
		new_node.cost = float("inf")
		new_node.parent = None
		return new_node

	def steer(self, source_node, dest_node):
		# take source_node and steer towards destination node
		dtheta = random.uniform(-self.max_curvature, self.max_curvature)
		dx = np.cos(source_node.pose[2] + dtheta/2)
		dy = np.sin(source_node.pose[2] + dtheta/2)
		vec = np.array([dx, dy, dtheta])
		new_node = Node(source_node.pose + self.growth * vec)
		return new_node

	def set_parent(self, new_node, near_nodes):
		# connects new_node along a minimum cost path
		if not near_nodes:
			near_nodes.append(self.nearest_node(new_node))
		cost_list = []
		for near_node in near_nodes:
			# CALL TO LOCAL PATH PLANNER
			p1 = time.time()
			px, py, pangle, mode, plength, u = plan.dubins_path_planning(near_node.pose[0], near_node.pose[1], near_node.pose[2], new_node.pose[0], new_node.pose[1], new_node.pose[2], self.max_curvature)
			p2 = time.time()
			self.local_planner_time += (p2 - p1)
			cost = self.cost(px, py, pangle, plength)
			if self.check_collision_path(px, py) and self.max_dist >= plength + near_node.dist:
				cost_list.append(near_node.cost + cost)
			else:
				cost_list.append(float("inf"))

		mincost = min(cost_list)
		min_node = near_nodes[cost_list.index(mincost)]
		if mincost == float("inf"):
			print("no parent found")
			# self.draw_near(near_nodes, new_node, new_node)
			return
		new_node.cost = mincost
		new_node.parent = min_node

		# CALL TO LOCAL PATH PLANNER
		p1 = time.time()
		px, py, pangle, mode, plength, u = plan.dubins_path_planning(min_node.pose[0], min_node.pose[1], min_node.pose[2], new_node.pose[0], new_node.pose[1], new_node.pose[2], self.max_curvature)
		p2 = time.time()
		self.local_planner_time += (p2 - p1)
		new_node.path_x = px
		new_node.path_y = py
		new_node.path_angle = pangle
		new_node.u = u
		new_node.dist = new_node.parent.dist + plength

	def get_best_last_node(self):
		cost_list = [node.cost for node in self.node_list if dist(node, self.end) < self.growth]
		if cost_list is not False:
			return None
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
		r = min(25.0 * ((math.log(nnode) / nnode)) ** (1 / d), self.growth * 10.0)
		dlist = [dist(node, new_node) for node in self.node_list]
		near_nodes = [self.node_list[dlist.index(i)] for i in dlist if i <= r]
		return near_nodes

	def rewire(self, new_node, near_nodes):

		for near_node in near_nodes:
			p1 = time.time()
			px, py, pangle, mode, plength, u = plan.dubins_path_planning(near_node.pose[0], near_node.pose[1], near_node.pose[2], new_node.pose[0], new_node.pose[1], new_node.pose[2], self.max_curvature)
			p2 = time.time()
			self.local_planner_time += (p2 - p1)
			temp_cost = new_node.cost + self.cost(px, py, pangle, plength)
			if near_node.cost > temp_cost and self.max_dist >= new_node.dist + plength:
				if self.check_collision_path(px, py):
					#print("REWIRE")
					#self.draw_near(near_nodes, new_node, near_node)
					near_node.parent = new_node
					near_node.cost = temp_cost
					near_node.path_x = px
					near_node.path_y = py
					near_node.path_angle = pangle
					near_node.u = u
					near_node.dist = near_node.parent.dist + plength
					#self.draw_near(near_nodes, new_node, near_node)

	def check_collision_path(self, px, py):
		# check for collision on path
		for kk in range(len(px)):
			if not self.check_collision(px[kk], py[kk]):
				return False    # collision
		return True      # safe

	def check_collision(self, nx, ny):
		if self.obstacles is not None:
			for (x, y, side) in self.obstacles:
					if ((nx > x - .8 * side / 2) & (nx < x + .8 * side / 2) & (ny > y - side / 2) & (
								ny < y + side / 2)):
						return False  # collision
		return True  # safe

	def nearest_node(self, sample):
		dlist = [dist(node, sample) for node in self.node_list]
		min_node = self.node_list[dlist.index(min(dlist))]
		return min_node

	def cost(self, px, py, pangle, plength):
		return plength

	def draw_graph(self, plot=None):
		for node in self.node_list:
			plt.quiver(node.pose[0], node.pose[1], math.cos(node.pose[2]), math.sin(node.pose[2]))
			if node.parent is not None:
				plt.plot(node.path_x, node.path_y, "-g")

		if self.obstacles is not None:
			for (x, y, side) in self.obstacles:
				plt.plot(x, y, "sk", ms=8 * side)

		plt.quiver(self.start.pose[0], self.start.pose[1], math.cos(self.start.pose[2]), math.sin(self.start.pose[2]), color="y")
		plt.quiver(self.end.pose[0], self.end.pose[1], math.cos(self.end.pose[2]), math.sin(self.end.pose[2]), color="r")
		plt.axis(self.space)
		plt.grid(True)
		plt.title("RRT* (Dubin's Curves, distance cost)")
		plt.pause(.1)   # need for animation

	def draw_near(self, near_nodes, new_node, temp_node):
		plt.clf()

		for node in self.node_list:
			plt.quiver(node.pose[0], node.pose[1], math.cos(node.pose[2]), math.sin(node.pose[2]))
			plt.text(node.pose[0], node.pose[1], str(node.dist), color="red", fontsize=12)
			if node.parent is not None:
				plt.plot(node.path_x, node.path_y, "-g")

		for node in near_nodes:
			plt.quiver(node.pose[0], node.pose[1], math.cos(node.pose[2]), math.sin(node.pose[2]), color="c")
		plt.pause(.1)
		if self.obstacles is not None:
			for (x, y, side) in self.obstacles:
				plt.plot(x, y, "sk", ms=8 * side)

		plt.quiver(temp_node.pose[0], temp_node.pose[1], math.cos(temp_node.pose[2]), math.sin(temp_node.pose[2]), color="m")
		plt.quiver(new_node.pose[0], new_node.pose[1], math.cos(new_node.pose[2]), math.sin(new_node.pose[2]), color="m")
		plt.plot(new_node.path_x, new_node.path_y, "m")

		plt.quiver(self.start.pose[0], self.start.pose[1], math.cos(self.start.pose[2]), math.sin(self.start.pose[2]), color="y")
		plt.quiver(self.end.pose[0], self.end.pose[1], math.cos(self.end.pose[2]), math.sin(self.end.pose[2]), color="r")
		plt.axis(self.space)
		plt.grid(True)
		plt.title("RRT* (Dubin's Curves)")
		plt.waitforbuttonpress()  # need for animation


def dist(node1, node2):
	# returns distance between two nodes
	return math.sqrt((node2.pose[0] - node1.pose[0]) ** 2 + (node2.pose[1] - node1.pose[1]) ** 2 + 5 * (node1.pose[2] - node2.pose[2]) ** 2)


def main():
	start_time = time.time()
	# obstacles are squares of [x,y,side length]
	obstacles = [
		(15, 17, 7),
		(4, 10, 6),
		(7, 23, 9),
		(22, 12, 5),
		(9, 15, 4)]

	# calling RRT*
	rrt_star = RRT_star(start=[15.0, 28.0, np.deg2rad(0.0)], goal=[15.0, 3.0, np.deg2rad(0.0)], space=[0, 30, 0, 30], obstacles=obstacles)
	path, u_optimal, tau_optimal = rrt_star.control_algorithm()
	print(u_optimal)

	# plotting code
	rrt_star.draw_graph()
	# if path is not None:
		# plt.plot([node.pose[0] for node in path], [node.pose[1] for node in path], '-r')
	plt.grid(True)
	print("--- %s seconds ---" % (time.time() - start_time))
	plt.show()


if __name__ == '__main__':
	main()
