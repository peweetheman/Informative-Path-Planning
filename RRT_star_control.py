from Node import Node
from true_field import true_field
import dubins_path_planner as plan
import Config
import random
import math
import copy
import numpy as np
import time
import matplotlib.pyplot as plt


class RRT_star:
	# Basic RRT* algorithm using distance as cost function

	def __init__(self, start, space, obstacles, var_x=None, gmrf_params=None, growth=1.5, max_iter=50, max_dist=40, min_dist=3, max_curvature=.8, plot=None):
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
		self.min_dist = min_dist
		self.obstacles = obstacles
		self.var_x = var_x
		self.gmrf_params = gmrf_params
		self.max_curvature = max_curvature
		self.local_planner_time = 0.0
		self.method_time = 0.0
		self.plot = plot

	def control_algorithm(self):
		for i in range(self.max_iter):
			sample = self.get_sample()
			nearest_node = self.nearest_node(sample)
			new_node = self.steer(nearest_node, sample)

			if self.check_collision(new_node.pose[0], new_node.pose[1]):
				near_nodes = self.get_near_nodes(new_node)
				self.set_parent(new_node, near_nodes)
				if new_node.parent is None:    # no possible path from any of the near nodes
					print("new_node.parent is none (printed in main algo)")
					continue
				self.node_list.append(new_node)
				self.rewire(new_node, near_nodes)
		# draw added edges
			#self.draw_graph(self.plot)
		# generate path
		last_node = self.get_best_last_node()
		path, u_optimal, tau_optimal = self.get_path(last_node)

		return path, u_optimal, tau_optimal, self.local_planner_time, self.method_time

	def get_sample(self):
		sample = Node([random.uniform(self.space[0], self.space[1]),
					  random.uniform(self.space[2], self.space[3]),
					  random.uniform(-math.pi, math.pi)])
		return sample

	def steer(self, source_node, dest_node):
		# take source_node and steer towards destination node
		dtheta = random.uniform(-self.max_curvature/2, self.max_curvature/2)
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
			px, py, pangle, mode, plength, u = plan.dubins_path_planning(near_node.pose[0], near_node.pose[1], near_node.pose[2], new_node.pose[0], new_node.pose[1], new_node.pose[2], self.max_curvature)
			path_var = self.path_var(px, py, pangle)
			if self.check_collision_path(px, py) and self.max_dist >= plength + near_node.dist:
				cost_list.append((near_node.total_var + path_var) / (near_node.dist + plength))
			else:
				cost_list.append(float("inf"))

		mincost = min(cost_list)
		min_node = near_nodes[cost_list.index(mincost)]
		if mincost == float("inf"):
			print("no parent found (in set)")
			return
		new_node.parent = min_node
		# CALL TO LOCAL PATH PLANNER
		px, py, pangle, mode, plength, u = plan.dubins_path_planning(min_node.pose[0], min_node.pose[1], min_node.pose[2], new_node.pose[0], new_node.pose[1], new_node.pose[2], self.max_curvature)
		new_node.path_x = px
		new_node.path_y = py
		new_node.path_angle = pangle
		new_node.total_var = new_node.parent.total_var + self.path_var(px, py, pangle)
		new_node.u = u
		new_node.dist = new_node.parent.dist + plength

	def get_best_last_node(self):
		cost_list = []
		for node in self.node_list:
			if node.dist >= self.min_dist:
				cost_list.append(node.total_var/node.dist)
			else:
				cost_list.append(float("inf"))
		best_node = self.node_list[cost_list.index(min(cost_list))]
		return best_node

	def get_path(self, last_node):
		path = [last_node]
		u_optimal = np.array((last_node.u))
		tau_optimal = np.vstack((last_node.path_x, last_node.path_y, last_node.path_angle))
		while True:
			last_node = last_node.parent
			if last_node is None:
				break
			u_optimal = np.concatenate((u_optimal, last_node.u), axis=0)
			tau_add = np.vstack((last_node.path_x, last_node.path_y, last_node.path_angle))
			tau_optimal = np.concatenate((tau_add, tau_optimal), axis=1)
			path.append(last_node)
		return path, u_optimal, tau_optimal

	def get_near_nodes(self, new_node):
		# gamma_star = 2(1+1/d) ** (1/d) volume(free)/volume(total) ** 1/d. We need a gamma > gamma_star for asymptotical completeness. See Kalman 2011. gamma = 1 satisfies
		d = 2  # dimension of the self.space
		nnode = len(self.node_list)
		r = min(20.0 * ((math.log(nnode) / nnode)) ** (1 / d), self.growth * 5.0)
		dlist = [dist(node, new_node) for node in self.node_list]
		near_nodes = [self.node_list[dlist.index(d)] for d in dlist if d <= r]
		return near_nodes

	def rewire(self, new_node, near_nodes):
		for near_node in near_nodes:
			p1 = time.time()
			px, py, pangle, mode, plength, u = plan.dubins_path_planning(new_node.pose[0], new_node.pose[1], new_node.pose[2], near_node.pose[0], near_node.pose[1], near_node.pose[2], self.max_curvature)
			p2 = time.time()
			self.local_planner_time += (p2 - p1)
			avg_var_per_length = (new_node.total_var + self.path_var(px, py, pangle)) / (new_node.dist + plength)
			if near_node.dist != 0:
				if near_node.total_var / near_node.dist > avg_var_per_length and self.max_dist >= new_node.dist + plength \
						and self.check_loop(near_node, new_node):
					if self.check_collision_path(px, py):
						near_node.parent = new_node
						near_node.path_x = px
						near_node.path_y = py
						near_node.path_angle = pangle
						near_node.u = u
						near_node.total_var = near_node.parent.total_var + self.path_var(px, py, pangle)
						near_node.dist = near_node.parent.dist + plength
						self.propagate_update_to_children(near_node)

	def propagate_update_to_children(self, parent_node):
		for node in self.node_list:
			if node.parent is not None:
				if node.parent == parent_node:
					node.total_var = parent_node.total_var + node.path_var
					node.dist = parent_node.dist + node.path_dist
					self.propagate_update_to_children(node)

	def check_loop(self, near_node, new_node):
		# checks to make sure that changing parents to temp_node does not create a loop
		temp = new_node.parent
		while temp is not None:
			if temp == near_node:
				return False       # creates a loop
			temp = temp.parent
		return True                # does not create a loop

	def check_collision_path(self, px, py):
		# check for collision on path
		for kk in range(len(px)):
			if not self.check_collision(px[kk], py[kk]):
				return False
		return True

	def check_collision(self, x_node, y_node):
		if self.obstacles is None:
			return True   # safe
		for (x, y, side) in self.obstacles:
			if (x_node > x - .8 * side / 2) and (x_node < x + .8 * side / 2) and (y_node > y - side / 2) and (y_node < y + side / 2):
				return False  # collision

		return True  # safe

	def nearest_node(self, sample):
		dlist = [dist(node, sample) for node in self.node_list]
		min_node = self.node_list[dlist.index(min(dlist))]
		return min_node

	# def cost(self, px, py, pangle, plength):          # more than 2/3 of time here and the rest of time in dubins path planner
	# 	control_cost = 0        # NOT USED!!!!!!
	# 	var_cost = np.zeros(len(px))
	#
	# 	(lxf, lyf, dvx, dvy, lx, ly, n, p, de, l_TH, p_THETA, xg_min, xg_max, yg_min, yg_max) = self.gmrf_params
	#
	# 	#iterate over path and calculate cost
	# 	for kk in range(len(px)):      # Iterate over length of trajectory
	# 		if not (self.space[0] <= px[kk] <= self.space[1]) or not (self.space[2] <= py[kk] <= self.space[3]):
	# 			var_cost[kk] = Config.border_variance_penalty
	# 			control_cost += 0
	# 		else:
	# 			p1 = time.time()
	# 			A_z = Config.interpolation_matrix(np.array([px[kk], py[kk], pangle[kk]]), n, p, lx, xg_min, yg_min, de)
	# 			self.method_time += (time.time() - p1)
	# 			var_cost[kk] = 1/(np.dot(A_z.T, self.var_x)[0][0])
	# 			control_cost += 0
	# 	return np.sum(var_cost) * plength

	def path_var(self, px, py, pangle):       # returns negative total variance along the path
		control_cost = 0  # NOT USED!!!!!!
		path_var = 0

		(lxf, lyf, dvx, dvy, lx, ly, n, p, de, l_TH, p_THETA, xg_min, xg_max, yg_min, yg_max) = self.gmrf_params
		p1 = time.time()
		A = np.zeros(shape=(n + p, 1)).astype(float)
		# iterate over path and calculate cost
		for kk in range(len(px)):  # Iterate over length of trajectory
			if not (self.space[0] <= px[kk] <= self.space[1]) or not (self.space[2] <= py[kk] <= self.space[3]):
				path_var += Config.border_variance_penalty  # value of 5
				control_cost += 0
			else:
				A += interpolation_matrix(np.array([px[kk], py[kk], pangle[kk]]), n, p, lx, xg_min, yg_min, de)
				control_cost += 0
		path_var -= np.dot(A.T, self.var_x)[0][0]
		self.method_time += (time.time() - p1)
		return path_var    # negative path var

	def draw_graph(self, plot=None):
		if plot is not None:  # use plot of calling
			for node in self.node_list:
				plot.quiver(node.pose[0], node.pose[1], math.cos(node.pose[2]), math.sin(node.pose[2]), color='b', angles='xy', scale_units='xy', scale=.8, width=.015)
				if node.parent is not None:
					plot.plot(node.path_x, node.path_y, color='green')

			if self.obstacles is not None:
				for (x, y, side) in self.obstacles:
					plot.plot(x, y, "sk", ms=8 * side)

			plot.quiver(self.start.pose[0], self.start.pose[1], math.cos(self.start.pose[2]), math.sin(self.start.pose[2]), color="b")
			plot.axis(self.space)
			plot.grid(True)
			plot.title("RRT* (avg variance per path length as cost function)")
			plot.pause(.1)  # need for animation

def dist(node1, node2):
	# returns distance between two nodes
	return math.sqrt((node2.pose[0] - node1.pose[0]) ** 2 +
					 (node2.pose[1] - node1.pose[1]) ** 2 +
					 3 * min((node1.pose[2] - node2.pose[2]) ** 2, (node1.pose[2] - node2.pose[2] + 2*math.pi) ** 2, (node1.pose[2] - node2.pose[2] - 2*math.pi) ** 2))

# Calculate new observation vector through shape function interpolation
def interpolation_matrix(x_local2, n, p, lx, xg_min, yg_min, de):
	"""INTERPOLATION MATRIX:
	Define shape function matrix that maps grid vertices to
	continuous measurement locations"""
	u1 = np.zeros(shape=(n + p, 1)).astype(float)
	nx = int((x_local2[0] - xg_min) / de[0])  # Calculates the vertice column x-number at which the shape element starts.
	ny = int((x_local2[1] - yg_min) / de[1])  # Calculates the vertice row y-number at which the shape element starts.
	# Calculate position value in element coord-sys in meters
	x_el = float(0.1 * (x_local2[0] / 0.1 - int(x_local2[0] / 0.1))) - de[0] / 2
	y_el = float(0.1 * (x_local2[1] / 0.1 - int(x_local2[1] / 0.1))) - de[1] / 2
	# Define shape functions, "a" is element width in x-direction
	u1[(ny * lx) + nx] = (1 / (de[0] * de[1])) * ((x_el - de[0] / 2) * (y_el - de[1] / 2))  # u for lower left corner
	u1[(ny * lx) + nx + 1] = (-1 / (de[0] * de[1])) * ((x_el + de[0] / 2) * (y_el - de[1] / 2))  # u for lower right corner
	u1[((ny + 1) * lx) + nx] = (-1 / (de[0] * de[1])) * ((x_el - de[0] / 2) * (y_el + de[1] / 2))  # u for upper left corner
	u1[((ny + 1) * lx) + nx + 1] = (1 / (de[0] * de[1])) * ((x_el + de[0] / 2) * (y_el + de[1] / 2))  # u for upper right corner
	return u1