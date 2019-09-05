import copy
import math
import random
import time
import numpy as np
import Config
from control_algorithms.base import dubins_path_planner as plan
from control_algorithms.base.Node import Node


class PRM:
	# PRM* algorithm using average variance per unit path length as cost function and Dubins path planner for local planning

	def __init__(self, start, PRM_params, gmrf_params, var_x, max_dist, plot):
		"""
		:param start: initial location of agent
		:param PRM_params: specified in config file
		:param gmrf_params: specified in config file
		:param var_x: variance of field as a 1D vector of variance of each node in GMRF
		:param max_dist: maximum distance that the algorithm solution will return
		:param plot: only used for plotting in the middle of running algorithm good for debugging
		"""
		self.start = Node(start)
		self.node_list = [self.start]
		(self.space, self.max_time, self.max_curvature, self.min_dist, self.obstacles) = PRM_params
		self.gmrf_params = gmrf_params
		self.max_dist = max(max_dist, 10)  # can't just take the max_dist in case at the end of the simulation this will allow no possible paths
		self.var_x = var_x
		# used to track runtimes
		self.local_planner_time = 0.0
		self.method_time = 0.0
		self.plot = plot

	def control_algorithm(self):
		start_time = time.time()
		while True:
			current_time = time.time() - start_time
			if current_time > self.max_time:
				break

			##  PRM start
			sample_node = self.get_sample()
			if self.check_collision(sample_node):
				nearest_node = self.nearest_node(sample_node)
				self.set_parent(sample_node, nearest_node)
				if sample_node.parent is None:  # no possible path from any of the near nodes
					continue
				self.node_list.append(sample_node)
			## PRM end

		# generate path
		last_node = self.get_best_last_node()
		if last_node is None:
			return None
		path, u_optimal, tau_optimal = self.get_path(last_node)
		return path, u_optimal, tau_optimal

	def get_sample(self):
		sample = Node([random.uniform(self.space[0]+.3, self.space[1]-.3),
					   random.uniform(self.space[2]+.3, self.space[3]-.3),
					   random.uniform(-math.pi, math.pi)])
		return sample

	def local_path(self, source_node, destination_node):
		# take source_node and find path to destination_node
		time1 = time.time()
		px, py, pangle, mode, plength, u = plan.dubins_path_planning(source_node.pose[0], source_node.pose[1], source_node.pose[2], destination_node.pose[0], destination_node.pose[1], destination_node.pose[2], self.max_curvature)
		self.local_planner_time += time.time() - time1
		new_node = copy.deepcopy(source_node)
		new_node.pose = destination_node.pose
		new_node.path_x = px
		new_node.path_y = py
		new_node.path_angle = pangle
		new_node.u = u

		new_node.path_dist = plength
		new_node.dist += plength
		new_node.path_var = self.path_var(px, py, pangle)
		new_node.total_var += new_node.path_var
		new_node.cost = new_node.total_var / new_node.dist
		new_node.parent = source_node
		return new_node

	def set_parent(self, new_node, nearest_node):
		# connects new_node to nearest node and tracks cost
		px, py, pangle, mode, plength, u = plan.dubins_path_planning(nearest_node.pose[0], nearest_node.pose[1], nearest_node.pose[2], new_node.pose[0], new_node.pose[1], new_node.pose[2], self.max_curvature)
		new_node.parent = nearest_node
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
				cost_list.append(node.cost)
			else:
				cost_list.append(float("inf"))
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

	def get_near_nodes(self, new_node):
		# gamma_star = 2(1+1/d) ** (1/d) volume(free)/volume(total) ** 1/d and we need gamma > gamma_star
		# for asymptotical completeness see Kalman 2011. gamma = 1 satisfies
		d = 2  # dimension of the self.space
		nnode = len(self.node_list)
		r = min(20.0 * ((math.log(nnode) / nnode)) ** (1 / d), 5.0)
		dlist = [dist(new_node, node) for node in self.node_list]
		near_nodes = [self.node_list[dlist.index(d)] for d in dlist if d <= r]
		return near_nodes

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

	def path_var(self, px, py, pangle):  # returns negative total variance along the path
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
		return path_var  # negative path var

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
			plot.title("PRM (avg variance per unit path length as cost function)")
			plot.pause(.1)  # need for animation


def dist(node1, node2):
	# returns distance between two nodes
	return math.sqrt((node2.pose[0] - node1.pose[0]) ** 2 +
					 (node2.pose[1] - node1.pose[1]) ** 2 +
					 3 * min((node1.pose[2] - node2.pose[2]) ** 2, (node1.pose[2] - node2.pose[2] + 2 * math.pi) ** 2, (node1.pose[2] - node2.pose[2] - 2 * math.pi) ** 2))


def interpolation_matrix(x_local2, n, p, lx, xg_min, yg_min, de):
	# Calculate new observation vector through shape function interpolation
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


def initialize(graph, source):
	d = {}  # Stands for destination
	p = {}  # Stands for predecessor
	for node in graph:
		d[node] = float('Inf')  # We start admiting that the rest of nodes are very very far
		p[node] = None
	d[source] = 0  # For the source we know how to reach
	return d, p


def relax(node, neighbour, graph, d, p):
	# If the distance between the node and the neighbour is lower than the one I have now
	if d[neighbour] > d[node] + graph[node][neighbour]:
		# Record this lower distance
		d[neighbour] = d[node] + graph[node][neighbour]
		p[neighbour] = node


def bellman_ford(graph, source):
	d, p = initialize(graph, source)
	for i in range(len(graph) - 1):  # Run this until is converges
		for u in graph:
			for v in graph[u]:  # For each neighbour of u
				relax(u, v, graph, d, p)  # Lets relax it
	return d, p