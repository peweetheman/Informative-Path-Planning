"""
Scripts related to Gaussian Processes.

author: Andreas Rene Geist
email: andreas.geist@tuhh.de
website: https://github.com/AndReGeist
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import Config

import os
import numpy as np
import scipy
import scipy.sparse as sp
from scipy import exp, sin, cos, sqrt, pi, interpolate
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import cm

# from scikits.sparse.cholmod import cholesky


"""Define sparse matrix save and load"""


def save_sparse_csr(filename, array):
	# note that .npz extension is added automatically
	np.savez(filename, data=array.data, indices=array.indices,
			 indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
	# here we need to add .npz extension manually
	loader = np.load(filename + '.npz')
	return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
						 shape=loader['shape'])


"""Calculate entry topology of the precision matrix Q"""


def calculate_precision_matrix(lx, ly, kappa, alpha, car1=False):
	"""TORUS VERTICE TOPOLOGY
	Define entries of the precision matrix to
	represent a conitional CAR 1 or 2 model"""
	field_info = np.arange(lx * ly).reshape((ly, lx))

	"""
	Indices for precision values of field vertice i,j:
					 a2,j
			  a1,d1  a1,j  a1,c1
		i,d2   i,d1   i,j  i,c1    i,c2
			  b1,d1  b1,j  b1,c1
					 b2,j
	Note: - all field indices with "1"-indice mark vertices directly around current field vertice
		  - field vertices with two "1"-indices are diagonal field vertices (relative to the current field vertice)

	Field structure:
	LEFT UPPER CORNER ###  NORTH BORDER  ###  RIGHT UPPER CORNER     ^
			#                                       #                |
			#                                       #                |
			#                                       #                |
		WEST BORDER       INNER  VERTICES       EAST BORDER         ly
			#                                       #                |
			#                                       #                |
			#                                       #                |
	LEFT LOWER CORNER  ###  SOUTH BORDER  ### RIGHT LOWER CORNER     v

			<----------------lx--------------------->
	"""
	if Config.set_gmrf_torus == True:
		"""
			Indices for precision values of field vertice i,j:
					 a2,j
			  a1,d1  a1,j  a1,c1
		i,d2   i,d1   i,j  i,c1    i,c2
			  b1,d1  b1,j  b1,c1
					 b2,j
		Note: - all field indices with "1"-indice mark vertices directly around current field vertice
			  - field vertices with two "1"-indices are diagonal field vertices (relative to the current field vertice)
		"""
		infmat = np.dot(-1, np.ones((lx * ly, 13)))
		for ii in range(0, ly):  # Step to next field vertice in y-direction
			for jj in range(0, lx):  # Step to next field vertice in x-direction
				# The GMRF field has ly entries in Y-direction (indices in field array 0 to ly-1)
				# Check if field vertice is inside field, on border or in corner of field
				# To define its neighbours
				if (ii + 2) <= (ly - 1):
					a1 = ii + 1;
					a2 = ii + 2
				# If moving two fields upwards in y-direction does not result in leaving the field
				# The uper field neighbours "a1" and "a2" are directly above the vertice being a1=ii+1; a2=ii+2
				elif (ii + 1) <= (ly - 1):
					a1 = ii + 1;
					a2 = 0
				# If moving two fields upwards in y-direction does result in a2 leaving the field
				# The uper field neighbour "a1" is still inside field, but "a2" is due to the torus topology now a2=0
				else:
					a1 = 0;
					a2 = 1
				# If moving one field upwards in y-direction does result in leaving the field (ii is on upper field border)
				# The uper field neighbour is in in y-direction with "a1" and "a2" due to the torus topology now  a1=0 and a2=1

				if (ii - 2) >= 0:
					b1 = ii - 1;
					b2 = ii - 2
				elif (ii - 1) >= 0:
					b1 = ii - 1;
					b2 = ly - 1
				else:
					b1 = ly - 1;
					b2 = ly - 2

				if (jj + 2) <= (lx - 1):
					c1 = jj + 1;
					c2 = jj + 2
				elif (jj + 1) <= (lx - 1):
					c1 = jj + 1;
					c2 = 0
				else:
					c1 = 0;
					c2 = 1

				if (jj - 2) >= 0:
					d1 = jj - 1;
					d2 = jj - 2
				elif (jj - 1) >= 0:
					d1 = jj - 1;
					d2 = lx - 1
				else:
					d1 = lx - 1;
					d2 = lx - 2
				# field i,j              a1,j             b1,j               i,c1             i,d1
				infmat[field_info[ii, jj], :] = np.array([field_info[ii, jj], field_info[a1, jj], field_info[b1, jj], field_info[ii, c1], field_info[ii, d1],
														  #               a2,j             b2,j               i,c2             i,d2
														  field_info[a2, jj], field_info[b2, jj], field_info[ii, c2], field_info[ii, d2],
														  #               a1,c1            a1,d1               b1,d1           b1,c1
														  field_info[a1, c1], field_info[a1, d1], field_info[b1, d1], field_info[b1, c1]])
		a = alpha + 4

		if car1 == True:
			Q_rc = np.zeros(shape=(2, 5 * lx * ly)).astype(int)  # Save row colum indices of Q in COO-sparse-format
			Q_d = np.zeros(shape=(1, 5 * lx * ly)).astype(float)  # Save data of Q in COO-sparse-format
			for i1 in range(0, (lx * ly)):  # For all GMRF Grid values
				a1 = int(5 * i1)  # Each GMRF node has 5 entries in the corresponding precision matrix
				Q_rc[0, a1:(a1 + 5)] = i1 * np.ones(shape=(1, 5))  # Row indices
				Q_rc[1, a1:(a1 + 5)] = np.hstack((i1, infmat[i1, 1:5]))  # Column indices
				Q_d[0, a1:(a1 + 5)] = np.hstack(((a * kappa) * np.ones(shape=(1, 1)),
												 (-1 * kappa) * np.ones(shape=(1, 4))))  # Data
			Q_temporary1 = sp.coo_matrix((Q_d[0, :], (Q_rc[0, :], Q_rc[1, :])), shape=(lx * ly, lx * ly)).tocsr()

			# if Config.set_gmrf_torus == False:
			#     Q_temporary = Q_temporary1.todense()
			#     Q_new = np.diag(np.diag(Q_temporary, k=0), k=0) + np.diag(np.diag(Q_temporary, k=-1), k=-1) + \
			#             np.diag(np.diag(Q_temporary, k=1), k=1) + np.diag(np.diag(Q_temporary, k=lx), k=lx) + \
			#             np.diag(np.diag(Q_temporary, k=-lx), k=-lx)
			#     return sp.coo_matrix(Q_new)
			# else:
			return Q_temporary1
		elif car1 == False:
			Q_rc = np.zeros(shape=(3, 13 * lx * ly)).astype(int)  # Save row colum indices of Q in COO-sparse-format
			Q_d = np.zeros(shape=(3, 13 * lx * ly)).astype(float)  # Save data of Q in COO-sparse-format
			for i2 in range(0, (lx * ly)):
				a1 = int(13 * i2)
				Q_rc[0, a1:(a1 + 13)] = i2 * np.ones(shape=(1, 13))  # Row indices
				Q_rc[1, a1:(a1 + 13)] = np.hstack((i2, infmat[i2, 1:5], infmat[i2, 5:9], infmat[i2, 9:13]))  # Column indices
				# Q_d[0, a1:(a1 + 13)] = np.hstack(((4 + a ** 2) * (1 / kappa) * np.ones(shape=(1, 1)), (-2 * a / kappa) * np.ones(shape=(1, 4)),
				#                                (1/kappa) * np.ones(shape=(1, 4)), (2 / kappa) * np.ones(shape=(1, 4))))  # Data
				Q_d[0, a1:(a1 + 13)] = np.hstack(((4 + a ** 2) * kappa * np.ones(shape=(1, 1)), (-2 * a * kappa) * np.ones(shape=(1, 4)),
												  kappa * np.ones(shape=(1, 4)), 2 * kappa * np.ones(shape=(1, 4))))  # Data
			Q_temporary1 = sp.coo_matrix((Q_d[0, :], (Q_rc[0, :], Q_rc[1, :])), shape=(lx * ly, lx * ly)).tocsr()
			return Q_temporary1

	elif Config.set_gmrf_torus == False:  # Neumann Boundary condition instead
		if car1 == True:
			n_a = 5  # Number of non-zero entries in the precision matrix for one GMRF vertice row
			q_center = (alpha + 4) * kappa
			q_border = (alpha + 3) * kappa
			q_corner = (alpha + 2) * kappa
			q_2 = -1 * kappa
			n_size = (lx - 2) * (ly - 2) * n_a + (lx - 2) * 2 * (n_a - 1) + (ly - 2) * 2 * (n_a - 1) + 4 * (n_a - 2)
			Q_rc = np.zeros(shape=(2, n_size)).astype(int)  # Save row colum indices of Q in COO-sparse-format
			Q_d = np.zeros(shape=(1, n_size)).astype(float)  # Save data of Q in COO-sparse-format
			ce = 0  # Current index inside the coo matrix that defines the GMRF precision
			for ii in range(0, ly):  # Step to next field vertice in y-direction
				for jj in range(0, lx):  # Step to next field vertice in x-direction
					i1 = field_info[ii, jj]
					# The GMRF field has ly entries in Y-direction (indices in field array 0 to ly-1)
					# Check if field vertice is inside field, on border or in corner of field
					# To define its neighbours
					if ii == 0 and jj == 0:  # Left lower corner
						Q_rc[0, ce:(ce + 3)] = i1 * np.ones(shape=(1, 3))
						Q_rc[1, ce:(ce + 3)] = [i1, 2, lx]  # Column indices
						Q_d[0, ce:(ce + 3)] = [q_corner, q_2, q_2]  # Data
						ce += 3
					elif ii == 0 and jj == lx - 1:  # Right lower corner
						Q_rc[0, ce:(ce + 3)] = i1 * np.ones(shape=(1, 3))
						Q_rc[1, ce:(ce + 3)] = [i1, i1 - 1, i1 + lx]  # Column indices
						Q_d[0, ce:(ce + 3)] = [q_corner, q_2, q_2]  # Data
						ce += 3
					elif ii == ly - 1 and jj == 0:  # Left upper corner
						Q_rc[0, ce:(ce + 3)] = i1 * np.ones(shape=(1, 3))
						Q_rc[1, ce:(ce + 3)] = [i1, i1 + 1, i1 - lx]  # Column indices
						Q_d[0, ce:(ce + 3)] = [q_corner, q_2, q_2]  # Data
						ce += 3
					elif ii == ly - 1 and jj == lx - 1:  # Right upper corner
						Q_rc[0, ce:(ce + 3)] = i1 * np.ones(shape=(1, 3))
						Q_rc[1, ce:(ce + 3)] = [i1, i1 - 1, i1 - lx]  # Column indices
						Q_d[0, ce:(ce + 3)] = [q_corner, q_2, q_2]  # Data
						ce += 3
					elif ii == ly - 1:  # North field border
						Q_rc[0, ce:(ce + 4)] = i1 * np.ones(shape=(1, 4))
						Q_rc[1, ce:(ce + 4)] = [i1, i1 - 1, i1 + 1, i1 - lx]  # Column indices
						Q_d[0, ce:(ce + 4)] = [q_border, q_2, q_2, q_2]  # Data
						ce += 4
					elif jj == lx - 1:  # East field border
						Q_rc[0, ce:(ce + 4)] = i1 * np.ones(shape=(1, 4))
						Q_rc[1, ce:(ce + 4)] = [i1, i1 - 1, i1 + lx, i1 - lx]  # Column indices
						Q_d[0, ce:(ce + 4)] = [q_border, q_2, q_2, q_2]  # Data
						ce += 4
					elif ii == 0:  # South field border
						Q_rc[0, ce:(ce + 4)] = i1 * np.ones(shape=(1, 4))
						Q_rc[1, ce:(ce + 4)] = [i1, i1 - 1, i1 + 1, i1 + lx]  # Column indices
						Q_d[0, ce:(ce + 4)] = [q_border, q_2, q_2, q_2]  # Data
						ce += 4
					elif jj == 0:  # West field border
						Q_rc[0, ce:(ce + 4)] = i1 * np.ones(shape=(1, 4))
						Q_rc[1, ce:(ce + 4)] = [i1, i1 + lx, i1 + 1, i1 - lx]  # Column indices
						Q_d[0, ce:(ce + 4)] = [q_border, q_2, q_2, q_2]  # Data
						ce += 4
					else:  # Center vertices
						Q_rc[0, ce:(ce + 5)] = i1 * np.ones(shape=(1, 5))
						Q_rc[1, ce:(ce + 5)] = [i1, i1 - 1, i1 + 1, i1 - lx, i1 + lx]  # Column indices
						Q_d[0, ce:(ce + 5)] = [q_center, q_2, q_2, q_2, q_2]  # Data  # Data
						ce += 5
			Q_temporary1 = sp.coo_matrix((Q_d[0, :], (Q_rc[0, :], Q_rc[1, :])), shape=(lx * ly, lx * ly)).tocsr()
			return Q_temporary1

		elif car1 == False:
			a = alpha + 4
			q0 = kappa * (4 + a ** 2)
			q1 = -2 * a * kappa
			q2 = 2 * kappa
			q3 = 1 * kappa
			n_size = 13 * (lx - 4) * (ly - 4) + 2 * (9 + 12) * ((lx - 4) + (ly - 4)) + 4 * 11 + 8 * 8 + 4 * 6
			Q_r = np.zeros(shape=(1, n_size)).astype(int)
			Q_c = np.zeros(shape=(1, n_size)).astype(int)
			Q_d = np.zeros(shape=(1, n_size)).astype(float)  # Save data of Q in COO-sparse-format
			ce = 0

			for ii in range(0, ly):  # Step to next field vertice in y-direction
				for jj in range(0, lx):  # Step to next field vertice in x-direction
					# The GMRF field has ly entries in Y-direction (indices in field array 0 to ly-1)
					# Check if field vertice is inside field, on border or in corner of field
					"""
					Entries of the precision element:    | Precision element confugrations (depends on the vertice postion inside the field):
								 k3                      |   I. o      II.    o     III.   o     IV.   o     V.    o
						  k2     k1     k2               |      o o         o o o        o o o       o o o       o o o
					k3    k1    ii,jj   k1    k3         |      0 o o     o o 0 o o      o 0 o o   o o 0 o o   o o 0 o o
						  k2     k1     k2               |                               o o o       o o o       o o o
								 k3                      |                                                         o
					"""
					i1 = field_info[ii, jj]
					k1 = []
					k2 = []
					k3 = []
					# Check if precision element types k1 and k2 are inside field for vertice ii,jj
					if (ii + 2) <= (ly - 1):  # North Border
						k1.append(i1 + lx)
						k2.append(i1 + 2 * lx)
					elif (ii + 1) <= (ly - 1):
						k1.append(i1 + lx)

					if (jj + 2) <= (lx - 1):  # East Border
						k1.append(i1 + 1)
						k2.append(i1 + 2)
					elif (jj + 1) <= (lx - 1):
						k1.append(i1 + 1)

					if (ii - 2) >= 0:  # South Border
						k1.append(i1 - lx)
						k2.append(i1 - 2 * lx)
					elif (ii - 1) >= 0:
						k1.append(i1 - lx)

					if (jj - 2) >= 0:  # West Border
						k1.append(i1 - 1)
						k2.append(i1 - 2)
					elif (jj - 1) >= 0:
						k1.append(i1 - 1)

					# Check if precision element type k3 is inside field for vertice ii,jj
					if (i1 + lx) in k1 and (i1 + 1) in k1:  # Upper right precision element k3
						k3.append(i1 + lx + 1)
					if (i1 + lx) in k1 and (i1 - 1) in k1:  # Upper left precision element k3
						k3.append(i1 + lx - 1)
					if (i1 - lx) in k1 and (i1 + 1) in k1:  # Lower right precision element k3
						k3.append(i1 - lx + 1)
					if (i1 - lx) in k1 and (i1 - 1) in k1:  # Lower left precision element k3
						k3.append(i1 - lx - 1)

					n_qe = 1 + len(k1) + len(k2) + len(k3)
					Q_r[0, ce:(ce + n_qe)] = i1 * np.ones(shape=(1, n_qe))
					Q_c[0, ce:(ce + n_qe)] = np.hstack((i1 * np.ones(shape=(1, 1)), np.array(k1).reshape(1, len(k1)), np.array(k2).reshape(1, len(k2)), np.array(k3).reshape(1, len(k3))))  # Column indices
					Q_d[0, ce:(ce + n_qe)] = np.hstack((q0 * np.ones(shape=(1, 1)), q1 * np.ones(shape=(1, len(k1))), q2 * np.ones(shape=(1, len(k2))), q3 * np.ones(shape=(1, len(k3)))))  # Data  # Data
					ce += n_qe

			return sp.coo_matrix((Q_d[0, :], (Q_r[0, :], Q_c[0, :])), shape=(lx * ly, lx * ly)).tocsr()


"""SAMPLE from GMRF"""
def sample_from_GMRF(gmrf_dim, kappa, alpha, car_var, plot_gmrf=False):
	x_min, x_max, y_min, y_max = Config.field_dim
	lxf, lyf, dvx, dvy = gmrf_dim
	lx1 = lxf + 2 * dvx  # Total number of GMRF vertices in x
	ly1 = lyf + 2 * dvy

	# Calculate precision matrix
	Q_storage = calculate_precision_matrix(lx1, ly1, kappa[0], alpha[0], car1=car_var)
	# Draw sampel from GMRF
	mue_Q = 10
	z_I = np.random.standard_normal(size=lx1 * ly1)
	x_Q = np.zeros(shape=(ly1, lx1, len(kappa)))

	print(Q_storage)
	L_Q = np.linalg.cholesky(Q_storage.todense())
	v_Q = np.linalg.solve(L_Q.T, z_I)
	x_Q_vec = mue_Q + v_Q
	x_Q = x_Q_vec.reshape((ly1, lx1))

	if plot_gmrf == True:
		if len(kappa) == 1:
			fig, ax = plt.subplots(1)
			# ax = ax.ravel()
			k = 0
			cf = ax.pcolor(np.linspace(x_min, x_max, num=lx1, endpoint=True),
						   np.linspace(y_min, y_max, num=ly1, endpoint=True), x_Q[:, :, k])
			ax.axis('tight')
			plt.colorbar(cf, ax=ax)
			ax.set_title('GMRF sample, kappa: ' + str(kappa[k]) + ', alpha: ' + str(alpha[k]))
			plt.xlabel('x (m)')
			plt.ylabel('y (m)')
			k += 1
			plt.show()
		else:
			fig, ax = plt.subplots(3, 2)
			k = 0
			for j in range(2):
				for i in range(3):
					cf = ax[i, j].pcolor(np.linspace(x_min, x_max, num=lx1, endpoint=True),
										 np.linspace(y_min, y_max, num=ly1, endpoint=True), x_Q[:, :, k])
					ax[i, j].axis('tight')
					plt.colorbar(cf, ax=ax[i, j])
					ax[i, j].set_title('GMRF sample, kappa: ' + str(kappa[k]) + ', alpha: ' + str(alpha[k]))
				plt.xlabel('x (m)')
				plt.ylabel('y (m)')
				k += 1
			plt.show()
	de = np.array([float(x_max - x_min) / (lxf - 1), float(y_max - y_min) / (lyf - 1)])  # Element width in x and y
	xg_min = x_min - dvx * de[0]  # Min GMRF field value in x
	xg_max = x_max + dvx * de[0]
	yg_min = y_min - dvy * de[1]
	yg_max = y_max + dvy * de[1]
	X = np.linspace(xg_min, xg_max, num=lx1, endpoint=True)  # Specifies column coordinates of field
	Y = np.linspace(yg_min, yg_max, num=ly1, endpoint=True)  # Specifies row coordinates of field
	f = scipy.interpolate.interp2d(X, Y, x_Q, kind='cubic')
	return f


class GMRF:
	def __init__(self, gmrf_dim, alpha_prior, kappa_prior, set_Q_init):
		"""Initialize GMRF dimensions and precision matrices"""

		"""Initialize GMRF dimensions"""
		x_min, x_max, y_min, y_max = Config.field_dim
		lxf, lyf, dvx, dvy = gmrf_dim

		lx = lxf + 2 * dvx  # Total number of GMRF vertices across x dimension
		print("size of lx: ", lx)
		ly = lyf + 2 * dvy  # Total number of GMRF vertices across x dimension
		print("size of ly: ", ly)
		n = lx * ly         # Total number of GMRF vertices
		print("size of n: ", n)
		de = np.array([float(x_max - x_min) / (lxf - 1), float(y_max - y_min) / (lyf - 1)])  # Element width in x and y
		print("value of de: ", de)
		xg_min = x_min - dvx * de[0]  # Min GMRF field value in x
		xg_max = x_max + dvx * de[0]

		yg_min = y_min - dvy * de[1]
		yg_max = y_max + dvy * de[1]

		"""Intialize GMRF PRECISION matrices"""
		p = 1  # Number of regression coefficients beta
		self.F = np.ones(shape=(n, p))  # Mean regression functions
		self.T = 1e-6 * np.ones(shape=(p, p))  # Precision matrix of the regression coefficients
		# Initialize hyperparameter prior
		THETA = []  # Matrix containing all discrete hyperparameter combinations
		for i in range(0, len(alpha_prior)):
			for j in range(0, len(kappa_prior)):
				THETA.append([kappa_prior[j], alpha_prior[i]])
		THETA = np.array(THETA).T
		l_TH = len(THETA[1])  # Number of hyperparameter pairs
		p_THETA = 1.0 / l_TH  # Prior probability for one theta

		self.diag_Q_t_inv = np.zeros(shape=(lx * ly + p, l_TH)).astype(float)
		F_sparse = sp.csr_matrix(self.F)
		FT_sparse = scipy.sparse.csr_matrix(self.F.T)
		T_inv = np.linalg.inv(self.T)  # Inverse of the Precision matrix of the regression coefficients
		T_sparse = sp.csr_matrix(self.T)

		if set_Q_init == True:
			for jj in range(0, l_TH):
				print("Initialize Matrix:", jj, "of", l_TH)

				"""Initialize Q_{x|eta}"""
				# _{field values|eta}             kappa          alpha
				Q_temporary = calculate_precision_matrix(lx, ly, THETA[0, jj], THETA[1, jj], car1=Config.set_GMRF_cartype)
				Q_eta_inv = np.linalg.inv(Q_temporary.todense())

				"""Q_{x|eta,y=/} & diag_Q_inv """
				A2 = Q_temporary.dot(-1 * F_sparse)
				B1 = -1 * FT_sparse.dot(Q_temporary)
				B2 = sp.csr_matrix.dot(FT_sparse, Q_temporary.dot(F_sparse)) + T_sparse
				H1 = sp.hstack([Q_temporary, A2])
				H2 = sp.hstack([B1, B2])
				filename = os.path.join('gp_scripts', 'Q_t_' + str(jj))
				Q_t = sp.vstack([H1, H2]).tocsr()
				setattr(self, filename, Q_t)
				np.savez(filename, data=Q_t.data, indices=Q_t.indices,
						 indptr=Q_t.indptr, shape=Q_t.shape)

				C1 = Q_eta_inv + np.dot(self.F, np.dot(T_inv, self.F.T))
				C2 = np.dot(self.F, T_inv)
				D1 = np.dot(self.F, T_inv).T
				Q_t_inv = np.vstack([np.hstack([C1, C2]),
									 np.hstack([D1, T_inv])])

				self.diag_Q_t_inv[:, jj] = Q_t_inv.diagonal()
			np.save(os.path.join('gp_scripts', 'diag_Q_t_inv.npy'), self.diag_Q_t_inv)

		else:
			print('Loading precalculated matrices')
			for j2 in range(0, l_TH):
				filename = os.path.join('gp_scripts', "Q_t_" + str(j2))
				loader = np.load(filename + '.npz')
				Q_t2 = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
									 shape=loader['shape'])
				filename2 = os.path.join('gp_scripts', "Q_t_" + str(j2))
				setattr(self, filename2, Q_t2)
			self.diag_Q_t_inv = np.load(os.path.join('gp_scripts', 'diag_Q_t_inv.npy'))

		"""Initialize adaptive GMRF algorithm matrices"""
		self.b = np.zeros(shape=(n + p, 1))  # Canonical mean
		self.c = 0.0  # Log-likelihood update vector
		self.h_theta = np.zeros(shape=(n + p, l_TH))
		self.g_theta = np.zeros(shape=(l_TH, 1))
		self.log_pi_y = np.zeros(shape=(l_TH, 1))
		self.pi_theta = np.zeros(shape=(l_TH, 1))
		self.mue_theta = np.zeros(shape=(n + p, l_TH))
		self.mue_x = np.zeros(shape=(n + p, 1))
		self.var_x = np.zeros(shape=(n + p, 1))
		print("size of p: ", p)
		self.params = (lxf, lyf, dvx, dvy, lx, ly, n, p, de, l_TH, p_THETA, xg_min, xg_max, yg_min, yg_max)

	def gmrf_bayese_update(self, x_auv, y_t):
		"""updates the GMRF Class beliefe of the true field
			Input: State, New measurement
			Output: Field Mean, Field Variance, Parameter Posterior
		"""
		(lxf, lyf, dvx, dvy, lx, ly, n, p, de, l_TH, p_THETA, xg_min, xg_max, yg_min, yg_max) = self.params
		"""Compute interpolation matrix"""
		u = Config.interpolation_matrix(x_auv, n, p, lx, xg_min, yg_min, de)
		u_sparse = sp.csr_matrix(u)
		"""Update canonical mean and observation-dependent likelihood terms"""
		self.b = self.b + (y_t / Config.sigma_w_squ) * u  # Canonical mean
		self.c = self.c - ((y_t ** 2) / (2 * Config.sigma_w_squ))  # Likelihood term

		for jj in range(0, l_TH):
			"""Calculate observation precision (?)"""
			filename = os.path.join('gp_scripts', "Q_t_" + str(jj))
			Q_temporary = getattr(self, filename)
			Q_temporary = Q_temporary
			self.h_theta[:, jj] = scipy.sparse.linalg.spsolve(Q_temporary, u_sparse).T
			# L_factor = cholesky(Q_temporary)
			"""Update Precision Matrix"""
			self.diag_Q_t_inv[:, jj] = np.subtract(self.diag_Q_t_inv[:, jj], (
				np.multiply(self.h_theta[:, jj], self.h_theta[:, jj]) / (Config.sigma_w_squ + np.dot(u.T, self.h_theta[:, jj]))))
			# Q_unc_sparse = 0.02*scipy.sparse.eye(n+p)
			Q_temporary = Q_temporary + (1 / Config.sigma_w_squ) * u_sparse.dot(u_sparse.T)
			setattr(self, filename, Q_temporary)
			if Config.set_Q_check == True:
				# Check precision matrix
				my_data = Q_temporary.todense()
				my_data[my_data == 0.0] = np.nan
				plt.matshow(my_data, cmap=cm.Spectral_r, interpolation='none')
				plt.draw()
				plt.pause(30)
				x = raw_input("Press [enter] to continue")

			self.g_theta[jj] = self.g_theta[jj] - (0.5 * np.log(1 + (1 / Config.sigma_w_squ) * np.dot(u.T, self.h_theta[:, jj])))

		for hh in range(0, l_TH):
			"""Compute canonical mean"""
			filename = os.path.join('gp_scripts', "Q_t_" + str(jj))
			Q_temporary = getattr(self, filename)
			self.mue_theta[:, hh] = scipy.sparse.linalg.spsolve(Q_temporary, self.b).T
			"""Compute Likelihood"""
			self.log_pi_y[hh] = self.c + self.g_theta[hh] + 0.5 * np.dot(self.b.T, self.mue_theta[:, hh])  # Compute likelihood

		"""Scale likelihood and Posterior distribution (theta|y)"""
		self.log_pi_exp = np.exp(self.log_pi_y - np.amax(self.log_pi_y))
		self.posterior = (1 / np.sum(self.log_pi_exp)) * self.log_pi_exp * p_THETA
		self.pi_theta = (1 / np.sum(self.posterior)) * self.posterior  # Compute posterior distribution
		"""Predictive mean and variance (x|y)"""
		for ji in range(0, n + p):
			self.mue_x[ji] = np.dot(self.mue_theta[[ji], :], self.pi_theta)  # Predictive Mean
			self.var_x[ji] = np.dot((self.diag_Q_t_inv[ji] +
									 (np.subtract(self.mue_theta[ji, :],
												  self.mue_x[ji] * np.ones(shape=(1, l_TH))) ** 2)), self.pi_theta)
		return self.mue_x, self.var_x, self.pi_theta
