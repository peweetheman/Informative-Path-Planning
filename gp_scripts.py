"""
Scripts related to Gaussian Processes.

author: Andreas Rene Geist
email: andreas.geist@tuhh.de
website: https://github.com/AndReGeist
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import Config

import numpy as np
import scipy
from scipy import exp, sin, cos, sqrt, pi, interpolate
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve

"""Define sparse matrix save and load"""
def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)
def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

"""Calculate entry topology of the precision matrix Q"""
def gmrf_Q(lx, ly, kappa, alpha, car1=False):
    """TORUS VERTICE TOPOLOGY
    Define entries of the precision matrix to
    represent a conitional CAR 1 or 2 model"""
    field_info = np.arange(lx * ly).reshape((ly, lx))
    infmat = np.dot(-1, np.ones((lx * ly, 13)))

    """ Indices for precision function of field i,j
                     a2,j
              a1,d1  a1,j  a1,c1
        i,d2   i,d1   i,j  i,c1    i,c2
              b1,d1  b1,j  b1,c1
                     b2,j
    """
    for ii in range(0, ly):
        for jj in range(0, lx):

            if (ii+2) <= (ly-1): a1 = ii+1; a2 = ii+2
            elif (ii+1) <= (ly-1): a1 = ii+1; a2 = 0
            else: a1 = 0; a2 = 1

            if (ii-2) >= 0: b1 = ii-1; b2 = ii-2
            elif (ii-1) >= 0: b1 = ii-1; b2 = ly-1
            else: b1 = ly-1; b2 = ly-2

            if (jj+2) <= (lx-1): c1 = jj+1; c2 = jj+2
            elif (jj+1) <= (lx-1): c1 = jj+1; c2 = 0
            else: c1 = 0; c2 = 1

            if (jj-2) >= 0: d1 = jj-1; d2 = jj-2
            elif (jj-1) >= 0: d1 = jj-1; d2 = lx-1
            else: d1 = lx-1; d2 = lx-2
            #                                        field i,j              a1,j             b1,j               i,c1             i,d1
            infmat[field_info[ii, jj], :] = np.array([field_info[ii, jj], field_info[a1, jj], field_info[b1, jj], field_info[ii, c1], field_info[ii, d1],
                                                            #               a2,j             b2,j               i,c2             i,d2
                                                                      field_info[a2, jj],field_info[b2, jj], field_info[ii, c2], field_info[ii, d2],
                                                            #               a1,c1            a1,d1               b1,d1           b1,c1
                                                                      field_info[a1,c1], field_info[a1,d1], field_info[b1,d1], field_info[b1,c1]])
    a = alpha + 4

    if car1 == True:
        Q_rc = np.zeros(shape=(3, 5 * lx * ly)).astype(int)  # Save Q in COO-sparse-format
        Q_d = np.zeros(shape=(3, 5 * lx * ly)).astype(float)  # Save Q in COO-sparse-format
        for i1 in range(0, (lx * ly)):
            a1 = int(5 * i1)
            Q_rc[0, a1:(a1 + 5)] = i1 * np.ones(shape=(1, 5))  # Row indices
            Q_rc[1, a1:(a1 + 5)] = np.hstack((i1, infmat[i1, 1:5]))  # Column indices
            Q_d[0, a1:(a1 + 5)] = np.hstack((a * (1 / kappa) * np.ones(shape=(1, 1)),
                                              -1 * (1/kappa) * np.ones(shape=(1, 4))))  # Data

        return Q_rc, Q_d
    else:
        Q_rc = np.zeros(shape=(3, 13 * lx * ly)).astype(int)  # Save Q in COO-sparse-format
        Q_d = np.zeros(shape=(3, 13 * lx * ly)).astype(float)  # Save Q in COO-sparse-format
        for i2 in range(0, (lx * ly)):
            a1 = int(13*i2)
            Q_rc[0, a1:(a1 + 13)] = i2*np.ones(shape=(1, 13))  # Row indices
            Q_rc[1, a1:(a1 + 13)] = np.hstack((i2, infmat[i2, 1:5], infmat[i2, 5:9], infmat[i2, 9:13]))  # Column indices
            Q_d[0, a1:(a1 + 13)] = np.hstack(((4 + a ** 2) * (1 / kappa) * np.ones(shape=(1, 1)), (-2 * a / kappa) * np.ones(shape=(1, 4)),
                                            (1/kappa) * np.ones(shape=(1, 4)), (2 / kappa) * np.ones(shape=(1, 4))))  # Data
        return Q_rc, Q_d

class GMRF:
    def __init__(self, gmrf_dim, alpha_prior, kappa_prior, set_Q_init):
        """Initialize GMRF dimensions and precision matrices"""

        """Initialize GMRF dimensions"""
        x_min, x_max, y_min, y_max = Config.field_dim
        lxf, lyf, dvx, dvy = gmrf_dim

        lx = lxf + 2 * dvx  # Total number of GMRF vertices in x
        ly = lyf + 2 * dvy
        n = lx * ly
        de = np.array([float(x_max - x_min)/(lxf-1), float(y_max - y_min)/(lyf-1)])  # Element width in x and y
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
        F_sparse = scipy.sparse.csr_matrix(self.F)
        FT_sparse = scipy.sparse.csr_matrix(self.F.T)
        T_inv = np.linalg.inv(self.T)  # Inverse of the Precision matrix of the regression coefficients
        T_sparse = scipy.sparse.csr_matrix(self.T)

        if set_Q_init == True:
            for jj in range(0, l_TH):
                # print("Initialization", jj)

                """Initialize Q_{x|eta}"""
                # _{field values|eta}             kappa          alpha
                Q_rc, Q_d = gmrf_Q(lx, ly, THETA[0, jj], THETA[1, jj], car1=Config.set_GMRF_cartype)
                Q_temporary = coo_matrix((Q_d[0, :], (Q_rc[0, :], Q_rc[1, :])), shape=(n, n)).tocsr()
                Q_eta_inv = np.linalg.inv(Q_temporary.todense())

                """Q_{x|eta,y=/} & diag_Q_inv """
                A2 = Q_temporary.dot(-1 * F_sparse)
                B1 = -1 * FT_sparse.dot(Q_temporary)
                B2 = scipy.sparse.csr_matrix.dot(FT_sparse, Q_temporary.dot(F_sparse)) + T_sparse
                H1 = hstack([Q_temporary, A2])
                H2 = hstack([B1, B2])
                filename = "Q_t_" + str(jj)
                Q_t = scipy.sparse.vstack([H1, H2]).tocsr()
                setattr(self, filename, Q_t)
                np.savez(filename, data=Q_t.data, indices=Q_t.indices,
                         indptr=Q_t.indptr, shape=Q_t.shape)

                C1 = Q_eta_inv + np.dot(self.F, np.dot(T_inv, self.F.T))
                C2 = np.dot(self.F, T_inv)
                D1 = np.dot(self.F, T_inv).T
                Q_t_inv = np.vstack([np.hstack([C1, C2]),
                                     np.hstack([D1, T_inv])])

                self.diag_Q_t_inv[:, jj] = Q_t_inv.diagonal()
            np.save('diag_Q_t_inv.npy', self.diag_Q_t_inv)

        else:
            print('Loading precalculated matrices')
            for j2 in range(0, l_TH):
                filename = "Q_t_" + str(j2)
                loader = np.load(filename + '.npz')
                Q_t2 = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                  shape=loader['shape'])
                filename2 = "Q_t_" + str(j2)
                setattr(self, filename2, Q_t2)
            self.diag_Q_t_inv = np.load('diag_Q_t_inv.npy')

        """Initialize GMRF algorithm matrices"""
        self.b = np.zeros(shape=(n + p, 1))  # Canonical mean
        self.c = 0.0  # Log-likelihood update vector
        self.h_theta = np.zeros(shape=(n + p, l_TH))
        self.g_theta = np.zeros(shape=(l_TH, 1))
        self.log_pi_y = np.zeros(shape=(l_TH, 1))
        self.pi_theta = np.zeros(shape=(l_TH, 1))
        self.mue_theta = np.zeros(shape=(n + p, l_TH))
        self.mue_x = np.zeros(shape=(n + p, 1))
        self.var_x = np.zeros(shape=(n + p, 1))

        self.params = (lxf, lyf, dvx, dvy, lx, ly, n, p, de, l_TH, p_THETA, xg_min, xg_max, yg_min, yg_max)

    def gmrf_bayese_update(self, x_auv, y_t):
        """updates the GMRF Class beliefe of the true field
            Input: State, New measurement
            Output: Field Mean, Field Variance, Parameter Posterior
        """
        (lxf, lyf, dvx, dvy, lx, ly, n, p, de, l_TH, p_THETA, xg_min, xg_max, yg_min, yg_max) = self.params
        """Compute interpolation matrix"""
        u = Config.interpolation_matrix(x_auv, n, p, lx, xg_min, yg_min, de)
        u_sparse = scipy.sparse.csr_matrix(u)

        """Update canonical mean and observation-dependent likelihood terms"""
        self.b = self.b + (y_t / Config.sigma_w_squ) * u  # Canonical mean
        self.c = self.c - ((y_t ** 2) / (2 * Config.sigma_w_squ))  # Likelihood term

        for jj in range(0, l_TH):
            """Calculate observation precision (?)"""
            filename = "Q_t_" + str(jj)
            Q_temporary = getattr(self, filename)
            self.h_theta[:, jj] = scipy.sparse.linalg.spsolve(Q_temporary, u_sparse).T

            """Update Precision Matrix"""
            self.diag_Q_t_inv[:, jj] = np.subtract(self.diag_Q_t_inv[:, jj], (
            np.multiply(self.h_theta[:, jj], self.h_theta[:, jj]) / (Config.sigma_w_squ + np.dot(u.T, self.h_theta[:, jj]))))
            Q_temporary = Q_temporary + (1 / Config.sigma_w_squ) * u_sparse.dot(u_sparse.T)
            setattr(self, filename, Q_temporary)
            # Check precision matrix
            # my_data = Q_t.todense()
            # my_data[my_data == 0.0] = np.nan
            # plt.matshow(my_data, cmap=cm.Spectral_r, interpolation='none')
            # plt.show()

            self.g_theta[jj] = self.g_theta[jj] - (0.5 * np.log(1 + (1 / Config.sigma_w_squ) * np.dot(u.T, self.h_theta[:, jj])))

        for hh in range(0, l_TH):
            """Compute canonical mean"""
            filename = "Q_t_" + str(jj)
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