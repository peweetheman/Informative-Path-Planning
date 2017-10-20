import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from random import randint
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Efficient-GMRF VERSION STATUS
"""
Goal: Combine Solowjow-Paper GMRF with Xu et al. GMRF

Input:
(1) prior distribution of theta
(2) spatial sites S*, x_field, y_field
(3) extended sites S, x_grid, y_grid
(4) regression function f(.)
Output:
(1) predictive mean mue_xy
(2) predictive variance sig_xy_sq

Variables:
p := full latent field (z.T, betha.T)
"""

"""Field initialization"""
# -------------------------------------------------------------------------
# Field size
x_min = 0
x_max = 10
y_min = 0
y_max = 5

# INITIALIZATION GMRF
lxf = 50  # Number of x-axis GMRF vertices inside field
lyf = 25
de = np.array([float(x_max - x_min)/(lxf-1), float(y_max - y_min)/(lyf-1)])  # Element width in x and y

dvx = 5  # Number of extra GMRF vertices at border of field
dvy = 5
xg_min = x_min - dvx * de[0]  # Min GMRF field value in x
xg_max = x_max + dvx * de[0]
yg_min = y_min - dvy * de[1]
yg_max = y_max + dvy * de[1]

lx = lxf + 2*dvx; ly = lyf + 2*dvy  # Total number of GMRF vertices in x and y
xf_grid = np.atleast_2d(np.linspace(x_min, x_max, lxf)).T  # GMRF grid inside field
yf_grid = np.atleast_2d(np.linspace(y_min, y_max, lyf)).T
x_grid = np.atleast_2d(np.linspace(xg_min, xg_max, lx)).T  # Total GMRF grid
y_grid = np.atleast_2d(np.linspace(yg_min, yg_max, ly)).T


# ---------------------------------------------------------------------------------
""" Define field topology of the vector field values"""
def gmrf_Q(lx, ly, kappa, alpha, car1=False):

    field_info = np.arange(lx * ly).reshape((ly, lx))
    infmat = np.dot(-1, np.ones((lx * ly, 13)))

    # TORUS VERTICE TOPOLOGY
    # Define Observation matrix that map grid vertices to continuous measurement locations
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
    Q = np.zeros(shape=(lx * ly, lx * ly))

    if car1 == True:
        for i1 in range(0, (lx * ly)):
            Q[i1, i1] = a * (1/kappa)
            Q[i1, infmat[i1, 1:5].astype(int)] = -1 * (1/kappa)
        return Q
    else:
        for i2 in range(0, (lx * ly)):
            Q[i2, i2] = (4 + a **2) * (1/kappa)
            Q[i2, infmat[i2, 1:5].astype(int)] = -2 * a * (1/kappa)
            Q[i2, infmat[i2, 5:9].astype(int)] = 1 * (1/kappa)
            Q[i2, infmat[i2, 9:13].astype(int)] = 2 * (1/kappa)
        return Q


# ---------------------------------------------------------------------------------------------
"""SAMPLE from GMRF"""
def sample_from_GMRF(lx1, ly1, kappa, alpha, car_var, plot_gmrf):
    # Calculate precision matrix
    Q_storage = np.zeros(shape=(lx1 * ly1, lx1 * ly1, len(kappa)))

    for i in range(len(kappa)):
        Q_storage[:, :, i] = gmrf_Q(lx1, ly1, kappa[i], alpha[i], car1=car_var[i])

    # Check Infmat and Q
    """
    print(ly, lx, ly*lx)
    print(infmat[0])
    print(infmat[24, :])
    print(infmat[24, 1:5])
    print(Q[i, infmat[i, 1:5].astype(int)] )
    print(Q[24, :])
    print(np.linalg.matrix_rank(Q, tol=None))
    """
    # Draw sampel from GMRF
    mue_Q = 10
    z_I = np.random.standard_normal(size=lx1 * ly1)
    x_Q = np.zeros(shape=(ly1, lx1, len(kappa)))

    for i in range(0, Q_storage.shape[2]):
        L_Q = np.linalg.cholesky(Q_storage[:, :, i])
        v_Q = np.linalg.solve(L_Q.T, z_I)
        x_Q_vec = mue_Q + v_Q
        x_Q[:, :, i] = x_Q_vec.reshape((ly1, lx1))

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
                        cf = ax[i,j].pcolor(np.linspace(x_min, x_max, num=lx, endpoint=True),
                                        np.linspace(y_min, y_max, num=ly, endpoint=True), x_Q[:, :, k])
                        ax[i, j].axis('tight')
                        plt.colorbar(cf, ax=ax[i,j])
                        ax[i, j].set_title('GMRF sample, kappa: ' + str(kappa[k]) + ', alpha: ' + str(alpha[k]))
                    plt.xlabel('x (m)')
                    plt.ylabel('y (m)')
                    k += 1
                plt.show()

    return x_Q

"""# e. g.
car_var = [False, False, False, True, True, True]  # To which car() model corresponds the hyperparameters?
#kappa = [4, 1, 0.25]  # Kappa for CAR(2) from paper "Efficient Bayesian spatial"
#alpha = [0.0025, 0.01, 0.04]

#kappa = [1, 1, 1]  # Kappa for CAR(1)
#alpha = [0.1, 0.001, 0.00001]

kappa = [4, 1, 0.25, 1, 1, 1]  # Kappa for CAR(1) and CAR(2)
alpha = [0.0025, 0.01, 0.04, 0.1, 0.001, 0.00001]

sample_from_GMRF(lx, ly, kappa, alpha, car_var)
"""  # Plot example gmrfs

# -------------------------------------------------------------------------
"""TEMPERATURE FIELD (Ground truth)"""
"""Analytic field"""
#z = np.array([[10, 10.625, 12.5, 15.625, 20],[5.625, 6.25, 8.125, 11.25, 15.625],[3, 3.125, 5., 12, 12.5],[5, 2, 3.125, 10, 10.625],[5, 15, 15, 5.625, 9]])
#X = np.atleast_2d([0, 2, 4, 6, 9])  # Specifies column coordinates of field
#Y = np.atleast_2d([0, 1, 3, 5, 10])  # Specifies row coordinates of field
#x_field = np.arange(x_min, x_max, 1e-2)
#y_field = np.arange(y_min, y_max, 1e-2)
#f = scipy.interpolate.interp2d(X, Y, z, kind='cubic')
#z_field = f(x_field, y_field)

"""Field from GMRF"""
car_var = [False]  # Use car(1)?
kappa_field = [1]  # Kappa for Choi CAR(2) true field/ Solowjow et. al CAR(1)
alpha_field = [0.01]  # Alpha for Choi CAR(2) true field/ Solowjow et. al CAR(1)
f_x = 50  # Size for true field in x
f_y = 25  # Size for true field in y

z = sample_from_GMRF(f_x, f_y, kappa_field, alpha_field, car_var, 'False')  # GMRF as in paper
X = np.linspace(x_min, x_max, num=f_x, endpoint=True)  # Specifies column coordinates of field
Y = np.linspace(y_min, y_max, num=f_y, endpoint=True)  # Specifies row coordinates of field
f = scipy.interpolate.interp2d(X, Y, z, kind='cubic')

x_field = np.arange(x_min, x_max, 1e-2)
y_field = np.arange(y_min, y_max, 1e-2)
z_field = f(x_field, y_field)

plt.figure()
cp = plt.contourf(x_field, y_field, z_field)
plt.colorbar(cp); plt.title('Temperature Field'); plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.show()


# ---------------------------------------------------------------------------------------------
"""SEQUENTIAL BAYESIAN PREDICTIVE ALGORITHM"""
# Initialize vectors and matrices
carGMRF = [False]  # Use car(1)?
p = 1  # Number of regression coefficients beta
n = lx*ly  # Number of GMRF vertices
b = np.zeros(shape=(n+p, 1)).astype(float)  # Canonical mean
#b[-1] = 10  # Initial mean field value (?)
u = np.zeros(shape=(n+p, 1)).astype(float)  # Observation topology vector
c = 0.0  # Log-likelihood update vector

"""Define hyperparameter prior"""
# Choi paper
#kappa_prior = np.array([0.0625 * (2 ** 0), 0.0625 * (2 ** 2), 0.0625 * (2 ** 4), 0.0625 * (2 ** 6), 0.0625 * (2 ** 8)]).astype(float)
#alpha_prior = np.array([0.000625 * (1 ** 2), 0.000625 * (2 ** 2), 0.000625 * (4 ** 2), 0.000625 * (8 ** 2), 0.000625 * (16 ** 2)]).astype(float)

# Choi Parameter (size 2)
kappa_prior = np.array([0.0625 * (2 ** 2), 0.0625 * (2 ** 4)]).astype(float)
alpha_prior = np.array([0.000625 * (2 ** 2), 0.000625 * (4 ** 2)]).astype(float)

# Choi Parameter (size 3)
#kappa_prior = np.array([0.0625 * (2 ** 2), 0.0625 * (2 ** 4), 0.0625 * (2 ** 6)]).astype(float)
#alpha_prior = np.array([0.000625 * (2 ** 2), 0.000625 * (4 ** 2), 0.000625 * (8 ** 2)]).astype(float)

# Solowjow Parameter for CAR(1) (size 1)
#kappa_prior = np.array([1]).astype(float)
#alpha_prior = np.array([0.01]).astype(float)

# Same theta values (size 5)
#kappa_prior = np.array([0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4)]).astype(float)
#alpha_prior = np.array([0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2)]).astype(float)

# Same theta values (size 10)
#kappa_prior = np.array([0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4), 0.0625 * (2 ** 4)]).astype(float)
#alpha_prior = np.array([0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2), 0.000625 * (4 ** 2)]).astype(float)

# Extended Choi paper
#kappa_prior = np.array([1000, 100, 10, 0.0625 * (2 ** 0), 0.0625 * (2 ** 2), 0.0625 * (2 ** 4), 0.0625 * (2 ** 6), 0.0625 * (2 ** 8), 0.0625 * (2 ** 9), 0.0625 * (2 ** 10)]).astype(float)
#alpha_prior = np.array([0.000625 * (1 ** -1), 0.000625 * (1 ** 0), 0.000625 * (1 ** 1), 0.000625 * (1 ** 2), 0.000625 * (2 ** 2), 0.000625 * (4 ** 2), 0.000625 * (8 ** 2), 0.000625 * (16 ** 2), 0.000625 * (32 ** 2), 0.000625 * (64 ** 2), 0.000625 * (128 ** 2)]).astype(float)

THETA = []  # Matrix containing all discrete hyperparameter combinations
for i in range(0, len(alpha_prior)):
    for j in range(0, len(kappa_prior)):
        THETA.append([kappa_prior[j], alpha_prior[i]])
THETA = np.array(THETA).T
l_TH = len(THETA[1])  # Number of hyperparameter pairs
p_THETA = 1.0 / l_TH  # Prior probability for one theta

"""Initialize mean regression functions"""
F = np.ones(shape=(n, p)).astype(float)  # Mean regression functions

"""Initialize matrices and vectors"""
T = 1e-6 * np.ones(shape=(p, p)).astype(float)  # Precision matrix of the regression coefficients
T_inv = np.linalg.inv(T)

Q_eta = np.zeros(shape=(n, n, l_TH)).astype(float)
Q_eta_inv = np.zeros(shape=(n, n, l_TH)).astype(float)
Q_t = np.zeros(shape=(n+p, n+p, l_TH)).astype(float)
Q_t_inv = np.zeros(shape=(n+p, n+p, l_TH)).astype(float)
Q_t_inv2 = np.zeros(shape=(n+p, n+p, l_TH)).astype(float)
L_Qt = np.zeros(shape=(n+p, n+p, l_TH)).astype(float)

h_theta = np.zeros(shape=(n+p, l_TH)).astype(float)
diag_Q_t_inv = np.zeros(shape=(n+p, l_TH)).astype(float)
mue_theta = np.zeros(shape=(n+p, l_TH)).astype(float)
mue_x = np.zeros(shape=(n+p, 1)).astype(float)
var_x = np.zeros(shape=(n+p, 1)).astype(float)
g_theta = np.zeros(shape=(l_TH, 1)).astype(float)
log_pi_y = np.zeros(shape=(l_TH, 1)).astype(float)
pi_theta = np.zeros(shape=(l_TH, 1)).astype(float)

"""Initialize precision matrix for different thetas"""
for jj in range(0, l_TH):
    """Initialize Q_{x|eta}"""
    # _{field values|eta}             kappa          alpha
    Q_eta[:, :, jj] = gmrf_Q(lx, ly, THETA[0, jj], THETA[1, jj], car1=carGMRF)
    Q_eta_inv[:, :, jj] = np.linalg.inv(Q_eta[:, :, jj])

    """Q_{x|eta,y=/}"""
    Q_t[:, :, jj] = np.vstack((np.hstack((Q_eta[:, :, jj],  np.dot(-1 * Q_eta[:, :, jj], F))),
                               np.hstack((np.dot(-1 * F.T, Q_eta[:, :, jj]),
                                          np.dot(F.T, np.dot(Q_eta[:, :, jj], F)) + T))))
    # Check Q_eta
    #A = Q_eta[:, :, jj]
    #B = np.dot(-Q_eta[:, :, jj], F)
    #C = np.dot(-F.T, Q_eta[:, :, jj])
    #D = np.dot(-F.T, np.dot(Q_eta[:, :, jj], F)) + T
    #print(A.shape, B.shape)
    #print(C.shape, D.shape)
    Q_t_inv[:, :, jj] = np.vstack((np.hstack((Q_eta_inv[:, :, jj] + np.dot(F, np.dot(T_inv, F.T)),  np.dot(F, T_inv))),
                                   np.hstack((np.dot(F, T_inv).T, T_inv))))
    diag_Q_t_inv[:, jj] = np.diagonal(Q_t_inv[:, :, jj])


#----------------------------------
"""START SIMULATION"""
sigma_w_squ = 0.2 ** 2  # Measurement variance
#n_y = 4  # Number of measurements per update

# Begin for-slope for all N observation at time t
while True:
    x = raw_input("Press [enter] to continue or [q] to quit")
    if x == 'q':
        break

    """Create discrete measurement"""
    #for nm in range(n_m)
    nxf = randint(0, len(xf_grid) - 2)  # Measurement at random grid
    nyf = randint(0, len(yf_grid) - 2)
    s_obs = [yf_grid[nyf], xf_grid[nxf]]
    sd_obs = [int((s_obs[0]) * 1e2), int((s_obs[1]) * 1e2)]
    y_t = np.array(z_field[sd_obs[0], sd_obs[1]])
    print('x/y in m,s_obs[0]', ' | ', s_obs[1], 'y_t', y_t)
    nx = nxf + dvx  # Calculates the vertice column x-number at which the shape element starts.
    ny = nyf + dvy  # Calculates the vertice row y-number at which the shape element starts.
    kk = ny * lx + nx
    #print('nxf', nxf, 'nyf', nyf, 's_obs', s_obs, 'sd_obs', sd_obs, 'kk', kk)
    #print('Before Update Q_t[kk, kk, jj]', Q_t[kk, kk, jj])

    """Set observation index location"""
    u = np.zeros(shape=(n + p, 1)).astype(float)  # Observation topology vector
    u[kk] = 1

    """Update canonical mean"""
    b = b + (y_t/sigma_w_squ) * u

    """Compute observation-dependent likelihood terms"""
    c = c - ((y_t ** 2) / (2 * sigma_w_squ))  # Likelihood term

    for jj in range(0, l_TH):
        print(jj)
        """Calculate observation precision (?)"""
        L_Qt[:, :, jj] = np.linalg.cholesky(Q_t[:, :, jj])
        v_h = np.linalg.solve(L_Qt[:, :, jj], u)
        h_theta[:, jj] = np.linalg.solve(L_Qt[:, :, jj].T, v_h).T
        #h_theta[:, jj] = np.linalg.solve(Q_t[:, :, jj], u).T

        """Update Precision Matrix"""
        diag_Q_t_inv[:, jj] = np.subtract(diag_Q_t_inv[:, jj],  (np.multiply(h_theta[:, jj], h_theta[:, jj]) / (sigma_w_squ + np.dot(u.T, h_theta[:, jj]))))
        Q_t[:, :, jj] = Q_t[:, :, jj] + (1 / sigma_w_squ) * np.dot(u, u.T)

        g_theta[jj] = g_theta[jj] - (0.5 * np.log(1 + (1 / sigma_w_squ) * np.dot(u.T, h_theta[:, jj])))
    # End for-slope for all N observation at time t

    for hh in range(0, l_TH):
        print(hh)
        """Compute canonical mean"""
        L_Qt[:, :, hh] = np.linalg.cholesky(Q_t[:, :, hh])
        v_t = np.linalg.solve(L_Qt[:, :, hh], b)
        mue_theta[:, hh] = np.linalg.solve(L_Qt[:, :, hh].T, v_t).T
        #mue_theta[:, [hh]] = np.linalg.solve(Q_t[:, :, hh], b)

        """Compute Likelihood"""
        log_pi_y[hh] = c + g_theta[hh] + 0.5 * np.dot(b.T, mue_theta[:, hh]) # - (1 / 2) * np.log(2*np.pi*sigma_w_squ)  # Compute likelihood
        #print('0.5 * np.dot(b.T, mue_theta[:, hh])', 0.5 * np.dot(b.T, mue_theta[:, hh]))
    #print('c', c)
    #print('g_theta', g_theta)

    """Scale likelihood and Posterior distribution (theta|y)"""
    log_pi_exp = np.exp(log_pi_y - np.amax(log_pi_y))
    posterior = (1 / np.sum(log_pi_exp)) * log_pi_exp * p_THETA

    pi_theta = (1 / np.sum(posterior)) * posterior  # Compute posterior distribution
    #pi_theta = posterior  # Compute posterior distribution
    #pi_theta = np.zeros(shape=(25, 1))  # Posterior for DEBUGGING
    #pi_theta[12, 0] = 1

    """Predictive mean and variance (x|y)"""
    for ji in range(0, n+p):
        mue_x[ji] = np.dot(mue_theta[[ji], :], pi_theta)  # Predictive Mean
        var_x[ji] = np.dot((diag_Q_t_inv[ji] + (np.subtract(mue_theta[ji, :], mue_x[ji] * np.ones(shape=(1, len(THETA[1])))) ** 2)),
                           pi_theta)

    # --------------------------------------------
    """PLOT RESULT"""
    xv, yv = np.meshgrid(x_grid, y_grid)
    mue_x_plot = mue_x[0:(lx*ly)].reshape((ly, lx))
    var_x_plot = var_x[0:(lx*ly)].reshape((ly, lx))

    xv_list = xv.reshape((lx*ly, 1))
    yv_list = yv.reshape((lx*ly, 1))
    labels = ['{0}'.format(i) for i in range(lx*ly)]  # Labels for annotating GMRF nodes

    """Plot GMRF mean values"""
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(131)
    c1 = ax1.contourf(np.linspace(xg_min, xg_max, num=lx, endpoint=True), np.linspace(yg_min, yg_max, num=ly, endpoint=True),
                     mue_x_plot, vmin=-1, vmax=22)
    plt.colorbar(c1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.scatter(xv, yv, marker='+', facecolors='k')
    plt.plot([x_min,x_min,x_max,x_max, x_min], [y_min,y_max,y_max,y_min, y_min], "c")
    ax1.set_title('GMRF')

    ax2 = fig.add_subplot(132)
    c2 = ax2.contourf(np.linspace(xg_min, xg_max, num=lx, endpoint=True), np.linspace(yg_min, yg_max, num=ly, endpoint=True),
                    var_x_plot)
    """Plot GMRF variance"""
    plt.colorbar(c2)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.scatter(xv, yv, marker='+', facecolors='k')
    plt.plot([x_min,x_min,x_max,x_max, x_min], [y_min,y_max,y_max,y_min, y_min], "c")
    ax2.set_title('Variance')

    """ # Label GMRF vertices
    for label, x, y in zip(labels, xv_list, yv_list):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-2, 2),
            textcoords='offset points', ha='center', va='center',
            #bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
    """

    """Plot Hyperparameter estimate"""
    _alpha_v, _kappa_v = np.meshgrid(alpha_prior, kappa_prior)
    alpha_v, kappa_v = _alpha_v.ravel(), _kappa_v.ravel()
    bottom = np.zeros_like(pi_theta)
    _x = _y = np.arange(len(alpha_prior))
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    ax3 = fig.add_subplot(133, projection='3d')
    colors = plt.cm.jet(np.arange(len(x)) / float(np.arange(len(x)).max()))
    #colors = plt.cm.jet(pi_theta.flatten() / float(pi_theta.max()))  # Color height dependent
    ax3.bar3d(x, y, bottom, 1, 1, pi_theta,color=colors ,alpha=0.5)
    ticksx = np.arange(0.5, len(alpha_prior)+0.5, 1)
    plt.xticks(ticksx, alpha_prior)
    plt.yticks(ticksx, kappa_prior)
    ax3.set_xlabel('alpha')
    ax3.set_ylabel('kappa')
    ax3.set_zlabel('p(theta)')
    ax3.set_title('Hyperparameter Posterior')

    plt.show()
    #plt.pause(0.001)
    #plt.draw



