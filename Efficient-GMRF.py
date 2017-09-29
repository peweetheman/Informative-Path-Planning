import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from random import randint

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

# -------------------------------------------------------------------------
# TEMPERATURE FIELD (Ground truth)
z = np.array([[10, 10.625, 12.5, 15.625, 20],
           [5.625, 6.25, 8.125, 11.25, 15.625],
           [3, 3.125, 5., 12, 12.5],
           [5, 2, 3.125, 10, 10.625],
           [5, 15, 15, 5.625, 9]])
X = np.atleast_2d([0, 2, 4, 6, 9])  # Specifies column coordinates of field
Y = np.atleast_2d([0, 1, 3, 5, 10])  # Specifies row coordinates of field

f = scipy.interpolate.interp2d(X, Y, z, kind='cubic')
x_min = 0
x_max = 10
y_min = 0
y_max = 5
x_field = np.arange(x_min, x_max, 1e-2)
y_field = np.arange(y_min, y_max, 1e-2)
z_field = f(x_field, y_field)

# PLOT TEMPERATURE FIELD and STREAMING FIELD
""" 
plt.figure()
cp = plt.contourf(x_field, y_field, z_field)
plt.colorbar(cp); plt.title('Temperature Field'); plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.show('None')

# Streamingfield
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
Y, X = np.meshgrid.sparse(x, y)
u = -1 - X**2 + Y
v = 1 + X - Y**2
"""


# -------------------------------------------------------------------------
# INITIALIZATION GMRF
lxf = 20  # Number of x-axis GMRF vertices inside field
lyf = 10
de = np.array([float(x_max - x_min)/(lxf-1), float(y_max - y_min)/(lyf-1)])
# Element width in x and y

dvx = 5  # Number of extra GMRF vertices at border of field
dvy = 5
xg_min = x_min - dvx * de[0]  # Min GMRF field value in x
xg_max = x_max + dvx * de[0]
yg_min = y_min - dvy * de[1]
yg_max = y_max + dvy * de[1]

lx = lxf + 2*dvx; ly = lyf + 2*dvy # Total number of GMRF vertices in x and y
#print('xf_min: ', xg_min,',xf_max: ', xg_max, ',de: ', de)
xf_grid = np.atleast_2d(np.linspace(x_min, x_max, lxf)).T  # GMRF grid inside field
yf_grid = np.atleast_2d(np.linspace(y_min, y_max, lyf)).T
x_grid = np.atleast_2d(np.linspace(xg_min, xg_max, lx)).T  # Total GMRF grid
y_grid = np.atleast_2d(np.linspace(yg_min, yg_max, ly)).T
#print('x_grid.shape:', x_grid.shape, ', x_grid min. Value: ', x_grid[0], 'x_grid min. Value:',  x_grid[len(x_grid) - 1])

# INITIALIZE VECTORS AND MATRICES
n = lx*ly  # Number of GMRF vertices
p = 4  # Number of regression coefficients betha
b = np.zeros(shape=(n+p, 1))  # Canonical mean
u = np.zeros(shape=(n+p, 1))  # Observation topology vector
c = 0  # Log-likelihood update vector

# Define hyperparameters
alpha_prior = np.array([0.000625 * (1 ** 2), 0.000625 * (2 ** 2), 0.000625 * (4 ** 2), 0.000625 * (8 ** 2), 0.000625 * (16 ** 2)])
kappa_prior = np.array([0.0625 * (2 ** 0), 0.0625 * (2 ** 2), 0.0625 * (2 ** 4), 0.0625 * (2 ** 6), 0.0625 * (2 ** 8)])
prob_theta_prior = 1/25

THETA = []  # Matrix containing all discrete hyperparameter combinations
for i in range(0, len(alpha_prior)):
    for j in range(0, len(kappa_prior)):
        THETA.append([kappa_prior[j], alpha_prior[i]])
THETA = np.array(THETA).T

# Calculate precision matrix for different thetas
for jj in range(0, len(THETA[1, :])):
    theta = THETA[:, jj]

# Find out the field topology of the vector field values
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

for i in range(0, ly):
    for j in range(0, lx):

        if (i+2) <= (ly-1): a1 = i+1; a2 = i+2
        elif (i+1) <= (ly-1): a1 = i+1; a2 = 0
        else: a1 = 0; a2 = 1

        if (i-2) >= 0: b1 = i-1; b2 = i-2
        elif (i-1) >= 0: b1 = i-1; b2 = ly-1
        else: b1 = ly-1; b2 = ly-2

        if (j+2) <= (lx-1): c1 = j+1; c2 = j+2
        elif (j+1) <= (lx-1): c1 = j+1; c2 = 0
        else: c1 = 0; c2 = 1

        if (j-2) >= 0: d1 = j-1; d2 = j-2
        elif (j-1) >= 0: d1 = j-1; d2 = lx-1
        else: d1 = lx-1; d2 = lx-2
        #                                        field i,j              a1,j             b1,j               i,c1             i,d1
        infmat[field_info[i, j], :] = np.array([field_info[i, j], field_info[a1, j], field_info[b1, j], field_info[i, c1], field_info[i, d1],
                                                        #               a2,j             b2,j               i,c2             i,d2
                                                                  field_info[a2, j],field_info[b2, j], field_info[i, c2], field_info[i, d2],
                                                        #               a1,c1            a1,d1               b1,d1           b1,c1
                                                                  field_info[a1,c1], field_info[a1,d1], field_info[b1,d1], field_info[b1,c1]])


# DEFINE PRECISION MATRIX
def gmrf_Q(kappa, alpha, infmat, car1=False):
    a = alpha + 4
    Q = np.zeros(shape=(lx * ly, lx * ly))

    if car1 == True:
        for i in range(0, (lx * ly)):
            Q[i, i] = a * kappa
            Q[i, infmat[i, 1:5].astype(int)] = -kappa
        return Q
    else:
        for i in range(0, (lx * ly)):
            Q[i, i] = (4 + a **2) * kappa
            Q[i, infmat[i, 1:5].astype(int)] = -2 * a * kappa
            Q[i, infmat[i, 5:9].astype(int)] = kappa
            Q[i, infmat[i, 9:13].astype(int)] = 2 * kappa
        return Q

# Calculate precision matrix
# kappa = [4, 1, 0.25]  # Kappa for CAR(2) from paper "Efficient Bayesian spatial"
# alpha = [0.0025, 0.01, 0.04]
# kappa = [1, 1, 1]  # Kappa for CAR(1)
# alpha = [0.1, 0.001, 0.00001]

kappa = [4, 1, 0.25, 1, 1, 1]  # Kappa for CAR(1)
alpha = [0.0025, 0.01, 0.04, 0.1, 0.001, 0.00001]

Q_storage = np.zeros(shape=(lx * ly, lx * ly, len(kappa)))
print(len(kappa))
car_var = [False, False, False, True, True, True]

for i in range(len(kappa)):
    Q_storage[:, :, i] = gmrf_Q(kappa[i], alpha[i], infmat, car1=car_var[i])

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
# sampel from gmrf
mue_Q = 10
z_I = np.random.standard_normal(size=lx * ly)
x_Q = np.zeros(shape=(ly, lx, len(kappa)))

for i in range(0, Q_storage.shape[2]):
    L_Q = np.linalg.cholesky(Q_storage[:, :, i])
    v_Q = np.linalg.solve(L_Q.T, z_I)
    x_Q_vec = mue_Q + v_Q
    x_Q[:, :, i] = x_Q_vec.reshape((ly, lx))


fig, ax = plt.subplots(3, 2)
#ax = ax.ravel()
k = 0
for j in range(2):
    for i in range(3):
        cf = ax[i,j].pcolor(np.linspace(0, x_Q.shape[1], num=lx, endpoint=True),
                        np.linspace(0, x_Q.shape[0], num=ly, endpoint=True), x_Q[:, :, k])
        ax[i,j].axis('tight')
        plt.colorbar(cf, ax=ax[i,j])
        ax[i,j].set_title('GMRF sample, kappa: ' + str(kappa[k]) + ', alpha: ' + str(alpha[k]))
    #plt.xlabel('x (m)')
    #plt.ylabel('y (m)')
        k += 1
plt.show()
#plt.pause(100)


