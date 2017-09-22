import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from random import randint


# -------------------------------------------------------------------------
# TEMPERATURE FIELD (Ground truth)
z = np.array([[10, 10.625, 12.5, 15.625, 20],
           [5.625, 6.25, 8.125, 11.25, 15.625],
           [3, 3.125, 5., 12, 12.5],
           [5, 2, 3.125, 10, 10.625],
           [5, 15, 15, 5.625, 10]])
X = np.atleast_2d([0, 2, 4, 6, 10])  # Specifies column coordinates of field
Y = np.atleast_2d([0, 1, 3, 5, 10])  # Specifies row coordinates of field

f = scipy.interpolate.interp2d(X, Y, z, kind='cubic')
x_min = 0; x_max = 10; y_min = 0; y_max = 10
x_field = np.arange(x_min, x_max, 1e-2)
y_field = np.arange(y_min, y_max, 1e-2)
z_field = f(x_field, y_field)
#print(x_field.shape, y_field.shape, z_field.shape)

# PLOT TEMPERATURE FIELD
# plt.figure()
# cp = plt.contourf(x_field, y_field, z_field)
# plt.colorbar(cp); plt.title('Temperature Field'); plt.xlabel('x (m)'); plt.ylabel('y (m)')
# plt.show('None')

# Streamingfield
#x = np.linspace(-3, 3, 100)
#y = np.linspace(-3, 3, 100)
#Y, X = np.meshgrid.sparse(x, y)
#u = -1 - X**2 + Y
#v = 1 + X - Y**2

# # Mesh the input space for evaluations of the real function, the prediction and
# # its MSE
# Number of GMRF vertices in x and y
lx = 31; ly = 31

# Element width in x and y
de = np.array([float(x_max - x_min)/(lx-1), float(y_max - y_min)/(ly-1)])
print(de)
x_grid = np.atleast_2d(np.linspace(x_min, x_max, lx)).T
y_grid = np.atleast_2d(np.linspace(x_min, x_max, ly)).T

rho = 4.001
mue_0 = 0 * np.ones(shape=(lx*ly,1))


# -------------------------------------------------------------------------
# Find out the field topology of the vector field values
def get_adjacent_cells(lx, ly):
    field_info = np.arange(lx * ly).reshape((lx, ly))
    infmat = np.dot(-1, np.ones((lx * ly, 5)))

    for i in range(0, lx):
        for j in range(0, ly):
            if (i+1) <= lx-1:  a = field_info[i+1, j]
            else:  a = -1
            if (j+1) <= ly-1:  b = field_info[i, j+1]
            else:  b = -1
            if (i-1) >= 0:  c = field_info[i-1, j]
            else:  c = -1
            if (j-1) >= 0:  d = field_info[i, j-1]
            else:  d = -1

            infmat[field_info[i, j], :] = np.array([field_info[i, j], a, b, c, d])

    return infmat
topology_inf = get_adjacent_cells(lx, ly)


# --------------------------------------------------------------------------------------------------------------
# OBSERVATION MATRIX
# Define Observation matrix that map grid vertices to continuous measurement locations
def gmrf_prior(lx, ly, rho, infmat):
    # lx = Number of vertices columns (~x)
    # ly =  Number of vertices rows (~y)
    #mue = 10 * np.ones(shape=(lx * ly, 1))
    Q = np.zeros(shape=(lx * ly, lx * ly))
    for i in range(0, (lx * ly)):
        Q[i, i] = rho
        a, b, c, d = infmat[i, 1:5].astype(int)
        if a >= 0:
            Q[i, a] = -1
            Q[a, i] = -1
        if b >= 0:
            Q[i, b] = -1
            Q[b, i] = -1
        if c >= 0:
            Q[i, c] = -1
            Q[c, i] = -1
        if d >= 0:
            Q[i, d] = -1
            Q[d, i] = -1

    return Q

Q = gmrf_prior(lx, ly, rho, topology_inf)
# CHECK Q
#for i in range(0, (lx * ly)):
#   if Q_inv[5, i] > 0:
#      print(i)
#print(Q_inv)



# --------------------------------------------------------------------------------------------------------------
# OBSERVATION MATRIX
# Define Observation matrix that map grid vertices to continuous measurement locations
def observation_matrix(s_obs, lx, ly, a, b, A_prev):
    # lx = Number of vertices columns (~x)
    # ly =  Number of vertices rows (~y)

    nx = int(s_obs[0]/a) # Calculates the vertice column x-number at which the shape element starts.
    ny = int(s_obs[1]/b) # Calculates the vertice row y-number at which the shape element starts.
    # a,b is the distance between two adjacent vertices in meters

    # Calculate position value in element coord-sys in meters
    x_el = float(0.1 * (s_obs[0]/0.1 - int(s_obs[0]/0.1))) - a/2
    y_el = float(0.1 * (s_obs[1]/0.1 - int(s_obs[1]/0.1))) - b/2
    A = np.transpose(np.zeros(shape=(lx*ly, 1)))
    n_lu = (ny * lx) + nx
    n_ru = (ny * lx) + nx + 1
    n_lo = ((ny + 1) * lx) + nx
    n_ro = ((ny + 1) * lx) + nx + 1

    # Define shape functions, "a" is element width in x-direction
    A[0, n_lu] = (1 / (a * b)) * ((x_el - a/2) * (y_el - b/2))
    A[0, n_ru] = (-1 / (a * b)) * ((x_el + a/2) * (y_el - b/2))
    A[0, n_lo] = (-1 / (a * b)) * ((x_el - a/2) * (y_el + b/2))
    A[0, n_ro] = (1 / (a * b)) * ((x_el + a/2) * (y_el + b/2))

    if A_prev.size == 0:
        A_new = A
    else:
        A_new = np.vstack((A_prev, A))
    return A_new


# --------------------------------------------------------------------------------------------------------------
# GMRF POSTERIOR
# Update GMRF with new observation
sigz = 0.001
def GMRF_posterior(mue_w, Q, A, sigz, z_obs):
    # The variance of the latent field given the new observation
    Q_w = Q + (1/(sigz**2)) * np.matrix.dot(A.T, A)
    L_Q = np.linalg.cholesky(Q_w)
    Q_invAtZmue = np.linalg.solve(L_Q.T, np.linalg.solve(L_Q, np.matrix.dot(A.T, (z_obs - np.matrix.dot(A, mue_w)))))
    #print(A.T.shape, 'dwad',z_obs - np.matrix.dot(A, mue_w))
    mue_w2 = mue_w + (1/(sigz**2)) * Q_invAtZmue
    #print('mue_w2',mue_w2.shape, Q_w.shape)
    FuncOutput = np.array([[mue_w2], [Q_invAtZmue]])
    return {'mue_w2':mue_w2, 'Q_w2':Q_w}


# --------------------------------------------------------------------------------------------------------------
# RUN SIMULATION
def run_GMRF(lx, ly, de, mue_0, Q, sigz, x_min, x_max, y_min, y_max, x_grid, y_grid):
    n = 0
    A = np.array([])
    z_obs = np.array([])
    z_vert = np.array([])
    plt.ion()
    plt.show()

    while True:
        #x = raw_input('Press enter for a random letter...')
        x = raw_input("Press [enter] to continue or [q] to quit")
        if x == 'q':
            break

        #--------------------------------------------------------------------------------------------------------------
        #  CREATE OBSERVATIONS

        # # RANDOM
        # sd_obs = np.array([randint(0, len(y_field)-1), randint(0, len(x_field)-1)])
        # s_obs = np.array([1e-2 * sd_obs[0], 1e-2 * sd_obs[1]])
        #
        # # TRAJECTORY
        # #s_obs = np.array([9.9, 0.1]) + n * np.array([-0.5, 0.5])
        # #n += 1
        # #print(n)
        # #sd_obs = np.array([s_obs[0] * 100, s_obs[1] * 100]).astype(int)
        #
        # # CHOOSE
        # #s_obs = np.array([9.9, 9.9])
        # #sd_obs = np.array([s_obs[0] * 100, s_obs[1] * 100]).astype(int)
        #
        # # CALCULATE MEASUREMENT
        # z_new = np.array(z_field[sd_obs[1], sd_obs[0]])
        # if z_obs.size == 0:
        #     z_obs = z_new
        # else:
        #     z_obs = np.vstack((z_obs, z_new))
        # print('Observation value: ', z_obs, 'x: ', s_obs[0], 'y: ', s_obs[1])
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # OBSERVATION MATRIX
        # A = observation_matrix(s_obs, lx, ly, de[0], de[1], A)
        # #rows, cols = np.nonzero(A)
        # # print('Non zero >row,col< values in A: ',rows, cols, 'Values of A: ', A[rows, cols])
        #
        # # --------------------------------------------------------------------------------------------------------------
        # # CALCULATE MEAN
        # #mue_0 = np.mean(z_obs) * np.ones(shape=(lx*ly,1))

        # Calculate 10 MEASUREMENTS ON GRID
        ni = 6 # Min 2 for one measurement
        nj = 7
        for i in range(1, ni):
            for j in range(1, nj):
                di = float(x_max - x_min) / ni
                dj = float(y_max - y_min) / nj
                #print('x_max - x_min',x_max - x_min)
                xi = di * i
                #print('xi', xi, 'i', i)
                yj = dj * j
                #print('xi',xi,'yj',yj)
                s_obs = np.array([xi, yj])
                sd_obs = np.array([s_obs[0] * 100, s_obs[1] * 100]).astype(int)
                z_new = np.array(z_field[sd_obs[1], sd_obs[0]])
                if z_obs.size == 0:
                     z_obs = z_new
                     s_obs_plot = s_obs
                else:
                     z_obs = np.vstack((z_obs, z_new))
                s_obs_plot = np.vstack((s_obs_plot, s_obs))
                #print('Observation value: ', z_obs, 'x: ', s_obs[0], 'y: ', s_obs[1])
                A = observation_matrix(s_obs, lx, ly, de[0], de[1], A)


        # --------------------------------------------------------------------------------------------------------------
        # GMRF POSTERIOR
        out1 = GMRF_posterior(mue_0, Q, A, sigz, z_obs)
        mue_w = out1.get('mue_w2')
        Q_w2 = out1.get('Q_w2')

        # --------------------------------------------------------------------------------------------------------------
        # PLOT MEAN
        xv, yv = np.meshgrid(x_grid, y_grid)

        mue_m = mue_w.reshape((lx, ly))
        fig = plt.figure()
        #contourf, pcolor
        cp = plt.contourf(np.linspace(x_min, x_max, num=lx, endpoint=True),
                      np.linspace(y_min, y_max, num=ly, endpoint=True), mue_m,
                     vmin=-1, vmax=20)
        plt.colorbar(cp); plt.title('GMRF'); plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.scatter(xv, yv, marker='+',facecolors='k')
        plt.scatter(s_obs_plot[:,0], s_obs_plot[:,1], facecolors='none', edgecolors='r')
        #plt.draw()
        #plt.pause(0.001)

        # PLOT prediction error variances
        x_vertices = np.linspace(x_min, x_max, lx)
        y_vertices = np.linspace(y_min, y_max, ly)
        z_vertices = f(x_vertices, y_vertices)

        e_var_plot = (np.subtract(z_vertices, mue_m))**2
        #e_var = z_vertices
        print(z_vert)

        fig = plt.figure()
        #contourf, pcolor
        cp = plt.pcolor(np.linspace(x_min, x_max, num=lx, endpoint=True),
                      np.linspace(y_min, y_max, num=ly, endpoint=True), e_var_plot,
                     vmin=0, vmax=50)
        plt.colorbar(cp); plt.title('Error Variance'); plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.scatter(xv, yv, marker='+',facecolors='k')
        plt.scatter(s_obs_plot[:,0], s_obs_plot[:,1], facecolors='none', edgecolors='r')
        plt.draw()
        plt.pause(100)


run_GMRF(lx, ly, de, mue_0, Q, sigz, x_min, x_max, y_min, y_max, x_grid, y_grid)