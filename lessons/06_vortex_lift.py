import numpy as np
import matplotlib.pyplot as plt

N = 50                                # Number of points in each direction
x_start, x_end = -2.0, 2.0            # x-direction boundaries
y_start, y_end = -1.0, 1.0            # y-direction boundaries
x = np.linspace(x_start, x_end, N)    # computes a 1D-array for x
y = np.linspace(y_start, y_end, N)    # computes a 1D-array for y
X, Y = np.meshgrid(x, y)

kappa = 1.0                           # strength of the doublet
x_doublet, y_doublet = 0.0, 0.0       # location of the doublet

u_inf = 1.0


def get_velocity_doublet(strength, xd, yd, X, Y):

    u = - strength / (2 * np.pi) * ((X - xd)**2 - (Y - yd)
                                    ** 2) / ((X - xd)**2 + (Y - yd)**2)**2
    v = - strength / (2 * np.pi) * 2 * (X - xd) * \
        (Y - yd) / ((X - xd)**2 + (Y - yd)**2)**2

    return u, v


def get_stream_function_doublet(strength, xd, yd, X, Y):

    psi = - strength / (2 * np.pi) * (Y - yd) / ((X - xd)**2 + (Y - yd)**2)

    return psi

# computes the velocity field on the mesh grid
u_doublet, v_doublet = get_velocity_doublet(kappa, x_doublet, y_doublet, X, Y)

# computes the stream-function on the mesh grid
psi_doublet = get_stream_function_doublet(kappa, x_doublet, y_doublet, X, Y)

# freestream velocity components
u_freestream = u_inf * np.ones((N, N), dtype=float)
v_freestream = np.zeros((N, N), dtype=float)

# stream-function of the freestream flow
psi_freestream = u_inf * Y

# superposition of the doublet on the freestream flow
u = u_freestream + u_doublet
v = v_freestream + v_doublet
psi = psi_freestream + psi_doublet

# plots the streamlines
size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u, v, density=2, linewidth=1,
               arrowsize=1, arrowstyle='->')
plt.scatter(x_doublet, y_doublet, color='#CD2305', s=80, marker='o')

# calculates the cylinder radius and adds it to the figure
R = np.sqrt(kappa / (2 * np.pi * u_inf))
circle = plt.Circle((0, 0), radius=R, color='#CD2305', alpha=0.5)
plt.gca().add_patch(circle)

# calculates the stagnation points and adds it to the figure
x_stagn1, y_stagn1 = +np.sqrt(kappa / (2 * np.pi * u_inf)), 0
x_stagn2, y_stagn2 = -np.sqrt(kappa / (2 * np.pi * u_inf)), 0
plt.scatter([x_stagn1, x_stagn2], [y_stagn1, y_stagn2],
            color='g', s=80, marker='o')

plt.show()

gamma = 4 * np.pi * u_inf * R

x_vortex, y_vortex = 0.0, 0.0    # location of the vortex


def get_velocity_vortex(strength, xv, yv, X, Y):

    u = + strength / (2 * np.pi) * (Y - yv) / ((X - xv)**2 + (Y - yv)**2)
    v = - strength / (2 * np.pi) * (X - xv) / ((X - xv)**2 + (Y - yv)**2)
    return u, v


def get_stream_function_vortex(strength, xv, yv, X, Y):

    psi = strength / (4 * np.pi) * np.log((X - xv)**2 + (Y - yv)**2)

    return psi

# computes the velocity field on the mesh grid
u_vortex, v_vortex = get_velocity_vortex(gamma, x_vortex, y_vortex, X, Y)

# computes the stream-function on the mesh grid
psi_vortex = get_stream_function_vortex(gamma, x_vortex, y_vortex, X, Y)

# superposition of the doublet and the vortex on the freestream flow
u = u_freestream + u_doublet + u_vortex
v = v_freestream + v_doublet + v_vortex
psi = psi_freestream + psi_doublet + psi_vortex

# calculates cylinder radius
R = np.sqrt(kappa / (2 * np.pi * u_inf))

# calculates the stagnation points
x_stagn1, y_stagn1 = + \
    np.sqrt(R**2 - (gamma / (4 * np.pi * u_inf))**2), - \
    gamma / (4 * np.pi * u_inf)
x_stagn2, y_stagn2 = - \
    np.sqrt(R**2 - (gamma / (4 * np.pi * u_inf))**2), - \
    gamma / (4 * np.pi * u_inf)

# plots the streamlines
size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u, v, density=2, linewidth=1,
               arrowsize=1.5, arrowstyle='->')
circle = plt.Circle((0, 0), radius=R, color='#CD2305', alpha=0.5)
plt.gca().add_patch(circle)
plt.scatter(x_vortex, y_vortex, color='#CD2305', s=80, marker='o')
plt.scatter([x_stagn1, x_stagn2], [y_stagn1, y_stagn2],
            color='g', s=80, marker='o')

plt.show()

# calculates the surface tangential velocity on the cylinder
theta = np.linspace(0, 2 * np.pi, 100)
u_theta = -2 * u_inf * np.sin(theta) - gamma / (2 * np.pi * R)

# computes the surface pressure coefficient
cp = 1.0 - (u_theta / u_inf)**2

# if there was no vortex
u_theta_no_vortex = -2 * u_inf * np.sin(theta)
cp_no_vortex = 1.0 - (u_theta_no_vortex / u_inf)**2

# plots the surface pressure coefficient
size = 6
plt.figure(figsize=(size, size))
plt.grid(True)
plt.xlabel(r'$\theta$', fontsize=18)
plt.ylabel(r'$C_p$', fontsize=18)
plt.xlim(theta.min(), theta.max())
plt.plot(theta, cp, color='#CD2305', linewidth=2, linestyle='-')
plt.plot(theta, cp_no_vortex, color='g', linewidth=2, linestyle='-')
plt.legend(['with vortex', 'without vortex'], loc='best', prop={'size': 16})

plt.show()
