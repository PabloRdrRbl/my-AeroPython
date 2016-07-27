import numpy as np
import matplotlib.pyplot as plt


N = 30  # Number of vortices

nx = 100
ny = 100

x_start, x_end = -2.5, 2.5
y_start, y_end = -0.5, 0.5

x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)

X, Y = np.meshgrid(x, y)

gamma = 5


def get_vortex_velocity(xv, yv, gamma, X, Y):

    xv = np.atleast_1d(xv)
    yv = np.atleast_1d(yv)

    n = np.size(xv)

    xv = xv.reshape(n, 1, 1)
    yv = yv.reshape(n, 1, 1)

    uu = (Y - yv) / ((X - xv)**2 + (Y - yv)**2)
    vv = (X - xv) / ((X - xv)**2 + (Y - yv)**2)

    u = (gamma / (2 * np.pi)) * np.sum(uu, axis=0)
    v = -(gamma / (2 * np.pi)) * np.sum(vv, axis=0)

    return u, v

xv = np.linspace(x_start, x_end, N)
yv = np.zeros(N)

u, v = get_vortex_velocity(xv, yv, gamma, X, Y)

size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u, v, density=2, linewidth=1,
               arrowsize=1, arrowstyle='->')
# plt.scatter(xv, yv, color='#CD2305', s=80, marker='o')

plt.show()


def get_infinite_vortex(a, gamma, X, Y):

    u = (gamma / 2 / a) * (np.sinh(2 * np.pi * Y / a) /
                           (np.cosh(2 * np.pi * Y / a) -
                            np.cos(2 * np.pi * X / a)))

    v = - (gamma / 2 / a) * (np.sin(2 * np.pi * X / a) /
                             (np.cosh(2 * np.pi * Y / a) -
                              np.cos(2 * np.pi * X / a)))

    return u, v

a = (x_end - x_start) / N

u, v = get_infinite_vortex(a, gamma, X, Y)

size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u, v, density=2, linewidth=1,
               arrowsize=1, arrowstyle='->')

plt.show()
