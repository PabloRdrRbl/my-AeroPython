import numpy as np
import matplotlib.pyplot as plt


N = 50

x_start, x_end = -2.0, 2.0
y_start, y_end = -1.0, 1.0

x = np.linspace(x_start, x_end, N)
y = np.linspace(y_start, y_end, N)

X, Y = np.meshgrid(x, y)

gamma = 5.0
x_vortex, y_vortex = 0.0, 0.0


def get_velocity_vortex(strength, xv, yv, X, Y):

    u = + strength / (2 * np.pi) * (Y - yv) / ((X - xv)**2 + (Y - yv)**2)
    v = - strength / (2 * np.pi) * (X - xv) / ((X - xv)**2 + (Y - yv)**2)

    return u, v


def get_stream_function_vortex(strength, xv, yv, X, Y):

    psi = strength / (4 * np.pi) * np.log((X - xv)**2 + (Y - yv)**2)

    return psi


u_vortex, v_vortex = get_velocity_vortex(gamma, x_vortex, y_vortex, X, Y)

psi_vortex = get_stream_function_vortex(gamma, x_vortex, y_vortex, X, Y)

size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u_vortex, v_vortex, density=2,
               linewidth=1, arrowsize=1, arrowstyle='->')
plt.scatter(x_vortex, y_vortex, color='#CD2305', s=80, marker='o')

plt.show()

strength_sink = -1.0
x_sink, y_sink = 0.0, 0.0


def get_velocity_sink(strength, xs, ys, X, Y):

    u = strength / (2 * np.pi) * (X - xs) / ((X - xs)**2 + (Y - ys)**2)
    v = strength / (2 * np.pi) * (Y - ys) / ((X - xs)**2 + (Y - ys)**2)

    return u, v


def get_stream_function_sink(strength, xs, ys, X, Y):

    psi = strength / (2 * np.pi) * np.arctan2((Y - ys), (X - xs))

    return psi

u_sink, v_sink = get_velocity_sink(strength_sink, x_sink, y_sink, X, Y)

psi_sink = get_stream_function_sink(strength_sink, x_sink, y_sink, X, Y)

u = u_vortex + u_sink
v = v_vortex + v_sink
psi = psi_vortex + psi_sink

size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u, v, density=2, linewidth=1,
               arrowsize=1, arrowstyle='->')
plt.scatter(x_vortex, y_vortex, color='#CD2305', s=80, marker='o')

plt.show()
