import numpy as np
import matplotlib.pyplot as plt

N = 50
x_start, x_end = -2.0, 2.0
y_start, y_end = -1.0, 1.0
x = np.linspace(x_start, x_end, N)
y = np.linspace(y_start, y_end, N)
X, Y = np.meshgrid(x, y)

kappa = 1.0
x_doublet, y_doublet = 0.0, 0.0


def get_velocity_doublet(strength, xd, yd, X, Y):
    u = (- strength / (2 * np.pi) *
         ((X - xd)**2 - (Y - yd) ** 2) / ((X - xd)**2 + (Y - yd)**2)**2)
    v = (- strength / (2 * np.pi) * 2 * (X - xd) * (Y - yd) /
         ((X - xd)**2 + (Y - yd)**2)**2)

    return u, v


def get_stream_function_doublet(strength, xd, yd, X, Y):
    psi = - strength / (2 * np.pi) * (Y - yd) / ((X - xd)**2 + (Y - yd)**2)

    return psi

u_doublet, v_doublet = get_velocity_doublet(kappa, x_doublet, y_doublet, X, Y)

psi_doublet = get_stream_function_doublet(kappa, x_doublet, y_doublet, X, Y)

size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u_doublet, v_doublet,
               density=2, linewidth=1, arrowsize=1, arrowstyle='->')
plt.scatter(x_doublet, y_doublet, color='#CD2305', s=80, marker='o')

plt.show()

# Uniform flow past a doublet

u_inf = 1.0

u_freestream = u_inf * np.ones((N, N), dtype=float)
v_freestream = np.zeros((N, N), dtype=float)

psi_freestream = u_inf * Y

# Superposition of the doublet on the freestream flow
u = u_freestream + u_doublet
v = v_freestream + v_doublet
psi = psi_freestream + psi_doublet

# Plots the streamlines
size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u, v, density=2, linewidth=1,
               arrowsize=1, arrowstyle='->')
plt.contour(X, Y, psi, levels=[
    0.], colors='#CD2305', linewidths=2, linestyles='solid')
plt.scatter(x_doublet, y_doublet, color='#CD2305', s=80, marker='o')

# Calculates the stagnation points
x_stagn1, y_stagn1 = +np.sqrt(kappa / (2 * np.pi * u_inf)), 0
x_stagn2, y_stagn2 = -np.sqrt(kappa / (2 * np.pi * u_inf)), 0

# Adds the stagnation points to the figure
plt.scatter([x_stagn1, x_stagn2], [y_stagn1, y_stagn2],
            color='g', s=80, marker='o')

plt.show()

# Computes the pressure coefficient field
cp = 1.0 - (u**2 + v**2) / u_inf**2

# Plots the pressure coefficient field
size = 10
plt.figure(
    figsize=(1.1 * size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
contf = plt.contourf(
    X, Y, cp, levels=np.linspace(-2.0, 1.0, 100), extend='both')
cbar = plt.colorbar(contf)
cbar.set_label('$C_p$', fontsize=16)
cbar.set_ticks([-2.0, -1.0, 0.0, 1.0])
cbar.ax.invert_yaxis()
plt.scatter(x_doublet, y_doublet, color='#CD2305', s=80, marker='o')
plt.contour(X, Y, psi, levels=[
    0.], colors='#CD2305', linewidths=2, linestyles='solid')
plt.scatter([x_stagn1, x_stagn2], [y_stagn1, y_stagn2],
            color='g', s=80, marker='o')

plt.show()

# Cp-angle
theta = np.linspace(0, 2 * np.pi, 100)
cp = 1 - 4 * (np.sin(theta))**2

plt.figure()
plt.plot(theta, cp, color='k', lw=3)

plt.show()
