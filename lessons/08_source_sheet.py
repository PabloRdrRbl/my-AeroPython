import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate


N = 100                               # Number of points in each direction
x_start, x_end = -1.0, 1.0            # x-direction boundaries
y_start, y_end = -1.5, 1.5            # y-direction boundaries

x = np.linspace(x_start, x_end, N)    # computes a 1D-array for x
y = np.linspace(y_start, y_end, N)    # computes a 1D-array for y

X, Y = np.meshgrid(x, y)

u_inf = 1.0     # free-stream speed

# calculates the free-stream velocity components
u_freestream = u_inf * np.ones((N, N), dtype=float)
v_freestream = np.zeros((N, N), dtype=float)


class Source:

    def __init__(self, strength, x, y):

        self.strength = strength
        self.x, self.y = x, y

    def velocity(self, X, Y):

        self.u = (self.strength / (2 * np.pi) * (X - self.x) /
                  ((X - self.x)**2 + (Y - self.y)**2))

        self.v = (self.strength / (2 * np.pi) * (Y - self.y) /
                  ((X - self.x)**2 + (Y - self.y)**2))

    def stream_function(self, X, Y):
        self.psi = (self.strength / (2 * np.pi) *
                    np.arctan2((Y - self.y), (X - self.x)))


N_sources = 11
strength = 4.50
strength_source = strength / N_sources

x_source = np.zeros(N_sources, dtype=float)
y_source = np.linspace(-1.0, 1.0, N_sources)

sources = np.empty(N_sources, dtype=object)

for i in range(N_sources):
    sources[i] = Source(strength_source, x_source[i], y_source[i])
    sources[i].velocity(X, Y)

u = u_freestream.copy()
v = v_freestream.copy()

for source in sources:
    u += source.u
    v += source.v

size = 6
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u, v, density=2, linewidth=1,
               arrowsize=1, arrowstyle='->')
plt.scatter(x_source, y_source, color='#CD2305', s=80, marker='o')
velocity = plt.contourf(X, Y, np.sqrt(u**2 + v**2),
                        levels=np.linspace(0.0, 0.1, 10))
cbar = plt.colorbar(velocity, ticks=[0, 0.05, 0.1], orientation='horizontal')
cbar.set_label('Velocity magnitude', fontsize=16)

plt.show()


# Infinite line of sources

sigma = 2

y_min, y_max = -1.0, 1.0

u_sheet = np.empty((N, N), dtype=float)
v_sheet = np.empty((N, N), dtype=float)

for i in range(N):
    for j in range(N):

        integrand = lambda s: X[i, j] / (X[i, j]**2 + (Y[i, j] - s)**2)

        u_sheet[i, j] = (sigma / (2 * np.pi) *
                         integrate.quad(integrand, y_min, y_max)[0])

        integrand = lambda s: (Y[i, j] - s) / (X[i, j]**2 + (Y[i, j] - s)**2)

        v_sheet[i, j] = (sigma / (2 * np.pi) *
                         integrate.quad(integrand, y_min, y_max)[0])

# superposition of the source-sheet to the uniform flow
u = u_freestream + u_sheet
v = v_freestream + v_sheet

size = 8
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u, v, density=2, linewidth=1,
               arrowsize=1, arrowstyle='->')
plt.axvline(0.0, (y_min - y_start) / (y_end - y_start), (y_max - y_start) / (y_end - y_start),
            color='#CD2305', linewidth=4)
velocity = plt.contourf(X, Y, np.sqrt(u**2 + v**2),
                        levels=np.linspace(0.0, 0.1, 10))
cbar = plt.colorbar(velocity, ticks=[0, 0.05, 0.1], orientation='horizontal')
cbar.set_label('Velocity magnitude', fontsize=16)

plt.show()
