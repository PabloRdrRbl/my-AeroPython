import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate


with open('./resources/naca0012.dat') as file_name:
    x, y = np.loadtxt(file_name, dtype=float, delimiter='\t',
                      unpack=True)

val_x, val_y = 0.1, 0.2
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
x_start, x_end = (x_min - val_x * (x_max - x_min),
                  x_max + val_x * (x_max - x_min))
y_start, y_end = (y_min - val_y * (y_max - y_min),
                  y_max + val_y * (y_max - y_min))

size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))

plt.grid(True)

plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)

plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)

plt.plot(x, y, color='k', linestyle='-', linewidth=2)

plt.show()


class Panel:

    def __init__(self, xa, ya, xb, yb):

        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb

        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2

        self.length = np.sqrt((xb - xa)**2 + (yb - ya)**2)

        if xb - xa <= 0.0:
            self.beta = np.arccos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = np.pi + np.arccos(-(yb - ya) / self.length)

        if self.beta <= np.pi:
            self.loc = 'upper'
        else:
            self.loc = 'lower'

        self.sigma = 0.0
        self.vt = 0.0
        self.cp = 0.0


def define_panels(x, y, N=40):

    R = (x.max() - x.min()) / 2

    x_center = (x.max() + x.min()) / 2

    x_circle = x_center + R * np.cos(np.linspace(0, 2 * np.pi, N + 1))

    x_ends = np.copy(x_circle)

    y_ends = np.empty_like(x_ends)

    x, y = np.append(x, x[0]), np.append(y, y[0])

    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1

        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b

    y_ends[N] = y_ends[0]

    panels = np.empty(N, dtype=object)

    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])

    return panels


N = 40                            # number of panels
panels = define_panels(x, y, N)   # discretizes of the geometry into panels

# plots the geometry and the panels
val_x, val_y = 0.1, 0.2
x_min, x_max = min(panel.xa for panel in panels), max(
    panel.xa for panel in panels)
y_min, y_max = min(panel.ya for panel in panels), max(
    panel.ya for panel in panels)
x_start, x_end = x_min - val_x * \
    (x_max - x_min), x_max + val_x * (x_max - x_min)
y_start, y_end = y_min - val_y * \
    (y_max - y_min), y_max + val_y * (y_max - y_min)

size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.plot(x, y, color='k', linestyle='-', linewidth=2)
plt.plot(np.append([panel.xa for panel in panels], panels[0].xa),
         np.append([panel.ya for panel in panels], panels[0].ya),
         linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')

plt.show()


class Freestream:

    def __init__(self, u_inf=1.0, alpha=0.0):

        self.u_inf = u_inf
        self.alpha = alpha * np.pi / 180

u_inf = 1.0
alpha = 0.0
freestream = Freestream(u_inf, alpha)


def integral(x, y, panel, dxdz, dydz):

    def func(s):

        return (((x - (panel.xa - np.sin(panel.beta) * s)) * dxdz +
                 (y - (panel.ya + np.cos(panel.beta) * s)) * dydz) /
                ((x - (panel.xa - np.sin(panel.beta) * s))**2 +
                 (y - (panel.ya + np.cos(panel.beta) * s))**2))

    return integrate.quad(lambda s: func(s), 0.0, panel.length)[0]


def build_matrix(panels):

    N = len(panels)
    A = np.empty((N, N), dtype=float)
    np.fill_diagonal(A, 0.5)

    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):

            if i != j:
                A[i, j] = (0.5 / np.pi *
                           integral(p_i.xc, p_i.yc, p_j,
                                    np.cos(p_i.beta), np.sin(p_i.beta)))

    return A


def build_rhs(panels, freestream):

    b = np.empty(len(panels), dtype=float)

    for i, panel in enumerate(panels):

        b[i] = - freestream.u_inf * np.cos(freestream.alpha - panel.beta)

    return b

A = build_matrix(panels)

b = build_rhs(panels, freestream)

sigma = np.linalg.solve(A, b)

for i, panel in enumerate(panels):
    panel.sigma = sigma[i]


def get_tangencial_velocity(panels, freestream):

    N = len(panels)
    A = np.empty((N, N), dtype=float)
    np.fill_diagonal(A, 0.0)

    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                A[i, j] = (0.5 / np.pi *
                           integral(p_i.xc, p_i.yc, p_j,
                                    - np.sin(p_i.beta),
                                    np.cos(p_i.beta)))

    b = freestream.u_inf * np.sin([freestream.alpha - panel.beta for
                                   panel in panels])

    sigma = np.array([panel.sigma for panel in panels])

    vt = np.dot(A, sigma) + b

    for i, panel in enumerate(panels):
        panel.vt = vt[i]

get_tangencial_velocity(panels, freestream)


def get_pressure_coefficient(panels, freestream):

    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2


get_pressure_coefficient(panels, freestream)

voverVsquared = np.array([0, 0.64, 1.01, 1.241, 1.378, 1.402, 1.411,
                          1.411, 1.399, 1.378, 1.35, 1.288, 1.228,
                          1.166, 1.109, 1.044, 0.956, 0.906, 0])

xtheo = np.array([0, 0.5, 1.25, 2.5, 5.0, 7.5, 10, 15, 20, 25, 30, 40,
                  50, 60, 70, 80, 90, 95, 100])
xtheo = xtheo / 100

val_x, val_y = 0.1, 0.2
x_min, x_max = min(panel.xa for panel in panels), max(
    panel.xa for panel in panels)
cp_min, cp_max = min(panel.cp for panel in panels), max(
    panel.cp for panel in panels)
x_start, x_end = x_min - val_x * \
    (x_max - x_min), x_max + val_x * (x_max - x_min)
y_start, y_end = cp_min - val_y * \
    (cp_max - cp_min), cp_max + val_y * (cp_max - cp_min)

plt.figure(figsize=(10, 6))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('$C_p$', fontsize=16)
plt.plot([panel.xc for panel in panels if panel.loc == 'upper'],
         [panel.cp for panel in panels if panel.loc == 'upper'],
         color='r', linewidth=1, marker='x', markersize=8)
plt.plot([panel.xc for panel in panels if panel.loc == 'lower'],
         [panel.cp for panel in panels if panel.loc == 'lower'],
         color='b', linewidth=0, marker='d', markersize=6)
plt.plot(xtheo, 1 - voverVsquared, color='k', linestyle='--', linewidth=2)
plt.legend(['upper', 'lower'], loc='best', prop={'size': 14})
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.gca().invert_yaxis()
plt.title('Number of panels : %d' % N)

plt.show()

accuracy = sum([panel.sigma * panel.length for panel in panels])
print('--> sum of source/sink strengths:', accuracy)


def get_velocity_field(panels, freestream, X, Y):

    Nx, Ny = X.shape
    u, v = np.empty((Nx, Ny), dtype=float), np.empty((Nx, Ny), dtype=float)

    for i in range(Nx):
        for j in range(Ny):

            u[i, j] = (freestream.u_inf * np.cos(freestream.alpha) +
                       0.5 / np.pi *
                       sum([p.sigma * integral(X[i, j], Y[i, j], p, 1, 0)
                            for p in panels]))

            v[i, j] = (freestream.u_inf * np.sin(freestream.alpha) +
                       0.5 / np.pi *
                       sum([p.sigma * integral(X[i, j], Y[i, j], p, 0, 1)
                            for p in panels]))

    return u, v

# defines a mesh grid
Nx, Ny = 20, 20                  # number of points in the x and y directions

val_x, val_y = 1.0, 2.0

x_min, x_max = min(panel.xa for panel in panels), max(
    panel.xa for panel in panels)

y_min, y_max = min(panel.ya for panel in panels), max(
    panel.ya for panel in panels)

x_start, x_end = (x_min - val_x *
                  (x_max - x_min), x_max + val_x * (x_max - x_min))

y_start, y_end = (y_min - val_y *
                  (y_max - y_min), y_max + val_y * (y_max - y_min))

X, Y = np.meshgrid(np.linspace(x_start, x_end, Nx),
                   np.linspace(y_start, y_end, Ny))

# computes the velicity field on the mesh grid
u, v = get_velocity_field(panels, freestream, X, Y)

# plots the velocity field
size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.streamplot(X, Y, u, v, density=1, linewidth=1,
               arrowsize=1, arrowstyle='->')
plt.fill([panel.xc for panel in panels],
         [panel.yc for panel in panels],
         color='k', linestyle='solid', linewidth=2, zorder=2)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.title('Streamlines around a NACA 0012 airfoil, AoA = %.1f' % alpha)

plt.show()

# computes the pressure field
cp = 1.0 - (u**2 + v**2) / freestream.u_inf**2

# plots the pressure field
size = 12
plt.figure(
    figsize=(1.1 * size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
contf = plt.contourf(
    X, Y, cp, levels=np.linspace(-2.0, 1.0, 100), extend='both')
cbar = plt.colorbar(contf)
cbar.set_label('$C_p$', fontsize=16)
cbar.set_ticks([-2.0, -1.0, 0.0, 1.0])
plt.fill([panel.xc for panel in panels],
         [panel.yc for panel in panels],
         color='k', linestyle='solid', linewidth=2, zorder=2)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.title('Contour of pressure field')

plt.show()
