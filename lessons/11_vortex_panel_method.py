import numpy as np
from scipy import integrate, linalg

import matplotlib.pyplot as plt


x, y = np.loadtxt('./resources/naca0012.dat', dtype=float, unpack=True)

val_x, val_y = 0.1, 0.2

xp_min, xp_max = x.min(), x.max()
yp_min, yp_max = y.min(), y.max()

xp_start, xp_end = (xp_min - val_x *
                    (xp_max - xp_min), xp_max + val_x * (xp_max - xp_min))
yp_start, yp_end = (yp_min - val_y *
                    (yp_max - yp_min), yp_max + val_y * (yp_max - yp_min))

size = 10
plt.figure(figsize=(size, (yp_end - yp_start) / (xp_end - xp_start) * size))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(xp_start, xp_end)
plt.ylim(yp_start, yp_end)
plt.plot(x, y, color='k', linestyle='-', linewidth=2)


class Panel:

    def __init__(self, xa, ya, xb, yb):

        self.xa, self.ya = xa, ya  # panel starting-point
        self.xb, self.yb = xb, yb  # panel ending-point

        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2         # panel center
        self.length = np.sqrt((xb - xa)**2 + (yb - ya)**2)  # panel length

        # orientation of panel (angle between x-axis and panel's normal)
        if xb - xa <= 0.0:
            self.beta = np.arccos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = np.pi + np.arccos(-(yb - ya) / self.length)

        # panel location
        if self.beta <= np.pi:
            self.loc = 'upper'  # upper surface
        else:
            self.loc = 'lower'  # lower surface

        self.sigma = 0.0  # source strength
        self.vt = 0.0    # tangential velocity
        self.cp = 0.0    # pressure coefficien


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
            if ((x[I] <= x_ends[i] <= x[I + 1]) or
                    (x[I + 1] <= x_ends[i] <= x[I])):
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

panels = define_panels(x, y, N=40)

val_x, val_y = 0.1, 0.2

xp_min = min(panel.xa for panel in panels)
xp_max = max(panel.xa for panel in panels)
yp_min = min(panel.ya for panel in panels)
yp_max = max(panel.ya for panel in panels)

xp_start, xp_end = (xp_min - val_x *
                    (xp_max - xp_min), xp_max + val_x * (xp_max - xp_min))

yp_start, yp_end = (yp_min - val_y *
                    (yp_max - yp_min), yp_max + val_y * (yp_max - yp_min))


size = 10
plt.figure(figsize=(size, (yp_end - yp_start) / (xp_end - xp_start) * size))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(xp_start, xp_end)
plt.ylim(yp_start, yp_end)
plt.plot(x, y, color='k', linestyle='-', linewidth=2)
plt.plot(np.append([panel.xa for panel in panels], panels[0].xa),
         np.append([panel.ya for panel in panels], panels[0].ya),
         linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')

plt.show()


class Freestream:

    def __init__(self, u_inf=1.0, alpha=0.0):

        self.u_inf = u_inf
        self.alpha = alpha * np.pi / 180

freestream = Freestream(u_inf=1.0, alpha=4.0)
