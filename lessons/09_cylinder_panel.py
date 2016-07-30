import numpy as np
import matplotlib.pyplot as plt

form scipy import integrate

u_inf = 1

# defines the cylinder
R = 1.0
theta = np.linspace(0, 2 * np.pi, 100)

# coordinates of the cylinder
x_cylinder, y_cylinder = R * np.cos(theta), R * np.sin(theta)

size = 4
plt.figure(figsize=(size, size))

plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)

plt.plot(x_cylinder, y_cylinder, color='b', linestyle='-', linewidth=2)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)

plt.show()


class Panel:

    def __init__(self, xa, ya, xb, yb):

        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb

        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2

        self.length = np.sqrt((xb - xa)**2 + (yb - ya)**2)

        # We are doing the panels counterclock-wise
        if xb - xa <= 0.0:

            self.beta = np.acos((yb - ya) / self.length)

        elif xb - xa > 0.0:

            self.beta = np.pi + np.acos(-(yb - ya) / self.length)

        self.sigma = 0.0
        self.vt = 0.0
        self.cp = 0.0


N_panels = 10                    # number of panels desired

# defining the end-points of the panels
x_ends = R * np.cos(np.linspace(0, 2 * np.pi, N_panels + 1))
y_ends = R * np.sin(np.linspace(0, 2 * np.pi, N_panels + 1))

# defining the panels
panels = np.empty(N_panels, dtype=object)
for i in range(N_panels):
    panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])

# plotting the panels
size = 6
plt.figure(figsize=(size, size))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.plot(x_cylinder, y_cylinder, color='b', linestyle='-', linewidth=1)
plt.plot(x_ends, y_ends, color='#CD2305', linestyle='-', linewidth=2)
plt.scatter([p.xa for p in panels], [
            p.ya for p in panels], color='#CD2305', s=40)
plt.scatter([p.xc for p in panels], [p.yc for p in panels],
            color='k', s=40, zorder=3)
plt.legend(['cylinder', 'panels', 'end-points', 'center-points'],
           loc='best', prop={'size': 16})
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)

plt.show()
