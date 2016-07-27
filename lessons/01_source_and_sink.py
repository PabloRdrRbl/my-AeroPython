import numpy as np
import matplotlib.pyplot as plt


# Grid parameters
N = 50
x_start, x_end = -2.0, 2.0
y_start, y_end = -1.0, 1.0

x = np.linspace(x_start, x_end, N)
y = np.linspace(y_start, y_end, N)

X, Y = np.meshgrid(x, y)

size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.scatter(X, Y, s=10, color='#CD2305', marker='o', linewidth=0)

plt.show()

# Source flow

strength_source = 5.0
x_source, y_source = -1.0, 0.0

# Velocity of the field
u_source = (strength_source / (2 * np.pi) * (X - x_source) /
            ((X - x_source)**2 + (Y - y_source)**2))

v_source = (strength_source / (2 * np.pi) * (Y - y_source) /
            ((X - x_source)**2 + (Y - y_source)**2))

# Plotting stream lines
size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u_source, v_source, density=2,
               linewidth=1, arrowsize=2, arrowstyle='->')
plt.scatter(x_source, y_source, color='#CD2305', s=80, marker='o', linewidth=0)

plt.show()

# Sink flow
strength_sink = -5.0
x_sink, y_sink = 1.0, 0.0

# Velocity of the field
u_sink = (strength_sink / (2 * np.pi) * (X - x_sink) /
          ((X - x_sink)**2 + (Y - y_sink)**2))

v_sink = (strength_sink / (2 * np.pi) * (Y - y_sink) /
          ((X - x_sink)**2 + (Y - y_sink)**2))

# Plotting stream lines
size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u_sink, v_sink, density=2,
               linewidth=1, arrowsize=2, arrowstyle='->')
plt.scatter(x_sink, y_sink, color='#CD2305', s=80, marker='o', linewidth=0)

plt.show()
plt.close()
# Source-sink pair
# Somputes the velocity of the pair source/sink by superposition
u_pair = u_source + u_sink
v_pair = v_source + v_sink

V = np.sqrt(u_pair**2 + v_pair**2)

# Plots the streamlines of the pair source/sink
size = 10
plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
# plt.streamplot(X, Y, u_pair, v_pair, density=2.0,
#               linewidth=1, arrowsize=2, arrowstyle='->')
plt.contourf(X, Y, V, 50, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.scatter([x_source, x_sink], [y_source, y_sink],
            color='#CD2305', s=80, marker='o', linewidth=0)

plt.show()
