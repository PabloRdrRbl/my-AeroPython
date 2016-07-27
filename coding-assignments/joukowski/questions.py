import numpy as np
import matplotlib.pyplot as plt


def map_function(z, c):

    return z + (c**2 / z)


if __name__ == '__main__':

    # Q1
    c = 1
    theta = np.linspace(0, 2 * np.pi, 100)
    R = 1.5

    x = R * np.cos(theta)
    y = R * np.sin(theta)

    plt.plot(x, y, color='k', lw=2)
    plt.axis('equal')
    plt.show()

    z = x + y * 1j

    chi = map_function(z, c)

    plt.plot(np.real(chi), np.imag(chi), color='k', lw=2)
    plt.axis('equal')
    plt.show()

    # Q2
    c = 1
    theta = np.linspace(0, 2 * np.pi, 100)
    R = 1.2

    x = R * np.cos(theta) - (c - R)
    y = R * np.sin(theta)

    plt.plot(x, y, color='k', lw=2)
    plt.axis('equal')
    plt.show()

    z = x + y * 1j

    chi = map_function(z, c)

    plt.plot(np.real(chi), np.imag(chi), color='k', lw=2)
    plt.axis('equal')
    plt.show()

    # Q3
    c = 1

    dx, dy = 0.1, 0.1
    xc, yc = -dx, dy

    theta = np.linspace(0, 2 * np.pi, 100)
    R = ((c - xc)**2 + yc**2)**0.5

    x = R * np.cos(theta) - xc
    y = R * np.sin(theta) - yc

    plt.plot(x, y, color='k', lw=2)
    plt.axis('equal')
    plt.show()

    z = x + y * 1j

    chi = map_function(z, c)

    plt.plot(np.real(chi), np.imag(chi), color='k', lw=2)
    plt.axis('equal')
    plt.show()
