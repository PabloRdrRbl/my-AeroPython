import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm


def velocity(sigma, xs, ys, X, Y):
    """
    Generalizing the one source case:

        xs, ys --> (1,)
        X, Y --> (nx, ny)
        sigma --> (1,)

    To the several sources one:

        xs, ys --> (ns, 1, 1)
        X, Y --> (ns, nx, ny)
        sigma --> (ns, 1, 1)

    """

    sigma = np.atleast_1d(sigma)
    xs = np.atleast_1d(xs)
    ys = np.atleast_1d(ys)

    # Handling one or n sourcefs arrays' dimensions

    nx = np.size(xs)  # Number of sources
    ny = np.size(ys)  # Number of sources

    assert (nx == ny), "xs and ys must have the same length"

    xs = xs.reshape((nx, 1, 1))
    ys = ys.reshape((ny, 1, 1))

    assert (xs.shape == (nx, 1, 1))

    # Creates 3D reapeating the 2D grid. So we
    # get the velocity field for each source.
    # If the broadcasting works as I think it
    # may be unecessary
    ns = np.size(sigma)  # Number of sources

    assert (nx == ns), "sigma and xs must have the same length"

    XX = np.repeat(X[None, ...], ns, axis=0)
    YY = np.repeat(Y[None, ...], ns, axis=0)

    sigma = sigma.reshape((ns, 1, 1))

    # Using uu given that it may have several
    # velocity fields which will be superposed
    # later
    uu = (sigma / (2 * np.pi)) * ((XX - xs) /
                                  ((XX - xs)**2 + (YY - ys)**2))

    vv = (sigma / (2 * np.pi)) * ((YY - ys) /
                                  ((XX - xs)**2 + (YY - ys)**2))

    assert (uu.shape == XX.shape)
    assert (vv.shape == XX.shape)

    # Superposing the velocity solutions
    # for each source

    u = np.sum(uu, axis=0, dtype=float)
    v = np.sum(vv, axis=0, dtype=float)

    return u, v


def stream_function(sigma, xs, ys, X, Y):
    """
    Generalizing the one source case:

        xs, ys --> (1,)
        X, Y --> (nx, ny)
        sigma --> (1,)

    To the several sources one:

        xs, ys --> (ns, 1, 1)
        X, Y --> (ns, nx, ny)
        sigma --> (ns, 1, 1)

    """

    sigma = np.atleast_1d(sigma)
    xs = np.atleast_1d(xs)
    ys = np.atleast_1d(ys)

    # Handling one or n sourcefs arrays' dimensions

    nx = np.size(xs)  # Number of sources
    ny = np.size(ys)  # Number of sources

    assert (nx == ny), "xs and ys must have the same length"

    xs = xs.reshape((nx, 1, 1))
    ys = ys.reshape((ny, 1, 1))

    # Creates 3D reapeating the 2D grid. So we
    # get the velocity field for each source.
    # If the broadcasting works as I think it
    # may be unecessary
    ns = np.size(sigma)  # Number of sources

    assert (nx == ns), "sigma and xs must have the same length"

    XX = np.repeat(X[None, ...], ns, axis=0)
    YY = np.repeat(Y[None, ...], ns, axis=0)

    sigma = sigma.reshape((ns, 1, 1))

    psi = (sigma / (2 * np.pi)) * np.arctan2((YY - ys), (XX - xs))

    assert (psi.shape == XX.shape)

    # Superposing the stream function solutions
    # for each source

    psi = np.sum(psi, axis=0, dtype=float)

    return psi


def plot_velocity(u, v, psi, x_start, x_end, y_start, y_end):

    size = 10
    plt.figure(figsize=(size, (y_end - y_start) / (x_end - x_start) * size))

    plt.grid(True)

    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)

    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_end)

    plt.streamplot(X, Y, u, v, density=2, linewidth=1,
                   arrowsize=1, arrowstyle='->')

    plt.contour(X, Y, psi,
                colors='#CD2305', linewidths=2, linestyles='solid')

    plt.scatter(xs, ys, color='#CD2305', s=10, marker='o')


def plot_cp(cp, x_start, x_end, y_start, y_end):

    size = 10
    plt.figure(figsize=(1.1 * size,
                        (y_end - y_start) / (x_end - x_start) * size))

    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)

    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_end)

    levels = np.linspace(np.min(cp), np.max(cp), 100)

    contf = plt.contourf(X, Y, cp, levels)

    cbar = plt.colorbar(contf)
    cbar.set_label('$C_p$', fontsize=16)
    cbar.ax.invert_yaxis()

    plt.scatter(xs, ys, color='#CD2305', s=10, marker='o')


if __name__ == '__main__':
    if True:
        # Grid parameters

        nx = 51
        ny = 51
        x_start, x_end = -1.0, 2.0
        y_start, y_end = -0.5, 0.5

        x = np.linspace(x_start, x_end, nx)
        y = np.linspace(y_start, y_end, ny)

        X, Y = np.meshgrid(x, y)

        # Data loading

        # Resaping in the funcition
        xs = np.loadtxt('./resources/NACA0012_x.txt')
        ys = np.loadtxt('./resources/NACA0012_y.txt')
        sigma = np.loadtxt('./resources/NACA0012_sigma.txt')

    else:
        nx = 200
        ny = 200

        x_start, x_end = -4.0, 4.0
        y_start, y_end = -2.0, 2.0

        x = np.linspace(x_start, x_end, nx)
        y = np.linspace(y_start, y_end, ny)

        X, Y = np.meshgrid(x, y)

        xs = 1
        ys = 0

        sigma = -5

    # Velocity field

    u_inf = 1  # (m/s)

    u, v = velocity(sigma, xs, ys, X, Y)

    u_freestream = u_inf * np.ones((nx, ny), dtype=float)
    v_freestream = np.zeros((nx, ny), dtype=float)

    u = u + u_freestream
    v = v + v_freestream

    # Stream function

    psi = stream_function(sigma, xs, ys, X, Y)

    psi_freestream = u_inf * Y

    psi = psi + psi_freestream

    # Velocity plot

    plot_velocity(u, v, psi, x_start, x_end, y_start, y_end)
    plt.savefig('./images/velocity.png')
    plt.close()

    # Pressure coeficient

    cp = 1.0 - (u**2 + v**2) / u_inf**2

    assert (cp.shape == X.shape), "Cp shape is not correct"

    # Cp plot

    plot_cp(cp, x_start, x_end, y_start, y_end)
    plt.savefig('./images/cp.png')
    plt.close()

    print(np.where(cp == np.max(cp)))
    print(np.max(cp))
