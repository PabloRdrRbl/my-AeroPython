import numpy as np
import matplotlib.pyplot as plt


# Airfoil

c = 1
AOA = 20

xc, yc = -0.15, 0
R = 1.15

# Other

# xc, yc = -0.2, 0.3
# R = c * np.sqrt((1 + abs(xc))**2 + yc**2)

# Cylindrical grid

Nr = 100
Nth = 145

radial = np.linspace(R, 5, Nr)
axial = np.linspace(0, 2 * np.pi, Nth)

X = radial * np.cos(axial.reshape((Nth, 1))) + xc
Y = radial * np.sin(axial.reshape((Nth, 1))) + yc

# Flow

u_inf = 1

# Functions


def axis_rotation(x, y, xc, yc, AOA):
    """
    Originaly the cylinder is at (xc, yc), the axis
    transformation moves it to the center
    """

    # To radians
    AOA = AOA * np.pi / 180

    xp = (x - xc) * np.cos(AOA) + (y - yc) * np.sin(AOA)
    yp = -(x - xc) * np.sin(AOA) + (y - yc) * np.cos(AOA)

    return xp, yp


def map_function(z, c):

    return z + (c**2 / z)


def get_velocity_doublet(kappa, xc, yc, X, Y):

    u = - kappa / (2 * np.pi) * (((X - xc)**2 - (Y - yc)**2) /
                                 ((X - xc)**2 + (Y - yc)**2)**2)

    v = (- kappa / np.pi * (X - xc) * (Y - yc) /
         ((X - xc)**2 + (Y - yc)**2)**2)

    return u, v


def get_stream_function_doublet(kappa, xc, yc, X, Y):

    psi = - kappa / (2 * np.pi) * (Y - yc) / ((X - xc)**2 + (Y - yc)**2)

    return psi


def get_velocity_vortex(gamma, xc, yc, X, Y):

    u = + gamma / (2 * np.pi) * (Y - yc) / ((X - xc)**2 + (Y - yc)**2)
    v = - gamma / (2 * np.pi) * (X - xc) / ((X - xc)**2 + (Y - yc)**2)
    return u, v


def get_stream_function_vortex(gamma, xc, yc, X, Y):

    psi = gamma / (4 * np.pi) * np.log((X - xc)**2 + (Y - yc)**2)

    return psi


def plot_plane(x, y, title=None):

    plt.figure()

    plt.xlim(-5, 5)
    plt.ylim(-6, 6)

    plt.title(title)

    plt.scatter(x, y, s=0.5, c='k')

    plt.axis('equal')


def plot_stream(x, y, psi, title=None):

    plt.figure()

    plt.xlim(-5, 5)
    plt.ylim(-6, 6)

    plt.title(title)

    levels = np.linspace(psi.min(), psi.max(), 45)

    plt.contour(x, y, psi, colors='k',
                levels=levels, linewidths=1, linestyles='solid')

    plt.axis('equal')


def plot_velocity(x, y, u, v, title=None):

    plt.figure()

    plt.xlim(-5, 5)
    plt.ylim(-6, 6)

    plt.title(title)

    plt.quiver(x, y, u, v)

    plt.axis('equal')


def plot_cp(x, y, cp, title=None):

    plt.figure()

    plt.xlim(-5, 5)
    plt.ylim(-6, 6)

    plt.title(title)

    # cmap = plt.cm.coolwarm

    levels = np.linspace(-1.0, 1.0, 300)
    contf = plt.contourf(x, y, cp,
                         levels=levels, extend='both')  # , cmap=cmap)

    cbar = plt.colorbar(contf)
    cbar.set_label('$C_p$', fontsize=16)
    cbar.ax.invert_yaxis()
    cbar.set_ticks([-2.0, -1.0, 0.0, 1.0])

    p = plt.Polygon(list(zip(x[:, 0], y[:, 0])), color="#cccccc", zorder=10)
    plt.gca().add_patch(p)

    plt.gca().set_aspect(1)

    plt.axis('equal')


if __name__ == '__main__':

    # Plot cylindrical grid

    Xr, Yr = axis_rotation(X, Y, xc, yc, AOA)

    plot_plane(Xr, Yr, title=r'$z-$plane')
    plt.savefig('./images/z-plane.png')
    plt.close()

    # Conformal mapping

    z = X + Y * 1j  # xi is done with z, not z'

    xi = map_function(z, c)

    # Plot transformed grid

    plot_plane(np.real(xi), np.imag(xi),  title=r'$\xi-$plane')
    plt.savefig('./images/xi-plane.png')
    plt.close()

    # Doublet
    # Doublet must be always placed at (0, 0),
    # and computed in the z-plane, not z'.
    # The same applies for the velocity

    kappa = 2 * np.pi * u_inf * R**2

    u_doublet, v_doublet = get_velocity_doublet(kappa, 0, 0, Xr, Yr)

    psi_doublet = get_stream_function_doublet(kappa, 0, 0, Xr, Yr)

    # Freestream

    u_freestream = u_inf * np.ones((Nth, Nr), dtype=float)
    v_freestream = np.zeros((Nth, Nr), dtype=float)

    psi_freestream = u_inf * Yr  # At z, not z'

    # Vortex

    # gamma calculation taken from Anderson's book

    theta = - AOA

    gamma = - 4 * np.pi * u_inf * R * np.sin(theta * np.pi / 180)

    u_vortex, v_vortex = get_velocity_vortex(gamma, 0, 0, Xr, Yr)

    psi_vortex = get_stream_function_vortex(gamma, 0, 0, Xr, Yr)

    # Cylinder flow

    u_cylinder = u_doublet + u_freestream + u_vortex
    v_cylinder = v_doublet + v_freestream + v_vortex

    psi_cylinder = psi_doublet + psi_freestream + psi_vortex

    plot_stream(X, Y, psi_cylinder, title=r'$\psi$ for $z-$plane')
    plt.savefig('./images/phi-z-plane.png')
    plt.close()

    # Airfoil stream

    plot_stream(np.real(xi), np.imag(xi), psi_cylinder,
                title=r'$\psi$ for $\xi-$plane')
    plt.savefig('./images/phi-xi-plane.png')
    plt.close()

    # Airfoil velocity

    # Velocity vectors need to be rotated back from the z' plane
    # to the z plane
    u_cylinder, v_cylinder = axis_rotation(
        u_cylinder, v_cylinder, 0, 0, AOA)

    z = X + Y * 1j  # xi is done using z not z'

    xi = map_function(z, c)

    W = (u_cylinder - v_cylinder * 1j) / (1 - (c / z)**2)

    u_airfoil = np.real(W)
    v_airfoil = - np.imag(W)

    plot_velocity(X, Y, u_cylinder, v_cylinder,
                  title=r'Velocity for $z-$plane')
    plt.savefig('./images/velocity-z-plane.png')
    plt.close()

    plot_velocity(np.real(xi), np.imag(xi), u_airfoil, v_airfoil,
                  title=r'Velocity for $\xi-$plane')
    plt.savefig('./images/velocity-xi-plane.png')
    plt.close()

    # Pressure coefficient

    cp_cylinder = 1 - (u_cylinder**2 + v_cylinder**2) / u_inf**2
    cp_airfoil = 1 - (u_airfoil**2 + v_airfoil**2) / u_inf**2

    plot_cp(X, Y, cp_cylinder, title=r'$C_p$ for $z-$plane')
    plt.savefig('./images/cp-z-plane.png')
    plt.close()

    plot_cp(np.real(xi), np.imag(xi), cp_airfoil,
            title=r'$C_p$ for $\xi-$plane')
    plt.savefig('./images/cp-xi-plane.png')
    plt.close()

    # Questions

    print(gamma)
    print(np.where(cp_cylinder[:, 0] == 1))
    print(u_airfoil[49, 0], v_airfoil[49, 0])
    print(cp_airfoil[110, 0])

#    plt.plot(axial[1:-1], (u_airfoil[1:-1, 0]), c='r', lw=2)
#    plt.plot(axial[1:-1], (v_airfoil[1:-1, 0]), c='g', lw=2)
#    plt.plot(axial[3:-3], cp_airfoil[3:-3, 0], c='b', lw=2)
#    plt.show()
