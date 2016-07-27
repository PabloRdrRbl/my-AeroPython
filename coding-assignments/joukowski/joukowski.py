import numpy as np
import matplotlib.pyplot as plt


# Airfoil

xc, yc = -0.15, 0
R = 1.15

c = 1

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

    levels = np.linsspace(-2.0, 1.0, 100)
    contf = plt.contourf(x, y, cp,
                         levels=levels, extend='both')

    cbar = plt.colorbar(contf)
    cbar.set_label('$C_p$', fontsize=16)
    cbar.ax.invert_yaxis()

    plt.axis('equal')


if __name__ == '__main__':

    # Plot cylindrical grid

    plot_plane(X, Y, title=r'$z-$plane')
    plt.savefig('./images/z-plane.png')
    plt.close()

    # Conformal mapping

    z = X + Y * 1j

    xi = map_function(z, c)

    # Plot transformed grid

    plot_plane(np.real(xi), np.imag(xi),  title=r'$\xi-$plane')
    plt.savefig('./images/xi-plane.png')
    plt.close()

    # Doublet

    kappa = 2 * np.pi * u_inf * R**2

    u_doublet, v_doublet = get_velocity_doublet(kappa, xc, yc, X, Y)

    psi_doublet = get_stream_function_doublet(kappa, xc, yc, X, Y)

    # Freestream

    u_freestream = u_inf * np.ones((Nth, Nr), dtype=float)
    v_freestream = np.zeros((Nth, Nr), dtype=float)

    psi_freestream = u_inf * Y

    # Cylinder flow

    u_cylinder = u_doublet + u_freestream
    v_cylinder = v_doublet + v_freestream

    psi_cylinder = psi_doublet + psi_freestream

    plot_stream(X, Y, psi_cylinder, title=r'$\psi$ for $z-$plane')
    plt.savefig('./images/phi-z-plane.png')
    plt.close()

    # Airfoil stream

    plot_stream(np.real(xi), np.imag(xi), psi_cylinder,
                title=r'$\psi$ for $\xi-$plane')
    plt.savefig('./images/phi-xi-plane.png')
    plt.close()

    # Airfoil velocity

    z = X + Y * 1j

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

    print(kappa)
    print(u_airfoil[61, 0], v_airfoil[61, 0])
    print(cp_airfoil.min())
