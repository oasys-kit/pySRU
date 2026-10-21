# coding: utf-8
#
# Polarization of the far field of a planar undulator (RADIATION_METHOD_APPROX_FARFIELD), compared with an
# independent time-domain calculation: the Lienard-Wiechert far field
#     E(t) ~ n x ((n - beta) x beta_dot) / (1 - n.beta)^3,   at the observer time t_obs = t - n.r/c,
# Fourier transformed with exp(+i w t_obs) (the same convention as pySRU, i.e., the real field is Re[E(w) exp(-i w t)]).
#
# The test checks the boundary terms of the integration by parts: they are vectors (n x (n x beta) at the ends of the
# trajectory). When only their first component was used for the three field components, E_y was wrong where the
# field is weak, breaking the mirror symmetry y -> -y (trajectory in the plane y = 0) and the agreement with the
# time-domain calculation (errors up to ~0.5 in the normalized Stokes parameters, between the rings of the emission).
#
__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"
__date__ = "21/10/2026"

import unittest
import numpy as np
import scipy.constants as codata
from pySRU.Simulation import create_simulation
from pySRU.ElectronBeam import ElectronBeam
from pySRU.MagneticStructureUndulatorPlane import MagneticStructureUndulatorPlane as Undulator
from pySRU.TrajectoryFactory import TRAJECTORY_METHOD_ANALYTIC
from pySRU.RadiationFactory import RADIATION_METHOD_APPROX_FARFIELD

E_GEV, K, LAMBDA_U, N_PERIODS = 6.0, 1.68, 0.018, 111
GAMMA = E_GEV * 1e9 / 0.511e6
# photon energy slightly below the first harmonic
PHOTON_ENERGY = 0.995 * 2 * GAMMA ** 2 * codata.h * codata.c / codata.e / LAMBDA_U / (1 + K ** 2 / 2)
DISTANCE = 100.0
# angles (rad) where the field is weak (between rings) and strong
ANGLES = [(2.3e-4, 1.0e-4), (2.5e-4, 5.2e-5), (9.5e-5, 9.2e-5), (1.5e-4, 1.0e-4), (4e-5, 3e-5)]


def stokes(ex, ey):
    """Normalized Stokes parameters (P1, P2, P3) of the field components (ex, ey)."""
    s0 = abs(ex) ** 2 + abs(ey) ** 2
    return np.array([abs(ex) ** 2 - abs(ey) ** 2, 2 * np.real(ex * np.conj(ey)),
                     2 * np.imag(np.conj(ex) * ey)]) / s0


def simulation(angles, points_per_period=200):
    x = np.array([a[0] for a in angles]) * DISTANCE
    y = np.array([a[1] for a in angles]) * DISTANCE
    return create_simulation(magnetic_structure=Undulator(K=K, period_length=LAMBDA_U, length=LAMBDA_U * N_PERIODS),
                             electron_beam=ElectronBeam(Electron_energy=E_GEV, I_current=0.2),
                             magnetic_field=None, photon_energy=PHOTON_ENERGY,
                             traj_method=TRAJECTORY_METHOD_ANALYTIC, Nb_pts_trajectory=points_per_period * N_PERIODS,
                             rad_method=RADIATION_METHOD_APPROX_FARFIELD, initial_condition=None,
                             distance=DISTANCE, X=x, Y=y, XY_are_list=True)


def stokes_lienard_wiechert(trajectory, angle_x, angle_y):
    """
    Normalized Stokes parameters of the far field of the trajectory, calculated in the time domain from the
    Lienard-Wiechert fields and Fourier transformed numerically. It is an independent reference for the
    frequency-domain expressions of RadiationFactory (which integrate by parts and add boundary terms).

    1) Electric field (radiation part) of a charge e with velocity beta*c and acceleration (d beta / dt)*c, observed
       in the direction of the unit vector n, at a large distance R (Jackson, 1999, Eq. 14.14):

           E(t) = e / (4 pi epsilon_0 c) [ n x ((n - beta) x d(beta)/dt) / ((1 - n.beta)^3 R) ]_ret

       where [ ]_ret is evaluated at the retarded (emission) time t'. In the far field, n and R are constant, and the
       constant factors are dropped (they cancel in the normalized Stokes parameters).

    2) Observer time. The field emitted at t' at the position r(t') is received at

           t_obs = t' + |D n - r(t')| / c  ~  t' + D / c - n.r(t') / c

       The constant D / c only adds a global phase and is dropped: t_obs = t' - n.r(t') / c, with
       d t_obs / d t' = 1 - n.beta (Jackson, 1999, Sec. 14.5).

    3) Spectral amplitude, with the Fourier transform

           E(omega) = integral E(t_obs) exp(+i omega t_obs) d t_obs

       so that the real field is E(t) = (1 / 2 pi) integral E(omega) exp(-i omega t) d omega, i.e., the time
       dependence exp(-i omega t), as in pySRU (whose integrand contains exp(+i omega (t - n.r / c))) (Jackson, 1999,
       Sec. 14.5). The integral is computed with the trapezoidal rule on the (non uniform) grid of t_obs given
       by the trajectory points. The abrupt start and end of the trajectory are included, as in the frequency-domain
       expressions with boundary terms.

    4) Stokes parameters of the transverse components (E_x, E_y) (Born & Wolf, 1999):

           S0 = |E_x|^2 + |E_y|^2,   S1 = |E_x|^2 - |E_y|^2,
           S2 = 2 Re(E_x E_y*),      S3 = 2 Im(E_x* E_y),        (P1, P2, P3) = (S1, S2, S3) / S0

       in the right-handed frame (x, y, z), z along the beam. With the time dependence exp(-i omega t), S3 > 0 is
       left-handed (counter-clockwise rotation for an observer looking toward the source). Near the axis
       (angles ~ 1 / gamma) the longitudinal component E_z is negligible.

    References
    ----------
    J. D. Jackson, Classical Electrodynamics, 3rd ed., Wiley, New York (1999), Chapter 14.
    M. Born and E. Wolf, Principles of Optics, 7th ed., Cambridge University Press (1999).
    A. Hofmann, The Physics of Synchrotron Radiation, Cambridge University Press (2004), Chapters 2-3.

    Parameters
    ----------
    trajectory : pySRU Trajectory
        The trajectory: t (s), x, y, z (in units of c, i.e., the positions in m are c * x, ...), v_x, v_y, v_z
        (beta), a_x, a_y, a_z (d beta / dt, in s^-1).
    angle_x, angle_y : float
        Observation angles (rad): n = (angle_x, angle_y, 1) / |(angle_x, angle_y, 1)|.

    Returns
    -------
    numpy array
        (P1, P2, P3) at the photon energy PHOTON_ENERGY.
    """
    T = trajectory
    r = np.vstack((T.x, T.y, T.z)) * codata.c
    b = np.vstack((T.v_x, T.v_y, T.v_z))
    bd = np.vstack((T.a_x, T.a_y, T.a_z))
    n = np.array([angle_x, angle_y, 1.0])
    n /= np.linalg.norm(n)
    E = np.cross(n[None, :], np.cross((n[:, None] - b).T, bd.T)) / ((1 - n @ b) ** 3)[:, None]
    t_obs = T.t - (n @ r) / codata.c
    omega = 2 * np.pi * PHOTON_ENERGY * codata.e / codata.h
    integrand = E[:, :2] * np.exp(1j * omega * t_obs)[:, None]
    J = ((integrand[1:] + integrand[:-1]) * 0.5 * np.diff(t_obs)[:, None]).sum(axis=0)   # trapezoidal rule
    return stokes(J[0], J[1])


class RadiationPolarizationTest(unittest.TestCase):

    def test_mirror_symmetry(self):
        # trajectory in the plane y = 0: E_x(x, -y) = E_x(x, y), E_y(x, -y) = -E_y(x, y), exactly
        angles = ANGLES + [(a[0], -a[1]) for a in ANGLES]
        E = simulation(angles, points_per_period=20).electric_field._electrical_field
        n = len(ANGLES)
        for i in range(n):
            np.testing.assert_allclose(E[n + i, 0], E[i, 0], rtol=1e-10)
            np.testing.assert_allclose(E[n + i, 1], -E[i, 1], rtol=1e-10)

    def test_against_lienard_wiechert(self):
        sim = simulation(ANGLES, points_per_period=200)
        E = sim.electric_field._electrical_field
        for i, (ax, ay) in enumerate(ANGLES):
            np.testing.assert_allclose(stokes(E[i, 0], E[i, 1]), stokes_lienard_wiechert(sim.trajectory, ax, ay),
                                       atol=0.02, err_msg="angles (%g, %g)" % (ax, ay))


if __name__ == "__main__":
    unittest.main()
