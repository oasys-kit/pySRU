import unittest
import numpy as np
import scipy.constants as codata
from pySRU.MagneticStructureUndulatorPlane import MagneticStructureUndulatorPlane as Undulator
from pySRU.ElectronBeam import ElectronBeam
from pySRU.SourceUndulatorPlane import SourceUndulatorPlane
from pySRU.TrajectoryFactory import TrajectoryFactory,TRAJECTORY_METHOD_ANALYTIC,TRAJECTORY_METHOD_INTEGRATION,\
                                                        TRAJECTORY_METHOD_ODE
from pySRU.RadiationFactory import RadiationFactory , RADIATION_METHOD_APPROX_FARFIELD,RADIATION_METHOD_NEAR_FIELD ,\
                                RADIATION_METHOD_FARFIELD


class RadiationFactoryTest(unittest.TestCase):

    #TODO des print sont cache qql part
    def test_create_radiation_undulator(self):
        undulator_test = Undulator(K=1.87, period_length=0.035, length=0.035 * 14)
        electron_beam_test = ElectronBeam(Electron_energy=1.3, I_current=1.0)
        source_test=SourceUndulatorPlane(magnetic_structure=undulator_test,electron_beam=electron_beam_test)
        traj_fact=TrajectoryFactory(Nb_pts=1001, method=TRAJECTORY_METHOD_ANALYTIC)
        traj=traj_fact.create_from_source(source_test)
        rad_fact = RadiationFactory(omega=source_test.harmonic_frequency(1), method=RADIATION_METHOD_APPROX_FARFIELD, Nb_pts=101)
        rad=rad_fact.create_for_single_electron(trajectory=traj,source=source_test)
        self.assertFalse(rad.X == None)
        self.assertFalse(rad.Y == None)
        self.assertFalse(rad.distance == None)

        rad_fact.method=RADIATION_METHOD_FARFIELD

        rad2=rad_fact.create_for_single_electron(trajectory=traj,source=source_test)
        err=rad.difference_with(rad2)

        self.assertTrue(rad.XY_are_like_in(rad2))
        self.assertTrue(rad.XY_are_like_in(err))
        self.assertTrue(rad.distance == rad2.distance)
        self.assertTrue(err.distance == rad2.distance)
        self.assertGreaterEqual(err.intensity.min(),0.0)
        self.assertLessEqual(err.max(), rad.max()*1e-3)


        traj_test2=TrajectoryFactory(Nb_pts=1001, method=TRAJECTORY_METHOD_INTEGRATION,
                                     initial_condition=traj_fact.initial_condition).create_from_source(source_test)
        rad3=rad_fact.create_for_single_electron(trajectory=traj_test2, source=source_test)
        err = rad2.difference_with(rad3)
        self.assertLessEqual(err.max(),rad2.max()*1e-3)
