import datetime
import unittest

import numpy
import pandas

from coral_model.environment import Processes, Constants, Environment


class TestProcesses(unittest.TestCase):

    def test_default(self):
        processes = Processes()
        self.assertTrue(processes.fme)
        self.assertTrue(processes.tme)
        self.assertTrue(processes.pfd)

    def test_tme_false(self):
        processes = Processes(tme=False)
        self.assertTrue(processes.fme)
        self.assertFalse(processes.tme)
        self.assertTrue(processes.pfd)

    def test_fme_false(self):
        processes = Processes(fme=False)
        self.assertFalse(processes.fme)
        self.assertFalse(processes.tme)
        self.assertTrue(processes.pfd)

    def test_pfd_false(self):
        processes = Processes(pfd=False)
        self.assertFalse(processes.fme)
        self.assertFalse(processes.tme)
        self.assertFalse(processes.pfd)


class TestConstants(unittest.TestCase):

    def test_default_function(self):
        constants = Constants(Processes(), turbulence_coef=.2)
        self.assertEqual(constants.Cs, .2)

    def test_default_lme(self):
        constants = Constants(Processes())
        self.assertEqual(constants.Kd0, .1)
        self.assertAlmostEqual(constants.theta_max, 1.5707963267948966)

    def test_default_fme(self):
        constants = Constants(Processes())
        self.assertEqual(constants.Cs, .17)
        self.assertEqual(constants.Cm, 1.7)
        self.assertEqual(constants.Cf, .01)
        self.assertEqual(constants.nu, 1e-6)
        self.assertEqual(constants.alpha, 1e-7)
        self.assertEqual(constants.psi, 2)
        self.assertEqual(constants.wcAngle, 0)
        self.assertEqual(constants.rd, 500)
        self.assertEqual(constants.numericTheta, .5)
        self.assertEqual(constants.err, 1e-3)
        self.assertEqual(constants.maxiter_k, 1e5)
        self.assertEqual(constants.maxiter_aw, 1e5)

    def test_default_tme(self):
        constants = Constants(Processes())
        self.assertEqual(constants.K0, 80)
        self.assertEqual(constants.ap, .4)
        self.assertEqual(constants.k, .6089)

    def test_default_pld(self):
        constants = Constants(Processes())
        self.assertEqual(constants.iota, .6)
        self.assertEqual(constants.ik_max, 372.32)
        self.assertEqual(constants.pm_max, 1)
        self.assertEqual(constants.betaI, .34)
        self.assertEqual(constants.betaP, .09)

    def test_default_ptd(self):
        constants = Constants(Processes())
        self.assertEqual(constants.Ea, 6e4)
        self.assertEqual(constants.R, 8.31446261815324)
        self.assertEqual(constants.k_var, 2.45)
        self.assertEqual(constants.nn, 60)

    def test_default_pfd(self):
        constants = Constants(Processes())
        self.assertEqual(constants.pfd_min, .68886964)
        self.assertEqual(constants.ucr, .17162374)

        constants = Constants(Processes(fme=False))
        self.assertEqual(constants.pfd_min, .68886964)
        self.assertEqual(constants.ucr, .5173)

    def test_default_pd(self):
        constants = Constants(Processes())
        self.assertEqual(constants.r_growth, .002)
        self.assertEqual(constants.r_recovery, .2)
        self.assertEqual(constants.r_mortality, .04)
        self.assertEqual(constants.r_bleaching, 8)

    def test_default_c(self):
        constants = Constants(Processes())
        self.assertEqual(constants.gC, .5)
        self.assertEqual(constants.omegaA0, 5)
        self.assertEqual(constants.omega0, .14587415)
        self.assertEqual(constants.kappaA, .66236107)

    def test_default_md(self):
        constants = Constants(Processes())
        self.assertEqual(constants.prop_form, .1)
        self.assertEqual(constants.prop_plate, .5)
        self.assertEqual(constants.prop_plate_flow, .1)
        self.assertAlmostEqual(constants.prop_space, .35355339059327373)
        self.assertEqual(constants.prop_space_light, .1)
        self.assertEqual(constants.prop_space_flow, .1)
        self.assertEqual(constants.u0, .2)
        self.assertEqual(constants.rho_c, 1600)

    def test_default_dc(self):
        constants = Constants(Processes())
        self.assertEqual(constants.sigma_t, 2e5)
        self.assertEqual(constants.Cd, 1)
        self.assertEqual(constants.rho_w, 1025)

    def test_default_cr(self):
        constants = Constants(Processes())
        self.assertEqual(constants.no_larvae, 1e6)
        self.assertEqual(constants.prob_settle, 1e-4)
        self.assertEqual(constants.d_larvae, 1e-3)


class TestEnvironment(unittest.TestCase):

    def test_default(self):
        environment = Environment()
        self.assertIsNone(environment.light)
        self.assertIsNone(environment.light_attenuation)
        self.assertIsNone(environment.temperature)
        self.assertIsNone(environment.aragonite)
        self.assertIsNone(environment.storm_category)

    def test_dates1(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        self.assertIn('2000-01-01', str(environment.dates[0]))
        self.assertIn('2001-01-01', str(environment.dates[366]))

    def test_dates2(self):
        environment = Environment()
        environment.set_dates(
            datetime.date(2000, 1, 1), datetime.date(2001, 1, 1)
        )
        self.assertIn('2000-01-01', str(environment.dates[0]))
        self.assertIn('2001-01-01', str(environment.dates[366]))

    def test_set_parameter1(self):
        environment = Environment()
        with self.assertRaises(TypeError) as context:
            environment.set_parameter_values('light', 600)
        self.assertTrue('No dates are defined.' in str(context.exception))

    def test_set_parameter2(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        with self.assertRaises(ValueError) as context:
            environment.set_parameter_values('keyword', 1)
        self.assertTrue('Entered parameter' in str(context.exception))

    def test_set_parameter3(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        environment.set_parameter_values('light', 600)
        for d, date in enumerate(environment.dates):
            self.assertEqual(date, environment.light.index[d])

    def test_set_parameter4(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        environment.set_parameter_values('light', 600)
        for light in environment.light.values:
            self.assertEqual(light, 600)

    def test_set_parameter5(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        light_list = [300 * (1 + numpy.cos(.1 * t)) for t in range(len(environment.dates))]
        environment.set_parameter_values('light', light_list)
        for i, light in enumerate(environment.light.values):
            self.assertEqual(light, light_list[i])

    def test_set_parameter6(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        environment.set_parameter_values('LAC', .1)
        for lac in environment.light_attenuation.values:
            self.assertEqual(lac, .1)

    def test_set_parameter7(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        environment.set_parameter_values('temperature', 300)
        for temp in environment.temperature.values:
            self.assertEqual(temp, 300)

    def test_set_parameter8(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        environment.set_parameter_values('temperature', 300)
        for temp in environment.temp_kelvin.values:
            self.assertEqual(float(temp), 300)

    def test_set_parameter9(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        environment.set_parameter_values('temperature', 300)
        for temp in environment.temp_celsius.values:
            self.assertEqual(float(temp), 300 - 273.15)

    def test_set_parameter10(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        environment.set_parameter_values('temperature', 30)
        for temp in environment.temp_kelvin.values:
            self.assertEqual(float(temp), 30 + 273.15)

    def test_set_parameter11(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        environment.set_parameter_values('temperature', 30)
        for temp in environment.temp_celsius.values:
            self.assertEqual(float(temp), 30)

    def test_set_parameter12(self):
        environment = Environment()
        environment.set_dates('2000-01-01', '2001-01-01')
        environment.set_parameter_values('aragonite', 5)
        for arag in environment.aragonite.values:
            self.assertEqual(float(arag), 5)
