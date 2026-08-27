import math
import unittest

import numpy as np

from gasdynbench.physics import (
    area_mach,
    fanno_inverse_fld,
    fanno_ratios,
    nozzle_back_pressure,
    nozzle_shock_area,
    oblique_beta,
    oblique_detachment,
    oblique_theta,
    rayleigh_inverse_t0,
    rayleigh_ratios,
    shock_tube_pressure_ratio,
    shock_tube_pressure_ratio_general,
    shock_tube_residual,
    shock_tube_residual_general,
)


class PhysicsTests(unittest.TestCase):
    def test_sonic_reference_states(self):
        np.testing.assert_allclose(rayleigh_ratios(1.0), np.ones(6), atol=1e-12)
        np.testing.assert_allclose(fanno_ratios(1.0), np.array([1, 1, 1, 1, 0]), atol=1e-12)
        self.assertAlmostEqual(float(area_mach(1.0)), 1.0, places=12)

    def test_rayleigh_branch_inverse(self):
        for m, branch in [(0.35, "subsonic"), (2.4, "supersonic")]:
            target = float(rayleigh_ratios(m)[4])
            self.assertAlmostEqual(rayleigh_inverse_t0(target, branch), m, places=9)

    def test_fanno_branch_inverse(self):
        for m, branch in [(0.4, "subsonic"), (2.2, "supersonic")]:
            target = float(fanno_ratios(m)[4])
            self.assertAlmostEqual(fanno_inverse_fld(target, branch), m, places=8)

    def test_oblique_branch_inverse(self):
        m = 3.0
        _, theta_max = oblique_detachment(m)
        theta = 0.55 * theta_max
        weak = oblique_beta(m, theta, "weak")
        strong = oblique_beta(m, theta, "strong")
        self.assertLess(weak, strong)
        self.assertAlmostEqual(float(oblique_theta(m, weak)), theta, places=9)
        self.assertAlmostEqual(float(oblique_theta(m, strong)), theta, places=9)

    def test_nozzle_inverse(self):
        ae, shock = 3.0, 1.8
        pb = nozzle_back_pressure(ae, shock)
        self.assertAlmostEqual(nozzle_shock_area(ae, pb), shock, places=8)

    def test_shock_tube_residual(self):
        p4, t4 = 40.0, 1.3
        p2 = shock_tube_pressure_ratio(p4, t4)
        self.assertTrue(1.0 < p2 < p4)
        self.assertLess(abs(float(shock_tube_residual(p2, p4, t4))), 1e-8)

    def test_general_shock_tube_reduces_to_equal_gas_relation(self):
        for p4, t4, gamma in [(5.0, 0.8, 1.3), (40.0, 1.3, 1.4), (250.0, 1.8, 1.6)]:
            standard = shock_tube_pressure_ratio(p4, t4, gamma)
            general = shock_tube_pressure_ratio_general(p4, t4, gamma, gamma, 1.0)
            self.assertAlmostEqual(standard, general, places=11)

    def test_distinct_gas_shock_tube_residual(self):
        inputs = (80.0, 1.4, 1.35, 1.66, 6.5)
        p2 = shock_tube_pressure_ratio_general(*inputs)
        self.assertTrue(1.0 < p2 < inputs[0])
        residual = shock_tube_residual_general(p2, *inputs)
        self.assertLess(abs(float(residual)), 1e-8)


if __name__ == "__main__":
    unittest.main()
