"""First-frame feasibility must survive a task objective pulling into an obstacle."""
import unittest
from types import SimpleNamespace
import numpy as np
from holosoma_retargeting.retarget_pipeline.clarabel_step import make_settings
from holosoma_retargeting.retarget_pipeline.collision_aware_initialization import (
    InitializationSettings, _restore, _refine_feasible,
)


class PlaneProblem:
    """One movable coordinate, one frozen coordinate, obstacle at x=0."""
    def __init__(self,locked=False):
        self.n=1;self.qadr=np.array([0]);self.lo=np.array([-.1]);self.hi=np.array([.1]);self.locked=locked
        self.cfg=SimpleNamespace(damping=.001,collision_weight=1e5,collision_tolerance=.001,
            trust_radius=.1,convergence_step=.0002)
        self.solver_settings=make_settings()

    def collisions(self,q,derivatives=True):
        d=np.array([-.02 if self.locked else q[0]])
        G=np.array([[0. if self.locked else 1.]])
        return d,G,np.array([0])

    def residuals(self,q,*unused):
        return np.array([10*(q[0]+.02)]),np.array([[10.]])


class CollisionInitializationTests(unittest.TestCase):
    def test_restore_then_track_keeps_nonlinear_feasibility_and_frozen_coordinate(self):
        ref=PlaneProblem();q=np.array([-.02,7.]);settings=InitializationSettings()
        restored,_,feasible=_restore(ref,q,settings)
        self.assertTrue(feasible)
        self.assertGreaterEqual(restored[0],-.0005)
        result,_=_refine_feasible(ref,restored,{},q,q,settings)
        # The pose objective prefers -20 mm. It cannot override the 1 mm limit.
        self.assertGreaterEqual(result[0],-.001+settings.feasibility_epsilon_m)
        self.assertLess(result[0],0.)
        self.assertEqual(result[1],7.)

    def test_unmovable_penetration_is_reported_infeasible(self):
        q=np.array([-.02,7.]);restored,_,feasible=_restore(PlaneProblem(locked=True),q,InitializationSettings())
        self.assertFalse(feasible)
        np.testing.assert_array_equal(restored,q)


if __name__=='__main__':unittest.main()
