"""Constrain the right-thumb tip to the intended halfspace during source contact.

Matched to the combined geometry-side/contact-priority candidate. One linearized
support-plane inequality is added without slack. Other collision constraints,
soft penalties, trust radius, rate limits, and nonlinear line search are intact.
Nonlinear side clearance is checked separately after the complete solve.
"""
from config import *
follower = module('hard_side_followup_solver', FOLLOWUP_CODE / 'solver.py')
np, mujoco = follower.np, follower.mujoco
import clarabel
from scipy import sparse
from pipeline_v2 import solve_clarabel_qp_step


class Refiner(follower.Refiner):
    def frame_target(self, *args):
        target = super().frame_target(*args)
        self.side_target = target
        return target

    def subproblem(self, q, residual, J, dist, gradient, previous):
        n, k = self.n, len(dist)
        H = 2*(J.T@J+self.cfg.damping*np.eye(n))
        P = sparse.block_diag([sparse.csc_matrix(H), self.cfg.collision_weight*sparse.eye(k)], format='csc')
        linear = np.r_[2*J.T@residual, np.zeros(k)]
        low, high = self.lo-q[self.qadr], self.hi-q[self.qadr]
        if previous is not None:
            low = np.maximum(low, previous[self.qadr]-self.cfg.joint_rate_limit-q[self.qadr])
            high = np.minimum(high, previous[self.qadr]+self.cfg.joint_rate_limit-q[self.qadr])
        I, Z = sparse.eye(n, format='csc'), sparse.csc_matrix((n, k))
        blocks = [sparse.hstack([I, Z]), sparse.hstack([-I, Z])]; rhs = [high, -low]
        if k:
            blocks += [sparse.hstack([-sparse.csc_matrix(gradient), -sparse.eye(k)]),
                       sparse.hstack([sparse.csc_matrix((k, n)), -sparse.eye(k)])]
            rhs += [dist+self.cfg.collision_tolerance, np.zeros(k)]
        hard_rows = 0
        if self.data['contact'][self.frame, 1, 0]:
            self.fk(q)
            clearance, G = self.support_side(self.side_target, 1, 0)
            blocks.append(sparse.csc_matrix(np.r_[-G, np.zeros(k)][None]))
            rhs.append(np.array([clearance+self.cfg.collision_tolerance]))
            hard_rows = 1
        A = sparse.vstack(blocks, format='csc'); b = np.concatenate(rhs)
        cones = [clarabel.NonnegativeConeT(len(b))]
        soc = sparse.vstack([sparse.csc_matrix((1, n+k)), sparse.hstack([-I, Z])], format='csc')
        A = sparse.vstack([A, soc], format='csc'); b = np.r_[b, self.cfg.trust_radius, np.zeros(n)]
        cones.append(clarabel.SecondOrderConeT(n+1))
        x, info = solve_clarabel_qp_step(P, linear, A, b, cones, self.solver_settings)
        info.update(slack_max_m=float(x[n:].max()) if k else 0., collision_rows=k,
                    hard_right_thumb_side_rows=hard_rows)
        return x[:n], info


geometry_solver = follower.parent.implementation.original
geometry_solver.Refiner = Refiner
target_at = follower.target_at


def load(seq, variant):
    return follower.load(seq, 'baseline' if variant in ('baseline', 'both') else 'combined')


geometry_solver.load = load


if __name__ == '__main__': geometry_solver.main(*sys.argv[1:])
