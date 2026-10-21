"""Regression for the visual-only shelf body pass, using actual collision queries."""
import tempfile
import unittest
from pathlib import Path
import mujoco
import numpy as np
from holosoma_retargeting.retarget_pipeline.object_colliders import (
    prepare_body_object_colliders, validate_body_object_colliders,
)

PKG = Path(__file__).resolve().parents[2]/'src/holosoma_retargeting/holosoma_retargeting'


class BodyColliderTests(unittest.TestCase):
    def test_shelf_collider_is_added_and_detects_overlap(self):
        base = PKG/'models/g1/g1_29dof_wuji_welded_w_shelf.xml'
        original = mujoco.MjModel.from_xml_path(str(base))
        with self.assertRaisesRegex(ValueError, 'active collision proxies'):
            validate_body_object_colliders(original, 'shelf')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.xml'
            prepare_body_object_colliders(base,path,'shelf',PKG/'models/decomp/shelf')
            model = mujoco.MjModel.from_xml_path(str(path))
            report = validate_body_object_colliders(model,'shelf')
            self.assertEqual(report['object_collision_geoms'],30)
            self.assertEqual(model.nq,original.nq)
            np.testing.assert_array_equal(model.jnt_range,original.jnt_range)
            data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            a=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'left_hip_yaw_link')
            b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'shelf_piece_023')
            # Move only the free object to intersect this robot link.
            data.qpos[-7:-4] += data.geom_xpos[a]-data.geom_xpos[b]+np.array([.01,.02,.01])
            mujoco.mj_forward(model,data)
            self.assertLess(mujoco.mj_geomDistance(model,data,a,b,.1,np.zeros(6)),0.)
            self.assertTrue(any(a in (c.geom1,c.geom2) and b in (c.geom1,c.geom2) for c in data.contact))

    def test_missing_decomposition_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'No object collision pieces'):
                prepare_body_object_colliders(PKG/'models/g1/g1_29dof_wuji_welded_w_shelf.xml',
                    Path(folder)/'model.xml','shelf',Path(folder)/'missing')

    def test_existing_disabled_proxy_is_enabled_and_detectable(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)/'input.xml';out=Path(folder)/'output.xml'
            base.write_text('''<mujoco><worldbody><geom type="plane" size="1 1 .1"/>
              <body name="robot" pos="0 0 .2"><freejoint/><geom type="sphere" size=".1"/></body>
              <body name="box_link" pos="0 0 .2"><freejoint/>
                <geom name="box_visual" type="box" size=".1 .1 .1" contype="0" conaffinity="0"/>
                <geom type="box" size=".1 .1 .1" contype="0" conaffinity="0"/>
              </body></worldbody></mujoco>''')
            prepare_body_object_colliders(base,out,'box',Path(folder)/'missing')
            model=mujoco.MjModel.from_xml_path(str(out))
            self.assertEqual(validate_body_object_colliders(model,'box')['object_collision_geoms'],1)
            data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            collider=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'box_collision_000')
            self.assertTrue(any(collider in (c.geom1,c.geom2) for c in data.contact))


if __name__=='__main__':unittest.main()
