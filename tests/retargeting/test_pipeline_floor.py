"""Regression: toe-joint grounding must not lower an already grounded object."""
import numpy as np
import unittest
from holosoma_retargeting.retarget_pipeline.pipeline import preprocess_motion_data_uniform, resolve_floor_mode
from holosoma_retargeting.retarget_pipeline.checks import object_floor_heights


def test_grounded_scene_keeps_object_floor_and_relative_hand_positions():
    scale = .8188585607940446
    toe_height = .026917246528181636
    human = np.array([[[0., 0., toe_height], [.2, 0., toe_height]]])
    hands = np.array([[[[.1, .2, .5]]]])
    obj = np.array([[1., 0., 0., 0., .1, .2, .1]])  # wxyz + xyz
    vertices = np.array([[-.1, -.1, -.1], [.1, .1, .1]])
    args = (human, hands, obj, ['L_Toe', 'R_Toe'], ('L_Toe', 'R_Toe'), scale)
    before = (human.copy(), hands.copy(), obj.copy())
    h, k, p, z = preprocess_motion_data_uniform(*args, floor_mode='source')
    assert z == 0.
    np.testing.assert_allclose(object_floor_heights(scale*vertices, p), 0., atol=1e-10)
    np.testing.assert_allclose(k-p[:,None,None,:3], scale*(hands-obj[:,None,None,4:]))
    np.testing.assert_allclose(h, scale*human)
    np.testing.assert_array_equal(p[:,3:], obj[:,:4])
    for actual, original in zip((human,hands,obj),before):np.testing.assert_array_equal(actual,original)
    # The legacy mode deliberately reproduces the diagnosed 22.04-mm error.
    _, _, legacy, _ = preprocess_motion_data_uniform(*args, floor_mode='toe')
    np.testing.assert_allclose(object_floor_heights(scale*vertices, legacy), -scale*toe_height, atol=1e-10)


def test_policy_is_scoped_and_legacy_mat_behavior_retained():
    assert resolve_floor_mode('auto','humoto') == 'source'
    assert resolve_floor_mode('auto','humoto_hiphi2smplx') == 'source'
    assert resolve_floor_mode('auto','omomo') == 'toe'
    assert resolve_floor_mode('toe','humoto') == 'toe'
    h=np.array([[[0.,0.,.12],[0.,0.,.14]]]);p=np.array([[1.,0.,0.,0.,0.,0.,.5]])
    _, hk, _, z=preprocess_motion_data_uniform(h,None,p,['l','r'],('l','r'),.8)
    assert hk is None
    np.testing.assert_allclose(z, .02, atol=1e-12)


def test_floor_check_uses_mesh_orientation_and_translation():
    vertices=np.array([[-2.,0.,0.],[1.,0.,0.]])
    # Rotate 90 degrees about Y: x=-2 maps to z=2, x=1 to z=-1.
    pose=np.array([[0.,0.,1.,np.sqrt(.5),0.,np.sqrt(.5),0.]])
    np.testing.assert_allclose(object_floor_heights(vertices,pose),0.,atol=1e-10)
    pose[:,2]-=.01
    np.testing.assert_allclose(object_floor_heights(vertices,pose),-.01,atol=1e-10)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_grounded_scene_keeps_object_floor_and_relative_hand_positions,
        test_policy_is_scoped_and_legacy_mat_behavior_retained,
        test_floor_check_uses_mesh_orientation_and_translation,
    ))


if __name__ == '__main__':
    unittest.main()
