"""The generic fast collider path must preserve disconnected solids and cavities."""
import importlib.util
from pathlib import Path
import numpy as np
import trimesh

path=Path(__file__).resolve().parents[2]/'src/wuji_wbt/tools/pipeline_v2_cross_dataset/convex_components.py'
spec=importlib.util.spec_from_file_location('convex_component_budget',path)
impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)

def test_disconnected_boxes_remain_two_exact_solids():
    first=trimesh.creation.box(extents=[.1,.2,.3]);second=first.copy();second.apply_translation([1,0,0])
    parts,audit=impl.candidate(trimesh.util.concatenate([first,second]))
    assert audit['accepted'] and len(parts)==2
    assert max(r['sampled_error_mm'] for r in audit['parts'])<1e-6
    # The one-metre gap is not filled by an enclosing global hull.
    assert all(not (p.bounds[0,0]<.5<p.bounds[1,0]) for p in parts)

def test_annulus_cavity_rejects_single_hull_shortcut():
    ring=trimesh.creation.annulus(r_min=.3,r_max=.5,height=.1,sections=32)
    parts,audit=impl.candidate(ring)
    assert parts is None and not audit['accepted']
    assert 'surface budget' in audit['reason']

def test_zero_thickness_surface_is_not_silently_thickened():
    plane=trimesh.Trimesh(vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),faces=[[0,1,2],[0,2,3]])
    parts,audit=impl.candidate(plane)
    assert parts is None and not audit['accepted']
