"""Require real, active object collision geometry in every Stage-A body solve."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def prepare_body_object_colliders(base_path, output_path, object_name, decomposition_dir):
    """Preserve the robot and visual mesh; activate proxies or add cached CoACD pieces.

    Materialize an unscaled run-local model, so the existing uniform scaling and
    geometry checks apply equally to the visible object and collision pieces.
    Never replace a concave visual object with its single convex hull.
    """
    base_path, output_path = Path(base_path).resolve(), Path(output_path)
    tree = ET.parse(base_path)
    root = tree.getroot()
    body = root.find(f".//body[@name='{object_name}_link']")
    if body is None:
        raise ValueError(f"Stage A requires object body {object_name}_link in {base_path}")
    compiler = root.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(root, 'compiler')
    for key in ('meshdir', 'texturedir'):
        path = Path(compiler.get(key, '.'))
        if not path.is_absolute():
            compiler.set(key, str((base_path.parent/path).resolve()))
    visual_name = f'{object_name}_visual'
    proxies = [g for g in body.iter('geom') if g.get('name') != visual_name]
    if not proxies:
        pieces = sorted(Path(decomposition_dir).glob('piece_*.obj'))
        if not pieces:
            raise ValueError(f"No object collision pieces for {object_name}: {decomposition_dir}. "
                             "Build/validate the decomposition before retargeting; visual-only Stage A is forbidden.")
        assets = root.find('asset')
        if assets is None:
            assets = ET.SubElement(root, 'asset')
        existing_meshes = {x.get('name') for x in assets.findall('mesh')}
        for i, path in enumerate(pieces):
            name = f'{object_name}_piece_{i:03d}'
            if name+'_mesh' in existing_meshes:
                raise ValueError(f"Unused conflicting collision mesh {name}_mesh in {base_path}")
            ET.SubElement(assets, 'mesh', name=name+'_mesh', file=str(path.resolve()))
            proxies.append(ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name+'_mesh',
                                         rgba='.3 .6 .9 0', friction='.9 .5 .5'))
    for i, geom in enumerate(proxies):
        if object_name not in geom.get('name', ''):
            geom.set('name', f'{object_name}_collision_{i:03d}')
        geom.set('contype', '1')
        geom.set('conaffinity', '1')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path)
    validate_body_object_colliders(mujoco.MjModel.from_xml_path(str(output_path)), object_name)
    return str(output_path)


def validate_body_object_colliders(model, object_name):
    """Reject visual-only, filtered-out or disabled object collisions before solving."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{object_name}_link')
    if bid < 0:
        raise ValueError(f"Missing object body {object_name}_link")
    descendants = {bid}
    for b in range(bid+1, model.nbody):
        if int(model.body_parentid[b]) in descendants:
            descendants.add(b)
    object_geoms = np.flatnonzero(np.isin(model.geom_bodyid, list(descendants)))
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    pieces = object_geoms[active[object_geoms]]
    visible = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{object_name}_visual')
    if not len(pieces) or visible < 0 or visible in pieces:
        raise ValueError(f"Stage A for {object_name} needs active collision proxies, not only the visual mesh. "
                         "Rebuild part0 with object colliders enabled.")
    if int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT):
        raise ValueError('Stage A has contact detection disabled')
    robots = np.flatnonzero(active & ~np.isin(np.arange(model.ngeom), object_geoms)
                            & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE))
    for g in robots:
        compatible = ((model.geom_contype[g] & model.geom_conaffinity[pieces]) != 0) | (
            (model.geom_contype[pieces] & model.geom_conaffinity[g]) != 0)
        if not np.all(compatible):
            raise ValueError(f'Robot geom {g} has filtered object collision pieces')
        for p in pieces:
            b1, b2 = sorted((int(model.geom_bodyid[g]), int(model.geom_bodyid[p])))
            if (b1 << 16) + b2 in model.exclude_signature:
                raise ValueError(f'Robot/object bodies {b1}, {b2} have an explicit contact exclusion')
    for g in pieces:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ''
        if object_name not in name:
            raise ValueError(f'Object collider {g} would be ignored by the body solver name filter')
    if not len(robots):
        raise ValueError('Stage A has no active robot collision geometry')
    return dict(object_collision_geoms=int(len(pieces)), robot_collision_geoms=int(len(robots)),
                candidate_robot_object_pairs=int(len(pieces)*len(robots)))
