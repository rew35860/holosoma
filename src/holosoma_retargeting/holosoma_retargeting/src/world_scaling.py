"""Uniform object scaling consistent with scaling the whole demonstration."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np


def scale_object_poses(poses, scale, floor_origin_z):
    """For x_world=R*v+t, implement s*(x_world-[0,0,z0]).

    Mesh vertices are scaled separately by s; rotation is unchanged.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    out = np.asarray(poses).copy()
    out[:, 4:7] = (out[:, 4:7] - [0., 0., floor_origin_z]) * scale
    return out


def scaled_scene(source, object_name, scale, output_dir):
    """Scale object meshes and body-relative geom offsets; retain robot size."""
    source = Path(source).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find('compiler')
    meshdir = source.parent / (compiler.get('meshdir', '') if compiler is not None else '')
    body = root.find(f".//body[@name='{object_name}_link']")
    if body is None:
        raise ValueError(f"Object body absent: {object_name}")
    mesh_names = {g.get('mesh') for g in body.iter('geom') if g.get('mesh')}
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('file'):
            mesh.set('file', str((meshdir / mesh.get('file')).resolve()))
        if mesh.get('name') in mesh_names:
            old = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            mesh.set('scale', ' '.join(map(str, old * scale)))
    if compiler is not None:
        compiler.attrib.pop('meshdir', None)
    for elem in body.iter():
        if elem is body:
            continue  # free-body world position is provided by the trajectory
        if elem.get('pos'):
            elem.set('pos', ' '.join(map(str, np.fromstring(elem.get('pos'), sep=' ') * scale)))
        if elem.tag == 'geom' and elem.get('size'):
            elem.set('size', ' '.join(map(str, np.fromstring(elem.get('size'), sep=' ') * scale)))
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(source.read_bytes() + repr(float(scale)).encode()).hexdigest()[:16]
    path = dest / f'{object_name}_{key}.xml'
    tree.write(path, encoding='unicode')
    return str(path.resolve())
