#!/usr/bin/env python
"""Assemble a combined G1 + Wuji-hand URDF (the "Dex3 approach").

Grafts both Wuji hand URDFs onto the Unitree G1 body URDF, with the docking connector as a
rigid link between each wrist flange and the palm -- mirroring Unitree's g1_with_brainco_hand
(wrist_yaw -> connector -> palm -> fingers). Self-contained: copies every referenced mesh into
<out>/meshes/ and writes RELATIVE paths, so it loads anywhere (e.g. the VS Code URDF visualizer).

Mount numbers match make_wuji_model.py (flange 0.0415 + adapter 0.026 -> palm 0.0675).

Usage (set the repo paths if not default):
    export WUJI_HAND_DESCRIPTION=~/Downloads/wuji-hand-description
    export G1_URDF=~/Downloads/unitree_ros/robots/g1_description/g1_29dof.urdf
    python wuji/make_wuji_urdf.py [--out models/g1_wuji]
"""
import os
import sys
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.spatial.transform import Rotation as R

WUJI = Path(os.environ.get("WUJI_HAND_DESCRIPTION", Path.home() / "wuji-hand-description"))
G1_URDF = Path(os.environ.get("G1_URDF", Path.home() / "Downloads/unitree_ros/robots/g1_description/g1_29dof.urdf"))
OUT = Path(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "models/g1_wuji")
MESHES = OUT / "meshes"

# mount (same as make_wuji_model.py); URDF wants rpy, so convert our quaternion once
FLANGE_X, ADAPTER = 0.0415, 0.026
MOUNT_X = FLANGE_X + ADAPTER
CONNECTOR_X = (FLANGE_X + MOUNT_X) / 2
RPY = R.from_quat([0.7071068, 0.0, 0.7071068, 0.0][1:] + [0.7071068]).as_euler("xyz")  # wxyz->rpy
RPY_STR = f"{RPY[0]:.6f} {RPY[1]:.6f} {RPY[2]:.6f}"
# per hand: (prefix, wrist link, hand-urdf root link)
HANDS = {"left": ("wj_", "left_wrist_yaw_link", "left_palm_link"),
         "right": ("wjr_", "right_wrist_yaw_link", "right_palm_link")}


def copy_meshes(root, urdf_path):
    """Resolve each <mesh filename> relative to its own urdf dir, copy into <out>/meshes/,
    and rewrite the path to meshes/<name> so the combined URDF is portable."""
    base = Path(urdf_path).parent
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if not fn:
            continue
        src = (base / fn).resolve()
        if src.is_file():
            dst = MESHES / src.name
            if not dst.exists():
                shutil.copy(src, dst)
            mesh.set("filename", f"meshes/{src.name}")


def prefix_hand(root, pfx):
    """Namespace a hand URDF: prefix every link/joint name and every parent/child reference."""
    for el in root.iter("link"):
        el.set("name", pfx + el.get("name"))
    for el in root.iter("joint"):
        el.set("name", pfx + el.get("name"))
    for tag in ("parent", "child"):
        for el in root.iter(tag):
            if el.get("link"):
                el.set("link", pfx + el.get("link"))


def fixed_joint(name, parent, child, xyz, rpy):
    j = ET.Element("joint", {"name": name, "type": "fixed"})
    ET.SubElement(j, "parent", {"link": parent})
    ET.SubElement(j, "child", {"link": child})
    ET.SubElement(j, "origin", {"xyz": xyz, "rpy": rpy})
    return j


def connector_link(name):
    stl = WUJI / "docking/meshes/hand_docking_link.STL"
    dst = MESHES / stl.name
    if not dst.exists():
        shutil.copy(stl, dst)
    link = ET.Element("link", {"name": name})
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    geo = ET.SubElement(vis, "geometry")
    ET.SubElement(geo, "mesh", {"filename": f"meshes/{stl.name}"})
    ET.SubElement(ET.SubElement(vis, "material", {"name": "connector"}), "color", {"rgba": "0.55 0.57 0.62 1"})
    return link


def main():
    MESHES.mkdir(parents=True, exist_ok=True)
    g1 = ET.parse(str(G1_URDF))
    groot = g1.getroot()
    copy_meshes(groot, G1_URDF)

    # drop the G1 default rubber hands (links + any joint referencing them)
    for link in [l for l in groot.findall("link") if "rubber_hand" in l.get("name", "")]:
        groot.remove(link)
    for j in [j for j in groot.findall("joint")
              if any("rubber_hand" in (c.get("link") or "") for c in j) or "rubber_hand" in j.get("name", "")]:
        groot.remove(j)

    for side, (pfx, wrist, palm_root) in HANDS.items():
        h = ET.parse(str(WUJI / f"urdf/{side}.urdf"))
        hroot = h.getroot()
        copy_meshes(hroot, WUJI / f"urdf/{side}.urdf")
        prefix_hand(hroot, pfx)
        for el in list(hroot):                                  # graft the hand's links + joints
            if el.tag in ("link", "joint"):
                groot.append(el)
        cl = f"{pfx}connector_link"
        groot.append(connector_link(cl))                        # the connector, like BrainCo base2_link
        # BrainCo-style chain: wrist -> connector -> palm. Connector at mid-gap (oriented +z->arm+x);
        # palm hangs off the connector by the remaining standoff along the connector's local z
        # (which maps to arm +x), so the palm's final pose is unchanged (0.0675, rpy [0,pi/2,0]).
        groot.append(fixed_joint(f"{pfx}connector_joint", wrist, cl, f"{CONNECTOR_X:.6f} 0 0", "0 1.570796 0"))
        groot.append(fixed_joint(f"{pfx}{palm_root}_joint", cl, pfx + palm_root,
                                 f"0 0 {MOUNT_X - CONNECTOR_X:.6f}", "0 0 0"))

    out_urdf = OUT / "g1_29dof_wuji.urdf"
    ET.indent(g1)
    g1.write(str(out_urdf))
    n_mesh = len(list(MESHES.glob("*")))
    print(f"wrote {out_urdf}  ({len(groot.findall('link'))} links, {len(groot.findall('joint'))} joints, {n_mesh} meshes)")


if __name__ == "__main__":
    main()
