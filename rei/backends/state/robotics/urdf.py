"""Non-mutating URDF joint locking, independent of the dynamics backend."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
import xml.etree.ElementTree as ET

import numpy as np

from ....core.model_config import normalize_model_config


@dataclass(frozen=True)
class ReducedUrdf:
    xml: str
    active_joint_names: tuple[str, ...]
    locked_joints: dict[str, float]


def load_joint_locks_toml(path: str | Path) -> dict[str, float]:
    """Read ``[model.locked_joints]`` from a problem spec or standalone TOML.

    An absent section means no locks. Positions must be finite TOML numbers
    (radians/metres); URDF names and limits are checked by lock_urdf_joints.
    This is model-loading configuration, not an optimization constraint.
    """
    with Path(path).open('rb') as stream:
        config = tomllib.load(stream)
    return normalize_model_config(config.get('model', {})).get('locked_joints', {})


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    value = np.array(default if text is None else [float(x) for x in text.split()])
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("URDF vector must contain three finite numbers")
    return value


def _rotation(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def _rpy(R: np.ndarray) -> np.ndarray:
    p = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if math.hypot(R[0, 0], R[1, 0]) > 1e-12:
        return np.array([math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])])
    return np.array([math.atan2(-R[1, 2], R[1, 1]), p, 0.0])


def lock_urdf_joints(xml: str, positions: Mapping[str, float]) -> ReducedUrdf:
    """Replace selected scalar joints by fixed transforms at the given positions.

    Positions are radians for revolute/continuous joints and metres for prismatic
    joints. Mimic followers are locked recursively with their master. Conflicting
    positions, limit violations, unknown joints and mimic cycles are rejected.
    Links, inertias, visual/collision elements and mesh URIs are preserved; no
    source files are modified. Relative mesh paths retain their source directory.
    Unselected mimic relationships remain in the XML for capable consumers.
    """
    root = ET.fromstring(xml)
    if root.tag != "robot":
        raise ValueError("Expected a URDF robot element")
    joints = {j.attrib['name']: j for j in root.findall('joint')}
    if len(joints) != len(root.findall('joint')):
        raise ValueError("Duplicate URDF joint names")
    locked = {name: float(q) for name, q in positions.items()}
    unknown = locked.keys() - joints.keys()
    if unknown:
        raise ValueError(f"Unknown joints: {sorted(unknown)}")
    visited, visiting = set(), set()

    def resolve(name: str) -> None:
        if name in visiting:
            raise ValueError(f"Mimic cycle at {name}")
        if name in visited:
            return
        visiting.add(name)
        mimic = joints[name].find('mimic')
        if mimic is not None:
            master = mimic.get('joint')
            if master not in joints:
                raise ValueError(f"Unknown mimic master for {name}: {master}")
            resolve(master)
            if master in locked:
                q = float(mimic.get('multiplier', '1')) * locked[master] + float(mimic.get('offset', '0'))
                if name in locked and not math.isclose(locked[name], q, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Conflicting mimic position for {name}")
                locked[name] = q
            elif name in locked:
                raise ValueError(f"Lock mimic master {master} before follower {name}")
        visiting.remove(name)
        visited.add(name)

    for name in joints:
        resolve(name)
    for name, q in locked.items():
        joint = joints[name]
        kind = joint.get('type')
        if kind not in ('revolute', 'continuous', 'prismatic', 'fixed'):
            raise ValueError(f"Cannot lock joint type {kind}: {name}")
        if not math.isfinite(q):
            raise ValueError(f"Non-finite position for {name}")
        if kind == 'fixed' and q != 0:
            raise ValueError(f"Fixed joint {name} only accepts zero")
        limit = joint.find('limit')
        if kind in ('revolute', 'prismatic') and limit is not None:
            if not float(limit.get('lower', '-inf')) <= q <= float(limit.get('upper', 'inf')):
                raise ValueError(f"Locked position outside limits for {name}")
        origin = joint.find('origin')
        if origin is None:
            origin = ET.SubElement(joint, 'origin')
        xyz = _vector(origin.get('xyz'), (0., 0., 0.))
        R = _rotation(_vector(origin.get('rpy'), (0., 0., 0.)))
        if kind != 'fixed':
            axis_node = joint.find('axis')
            axis = _vector(None if axis_node is None else axis_node.get('xyz'), (1., 0., 0.))
            norm = np.linalg.norm(axis)
            if norm == 0:
                raise ValueError(f"Zero axis for {name}")
            axis /= norm
            if kind == 'prismatic':
                xyz += R @ (axis*q)
            else:
                x, y, z = axis
                K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                R = R @ (np.eye(3) + math.sin(q)*K + (1-math.cos(q))*(K@K))
        origin.set('xyz', ' '.join(format(v, '.17g') for v in xyz))
        origin.set('rpy', ' '.join(format(v, '.17g') for v in _rpy(R)))
        joint.set('type', 'fixed')
        for tag in ('axis', 'limit', 'mimic', 'dynamics', 'safety_controller', 'calibration'):
            for element in joint.findall(tag):
                joint.remove(element)
    return ReducedUrdf(
        xml=ET.tostring(root, encoding='unicode'),
        active_joint_names=tuple(name for name, joint in joints.items() if joint.get('type') != 'fixed'),
        locked_joints=locked,
    )
