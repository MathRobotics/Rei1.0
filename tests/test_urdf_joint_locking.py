"""Joint locking preserves the full robot's constrained dynamics."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rei.backends.state.robotics.urdf import load_joint_locks_toml, lock_urdf_joints


def test_toml_locks_apply_to_model(robot_xml, tmp_path):
    config = tmp_path / 'problem.toml'
    config.write_text('[time]\nN = 2\n[model.locked_joints]\nfinger = 0.2\n')
    reduced = lock_urdf_joints(robot_xml, load_joint_locks_toml(config))
    assert reduced.locked_joints == {'finger': .2, 'follower': 0.}
    assert reduced.active_joint_names == ('left', 'right')


def test_toml_without_locks_is_compatible(tmp_path):
    config = tmp_path / 'problem.toml'
    config.write_text('[time]\nN = 2\n')
    assert load_joint_locks_toml(config) == {}


@pytest.mark.parametrize('content', [
    'model = 1', '[model]\nlocked_joints = ["finger"]',
    '[model]\nlocked_joint = 0.2',
    '[model.locked_joints]\nfinger = true',
    '[model.locked_joints]\nfinger = "0.2"',
    '[model.locked_joints]\nfinger = nan',
    '[model.locked_joints]\nfinger = inf',
])
def test_invalid_toml_lock_configuration(tmp_path, content):
    config = tmp_path / 'problem.toml'
    config.write_text(content)
    with pytest.raises(ValueError):
        load_joint_locks_toml(config)


@pytest.fixture
def robot_xml():
    root = ET.Element('robot', name='branched_gripper')
    for name in ('base', 'left', 'right', 'finger', 'follower'):
        link = ET.SubElement(root, 'link', name=name)
        inertia = ET.SubElement(link, 'inertial')
        ET.SubElement(inertia, 'origin', xyz='.1 .2 .3', rpy='.2 -.1 .3')
        ET.SubElement(inertia, 'mass', value='2')
        ET.SubElement(inertia, 'inertia', ixx='.2', iyy='.3', izz='.4', ixy='0', ixz='0', iyz='0')
    for name, kind, parent, child in (
        ('left', 'revolute', 'base', 'left'),
        ('right', 'revolute', 'base', 'right'),
        ('finger', 'prismatic', 'left', 'finger'),
        ('follower', 'prismatic', 'right', 'follower'),
    ):
        j = ET.SubElement(root, 'joint', name=name, type=kind)
        ET.SubElement(j, 'parent', link=parent)
        ET.SubElement(j, 'child', link=child)
        ET.SubElement(j, 'origin', xyz='.2 .1 .3', rpy='.3 -.5 .8')
        ET.SubElement(j, 'axis', xyz='1 2 3')
        ET.SubElement(j, 'limit', lower='-2', upper='2', effort='10', velocity='2')
        if name == 'follower':
            ET.SubElement(j, 'mimic', joint='finger', multiplier='-.5', offset='.1')
    return ET.tostring(root, encoding='unicode')


def test_preserves_links_and_resolves_mimic(robot_xml):
    positions = {'finger': .2}
    result = lock_urdf_joints(robot_xml, positions)
    before, after = ET.fromstring(robot_xml), ET.fromstring(result.xml)
    assert positions == {'finger': .2}
    assert result.locked_joints == {'finger': .2, 'follower': 0.}
    assert result.active_joint_names == ('left', 'right')
    assert [ET.tostring(l) for l in before.findall('link')] == [ET.tostring(l) for l in after.findall('link')]
    assert after.find("joint[@name='follower']/mimic") is None
    assert before.find("joint[@name='finger']").get('type') == 'prismatic'


@pytest.mark.parametrize('positions,match', [
    ({'missing': 0}, 'Unknown joints'), ({'finger': 3}, 'outside limits'),
    ({'finger': float('nan')}, 'Non-finite'),
    ({'finger': .2, 'follower': .2}, 'Conflicting mimic'),
    ({'follower': 0}, 'Lock mimic master'),
])
def test_rejects_invalid_locks(robot_xml, positions, match):
    with pytest.raises(ValueError, match=match):
        lock_urdf_joints(robot_xml, positions)


def test_rejects_mimic_cycles(robot_xml):
    root = ET.fromstring(robot_xml)
    ET.SubElement(root.find("joint[@name='finger']"), 'mimic', joint='follower')
    with pytest.raises(ValueError, match='cycle'):
        lock_urdf_joints(ET.tostring(root, encoding='unicode'), {})


@pytest.mark.parametrize('positions', [{'finger': .2}, {'left': .6}])
def test_dynamics_match_original_with_zero_locked_velocity(robot_xml, positions):
    Kots = pytest.importorskip('robokots.kots').Kots
    from robokots.urdf_io import urdf_xml_to_model_data
    reduced = lock_urdf_joints(robot_xml, positions)
    original = Kots.from_json_data(urdf_xml_to_model_data(robot_xml), order=3)
    model = Kots.from_json_data(urdf_xml_to_model_data(reduced.xml), order=3)
    original_joints = [j for j in original.robot_.joints if j.dof]
    active_joints = sorted((j for j in model.robot_.joints if j.dof), key=lambda j: j.dof_index)
    indices = {j.name: j.dof_index for j in original_joints}
    active = [indices[j.name] for j in active_joints]
    rng = np.random.default_rng(21)
    for _ in range(3):
        q, v, a = rng.normal(size=(3, model.dof()))
        full_q = np.zeros(original.dof()); full_v = full_q.copy(); full_a = full_q.copy()
        full_q[active], full_v[active], full_a[active] = q, v, a
        for name, value in reduced.locked_joints.items():
            full_q[indices[name]] = value
        reference = original.inverse_dynamics(full_q, full_v, full_a, backend='numpy')[active]
        np.testing.assert_allclose(model.inverse_dynamics(q, v, a, backend='numpy'), reference, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('backend', ['numpy', 'rust'])
def test_reduced_model_trajectory_derivatives(robot_xml, backend):
    Kots = pytest.importorskip('robokots.kots').Kots
    from robokots.urdf_io import urdf_xml_to_model_data
    from rei import load_problem_spec_toml
    from rei.optimize_backends.kots import compile_kots_trajectory_problem
    reduced = lock_urdf_joints(robot_xml, {'finger': .2})
    model = Kots.from_json_data(urdf_xml_to_model_data(reduced.xml), order=3, backend=backend)
    spec = load_problem_spec_toml(Path(__file__).resolve().parents[1] / 'examples/spec/robokots_traj_dynamics_d12.toml')
    spec['time'].update(N=2, dt=.1)
    spec['trajectory']['num_ctrl_points'] = 6
    compiled = compile_kots_trajectory_problem(spec, model=model, kots_backend=backend)
    assert compiled.trajectory_map.q_dim == 2
    runtime = compiled.runtime
    rng = np.random.default_rng(42)
    point = rng.normal(scale=.1, size=runtime.pack.n_total)
    direction = rng.normal(size=point.size)
    runtime.pack.set(point)
    r, J = runtime.linearize()
    assert np.isfinite(r).all() and np.isfinite(J).all()
    eps = 1e-6
    runtime.pack.set(point + eps*direction)
    plus = runtime.eval_stacked_terms(weighted=True)
    runtime.pack.set(point - eps*direction)
    minus = runtime.eval_stacked_terms(weighted=True)
    np.testing.assert_allclose((plus-minus)/(2*eps), J@direction, rtol=1e-6, atol=1e-6)
