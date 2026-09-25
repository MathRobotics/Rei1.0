"""Isolate Python trajectory chaining (no Rust kernel).

uv run python developer/benchmarks/kots_vjp_chain.py
Compare pre-6e25f22, the regressed zero-RHS algorithm, and the current path.
"""
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from rei.backends.state.robotics.kots import KotsTrajectoryStateBuilder
from rei.core.state_schema import make_key
from rei.core.trajectory import TrajectoryMap


def measure(fn):
    fn()
    times = []
    for _ in range(5):
        start = perf_counter()
        fn()
        times.append(1000 * (perf_counter() - start))
    return median(times)


def main():
    for n in (25, 100, 200):
        dof, order = 69, 4
        maps = TrajectoryMap.from_bspline_derivatives(steps=n, q_dim=dof,
            degree=3, num_ctrl_points=20, max_derivative_order=order - 1)
        builder = KotsTrajectoryStateBuilder(
            SimpleNamespace(dof=lambda: dof, order=lambda: order),
            trajectory_map=maps[0], trajectory_derivative_maps=dict(enumerate(maps)))
        grads = np.random.default_rng(0).normal(size=(n, dof * order))
        group = [(k, make_key(k=k, owner_type="total_joint", owner_name="robot",
            dtype="dynamics", field="torque"), np.zeros(dof), None) for k in range(n)]

        def legacy():
            return np.stack([builder._trajectory_motion_gradient_transpose_at(
                k=k, motion_grad=g) for k, g in enumerate(grads)])

        def regressed():
            values = np.zeros((maps[0].p_dim, n))
            for r, m in enumerate(maps):
                rhs = np.zeros((n, dof, n))
                rhs[np.arange(n), :, np.arange(n)] = grads[:, r::order]
                values += m.apply_transpose(rhs.reshape(n * dof, n))
            return values.T

        def current():
            out = [None] * n
            builder._chain_batched_param_vjp_group(out=out, group=group,
                motions=[], motion_grads=grads)
            return np.stack(out)

        reference = legacy()
        np.testing.assert_allclose(regressed(), reference, atol=1e-9)
        np.testing.assert_allclose(current(), reference, atol=1e-9)
        print(f"frames={n}, dof={dof}, controls=20, derivatives=0..3")
        for name, fn in (("legacy", legacy), ("6e25f22", regressed), ("current", current)):
            print(f"  {name:9s}: {measure(fn):.3f} ms")
        print(f"  removed zero RHS: {n * dof * n * 8 / 1e6:.2f} MB per derivative")


if __name__ == "__main__":
    main()
