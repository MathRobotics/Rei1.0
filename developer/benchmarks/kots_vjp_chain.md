# IOC VJP chaining regression in 6e25f22

The independent-request VJP path in `6e25f22` embedded each request in a
separate column of a zero tensor shaped `(trajectory_steps, dof, requests)`.
For 100 frames and 69 DoF this allocated 5.52 MB per derivative, then multiplied
the largely zero tensor by the B-spline basis. With one request per frame,
storage grew quadratically with the frame count. Reducing Python call count
had increased both memory traffic and arithmetic. The original tests checked
values, not this scaling, so they did not detect the performance regression.
That implementation also rejected matrix RHS batches that the previous
per-request path supported.

The replacement evaluates each independent contribution directly:

```
contribution[t, c, q] = basis[k[t], c] * motion_gradient[t, q]
```

This preserves request ownership, repeated time indices, and extra RHS
columns without the zero tensor. Generic dense maps use per-request transpose
products. Motion composition evaluates only requested times, while sharing
the resulting motions between fields with the same time grid. The separate
summed VJP retains its linear-sized full-grid accumulation.

Run `uv run python developer/benchmarks/kots_vjp_chain.py` to compare numerical
results and median times for the old per-frame, regressed, and current paths.
One local run (69 DoF, 20 controls, derivative orders 0–3):

| Frames | Pre-6e25f22 | 6e25f22 | Corrected |
|---:|---:|---:|---:|
| 25 | 0.383 ms | 0.592 ms | 0.144 ms |
| 100 | 2.127 ms | 11.965 ms | 0.659 ms |
| 200 | 4.412 ms | 17.900 ms | 1.255 ms |

This isolates Python trajectory chaining; it does not run the Rust VJP kernel
or the user's IOC workload. It establishes the local regression, but does not
attribute all of the reported 185 → 248 ms IOC slowdown or the 19 → 27 ms
kernel increase. Those require the original workload and profiling setup.
