# Examples

現在のライブラリ構成に合わせた最小サンプルです。

## 実行方法

```bash
PYTHONPATH=. python examples/01_minimize_quadratic.py
PYTHONPATH=. python examples/02_get_state_minimal.py
PYTHONPATH=. python examples/03_toml_problem.py
PYTHONPATH=. python examples/08_camera_calibration.py
```

Pinocchio / RoboKots 例は追加依存が必要です。

```bash
uv sync --group pinocchio
PYTHONPATH=. python examples/04_pinocchio_ik.py
PYTHONPATH=. python examples/06_pinocchio_trajectory_dynamics.py
PYTHONPATH=. python examples/06_pinocchio_trajectory_dynamics.py --plot
```

```bash
uv sync --group kots
PYTHONPATH=. python examples/05_robokots_ik.py
PYTHONPATH=. python examples/07_robokots_trajectory_dynamics.py
PYTHONPATH=. python examples/07_robokots_trajectory_dynamics.py --plot
PYTHONPATH=. python examples/09_kots_vision_composite.py
```

## ファイル一覧

- `01_minimize_quadratic.py`
  - `get_var` ベースの最小 NLS 問題を DSL(dict) で組み立てて解く
- `02_get_state_minimal.py`
  - `get_state` と `build_state()` の最小連携（backend 接続点）
- `03_toml_problem.py`
  - TOML (`examples/dsl/basic.toml`) を読み込んで解く
- `04_pinocchio_ik.py`
  - Pinocchio の `PinocchioStateBuilder` を使った最小 IK（`get_state(pos)`）
- `05_robokots_ik.py`
  - RoboKots の `KotsStateBuilder` を使った最小 IK（`get_state(pos)`）
- `06_pinocchio_trajectory_dynamics.py`
  - Pinocchio の trajectory 最適化（決定変数 `p`）に dynamics(`torque`) 正則化を加えた例
  - `--plot` で DSL の `term.attrs.plot` を使って時系列を描画
- `07_robokots_trajectory_dynamics.py`
  - RoboKots の trajectory 最適化（決定変数 `p`）に dynamics(`torque`, `torque_d1`, `torque_d2`) 正則化を加えた例
  - `--plot` で DSL の `term.attrs.plot` を使って時系列を描画
- `08_camera_calibration.py`
  - `dtype="vision"` の camera calibration 例（`--model linear|pinhole`）
  - `compile_camera_calibration_problem()` と `CameraCalibrationStateProvider` の組み合わせ例
- `09_kots_vision_composite.py`
  - `CompositeStateBuilder` で `KotsStateBuilder` と `CameraCalibrationStateProvider` を合成する統合例
  - ロボット位置誤差 (`dtype="kinematics"`) とカメラ再投影誤差 (`dtype="vision"`) を同時最適化
- `dsl/basic.toml`
  - 最小 TOML 問題定義
- `dsl/ik_pos.toml`
  - backend 連携 IK 用 TOML
- `dsl/pinocchio_traj_dynamics.toml`
  - Pinocchio の trajectory + dynamics(`torque`) 最適化用 TOML
- `dsl/robokots_traj_dynamics_d12.toml`
  - RoboKots の trajectory + dynamics(`torque`, `torque_d1`, `torque_d2`) 最適化用 TOML
- `dsl/robokots_traj_dynamics_d12_per_joint.toml`
  - `dsl/robokots_traj_dynamics_d12.toml` の per-joint 版（`expr.type="component"` で j0/j1 に分解）
  - `terms.attrs.joint_component` を付与した可視化・解析向け TOML
- `models/planar2.urdf`
  - Pinocchio 用の2自由度平面アーム
- `models/planar2.json`
  - RoboKots 用の2自由度平面アーム
