# Examples

`examples/` は `rei` の canonical API を確認するための実行サンプル集です。  
コマンドはリポジトリルートで実行してください。

## 事前準備

推奨:

```bash
python -m pip install -e .
```

インストールせずに実行する場合:

```bash
PYTHONPATH=. python examples/01_minimize_quadratic.py
```

## まず動かす例（追加依存なし）

```bash
python examples/01_minimize_quadratic.py
python examples/02_get_state_minimal.py
python examples/03_toml_problem.py
python examples/08_camera_calibration.py
python examples/10_stationarity_ioc.py
```

## Pinocchio 例

```bash
uv sync --group pinocchio
python examples/04_pinocchio_ik.py
python examples/06_pinocchio_trajectory_dynamics.py
python examples/06_pinocchio_trajectory_dynamics.py --plot
```

## RoboKots 例

```bash
uv sync --group kots
python examples/05_robokots_ik.py
python examples/07_robokots_trajectory_dynamics.py
python examples/07_robokots_trajectory_dynamics.py --plot
python examples/09_kots_vision_composite.py
python examples/11_forward_then_inverse_ioc.py
python examples/11_forward_then_inverse_ioc.py --plot
```

## サンプル一覧

- `01_minimize_quadratic.py`: `get_var` ベースの最小 NLS を DSL(dict) で定義して解く
- `02_get_state_minimal.py`: `get_state` と `build_state()` の最小接続例
- `03_toml_problem.py`: TOML 問題（`dsl/basic.toml`）を読み込んで解く
- `04_pinocchio_ik.py`: Pinocchio `PinocchioStateBuilder` を使った最小 IK
- `05_robokots_ik.py`: RoboKots `KotsStateBuilder` を使った最小 IK
- `06_pinocchio_trajectory_dynamics.py`: Pinocchio の軌道 + dynamics 正則化（`--plot` 対応）
- `07_robokots_trajectory_dynamics.py`: RoboKots の軌道 + dynamics(`torque`, `torque_d1`, `torque_d2`) 正則化（`--plot` 対応）
- `08_camera_calibration.py`: `dtype="vision"` の camera calibration（`--model linear|pinhole`）
- `09_kots_vision_composite.py`: `CompositeStateBuilder` でロボット状態とカメラ状態を合成して同時最適化
- `10_stationarity_ioc.py`: Stationarity 方程式ベースの IOC 風重み推定
- `11_forward_then_inverse_ioc.py`: RoboKots 軌道を順最適化し、解から重みを逆推定（`--plot` 対応）

## DSL / モデルファイル

- `dsl/basic.toml`: 最小 TOML 問題
- `dsl/ik_pos.toml`: IK 用 TOML
- `dsl/pinocchio_traj_dynamics.toml`: Pinocchio 軌道 + dynamics 用 TOML
- `dsl/robokots_traj_dynamics_d12.toml`: RoboKots 軌道 + dynamics(`torque`, `torque_d1`, `torque_d2`) 用 TOML
- `dsl/robokots_traj_dynamics_d12_per_joint.toml`: 上記 d12 の per-joint 版（`expr.type="component"` 利用）
- `models/planar2.urdf`: Pinocchio 用 2 自由度平面アーム
- `models/planar2.json`: RoboKots 用 2 自由度平面アーム
- `models/sample_robot.json`: RoboKots 用 3 自由度サンプルロボット
- `models/sample_robot.urdf`: `sample_robot.json` と同等の URDF 版
- `models/7_dof_arm.json`: RoboKots 用 7 自由度アーム
- `models/7_dof_arm.urdf`: `7_dof_arm.json` と同等の URDF 版
