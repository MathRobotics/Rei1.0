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
PYTHONPATH=. python examples/minimize_quadratic.py
```

## まず動かす例（追加依存なし）

```bash
python examples/minimize_quadratic.py
python examples/get_state_minimal.py
python examples/json_spec_problem.py
python examples/stationarity_ioc.py
```

## Pinocchio 例

```bash
uv sync --group pinocchio
python examples/pinocchio_ik.py
python examples/pinocchio_trajectory_dynamics.py
python examples/pinocchio_trajectory_dynamics.py --plot
```

## RoboKots 例

```bash
uv sync --group kots
python examples/robokots_ik.py
python examples/robokots_trajectory_dynamics.py
python examples/robokots_trajectory_dynamics.py --plot
```

## サンプル一覧

- `minimize_quadratic.py`: `get_var` ベースの最小 NLS を JSON spec 風 dict で定義して解く
- `get_state_minimal.py`: JSON spec 風 dict から `get_state` と `build_state()` を接続する最小例
- `json_spec_problem.py`: JSON spec ファイル（`spec/basic.json`）を読み込んで解く
- `pinocchio_ik.py`: JSON spec + Pinocchio `PinocchioStateBuilder` を使った最小 IK
- `robokots_ik.py`: JSON spec + RoboKots `KotsStateBuilder` を使った最小 IK
- `pinocchio_trajectory_dynamics.py`: JSON spec + Pinocchio の軌道 + dynamics 正則化（`--plot` 対応）
- `robokots_trajectory_dynamics.py`: JSON spec + RoboKots の軌道 + dynamics 正則化（`--plot` 対応）
- `stationarity_ioc.py`: JSON spec 風 dict + Stationarity 方程式ベースの IOC 風重み推定

## JSON spec / モデルファイル

- `spec/basic.json`: 最小 JSON spec 問題
- `spec/ik_pos.json`: IK 用 JSON spec
- `spec/pinocchio_traj_dynamics.json`: Pinocchio 軌道 + dynamics 用 JSON spec
- `spec/robokots_traj_dynamics_d12.json`: RoboKots 軌道 + dynamics 用 JSON spec
- `models/planar2.urdf`: Pinocchio 用 2 自由度平面アーム
- `models/planar2.json`: RoboKots 用 2 自由度平面アーム
- `models/sample_robot.json`: RoboKots 用 3 自由度サンプルロボット
- `models/sample_robot.urdf`: `sample_robot.json` と同等の URDF 版
