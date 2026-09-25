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
python examples/toml_spec_problem.py
python examples/stationarity_ioc.py
```

## Pinocchio 例

```bash
uv sync --group pinocchio
python examples/pinocchio_ik.py
python examples/pinocchio_trajectory_dynamics.py
python examples/pinocchio_trajectory_dynamics.py --plot
python examples/compare_robotics_backends.py --backend pinocchio
```

## RoboKots 例

```bash
uv sync --group kots
python examples/robokots_ik.py
python examples/robokots_trajectory_dynamics.py
python examples/robokots_trajectory_dynamics.py --plot
python examples/robokots_doc_noise_ioc.py --noise-std 0.001 --seed 42
python examples/compare_robotics_backends.py --backend robokots
```

## サンプル一覧

`robokots_doc_noise_ioc.py` は DOC → 各時刻の関節角への観測ノイズ付加 →
同じB-splineへの再フィット → IOC を実行し、ノイズ有無で推定結果を比較します。
`--noise-std` はrad単位、`--seed` は乱数seedです。`--output /tmp/demo.npz`
で元軌道・観測値・フィット後軌道・推定重みを保存できます。
既存IOCの制約乗数未対応という制限は、この例にも適用されます。

RoboKots / Pinocchio の軌道最適化例は LM が既定です。
反復を実行中に自動表示し（`--quiet` で抑制）、`--show-trials` で試行詳細も表示します。
`--history history.jsonl` で反復を `history.jsonl`、詳細を `history.lm_trials.jsonl` に分けて保存します。
詳細の保存先は `--trial-history trials.jsonl` で変更できます。
従来方式は `--solver gauss_newton` で再現し、`--show-line-search` と
`--line-search-history line_search.jsonl` はこの方式に限って使用できます。
`--history-vectors` で変数と
`Jᵀr` の全成分も含めます。保存先は新しいファイル名を指定してください。
詳細は [ソルバの履歴](../docs/solver-history.md) を参照してください。

- `minimize_quadratic.py`: `get_var` ベースの最小 NLS を Problem Spec 風 dict で定義して解く
- `get_state_minimal.py`: Problem Spec 風 dict から `get_state` と `build_state()` を接続する最小例
- `toml_spec_problem.py`: 標準 TOML spec ファイル（`spec/basic.toml`）を読み込んで解く
- `pinocchio_ik.py`: Problem Spec + Pinocchio `PinocchioStateBuilder` を使った最小 IK
- `robokots_ik.py`: Problem Spec + RoboKots `KotsStateBuilder` を使った最小 IK
- `pinocchio_trajectory_dynamics.py`: TOML spec + Pinocchio の軌道 + dynamics 正則化（`--plot` 対応）
- `robokots_trajectory_dynamics.py`: TOML spec + RoboKots の軌道 + dynamics 正則化（`--plot` 対応）
- `compare_robotics_backends.py`: Pinocchio / RoboKots の trajectory compile・solve・IOC 推定時間を比較
- `stationarity_ioc.py`: Problem Spec 風 dict + Stationarity 方程式ベースの IOC 風重み推定

## Problem Spec / モデルファイル

TOML spec の書き方は [DSL ガイド](../docs/dsl.md#problem-spec-の書き方) を参照してください。
軌道の例は、境界条件・全時刻の関節角制限・全時刻の正則化の順に並べています。
両バックエンドの spec は同じ条件で、`at` が指定点、`over = "all"` が全評価点を表します。

- `spec/basic.toml`: 標準 TOML spec の最小問題
- `spec/ik_pos.toml`: IK 用 TOML spec
- `spec/pinocchio_traj_dynamics.toml`: Pinocchio 軌道 + dynamics 用 TOML spec
- `spec/robokots_traj_dynamics_d12.toml`: RoboKots 軌道 + dynamics 用 TOML spec
- `models/planar2.urdf`: Pinocchio 用 2 自由度平面アーム
- `models/planar2.json`: RoboKots 用 2 自由度平面アーム
- `models/sample_robot.json`: RoboKots 用 3 自由度サンプルロボット
- `models/sample_robot.urdf`: `sample_robot.json` と同等の URDF 版
- `models/7_dof_arm.{json,urdf}`: RoboKots の JSON/URDF parity 用 7 自由度アーム
