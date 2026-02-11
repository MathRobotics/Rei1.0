# B-spline Debug Utilities

`TrajectoryMap.from_bspline` のデバッグ用スクリプトです。  
`[trajectory]` を含む TOML を読み、基底関数とサンプル軌道を PNG 出力します。

## 実行例

```bash
PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp \
  .venv/bin/python examples/debug/bspline/debug_bspline_traj.py \
  --dsl examples/dsl/kots_traj_pos.toml \
  --output examples/debug/bspline/out/kots_traj_pos_debug.png \
  --check-jacobian
```

微分（解析写像）まで検証する場合:

```bash
PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp \
  .venv/bin/python examples/debug/bspline/debug_bspline_traj.py \
  --dsl examples/dsl/kots_traj_pos_dq.toml \
  --output examples/debug/bspline/out/kots_traj_pos_dq_debug.png \
  --check-jacobian \
  --check-derivative \
  --derivative-order 1 \
  --derivative-wrt time
```

## 主なオプション

- `--steps`: `trajectory.steps` / `time.N+1` を上書き
- `--q-dim`: `trajectory.q_dim` を上書き
- `--show`: 画像保存に加えてウィンドウ表示
- `--check-jacobian`: 有限差分で `dq/dp` を検証
- `--check-derivative`: 微分軌道写像の生成・可視化・有限差分検証
- `--derivative-order`: 微分次数（`--check-derivative` 時）
- `--derivative-wrt`: 微分軸（`u` または `time`）
- `--dt`: `--derivative-wrt=time` 時の `dt` 上書き
- `--derivative-output`: 微分図の出力先 PNG

## 出力

- 軌道（`q_dim>=2` の場合は `q[0]-q[1]` 平面）
- 各次元の時系列 `q(k)`
- B-spline 基底関数 `N_i(u)`
- 基底行列ヒートマップ（`k x control index`）
- 微分デバッグ時: 微分軌道と微分基底の図、端末ログに微分基底行和と Jacobian 誤差
- 端末ログに基底行和と Jacobian 誤差
