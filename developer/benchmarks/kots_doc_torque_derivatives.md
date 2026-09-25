# トルク時間微分を含むDOCの計算時間

2026-09-25。既存の密行列 `gauss_newton` と、製品側の現在の
`gauss_newton_krylov` を比較した。前回の調査用の単一項合算VJPパッチは使用しない。

## 条件

- 201評価点、50制御点、5次B-spline、2秒、境界条件はnullspace削減。
- 速度・加速度ペナルティはそれぞれ `1e-4`。
  動力学の残差は表の各条件に置き換え、**各項の重みを `1e-10`** とした。
  時間微分はRoboKotsの `torque_d1` / `torque_d2` を使用する。
- 1階微分のみはモデル `order=4`、2階微分を含む場合は `order=5`。
  各比較のモデルorder・残差・初期値は両方式で同じ。
- `max_iters=30`、`tol_grad=1e-8`、履歴有効、画面出力無効。
  2関節は各方式1回ウォームアップ後3回測定、順番を交互に反転した中央値。
  **7自由度は別条件**：高負荷のため `max_iters=10`、ウォームアップなし各1回の参考値。
  7自由度の当初の30反復・反復測定はウォームアップ段階で中止し、時間比較には使用しない。
- 求解だけを計測。前処理構築とAPI計装を含み、モデル読込・コンパイル・
  nullspace構築は除外。最終勾配は計測区間外で密行列Jから独立に検証。
- RoboKots固定コミット `825dfe29cb4cae04b118298ea9a51a8d08e9de83`、Rust。
  Python 3.13.1、NumPy 2.5.3、macOS ARM64。
  `OPENBLAS_NUM_THREADS=1`、`VECLIB_MAXIMUM_THREADS=1`。測定は直列実行。

各残差は単位が異なるため、同じ数値の重みは物理的に同じ強さを意味しない。
ここでの重みは性能比較の条件であり、制御設計の推奨値ではない。

## 時間

| モデル | 動力学残差 | 密行列GN | Krylov GN | 収束時の高速化 | 状態（密 / Krylov） |
|---|---|---:|---:|---:|---|
| 2関節 | dτ/dt | 1.966秒 | 0.363秒 | 5.41倍 | converged / converged |
| 2関節 | d²τ/dt² | 21.705秒 | 13.332秒 | —（未収束を含む） | max_iters / max_iters |
| 2関節 | τ + dτ/dt + d²τ/dt² | 37.892秒 | 34.733秒 | —（未収束を含む） | max_iters / max_iters |
| 7自由度 | dτ/dt | 28.634秒 | 0.567秒 | —（未収束を含む） | max_iters / max_iters |
| 7自由度 | d²τ/dt² | 32.167秒 | 5.312秒 | —（未収束を含む） | max_iters / max_iters |
| 7自由度 | τ + dτ/dt + d²τ/dt² | 56.229秒 | 8.753秒 | —（未収束を含む） | max_iters / max_iters |

7自由度の時間は単発の参考値で、2関節と反復上限・測定方法が異なる。
未収束を含む行は、同じ解に到達するまでの時間比較ではない。
密行列GNの反復上限は採用ステップ、新方式は棄却を含む試行を数える点にも注意。

## 数値確認

| モデル・残差 | 勾配∞ノルム（密 / Krylov） | 目的関数（密 / Krylov） | 解の最大差 | 外側反復（密 / Krylov） | Krylov内側反復合計 |
|---|---:|---:|---:|---:|---:|
| 2関節・dτ/dt | 3.68e-13 / 2.62e-11 | 0.41556879 / 0.41556879 | 1.92e-12 | 4 / 4 | 8 |
| 2関節・d²τ/dt² | 96.1 / 588 | 36.863347 / 20.458456 | 2.26 | 30 / 30 | 735 |
| 2関節・τ + dτ/dt + d²τ/dt² | 96.1 / 276 | 36.871077 / 20.229739 | 2.09 | 30 / 30 | 1262 |
| 7自由度・dτ/dt | 1.52e+03 / 1.35e+06 | 1868.5324 / 10213275 | 3.3 | 10 / 10 | 24 |
| 7自由度・d²τ/dt² | 1.93e+06 / 3.45e+06 | 5272546.6 / 1146079.5 | 7.26 | 10 / 10 | 358 |
| 7自由度・τ + dτ/dt + d²τ/dt² | 1.67e+06 / 3.21e+06 | 6580097.7 / 1135110.4 | 6.45 | 10 / 10 | 359 |

前回の2関節・41評価点・10制御点では両時間微分とも収束したが、
今回の201評価点・50制御点へ、その結果をそのまま一般化できない。

## 再現

```bash
PYTHONPATH=/path/to/RoboKots:. OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python developer/benchmarks/kots_doc_operator.py \
  --model planar2.urdf --steps 201 --controls 50 --max-iters 30 \
  --repeat 3 --warmup 1 --torque-fields torque_d1 \
  --solvers gauss_newton gauss_newton_krylov --output /tmp/torque-d1.json
```

2階微分のみは `--torque-fields torque_d2`、3種類同時は
`--torque-fields torque torque_d1 torque_d2`。
7自由度は `--model 7_dof_arm.urdf --max-iters 10 --warmup 0 --repeat 1` に変更し、
各ソルバーを別プロセスで1回ずつ実行する。
[全測定値・API回数・解・実行環境](kots_doc_torque_derivative_results.json)。
