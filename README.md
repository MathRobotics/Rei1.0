# EiOpt

`eiopt` は `RoboKots/robokots/inward` を元にした、backend 非依存の最小 NLS(非線形最小二乗)ユーティリティです。

## Backend との接続点

backend(kots / pinocchio 等) と `eiopt` を繋ぐ唯一の接続点は `StateCache` が呼ぶ `build_state()` です。

- `build_state(x_all, *, pack=None, time=None, required=None) -> dict[StateKey, Any]`
  - `x_all`: 全決定変数ベクトル（`VariablePack` の順）
  - `pack`: `VariablePack`（受け取れる backend 実装なら渡されます）
  - `time`: `TimeGrid`
- `required`: 今回必要な `StateKey` の集合（これだけ計算すると速い）。`None` は「全部計算」。

## Backend の最小要件

backend 側で実装すべき最小要件は `build_state()` だけです。

- `build_state()` は **純粋関数に近い振る舞い**（同じ入力 → 同じ出力）であること
- `required` が渡された場合は **そのキーだけを計算して返す**こと
- 返り値は `dict[StateKey, Any]` で、**要求された key が必ず含まれる**こと
- 値は `numpy.ndarray` などの数値配列で、`Expr` が期待する shape に整形済みであること
- `pack`, `time` は受け取れれば十分（未使用でも OK）

`Expr.deps()` が `StateKey` を返し、`StateCache.update_if_needed(..., required=required)` がそれを使って
必要最小限の状態計算を行います。

## Problem 定義（TOML）

最適化問題は TOML で定義します（JSON は使用しません）。読み込みは `load_problem_toml()` を使います。

```python
from eiopt import compile_problem, load_problem_toml

dsl = load_problem_toml("examples/dsl/basic.toml")
runtime = compile_problem(dsl, build_state=build_state)
```

サンプルは `examples/dsl/basic.toml` を参照してください。

## Examples の実行

examples は `eiopt` を import するので、以下のどちらかで実行してください。

```bash
# (推奨) インストールして実行
python -m pip install -e .
python examples/main.py
```

```bash
# インストールせずに実行（repo 直下から）
PYTHONPATH=. python examples/main.py
```

Pinocchio 例（ロボティクス Pinocchio bindings が必要）:

```bash
PYTHONPATH=. python examples/main_pinocchio.py
```

引数つきの CLI 版:

```bash
PYTHONPATH=. python examples/cli/main_pinocchio.py --help
```

RoboKots 例（`robokots` と互換 `mathrobo` が必要）:

```bash
PYTHONPATH=. python examples/main_robokots.py
```

RoboKots 軌道最適化例（線形な `p -> q(k)` マップ）:

```bash
PYTHONPATH=. python examples/main_robokots_traj.py
```

※ この例は `matplotlib` を使って軌道を描画します。

### solve_gauss_newton の返り値

`solve_gauss_newton()` は最適化後の全決定変数ベクトル（`VariablePack` の順）も返します。

```python
x_star, cost, iters, rnorm, dxnorm, converged = solve_gauss_newton(runtime)
```

### 最適化後の Expr 値レポート

最適化後に、目的関数(terms)に含まれる residual と、DSLで名前が付いた Expr（例: `ee_pos`, `target_pos`）の値を
簡潔に確認したい場合は `format_solve_report()` を使えます。`x0` を渡すと初期値と最適化後の決定変数も併せて表示します。

```python
from eiopt import format_solve_report, solve_gauss_newton

x0 = runtime.pack.get().copy()
x_star, *_ = solve_gauss_newton(runtime)
print(format_solve_report(runtime, x0=x0, x_star=x_star))
```

### 最小テンプレート

```toml
[[variables]]
name = "q"
dim = 2
init = [0.0, 0.0]

[[terms]]
[terms.expr]
type = "get_state"

[terms.expr.key]
k = 0
owner_type = "link"
owner_name = "ee"
dtype = "kinematics"
field = "pos"

[terms.expr.jac]
var = "q"

[terms.cost]
type = "l2"
```

## StateKey の命名（推奨スキーマ）

`StateKey` は「何を計算して返すか」を表すキーです（`robokots/core/state.py` の考え方に寄せています）。

- `owner.owner_type`: `"link" | "joint" | ...`
- `owner.owner_name`: link/joint 名
- `k`: 時刻インデックス（`TimeGrid` の `k`）
- `dtype`: 大分類（推奨: `"kinematics"` / `"joint"` / `"dynamics"`）
- `field`: 量の名前（例: `"pos"`, `"rot"`, `"vel"`, `"acc"`, `"momentum"`, `"force"`, `"torque"`）
- `frame`: 座標系（推奨: `"world"` / `"local"`）
- `rel_frame`: 相対量が必要なときの相手フレーム（任意）

### Jacobian の field 命名

ヤコビアンは `"{field}_J_{var}"` を推奨します（例: `pos_J_q`）。

Python 側は `eiopt.core.state_schema.jac_field()` が使えます。

## 決定変数の取得（get_var）

`q` などの **決定変数そのもの** は backend の状態計算ではなく、`get_var` Expr で `VariablePack` から直接読み出すのを推奨します。
この場合 `build_state()` は kinematics/dynamics など「backend で計算が必要な状態」だけを担当できます。

```toml
[terms.expr]
type = "get_var"
var = "q" # 省略可（変数が1つなら自動）
```

軌道最適化などで `q` を時系列にスタックしている場合は、`k` を指定すると `q(k)` を返し、ヤコビアンは選択行列になります。

## RoboKots の軌道パラメータ最適化（TrajectoryMap / B-spline）

RoboKots 向けには、決定変数を軌道パラメータ `p` とし、`TrajectoryMap` を介して
`q(k)` と `dq/dp` を与える `KotsTrajectoryStateBuilder` が使えます。

- `StateKey.field="pos_J_p"` のようなヤコビアン要求に対して、内部で
  `J_state_p = J_state_q @ (dq/dp)` を適用します。
- `StateKey.k`（時刻インデックス）ごとに `q(k)` を構成して kinematics を更新します。

軌道近似は DSL の `[trajectory]` で指定できます。

```toml
[trajectory]
type = "bspline" # or "linear"
var = "p"
degree = 3
num_ctrl_points = 6
```

最小コード例（DSL から生成）:

```python
from eiopt import compile_problem
from eiopt.backends.kots import KotsTrajectoryStateBuilder

builder = KotsTrajectoryStateBuilder.from_dsl(kots, data, dsl=dsl)
runtime = compile_problem(dsl, build_state=builder.build_state)
```

DSL 側では `get_state.jac.var = "p"` を指定します。実例は
`examples/dsl/kots_traj_pos.toml` と `examples/main_robokots_traj.py` を参照してください。
`kots_traj_pos.toml` の `time.N` を増やすとステップ数を増やせます
（この example では `stack.range` と `target_pos_traj` が自動でステップ数に同期されます）。

`type = "linear"` の場合は `trajectory.linear.A`（または `trajectory.A`）と
必要に応じて `trajectory.linear.b` を指定します。

## 最小標準セット（pos/rot/frame + (optional) q）

backend から `eiopt` に提供する “標準の最小セット” として、まずは以下に絞るのが扱いやすいです。
（誤差の定義は後から Expr/Residual として設計できるように、ここでは **生の状態** を揃えます）

- `dtype="kinematics"`
  - `frame="world"`（ひとまず world 固定。DSL では省略可で、デフォルト world 扱い）
  - `field="pos"`: `(3,)` 位置ベクトル
  - `field="rot"`: `(9,)` 回転行列 `(3,3)` の row-major flatten
  - `field="frame"`: `(12,) = [pos(3), rot_flat(9)]`
  - `field="{field}_J_q"`: それぞれ上と同じ次元のヤコビアン（例: `pos_J_q` は `(3,nq)`）

- `dtype="joint"`（関節角度）
  - 推奨 owner: `owner_type="total_joint"`, `owner_name="robot"`
  - `field="q"`: `(nq,)`
  - `field="q_J_q"`: `(nq,nq)`（軌道最適化などで `q` を時系列にスタックした場合は選択行列になります）

ただし `q` は決定変数として `get_var` から直接読めるので、backend が `dtype="joint"` を返さなくても問題ないことが多いです。
既存の DSL などで `get_state(dtype="joint", field="q")` を使っている場合は、`get_var` に置き換えてください。

この “値とヤコビアンの次元が一致する” ルールにしておくと、後から `se3_error` などの誤差 Expr を作る際も、
必要な raw 情報を `StateCache` から取り出して組み合わせるだけで済みます。
