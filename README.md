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

`main_kots_traj.py` でも同じ実行ができます（ソルバ選択用の薄いラッパーです）:

```bash
PYTHONPATH=. python examples/main_kots_traj.py
```

※ この例は `matplotlib` を使って軌道を描画します。

### Kots 軌道問題の高レベルビルダー

`eiopt.backends.kots.compile_kots_trajectory_problem()` を使うと、
軌道マップ生成・導関数マップ生成・`p` 変数次元チェック・`KotsTrajectoryStateBuilder` 構築・`compile_problem()` までを一括で実行できます。

```python
from eiopt.backends.kots import compile_kots_trajectory_problem

compiled = compile_kots_trajectory_problem(
    dsl,
    model=kots,
    data=kots.state_dict_,
)

runtime = compiled.runtime
traj_map = compiled.trajectory_map
```

### solve_gauss_newton の返り値

`solve_gauss_newton()` は最適化後の全決定変数ベクトル（`VariablePack` の順）も返します。

```python
x_star, cost, iters, rnorm, dxnorm, converged = solve_gauss_newton(runtime)
```

### ソルバ切替（gauss_newton / scipy / cyipopt）

`solve_runtime()` でソルバを切り替えられます。`scipy` と `cyipopt` は optional dependency です。

```python
from eiopt import solve_runtime

x_star, cost, iters, rnorm, dxnorm, converged = solve_runtime(
    runtime,
    solver="scipy_minimize",  # "gauss_newton" | "scipy_minimize" | "cyipopt"
    max_iters=1000,
    scipy_method="L-BFGS-B",
)
```

### 実行時に特定 term の重みだけ変更

`compile_problem()` 後に、term index または `expr.name` を指定して重みを変更できます。
（対象コストは `scalar_weight` / `diag_weight` など `set_weight()` を持つもの）

```python
runtime.set_cost_weight("torque_traj_regularization", 1e-4)  # expr.name で指定
# runtime.set_cost_weight(7, 1e-4)  # index 指定も可
```

### term 属性でフィルタ（拘束条件の識別など）

`[[terms]]` には `expr` / `cost` 以外の項目を属性として持たせられます。
`[terms.attrs]` も併用でき、どちらも `runtime.find_term_indices()` で検索できます。
等式/不等式の区別には `constraint.kind = "eq" | "ineq"` が使えます。

```toml
[[terms]]
is_constraint = true

[terms.attrs]
group = "joint_limit"

[terms.constraint]
kind = "ineq" # "eq" | "ineq"

[terms.expr]
type = "hinge"
```

```python
constraint_idxs = runtime.find_term_indices(attr="is_constraint", value=True)
runtime.set_cost_weight_by_attr(attr="group", value="joint_limit", w=1e-1)

eq_constraint_idxs = runtime.find_constraint_term_indices(kind="eq")
ineq_constraint_idxs = runtime.find_constraint_term_indices(kind="ineq")
runtime.set_cost_weight_by_constraint(kind="ineq", w=1e-1)
```

### term別線形化と IOC 行列

termごとの線形化結果は `runtime.linearize_terms()` で取得できます。`weighted=False` を使うと未加重の `r_i, J_i` を取り出せます。

```python
terms = runtime.linearize_terms(weighted=False)
for t in terms:
    print(t.term_index, t.name, t.residual.shape, t.jacobian.shape, t.attrs)
```

IOC用に `A = [J_0^T r_0, ..., J_n^T r_n]` を作るヘルパーも使えます。

```python
from eiopt import build_term_gradient_matrix, estimate_weights_simplex

A, term_indices = build_term_gradient_matrix(runtime, weighted=False)
w_hat = estimate_weights_simplex(A)  # w>=0, sum(w)=1
```

### state の時系列一括取得

`runtime.collect_state_traj()` で `StateKey` ループを書かずに時系列をまとめて取得できます。

```python
ee_traj = runtime.collect_state_traj(
    owner_type="link",
    owner_name="ee",
    dtype="kinematics",
    field="pos",
    ks=range(11),
    expected_dim=3,
)
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

`format_solve_report()` は既定で `||J^T r||`, `rank(J)`, `svd(J)`, active term も表示します。
不要なら `include_diagnostics=False` を指定できます。

名前付き Expr の値をコード側で直接使う場合は `get_named_expr_value()` が使えます。

```python
from eiopt import get_named_expr_value

ee_traj = get_named_expr_value(runtime, name="ee_pos_traj")
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
- `dtype`: 大分類（推奨: `"kinematics"` / `"dynamics"`）
- `field`: 量の名前（例: `"pos"`, `"rot"`, `"vel"`, `"acc"`, `"momentum"`, `"force"`, `"torque"`）
- `frame`: 座標系（推奨: `"world"` / `"local"`）
- `rel_frame`: 相対量が必要なときの相手フレーム（任意）

### Jacobian の field 命名

ヤコビアンは `"{field}_J_{var}"` を推奨します（例: `pos_J_q`）。

Python 側は `eiopt.core.state_schema.jac_field()` が使えます。

`get_state` は単一 Jacobian (`expr.jac`) に加え、複数 Jacobian (`expr.jacs`) をサポートします。
複数変数がある場合は `var` を明示してください。

```toml
[terms.expr]
type = "get_state"

[terms.expr.key]
k = 0
owner_type = "link"
owner_name = "ee"
dtype = "kinematics"
field = "pos"

[[terms.expr.jacs]]
var = "p"

[[terms.expr.jacs]]
var = "q"
```

`field` は canonical 名のみ受け付けます。`torque` / `momentum` / `force` と、
トルク導関数 `torque_dN`（例: `torque_d1`, `torque_d2`）を使ってください。
`tau` / `h` / `wrench` / `dtau` / `tau_diff` / `torque_rate` などの旧表記は受け付けません。

## 決定変数の取得（get_var）

`q` などの **決定変数そのもの** は backend の状態計算ではなく、`get_var` Expr で `VariablePack` から直接読み出すのを推奨します。
この場合 `build_state()` は kinematics/dynamics など「backend で計算が必要な状態」だけを担当できます。

```toml
[terms.expr]
type = "get_var"
var = "q" # 変数が1つのときのみ省略可
```

軌道最適化などで `q` を時系列にスタックしている場合は、`k` を指定すると `q(k)` を返し、ヤコビアンは選択行列になります。

## 軌道スタックの直接取得（get_traj_var）

軌道パラメータ `p` を最適化している場合は、`get_traj_var` で
`TrajectoryMap` の `q_traj = A @ p + b`（`q(0),...,q(N)` のスタック）を
直接 residual として使えます。

- `expr.trajectory` を省略すると、トップレベルの `[trajectory]` を使います。
- `type="bspline"` で `q_dim` 未指定の場合、`var.dim / num_ctrl_points` から推定します（割り切れる場合）。

時刻差分（`q(k)-q(k-1)`）による平滑化の例:

```toml
[[terms]]
[terms.expr]
type = "time_diff"
name = "traj_smooth"
segment_dim = 2
wrt = "time" # "index"(既定) | "time"

[terms.expr.base]
type = "get_traj_var"
name = "q_traj"
var = "p"

[terms.cost]
type = "scalar_weight"
w = 1e-2
```

境界や関節上限のような時系列定数は `const_repeat` で簡潔に書けます。

```toml
[terms.expr.base.b]
type = "const_repeat"
var = "p"
value = [3.141592653589793, 3.141592653589793] # repeats は既定で time.N+1
```

この項は関節軌道 `q` を滑らかにしますが、`ee_pos` の軌道を直接滑らかにしたい場合は
`get_state(pos)` を `stack` して `time_diff`（必要なら2回）をかけた項も追加してください。

## RoboKots の軌道パラメータ最適化（TrajectoryMap / B-spline）

RoboKots 向けには、決定変数を軌道パラメータ `p` とし、`TrajectoryMap` を介して
`q(k)` と `dq/dp` を与える `KotsTrajectoryStateBuilder` が使えます。

- `StateKey.field="pos_J_p"` のようなヤコビアン要求に対して、内部で
  `J_state_p = J_state_state @ (dstate/dp)` を適用します。
- `StateKey.k`（時刻インデックス）ごとに `q(k)` を構成して kinematics を更新します。
- `KotsTrajectoryStateBuilder` は既定で `torque/torque_d1/momentum/force` を登録します。
  `compile_kots_trajectory_problem(...)` では `dynamics_fields=None`（既定）時に
  DSL が要求する dynamics field を自動検出して登録します。
  必要に応じて `dynamics_fields=(...)` を渡して対象 field を明示指定することもできます。
  Kots 側は `state_info/jacobian` が返せる field をそのまま登録できます。
  `compile_kots_trajectory_problem(...)` は DSL が要求する dynamics field と
  `dynamics_fields` 登録の不一致を検出し、線形化前に `ValueError` を返します。
  （RoboKots への問い合わせ時は `torque_dN -> torque_diffN` へ自動変換します）
  `torque_dN` を使う場合は RoboKots モデル次数が `order >= N+3` 必須です（例: `torque_d1` は `order >= 4`）。
  Pinocchio 側は標準で `torque`, `momentum`, `force` をサポートし、追加分は
  `dynamics_custom_handlers` で登録できます。
- `trajectory_derivative_maps` を渡すと、モデル次数に応じた `q, dq, ddq...` を内部状態に展開できます。

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
from eiopt.dsl import build_trajectory_map, build_trajectory_maps_with_derivatives, default_steps_from_time

traj = build_trajectory_map(
    dsl["trajectory"],
    default_steps=default_steps_from_time(dsl),
    default_q_dim=int(kots.dof()),
)
traj_maps = build_trajectory_maps_with_derivatives(
    dsl["trajectory"],
    max_derivative_order=max(0, int(kots.order()) - 1),
    derivative_wrt="time",
    default_steps=traj.steps,
    default_q_dim=traj.q_dim,
    default_dt=float(dsl["time"]["dt"]),
)
builder = KotsTrajectoryStateBuilder(
    kots,
    data,
    trajectory_map=traj,
    trajectory_derivative_maps={i: m for i, m in enumerate(traj_maps)},
    p_var="p",
    dynamics_fields=("torque", "torque_d1"),  # 例: 明示指定する場合
)
runtime = compile_problem(dsl, build_state=builder.build_state)
```

DSL 側では `get_state.jac.var = "p"` を指定します。実例は
`examples/dsl/kots_traj_pos.toml` と `examples/main_robokots_traj.py` を参照してください。
`kots_traj_pos.toml` の `time.N` を増やすとステップ数を増やせます
（始端・終端タスクを使う場合は、`get_state.key.k` の終端インデックスも合わせて更新してください）。

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

- `dtype="coord"`（関節角度）
  - 推奨 owner: `owner_type="total_joint"`, `owner_name="robot"`
  - `field="q"`: `(nq,)`
  - `field="q_J_q"`: `(nq,nq)`（軌道最適化などで `q` を時系列にスタックした場合は選択行列になります）

ただし `q` は決定変数として `get_var` から直接読めるので、backend が `dtype="coord"` を返さなくても問題ないことが多いです。
`dtype="joint"` は deprecated alias としてエラーになります。`get_state(dtype="coord", field="q")` を使ってください。

この “値とヤコビアンの次元が一致する” ルールにしておくと、後から `se3_error` などの誤差 Expr を作る際も、
必要な raw 情報を `StateCache` から取り出して組み合わせるだけで済みます。
