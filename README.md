# EiOpt

`eiopt` は `RoboKots/robokots/inward` を元にした、capability 指向の数値問題ツールキットです。

## Layered Namespaces

現在の canonical namespace は以下です。

- `eiopt.optimize`: 最適化責務の API 入口（`compile_nls_problem`, `solve` など）
- `eiopt.equations`: 方程式系 problem capability の入口（`RuntimeStationaritySource` など）
- `eiopt.flow`: フロー/制約/射影 problem capability の入口
- `eiopt.backends.state`: backend の state builder 入口
- `eiopt.optimize_backends`: optimize と backend を接続する compile helper

旧 namespace / alias (`eiopt.dsl`, `eiopt.model`, `eiopt.solvers`, `eiopt.expr` など) は削除済みです。

## Capability Adapters

Problem は capability 単位で扱えます。既存 runtime から能力を取り出す入口は以下です。

```python
from eiopt import (
    as_linear_equation_problem,
    as_constraint_problem,
    as_project_problem,
)

eq_problem = as_linear_equation_problem(runtime)
constraint_problem = as_constraint_problem(runtime, kind="eq")
project_problem = as_project_problem(runtime)  # 既定は恒等射影
```

## 削除済み Import Path

以下の旧フラット import path は削除済みです。canonical path を使用してください。

- `eiopt.backends.state.template` -> `eiopt.backends.state.dispatch.template`
- `eiopt.backends.state.composite` -> `eiopt.backends.state.dispatch.composite`
- `eiopt.backends.state.spatial` -> `eiopt.backends.state.robotics.spatial`
- `eiopt.backends.state.kots` -> `eiopt.backends.state.robotics.kots`
- `eiopt.backends.state.pinocchio` -> `eiopt.backends.state.robotics.pinocchio`
- `eiopt.backends.state.vision_pinhole` -> `eiopt.backends.state.vision.pinhole`

## Backend との接続点

backend(kots / pinocchio 等) と `eiopt` を繋ぐ唯一の接続点は `StateCache` が呼ぶ `build_state()` です。

- `build_state(x_all, *, pack=None, time=None, required=None) -> dict[StateKey, Any]`
  - `x_all`: 全決定変数ベクトル（`VariablePack` の順）
  - `pack`: `VariablePack`（受け取れる backend 実装なら渡されます）
  - `time`: `TimeGrid`
- `required`: 今回必要な `StateKey` の集合（これだけ計算すると速い）。`None` は「全部計算」。

複数 backend を同時利用したい場合は `eiopt.backends.state.dispatch.composite.CompositeStateBuilder` で
`build_state()` を合成できます（例: ロボット状態 provider + カメラ状態 provider）。

```python
from eiopt.backends.state.dispatch.composite import CompositeStateBuilder
from eiopt.optimize.builder import compile_nls_problem

state_builder = CompositeStateBuilder([robot_provider, camera_provider])
runtime = compile_nls_problem(dsl, build_state=state_builder.build_state)
```

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
from eiopt.optimize.builder import compile_nls_problem, load_problem_toml

dsl = load_problem_toml("examples/dsl/basic.toml")
runtime = compile_nls_problem(dsl, build_state=build_state)
```

サンプルは `examples/dsl/basic.toml` を参照してください。

## Examples の実行

現在の examples は canonical API の最小サンプルに絞っています。

```bash
# (推奨) インストールして実行
python -m pip install -e .
python examples/01_minimize_quadratic.py
python examples/02_get_state_minimal.py
python examples/03_toml_problem.py
python examples/10_stationarity_ioc.py
python examples/11_forward_then_inverse_ioc.py
python examples/08_camera_calibration.py
python examples/04_pinocchio_ik.py   # 要: pinocchio
python examples/05_robokots_ik.py    # 要: robokots
python examples/06_pinocchio_trajectory_dynamics.py  # 要: pinocchio
python examples/07_robokots_trajectory_dynamics.py   # 要: robokots
python examples/09_kots_vision_composite.py   # 要: robokots
```

```bash
# インストールせずに実行（repo 直下から）
PYTHONPATH=. python examples/01_minimize_quadratic.py
PYTHONPATH=. python examples/02_get_state_minimal.py
PYTHONPATH=. python examples/03_toml_problem.py
PYTHONPATH=. python examples/10_stationarity_ioc.py
PYTHONPATH=. python examples/11_forward_then_inverse_ioc.py
PYTHONPATH=. python examples/08_camera_calibration.py
PYTHONPATH=. python examples/04_pinocchio_ik.py
PYTHONPATH=. python examples/05_robokots_ik.py
PYTHONPATH=. python examples/06_pinocchio_trajectory_dynamics.py
PYTHONPATH=. python examples/07_robokots_trajectory_dynamics.py
PYTHONPATH=. python examples/09_kots_vision_composite.py
```

Pinocchio / RoboKots のサンプルは optional dependency が必要です。

```bash
uv sync --group pinocchio
uv sync --group kots
```

各サンプルの目的は `examples/README.md` を参照してください。
補足として、`examples/dsl/robokots_traj_dynamics_d12_per_joint.toml` は
`examples/dsl/robokots_traj_dynamics_d12.toml` の per-joint 版で、
`expr.type="component"` と `terms.attrs.joint_component` を使った可視化・解析向け DSL です。

### Kots 軌道問題の高レベルビルダー

`eiopt.optimize_backends.kots.compile_kots_trajectory_problem()` を使うと、
軌道マップ生成・導関数マップ生成・`p` 変数次元チェック・`KotsTrajectoryStateBuilder` 構築・`compile_nls_problem()` までを一括で実行できます。
この関数は入力 DSL オブジェクトを破壊的に書き換えません。

```python
from eiopt.optimize_backends.kots import compile_kots_trajectory_problem

compiled = compile_kots_trajectory_problem(
    dsl,
    model=kots,
    data=kots.state_dict_,
)

runtime = compiled.runtime
traj_map = compiled.trajectory_map
```

Pinocchio 側にも同様の helper があります。

```python
from eiopt.optimize_backends.pinocchio import compile_pinocchio_trajectory_problem

compiled = compile_pinocchio_trajectory_problem(
    dsl,
    model=pin_model,
    data=pin_data,
)

runtime = compiled.runtime
traj_map = compiled.trajectory_map
```

trajectory 前提を置かない backend の compile helper は
`eiopt.optimize_backends.problem_adapter.compile_problem_with_adapter()` で実装できます。
`dsl` 準備・state builder 構築・runtime 検証の 3 段を adapter に分離できます。

camera calibration 向けには `eiopt.optimize_backends.vision.compile_camera_calibration_problem()` も利用できます。

等式制約 (`constraint.kind="eq"`) を零空間で消去した reduced 問題を作る場合は、
`compile_kots_trajectory_problem()` の後段で backend 非依存 API を適用します。

```python
from eiopt.optimize.reductions import build_nullspace_equality_reduction
from eiopt.optimize.solvers import solve

use_nullspace = True
nullspace_eq = (
    build_nullspace_equality_reduction(
        runtime,
        # eq_term_indices=[...],              # 任意
        # objective_term_indices=[...],       # 任意
    )
    if use_nullspace
    else None
)
runtime_for_solve = runtime if nullspace_eq is None else nullspace_eq.runtime
out = solve(runtime_for_solve)
x_star_solve = out.solution
x_star = x_star_solve if nullspace_eq is None else nullspace_eq.lift(x_star_solve)
```

### solve_gauss_newton の返り値

`solve_gauss_newton()` は `SolveOutcome` を返します。
解ベクトルは `out.solution`、収束情報は `out.stats`、計測は `out.timing` です。

```python
out = solve_gauss_newton(runtime)
x_star = out.solution
print(out.stats.status, out.stats.converged, out.stats.objective)
```

計測結果を表形式で表示したい場合は `format_timing_report()` が使えます。

```python
from eiopt import format_timing_report

print(format_timing_report(out.timing, title="solver timing"))
```

### ソルバ切替（gauss_newton / scipy_minimize / cyipopt / liteopt）

`solve()` でソルバを切り替えられます。`scipy` / `cyipopt` / `liteopt` は optional dependency です。
solver 名は正規名のみ受け付けます（alias は非対応）。

```bash
python -m pip install -e ".[solver-scipy]"      # scipy solver を使う場合
python -m pip install -e ".[solver-cyipopt]"    # cyipopt solver を使う場合
python -m pip install -e ".[solver-liteopt]"    # liteopt solver を使う場合
python -m pip install -e ".[solvers]"           # 全 solver を入れる場合
```

```python
from eiopt.optimize.solvers import solve

out = solve(
    runtime,
    solver="liteopt",  # "gauss_newton" | "scipy_minimize" | "cyipopt" | "liteopt"
    options={"max_iters": 1000, "step_size": 1e-3, "tol_grad": 1e-4},
)
x_star = out.solution
print(out.stats.status, out.stats.objective)
```

各 solver の設定は `options` に統一されています。
外部 solver にそのまま渡したいオプションは `options["backend_options"]` を使います。

```python
out = solve(
    runtime,
    solver="scipy_minimize",
    options={
        "method": "L-BFGS-B",
        "max_iters": 200,
        "backend_options": {"maxcor": 20},
    },
)
x_star = out.solution
```

### 実行時に特定 term の重みだけ変更

`compile_nls_problem()` 後に、term index または `expr.name` を指定して重みを変更できます。
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

### term別線形化と Stationarity 行列

termごとの線形化結果は `runtime.linearize_terms()` で取得できます。`weighted=False` を使うと未加重の `r_i, J_i` を取り出せます。

```python
terms = runtime.linearize_terms(weighted=False)
for t in terms:
    print(t.term_index, t.name, t.residual.shape, t.jacobian.shape, t.attrs)
```

`LinearizedTerm` オブジェクトを作らずに stacked `(r, J)` だけ欲しい場合は
`runtime.linearize_stacked_terms()` を使えます（line-search 等のホットパス向け）。

```python
r, J = runtime.linearize_stacked_terms(weighted=True, term_indices=[0, 2, 5])
```

term ごとの行範囲も必要な解析向けには `runtime.linearize_stacked_terms_with_layout()` を使います。

```python
r, J, layout = runtime.linearize_stacked_terms_with_layout(weighted=False)
for s in layout:
    r_i = r[s.row_start:s.row_stop]
    J_i = J[s.row_start:s.row_stop, :]
    print(s.term_index, s.name, r_i.shape, J_i.shape)
```

Stationarity 方程式の解法用に `A = [J_0^T r_0, ..., J_n^T r_n]` を作るヘルパーも使えます。

```python
from eiopt import build_term_gradient_matrix, solve_simplex_min_norm

A, term_indices = build_term_gradient_matrix(runtime, weighted=False)
simplex_out = solve_simplex_min_norm(A)
w_hat = simplex_out.solution  # w>=0, sum(w)=1
```

Stationarity の組み立ては `RuntimeStationaritySource` と純関数群を組み合わせます。

```python
from eiopt import (
    RuntimeStationaritySource,
    filter_stationarity_contributions,
    build_stationarity_gradient_matrix,
    select_active_stationarity_indices,
    build_reference_simplex_init,
    solve_simplex_min_norm,
)

source = RuntimeStationaritySource(runtime)
x_opt = runtime.pack.get().copy()
source.set_point(x_opt)
contrib_all = source.term_contributions(required=source.required_list(None))
contrib = filter_stationarity_contributions(contrib_all, include_constraints=True)

A_col, term_indices = build_stationarity_gradient_matrix(contrib, n_total=source.n_total)
active_idx, *_ = select_active_stationarity_indices(contrib, mode="residual")
x0 = build_reference_simplex_init(contrib, active_idx)
simplex_out = solve_simplex_min_norm(A_col[:, active_idx], x0=x0)
w_hat = simplex_out.solution
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
out = solve_gauss_newton(runtime)
print(format_solve_report(runtime, x0=x0, outcome=out))
```

`format_solve_report()` は既定で `||J^T r||`, `rank(J)`, `svd(J)`, active term も表示します。
不要なら `include_diagnostics=False` を指定できます。

名前付き Expr の値をコード側で直接使う場合は `get_named_expr_value()` が使えます。

```python
from eiopt import get_named_expr_value

ee_traj = get_named_expr_value(runtime, name="ee_pos_traj")
```

### `term.attrs` ベースで時系列を描画

`term.attrs.plot` に可視化メタデータを置くと、DSL から描画対象を宣言できます。
描画責務は `eiopt.optimize` 側に閉じており、core/backend の責務分離を維持できます。

```toml
[[terms]]
[terms.expr]
type = "sub"
name = "torque_reg"

[terms.expr.a]
type = "get_state"

[terms.expr.a.key]
k = "last"
owner_type = "total_joint"
owner_name = "robot"
dtype = "dynamics"
field = "torque"

[terms.expr.a.jac]
var = "p"

[terms.expr.b]
type = "const"
var = "p"
value = { fill = 0.0 }

[terms.cost]
type = "scalar_weight"
w = 1e-4

[terms.attrs.plot]
type = "state_traj"
name = "joint_torque"
# key.* を省略した場合、同 term 内の最初の get_state から推定します
k0 = 0
k1 = "last"
```

```python
from eiopt import collect_plot_series_from_term_attrs, plot_term_attrs

series = collect_plot_series_from_term_attrs(runtime)
fig, ax, _series = plot_term_attrs(runtime, title="Trajectory diagnostics")
```

`plot` は dict 1 件だけでなく list も受け付けます。
同一 term に複数の時系列を設定したい場合は `attrs.plot = [{...}, {...}]` を使ってください。

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

### Camera calibration 向けの最小キー設計（`dtype="vision"`）

camera calibration の最小雛形では、次のキー設計を推奨します。

- `dtype = "vision"`
- `owner_type = "camera"`
- `owner_name = "cam0"`（カメラ名）
- `field = "reproj" | "intrinsics" | "extrinsics"`（必要に応じて拡張）

### Camera calibration DSL の canonical 形式

`compile_camera_calibration_problem()` は top-level `vision` セクションを canonical 入力として扱います。
`owner_name` と `observations` は必須です。

```toml
[vision]
p_var = "theta"            # 既定
owner_type = "camera"      # 既定
owner_name = "cam0"        # 必須
field = "reproj"           # 既定
k = 0                      # 既定
term_name = "camera_reproj_error"  # 既定
observations = [0.1, 0.2, 0.3]     # 必須
```

この形式では、`compile_camera_calibration_problem()` が再投影誤差 term を標準形で自動生成/置換します。
対象パラメータ変数は `vision.p_var`（既定 `theta`）と一致する必要があります。

Python helper:

```python
from eiopt.core.state_schema import vision_key, vision_jac_key

key = vision_key(k=0, owner_name="cam0", field="reproj")
key_j = vision_jac_key(k=0, owner_name="cam0", field="reproj", var="theta")
```

最小 provider 雛形:

```python
from eiopt.backends.state.vision.provider import CameraCalibrationStateProvider, VisionFieldHandler

provider = CameraCalibrationStateProvider(
    model=model,
    data=data,
    param_var="theta",
    field_handlers={
        "reproj": VisionFieldHandler(
            value_handler=reproj_value_fn,
            jac_handler=reproj_jac_fn,
        ),
    },
)
```

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
`k` は整数に加え、`"last"` / `"last-1"` / `"first"` が使えます（`time.N` から解決）。

## 軌道スタックの直接取得（get_traj_var）

軌道パラメータ `p` を最適化している場合は、`get_traj_var` で
`TrajectoryMap` の `q_traj = A @ p + b`（`q(0),...,q(N)` のスタック）を
直接 residual として使えます。

- `expr.trajectory` を省略すると、トップレベルの `[trajectory]` を使います。
- `type="bspline"` で `q_dim` 未指定の場合、`var.dim / num_ctrl_points` から推定します（割り切れる場合）。
- 全要素を同じ値で埋める場合は、明示的に `value = { fill = ... }` を使ってください。

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
value = { fill = 3.141592653589793 } # repeats は既定で time.N+1
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
  `torque_dN` を使う場合は RoboKots モデル次数が `order >= N+3` 必須です（例: `torque_d1` は `order >= 4`）。
  Pinocchio 側は標準で `torque`, `momentum`, `force` をサポートし、追加分は
  `dynamics_custom_handlers` で登録できます。
- `trajectory_derivative_maps` を渡すと、モデル次数に応じた `q, dq, ddq...` を内部状態に展開できます。
- `compile_kots_trajectory_problem(...)` は `trajectory.q_dim` 未指定時に `model.dof()` から自動推定します。
- 同 helper では `variables[].dim="auto"`（または dim 省略）を受け付け、`trajectory.p_dim` に合わせて補完します。
- 変数の全要素同値初期化は `init = { fill = ... }` で明示指定します（暗黙ブロードキャストは行いません）。

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
from eiopt.backends.state.robotics.kots import KotsTrajectoryStateBuilder
from eiopt.optimize.builder import compile_nls_problem
from eiopt.optimize.dsl import (
    build_trajectory_map,
    build_trajectory_maps_with_derivatives,
    default_steps_from_time,
)

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
runtime = compile_nls_problem(dsl, build_state=builder.build_state)
```

DSL 側では `get_state.jac.var = "p"` を指定します。
最小の `get_state` 実行例は `examples/02_get_state_minimal.py` を参照してください。
終端タスクの `k` には数値の代わりに `"last"` を使うと、`time.N` 変更時の修正が不要です。

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
`dtype="joint"` はサポート対象外です。`get_state(dtype="coord", field="q")` を使ってください。

この “値とヤコビアンの次元が一致する” ルールにしておくと、後から `se3_error` などの誤差 Expr を作る際も、
必要な raw 情報を `StateCache` から取り出して組み合わせるだけで済みます。
