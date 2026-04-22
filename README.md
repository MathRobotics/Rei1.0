# Rei

`rei` は capability 指向で数値問題を構築・線形化・最適化するための Python ツールキットです。  
DSL（dict/TOML）で問題を定義し、backend 側の `build_state()` を接続して解きます。

## インストール

Python `>=3.11` が必要です。

基本インストール:

```bash
python -m pip install -e .
```

optional solver:

```bash
python -m pip install -e ".[solver-scipy]"
python -m pip install -e ".[solver-cyipopt]"
python -m pip install -e ".[solver-liteopt]"
python -m pip install -e ".[solvers]"
```

optional backend dependency（`uv` を使う場合）:

```bash
uv sync --group pinocchio
uv sync --group kots
uv sync --group solver-liteopt
```

## クイックスタート（最小 NLS）

```python
from rei import compile_nls_problem, solve

dsl = {
    "variables": [{"name": "q", "dim": 2, "init": [0.0, 0.0]}],
    "terms": [
        {
            "expr": {
                "type": "sub",
                "a": {"type": "get_var", "var": "q"},
                "b": {"type": "const", "var": "q", "value": [3.0, -1.0]},
            },
            "cost": {"type": "l2"},
        }
    ],
}

runtime = compile_nls_problem(
    dsl,
    build_state=lambda *_args, **_kwargs: {},
)
out = solve(runtime, solver="gauss_newton")
print(out.solution, out.stats.status)
```

TOML を使う場合:

```python
from rei import compile_nls_problem, load_problem_toml, solve

dsl = load_problem_toml("examples/dsl/basic.toml")
runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
out = solve(runtime)
```

DSL の書き方をまとめたガイドは `docs/dsl.md` を参照してください。

## Canonical Namespace

- `rei.optimize`: 最適化 API 入口（`compile_nls_problem`, `solve` など）
- `rei.equations`: 方程式系 capability（`RuntimeStationaritySource` など）
- `rei.flow`: 制約/射影 capability
- `rei.backends.state`: backend state builder
- `rei.optimize_backends`: backend 別 compile helper

## Backend 接続点（`build_state()`）

`rei` と backend の接続点は `StateCache` が呼ぶ `build_state()` だけです。

```python
build_state(x_all, *, pack=None, time=None, required=None) -> dict[StateKey, Any]
```

- `x_all`: 全決定変数ベクトル
- `pack`: `VariablePack`（必要なら利用）
- `time`: `TimeGrid`
- `required`: 必要な `StateKey` 集合。`None` は「全部」

実装要件:

- 同じ入力に対して同じ出力を返すこと
- `required` 指定時は要求キーを返すこと
- 返り値は `dict[StateKey, Any]`
- `Expr` が期待する shape の数値配列を返すこと

## Capability Adapter

既存 runtime から capability を取り出せます。

```python
from rei import as_constraint_problem, as_linear_equation_problem, as_project_problem

eq_problem = as_linear_equation_problem(runtime)
constraint_problem = as_constraint_problem(runtime, kind="eq")
project_problem = as_project_problem(runtime)
```

## Backend 向け Compile Helper

軌道・カメラ較正などは `rei.optimize_backends` の helper が使えます。

```python
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize_backends.pinocchio import compile_pinocchio_trajectory_problem
from rei.optimize_backends.vision import compile_camera_calibration_problem
```

各 helper は `compiled.runtime` を返し、必要に応じて軌道マップなどの付加情報も返します。

## Solver 切替

`solve()` の solver 名は次の 4 つです（alias は非対応）。

- `"gauss_newton"`
- `"scipy_minimize"`（要 scipy）
- `"cyipopt"`（要 cyipopt）
- `"liteopt"`（要 liteopt）

```python
from rei import solve

out = solve(
    runtime,
    x0=[0.5],  # optional initial point override
    solver="liteopt",
    options={
        "method": "gd",  # default
        "verbose": True,
        "max_iters": 1000,
        "step_size": 1e-3,
        "tol_grad": 1e-4,
        "line_search": True,
        # optional non-finite guard (step-size retry)
        "nonfinite_retries": 8,
        "nonfinite_step_shrink": 0.2,
        "min_step_size": 1e-12,
    },
)

out_gn = solve(
    runtime,
    solver="liteopt",
    options={
        "method": "gn",
        "verbose": True,
        "max_iters": 200,
        "lambda_": 1e-8,
        "line_search_method": "strict_decrease",
        "line_search": True,
        "ls_beta": 0.5,
        "ls_min_step": 1e-8,
        "ls_max_steps": 12,
    },
)

out_ipopt = solve(
    runtime,
    solver="cyipopt",
    options={
        "max_iters": 200,
        "tol": 1e-8,
        "print_level": 0,  # backend option (auto-forwarded)
        # or: "backend_options": {"print_level": 0}
    },
)
```

返り値は `SolveOutcome` で、主に `out.solution`, `out.stats`, `out.timing` を使います。
初期点は `solve(..., x0=...)` で直接渡せます。`solve(..., options={"x0": ...})` も利用できますが、
`x0=` の方を推奨します。
`scipy_minimize` / `cyipopt` / `liteopt` では未知の top-level key を backend option として転送します。
ただし他 solver 用の key（例: `cyipopt` で `tol_dx`）はエラーにします。

## Examples

基本例:

```bash
python examples/01_minimize_quadratic.py
python examples/02_get_state_minimal.py
python examples/03_toml_problem.py
python examples/08_camera_calibration.py
python examples/10_stationarity_ioc.py
python examples/11_forward_then_inverse_ioc.py
```

Pinocchio:

```bash
uv sync --group pinocchio
python examples/04_pinocchio_ik.py
python examples/06_pinocchio_trajectory_dynamics.py
```

RoboKots:

```bash
uv sync --group kots
python examples/05_robokots_ik.py
python examples/07_robokots_trajectory_dynamics.py
python examples/09_kots_vision_composite.py
```

`examples/README.md` に各サンプルの目的と詳細があります。

## 移行メモ（削除済み Import Path）

旧フラット import path は削除済みです。
旧 namespace / alias（`rei.dsl`, `rei.model`, `rei.solvers`, `rei.expr` など）も削除済みです。

- `rei.backends.state.template` -> `rei.backends.state.dispatch.template`
- `rei.backends.state.composite` -> `rei.backends.state.dispatch.composite`
- `rei.backends.state.spatial` -> `rei.backends.state.robotics.spatial`
- `rei.backends.state.kots` -> `rei.backends.state.robotics.kots`
- `rei.backends.state.pinocchio` -> `rei.backends.state.robotics.pinocchio`
- `rei.backends.state.vision_pinhole` -> `rei.backends.state.vision.pinhole`
