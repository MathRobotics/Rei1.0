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

dsl = load_problem_toml("examples/specs/basic.toml")
problem, ctx, required = compile_problem(dsl, build_state=build_state)
```

サンプルは `examples/specs/basic.toml` を参照してください。

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
dtype = "frame"
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
- `dtype`: 大分類（推奨: `"frame"` / `"joint"` / `"dynamics"`）
- `field`: 量の名前（例: `"pos"`, `"rot"`, `"vel"`, `"acc"`, `"momentum"`, `"force"`, `"torque"`）
- `frame`: 座標系（推奨: `"world"` / `"local"`）
- `rel_frame`: 相対量が必要なときの相手フレーム（任意）

### Jacobian の field 命名

ヤコビアンは `"{field}_J_{var}"` を推奨します（例: `pos_J_q`）。

Python 側は `eiopt.core.state_schema.jac_field()` が使えます。

## 最小標準セット（pos/rot/frame + q）

backend から `eiopt` に提供する “標準の最小セット” として、まずは以下に絞るのが扱いやすいです。
（誤差の定義は後から Expr/Residual として設計できるように、ここでは **生の状態** を揃えます）

- `dtype="frame"`
  - `frame="world"`（ひとまず world 固定。DSL では省略可で、デフォルト world 扱い）
  - `field="pos"`: `(3,)` 位置ベクトル
  - `field="rot"`: `(9,)` 回転行列 `(3,3)` の row-major flatten
  - `field="frame"`: `(12,) = [pos(3), rot_flat(9)]`
  - `field="{field}_J_q"`: それぞれ上と同じ次元のヤコビアン（例: `pos_J_q` は `(3,nq)`）

- `dtype="joint"`（関節角度）
  - 推奨 owner: `owner_type="total_joint"`, `owner_name="robot"`
  - `field="q"`: `(nq,)`
  - `field="q_J_q"`: `(nq,nq)`（軌道最適化などで `q` を時系列にスタックした場合は選択行列になります）

`q/q_J_q` を backend が直接返さない場合は、`eiopt.adapters.with_standard_joint_q()` で `build_state` をラップすると自動で注入できます。

この “値とヤコビアンの次元が一致する” ルールにしておくと、後から `se3_error` などの誤差 Expr を作る際も、
必要な raw 情報を `StateCache` から取り出して組み合わせるだけで済みます。
