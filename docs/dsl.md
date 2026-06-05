# Rei DSL ガイド

`rei` の DSL は Python `dict` と TOML で同じ構造です。  
このドキュメントでは見やすさのため TOML で説明します。

## 最小例

```toml
[[variables]]
name = "q"
dim = 2
init = [0.0, 0.0]

[[terms]]
[terms.expr]
type = "sub"
name = "q_minus_target"

[terms.expr.a]
type = "get_var"
var = "q"

[terms.expr.b]
type = "const"
var = "q"
value = [0.5, -1.2]

[terms.cost]
type = "l2"
```

これは `q - [0.5, -1.2]` を最小化する一番小さい形です。

## 全体構造

```toml
[time]          # 任意
N = 20
dt = 0.1

[trajectory]    # 任意
type = "bspline"
var = "p"
degree = 4
num_ctrl_points = 8

[[variables]]   # 0 個以上。通常は 1 個以上
name = "..."
dim = 2
init = [0.0, 0.0]

[[terms]]       # 0 個以上
constraint = "eq"   # 任意。 "eq" / "ineq"

[terms.expr]        # 必須
type = "..."

[terms.cost]        # 任意。省略時は { type = "l2" }
type = "l2"

[terms.attrs]       # 任意。自由メタデータ
group = "objective"
```

ルートでよく使うキーは次の 4 つです。

- `variables`: 決定変数
- `terms`: 残差項
- `time`: 時間離散化。軌道問題や `k="last"` で使う
- `trajectory`: 軌道パラメータ `p` から時系列 `q(k)` を作る設定

## `variables`

各変数は次の形です。

- `name` 必須: 変数名
- `dim` 任意: 次元。`init` から決まるなら省略可
- `init` 任意: 初期値。省略時は零ベクトル

`dim` と `init` は両方とも省略できるわけではありません。実質的には次のどちらかが必要です。

- `dim` を与える
- `init` を与えて長さを決める

例:

```toml
[[variables]]
name = "x"
dim = 3
init = [1.0, 2.0, 3.0]
```

全成分を同じ値で埋める場合は `fill` を使います。

```toml
[[variables]]
name = "p"
dim = 12
init = { fill = 0.0 }
```

注意:

- `dim > 1` のとき `init = 0.0` のような暗黙ブロードキャストは不可です
- その場合は `init = { fill = 0.0 }` を使います
- 複数変数があるときは、式側で `var = "..."` を明示した方が安全です

## 変数名の既定値

「常に `q` や `p` が既定になる」というグローバル規則はありません。  
既定値は式の種類ごとに次のように決まります。

- `get_var.var` を省略:
  変数が 1 個だけなら、その唯一の変数名を使います。複数あるとエラーです。
- `get_state.jac.var` / `get_state.jacs[*].var` を省略:
  変数が 1 個だけなら、その唯一の変数名を使います。複数あるとエラーです。
- `get_traj_var.var` を省略:
  まず `[trajectory].var` を見ます。それも無ければ、変数が 1 個だけのときはその変数名を使います。複数あるとエラーです。
- `const.var` / `const_repeat.var` を省略:
  自動で特定の変数名は選びません。`var` は次元解釈や補助情報のための任意キーです。

TOML 読み込み (`load_problem_toml`) では補助的な正規化も入ります。

- `variables` が 1 個だけの TOML なら、`get_state` の `jac.var` が自動補完されます
- `variables` が複数ある場合は自動補完されません

trajectory 用 helper ではさらに別の既定があります。

- `prepare_trajectory_problem_dsl()` などの trajectory compile helper では、`[trajectory].var` を省略すると `"p"` が使われます

実運用では曖昧さを避けるため、複数変数の問題では `var` と `jac.var` を明示するのを推奨します。

## `terms`

各 term は「式 `expr` を評価し、その残差に `cost` をかける」単位です。

- `expr` 必須: 残差ベクトルを作る式
- `cost` 任意: 重み付け。省略時は `l2`
- `constraint` 任意: 制約扱いにしたいときに `eq` / `ineq`
- `attrs` 任意: 任意のメタデータ

制約は次のどちらでも書けます。

```toml
constraint = "eq"
```

```toml
[terms.constraint]
kind = "eq"
```

内部では `attrs.is_constraint = true` と `attrs.constraint_kind = "eq" | "ineq"` に正規化されます。

`attrs` は自由に使えます。よくある用途:

- `group = "objective"` のような後段フィルタ用タグ

プロットしたい term には term 直下に `plot` を書けます。state term なら
state key は式から推論されます。

```toml
[[terms]]
name = "q_init"
kind = "eq"
quantity = "joint_angles"
at = 0
target = { fill = 1.57 }
plot = "joint_q"
```

backend が計算する関節トルクも quantity として書けます。

```toml
[[terms]]
name = "torque_traj_regularization"
weight = 1e-10
quantity = "joint_torques"
stride = 50
plot = { name = "joint_torque", stride = 50 }
```

等式制約を nullspace reduction で厳密に消す対象にしたい場合は、
term 直下に `enforce = "nullspace"` を書きます。

```toml
[[terms]]
name = "qdot_init"
kind = "eq"
enforce = "nullspace"
quantity = "joint_velocities"
at = 0
target = { fill = 0.0 }
```

補足:

- `enforce` は内部では `attrs.enforce` に取り込まれます
- `constraint` は term の式そのものを変えるものではなく、constraint capability や KKT / reduction 系が参照する metadata です
- `solve()` の残差最小化では、constraint term も通常の term と同じく residual + cost として組み立てられます

## 式 `expr.type`

全ての式で `name` は任意です。付けておくとログや可視化で追いやすくなります。

### 1. `get_var`

決定変数をそのまま読む式です。

主なキー:

- `var`: 変数名
- `k`: 任意。時間積み変数を 1 ステップだけ切り出す

```toml
[terms.expr]
type = "get_var"
name = "joint_q"
var = "q"
```

`k` を使う場合は、変数が `time.N + 1` ステップ分の縦積みベクトルとして解釈できる必要があります。

`var` を省略できるのは、DSL 内の変数が 1 個だけのときです。

### 2. `const`

定数ベクトルです。

主なキー:

- `value` 必須: ベクトル値
- `var` 任意: どの変数の次元系で解釈するか
- `dim` 任意: 長さを明示したいとき

```toml
[terms.expr.b]
type = "const"
var = "q"
value = [1.0, -1.0]
```

全成分同値なら `fill` を使えます。

```toml
value = { fill = 0.0 }
```

### 3. `sub`

2 つの式の差 `a - b` を取ります。最もよく使う基本形です。

```toml
[terms.expr]
type = "sub"

[terms.expr.a]
type = "get_var"
var = "q"

[terms.expr.b]
type = "const"
var = "q"
value = [0.0, 0.0]
```

### 4. `get_state`

backend の `build_state()` が返す状態量を読む式です。IK や dynamics 制約で使います。

主なキー:

- `key.k`: 時刻 index
- `key.owner_type`, `key.owner_name`: 状態の持ち主
- `key.dtype`: 例 `coord`, `kinematics`, `dynamics`, `vision`
- `key.field`: 例 `q`, `pos`, `rot`, `frame`, `torque`, `torque_d1`
- `key.frame`, `key.rel_frame`: 任意
- `jac` または `jacs`: どの変数に対するヤコビアンを要求するか

```toml
[terms.expr.a]
type = "get_state"
name = "ee_pos"

[terms.expr.a.key]
k = 0
owner_type = "link"
owner_name = "ee"
dtype = "kinematics"
field = "pos"
frame = "world"

[terms.expr.a.jac]
var = "q"
```

複数変数に対するヤコビアンが必要なら `jacs` を使います。

```toml
[terms.expr]
type = "get_state"

[terms.expr.key]
k = 0
owner_type = "link"
owner_name = "end"
dtype = "kinematics"
field = "pos"

[[terms.expr.jacs]]
var = "p"

[[terms.expr.jacs]]
var = "q"
```

注意:

- 変数が 1 個だけなら `jac.var` は省略可能です
- 変数が複数あるなら `jac.var` / `jacs[*].var` を明示してください
- `dtype="joint"` や `field="tau"` / `field="dtau"` のような alias は不可です
- kinematics の `frame` は現在 `world` のみ対応です
- TOML 読み込み時は、変数が 1 個だけなら `jac.var` が自動補完されます

### 5. `stack`

内側の式を `k0..k1` にわたって評価して縦に連結します。

```toml
[terms.expr]
type = "stack"

[terms.expr.range]
k0 = 0
k1 = "last"
stride = 10

[terms.expr.inner]
type = "get_state"

[terms.expr.inner.key]
owner_type = "total_joint"
owner_name = "robot"
dtype = "dynamics"
field = "torque"

[terms.expr.inner.jac]
var = "p"
```

`stride` は任意です。未指定なら 1 で、全ステップを評価します。

### 6. `vstack`

複数の式をそのまま縦に連結します。時間範囲を展開する `stack` と違い、
`parts` に書いた式だけを評価します。

```toml
[terms.expr]
type = "vstack"

[[terms.expr.parts]]
type = "get_var"
var = "q"

[[terms.expr.parts]]
type = "get_var"
var = "dq"
```

### 7. `hinge`

`max(base, 0)` を返します。上限/下限制約違反の表現に向いています。

```toml
[terms.expr]
type = "hinge"

[terms.expr.base]
type = "sub"
```

### 8. `get_traj_var`

`[trajectory]` から作られる軌道 `q(k)` やその導関数を読みます。軌道最適化の中心です。

主なキー:

- `var`: 軌道パラメータ変数名。通常は `p`
- `k`: 任意。1 ステップだけ切り出す
- `derivative_order`: 任意。`0, 1, 2, ...`
- `derivative_wrt`: 任意。`"u"` または `"time"`
- `max_derivative_order`: 任意。`q, dq, ..., d^Nq` をまとめて返す

```toml
[terms.expr]
type = "get_traj_var"
name = "qdot_terminal"
var = "p"
derivative_order = 1
derivative_wrt = "time"
k = "last"
```

`derivative_order` と `max_derivative_order` は同時には使えません。

`var` を省略した場合は、まず `[trajectory].var` が使われます。  
`[trajectory].var` も無い場合は、変数が 1 個だけのときに限ってその変数名が使われます。

### 9. `const_repeat`

1 ステップ分のベクトルを全時刻に繰り返します。

主なキー:

- `value` 必須
- `repeats` または `steps` 任意。未指定なら `time.N + 1`
- `segment_dim` / `dim` 任意

```toml
[terms.expr.b]
type = "const_repeat"
var = "p"
value = { fill = 1.0 }
```

`value = { fill = ... }` を使う場合は、`segment_dim` / `dim` を与えるか、
`[trajectory]` から 1 ステップ次元を推定できる必要があります。

### 10. `time_diff`

時間方向の差分を取ります。`[y(1)-y(0), y(2)-y(1), ...]` を返します。

主なキー:

- `base` 必須
- `segment_dim` 任意: 1 ステップ分の次元。省略または `"auto"` なら `base` から推定
- `wrt` 任意: `"index"` または `"time"`
- `dt` 任意: `wrt="time"` 時の刻み幅。省略時は `[time].dt`
- `scale` 任意

```toml
[terms.expr]
type = "time_diff"
wrt = "time"

[terms.expr.base]
type = "get_var"
var = "q"
```

`get_traj_var` や、`time.N + 1` で分割できる `get_var` など、
1 ステップ次元を `base` から決められる場合は `segment_dim` を省略できます。
曖昧な場合だけ明示してください。

### 10. `component`

`(..., segment_dim)` の並びから 1 成分だけを取り出します。

主なキー:

- `base` 必須
- `segment_dim` 任意。省略または `"auto"` なら `base` から推定
- `index` 必須

```toml
[terms.expr]
type = "component"
index = 1

[terms.expr.base]
type = "get_var"
var = "x"
```

典型的には `get_traj_var`、時間方向に並んだ `get_var`、
あるいは `sub(get_state(...), const(...))` のように子の次元が揃っている式から
自動で 1 ステップ次元を決められます。

### 実例: 特定の関節だけを扱う

全関節ベクトルの式は、`component` で関節ごとに分けられます。

考え方は次の 2 段です。

1. まず `base` で「全関節分の残差」や「全関節分の軌道」を作る
2. その外側を `component` で包み、`index` 番目の関節だけを取り出す

たとえば `q_init_j0` は、概念的には

```text
component(index=0, base=(joint_q_init - target_q_init))
```

です。  
`joint_q_init - target_q_init` は 2 関節分のベクトルですが、
この例では `segment_dim` を省略しても 2 関節分だと推定され、
`index = 0` によって 0 番目の関節だけが残ります。
さらに `constraint.kind = "eq"` が付いているので、
これは「初期時刻で joint 0 を目標角度に一致させる等式制約」です。

`qdot_init_j0` も同じ形で、違うのは `base.a` が
`get_traj_var(... derivative_order = 1, derivative_wrt = "time", k = 0)` になっている点です。  
つまり「初期時刻の全関節速度ベクトル」を作り、そのうち joint 0 だけを取り出して
`0` に合わせています。

`qdot_traj_regularization_j0` では `constraint` が無く、
`base` は `get_traj_var(... derivative_order = 1, derivative_wrt = "time")` です。  
これは「全時刻の全関節速度」から joint 0 だけを抜き、
その大きさを目的関数として小さくする例です。

特定の関節だけに目標値を与えたいときは、`fill` を使うより
「全関節の式を作ってから `component` で 1 関節だけ抜く」方が一般的です。

```toml
[terms.expr]
type = "component"
index = 0

[terms.expr.base]
type = "sub"

[terms.expr.base.a]
type = "get_state"
# または get_var / get_traj_var

[terms.expr.base.b]
type = "const"
value = [1.57, 0.0]
```

この形なら、

- `constraint = "eq"` を付ければ「その関節だけの等式制約」
- `constraint = "ineq"` を付ければ「その関節だけの不等式制約」
- `constraint` を付けなければ「その関節だけの目的関数項」

として同じパターンで使えます。

## `cost.type`

使える標準 cost は 4 つです。

### `l2`

重みなしの基本形です。省略時の既定値でもあります。

```toml
[terms.cost]
type = "l2"
```

### `scalar_weight`

残差全体に 1 つの重みをかけます。

```toml
[terms.cost]
type = "scalar_weight"
w = 100.0
```

### `diag_weight`

残差の各成分に別々の重みをかけます。

```toml
[terms.cost]
type = "diag_weight"
w = [1.0, 10.0, 1.0]
```

### `huber`

Huber loss です。

```toml
[terms.cost]
type = "huber"
delta = 1.0
```

注意:

- `scalar_weight.w` と `diag_weight.w` は 0 以上
- `huber.delta` は正

## `[time]`

時間軸を使う場合は次を定義します。

```toml
[time]
N = 20
dt = 0.1
```

- ステップ数は `N + 1`
- `k = "last"` は最後のステップを意味します
- `k = "first"` と `k = "last-1"` も使えます

`time` が特に効くのは次の場面です。

- `get_state.key.k`
- `get_var.k`
- `get_traj_var.k`
- `stack.range.k0`, `stack.range.k1`
- `const_repeat` の繰り返し数
- `time_diff` や `get_traj_var(..., derivative_wrt="time")`

## `[trajectory]`

軌道問題で使います。標準では `bspline` と `linear` が使えます。

### `type = "bspline"`

```toml
[trajectory]
type = "bspline"
var = "p"
degree = 4
num_ctrl_points = 8
```

よく使う補助キー:

- `q_dim`: 1 ステップ分の次元
- `steps`: 時系列長。`[time]` があれば省略可
- `knot_vector`, `u_samples`: 必要なら明示

`q_dim` は、省略しても `var` の次元と `num_ctrl_points` から推定できる場合があります。

`[trajectory].var` は `get_traj_var` が参照する既定の軌道パラメータ変数名です。  
backend 用の trajectory helper では、これを省略すると `"p"` が使われます。

### `type = "linear"`

```toml
[trajectory]
type = "linear"
var = "p"
steps = 3
q_dim = 2
A = [
  [1.0, 0.0],
  [0.0, 1.0],
  [2.0, 0.0],
  [0.0, 2.0],
  [3.0, 0.0],
  [0.0, 3.0],
]
b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

`A` は `(steps * q_dim, p_dim)` の行列です。`b` は省略すると 0 になります。

## 関節角度や時間微分を入れる方法

関節角度 `q`、速度 `qdot`、加速度 `qddot` などを DSL に入れるときは、
まず「その量をどこから読むか」を決めます。

### 1. 決定変数そのものなら `get_var`

最適化変数がそのまま関節角度なら `get_var` を使います。

```toml
[[variables]]
name = "q"
dim = 2
init = [0.0, 0.0]

[[terms]]
[terms.expr]
type = "sub"

[terms.expr.a]
type = "get_var"
var = "q"

[terms.expr.b]
type = "const"
var = "q"
value = [0.0, 0.0]

[terms.cost]
type = "scalar_weight"
w = 1e-3
```

これは「関節角度 `q` を 0 に近づける正則化項」です。

### 2. 軌道パラメータ `p` から得るなら `get_traj_var`

軌道最適化では、決定変数が `q` そのものではなく軌道パラメータ `p` であることが多いです。  
このときは `get_traj_var` で `q(k)` やその時間微分を読みます。

```toml
[time]
N = 20
dt = 0.1

[trajectory]
type = "bspline"
var = "p"
degree = 4
num_ctrl_points = 8

[[variables]]
name = "p"
init = { fill = 0.0 }
```

関節角度軌道:

```toml
[terms.expr]
type = "get_traj_var"
var = "p"
```

関節速度軌道:

```toml
[terms.expr]
type = "get_traj_var"
var = "p"
derivative_order = 1
derivative_wrt = "time"
```

関節加速度軌道:

```toml
[terms.expr]
type = "get_traj_var"
var = "p"
derivative_order = 2
derivative_wrt = "time"
```

端点だけを取りたいなら `k = 0` や `k = "last"` を付けます。

### 3. backend が計算する量なら `get_state`

関節角度が backend state として供給されるなら `get_state` を使います。
IK の `q`、順動力学や逆動力学から得る `torque` なども同じです。

```toml
[terms.expr.a]
type = "get_state"

[terms.expr.a.key]
k = 0
owner_type = "total_joint"
owner_name = "robot"
dtype = "coord"
field = "q"

[terms.expr.a.jac]
var = "p"
```

backend が時間微分を field として直接返せるなら、それも `get_state` で読めます。

```toml
[terms.expr.inner.key]
owner_type = "total_joint"
owner_name = "robot"
dtype = "dynamics"
field = "torque_d1"
```

### 4. 離散時間差分として作るなら `time_diff`

任意の時系列ベクトルから「隣接ステップ差分」を作りたいときは `time_diff` を使います。

```toml
[terms.expr]
type = "time_diff"
segment_dim = 2
wrt = "time"

[terms.expr.base]
type = "get_traj_var"
var = "p"
```

これは一般に `dq/dt` の離散近似として使えます。  
`base` は `get_var` でも `stack(get_state(...))` でも構いません。

### 5. backend state を時系列化してから差分を取るなら `stack` + `time_diff`

backend が各時刻の状態を返すが、時間微分 field は直接持っていない場合は、
まず `stack` で `k=0..last` を縦に並べ、その後 `time_diff` をかけます。

```toml
[terms.expr]
type = "time_diff"
segment_dim = 2
wrt = "time"

[terms.expr.base]
type = "stack"

[terms.expr.base.range]
k0 = 0
k1 = "last"

[terms.expr.base.inner]
type = "get_state"

[terms.expr.base.inner.key]
owner_type = "total_joint"
owner_name = "robot"
dtype = "coord"
field = "q"

[terms.expr.base.inner.jac]
var = "p"
```

この形にしておくと、「backend が返す任意の時系列量」に対して同じ書き方を使えます。

## 評価関数に入れるか、制約として入れるか

DSL 上では、どちらも「残差を作る term」を書く点は同じです。  
違いは `constraint` を付けるかどうかです。

### 1. 評価関数に入れる

`constraint` を付けずに term を書きます。

```toml
[[terms]]
[terms.expr]
type = "get_traj_var"
name = "qdot_traj_regularization"
var = "p"
derivative_order = 1
derivative_wrt = "time"

[terms.cost]
type = "scalar_weight"
w = 1e-4
```

これは「関節速度を小さくしたい」という目的関数項です。

### 2. 等式制約として入れる

目標との差が 0 になってほしいなら `sub(a, b)` を作って `constraint = "eq"` を付けます。

```toml
[[terms]]
constraint = "eq"

[terms.expr]
type = "sub"
name = "qdot_terminal"

[terms.expr.a]
type = "get_traj_var"
var = "p"
derivative_order = 1
derivative_wrt = "time"
k = "last"

[terms.expr.b]
type = "const"
var = "p"
value = { fill = 0.0 }

[terms.cost]
type = "scalar_weight"
w = 100.0
```

これは「終端速度を 0 にしたい」という等式制約の書き方です。

### 3. 不等式制約として入れる

上限・下限は `bounds` で書けます。これは内部では violation が 0 なら
feasible になる `hinge` residual に展開されます。

```toml
[[terms]]
name = "joint_q_bounds"
kind = "ineq"
quantity = "joint_angles"
bounds = { lower = { fill = -3.14159 }, upper = { fill = 3.14159 } }
weight = 1e5
```

内部的には次の形と同じ意味です。

```text
vstack(
  hinge(q - upper),
  hinge(lower - q),
)
```

片側だけなら `bounds = { upper = ... }` または `bounds = { lower = ... }`
だけでも使えます。

## 実務上の使い分け

関節量を DSL に入れるときは、次の順で考えると整理しやすいです。

- 量が最適化変数そのものなら `get_var`
- 量が軌道 `q, qdot, qddot` なら `get_traj_var`
- 量が backend 計算値なら `get_state`
- 任意の時系列量の差分なら `time_diff`
- backend state の時系列化が必要なら `stack` + `time_diff`

その上で、

- ただの評価項なら `constraint` を付けない
- 等式として扱いたいなら `constraint = "eq"`
- 不等式として扱いたいなら `constraint = "ineq"`

と分けるのが一番一般的です。

## `get_state` と `build_state()`

`get_state` を使う場合、backend 側の `build_state()` は「値」と「ヤコビアン」を返せる必要があります。

たとえば次の DSL:

```toml
[terms.expr.a.key]
k = 0
owner_type = "total_joint"
owner_name = "robot"
dtype = "coord"
field = "q"

[terms.expr.a.jac]
var = "p"
```

backend には概念的に次の 2 つが要求されます。

- 値: `(k=0, owner=robot, dtype=coord, field=q)`
- ヤコビアン: `(k=0, owner=robot, dtype=coord, field=q_J_p)`

つまり、DSL は「何を読むか」を宣言し、実際の数値は `build_state()` が供給します。

## よくある書き方

### 1. 静的 IK

`get_state - const` をそのまま最小化します。

```toml
[[variables]]
name = "q"
dim = 2
init = [0.0, 0.0]

[[terms]]
[terms.expr]
type = "sub"

[terms.expr.a]
type = "get_state"

[terms.expr.a.key]
k = 0
owner_type = "link"
owner_name = "ee"
dtype = "kinematics"
field = "pos"
frame = "world"

[terms.expr.a.jac]
var = "q"

[terms.expr.b]
type = "const"
var = "q"
value = [1.5, 0.5, 0.0]
```

### 2. 軌道の端点拘束

`k = 0` や `k = "last"` を使って初期・終端条件を書きます。

```toml
[time]
N = 20
dt = 0.1

[trajectory]
type = "bspline"
var = "p"
degree = 4
num_ctrl_points = 8

[[variables]]
name = "p"
init = { fill = 0.0 }

[[terms]]
constraint = "eq"

[terms.expr]
type = "sub"

[terms.expr.a]
type = "get_traj_var"
var = "p"
k = "last"

[terms.expr.b]
type = "const"
var = "p"
value = [1.57, 0.0]

[terms.cost]
type = "scalar_weight"
w = 100.0
```

## 迷ったときの指針

- まずは `get_var`, `const`, `sub`, `scalar_weight` だけで組み立てる
- backend 状態が必要になったら `get_state` を足す
- 軌道問題なら `[time]`, `[trajectory]`, `get_traj_var` を使う
- 制約として扱いたい項目には `constraint = "eq"` または `"ineq"` を付ける
- 多変数問題では `var` / `jac.var` を省略しない

通常は `examples/spec/*.toml` を入口にしてください。DSL は spec 変換後の
内部表現や高度な式を直接扱うときのために残しています。
