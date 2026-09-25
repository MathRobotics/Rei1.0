# ソルバの履歴

## LM（現在の既定）

`solve(runtime)` は LM を実行します。反復履歴は `outcome.history`、試行の詳細は
`outcome.trial_history` です。`history_path="history.jsonl"` を指定すると
`history.jsonl` と `history.lm_trials.jsonl` に分かれます。詳細の保存先は
`trial_history_path` で変更できます。表示には `λ`・`ρ`・採否を含みます。
LM では棄却も一反復です。詳しくは [LM の基準実装](levenberg-marquardt.md) を参照してください。

verbose の列見出しは初回と20反復ごと（20・40・60…）に再表示します。
同じ反復の再試行や最終行では重複表示しません。LM・Gauss–Newton 共通で、
保存する JSONL や `format_solver_history()` の一括表示には影響しません。

以下は `solver="gauss_newton"` で再現できる従来方式の説明です。

## Gauss–Newton（従来方式）

Rei の Gauss–Newton は、反復履歴と line search の詳細を分けて保存します。
LiteOpt と IOC の専用ソルバは今回の対象外です。

Gauss–Newton は外部の `on_iter` なしで、初期状態・各反復・探索失敗・最終結果を
実行中に表表示します（`verbose = True` が既定）。表示を止める場合は
`options={"verbose": False}` を指定してください。保存の有無とは独立です。
line search の詳細は通常の反復表示には混ぜません。
表の数値は有効数字3桁（例：`1.23e-04`）で表示します。JSONL の数値は丸めません。
表の `Δobj` は `delta_objective`（目的関数値の差分）です。

```python
from rei import solve, format_solver_history

outcome = solve(runtime, solver="gauss_newton", options={
    "history_path": "history.jsonl",
    "line_search_history_path": "line_search.jsonl",
})
# 反復履歴は solve の実行中に自動表示される
# 探索の詳細が必要な場合だけ表示
print(format_solver_history(outcome.line_search_history))
```

- `history.jsonl` / `outcome.history`：初期状態・反復・探索失敗・最終状態。
- `line_search.jsonl` / `outcome.line_search_history`：各探索の全試行と探索終了。

`line_search_history_path` を省略すると、`history_path` と同じディレクトリに
`<history の拡張子を除いた名前>.line_search.jsonl` を作成します。
例えば `history.jsonl` の詳細は `history.line_search.jsonl` です。
分離後の形式は `schema_version = 2` です。

JSONL は 1 イベント 1 行です。実行中に追記するので、途中終了しても保存済みの
行は読めます。保存先の親ディレクトリは事前に作成してください。
既存ファイルへの追記・上書きはせず、同名のファイルがあればエラーにします。

## 記録するイベント

| `event` | 内容 |
|---|---|
| `initial` | 初期状態。`iteration = 0` |
| `line_search_trial` | 候補の評価。棄却された試行も含む |
| `line_search_end` | 探索結果と試行数・終了理由 |
| `iteration_end` | 採用された更新後の状態 |
| `iteration_failed` | 更新されなかった探索の要約。状態は探索開始時の点 |
| `iteration_retry` | 未収束のため設定を変更して再試行する通知 |
| `line_search_retry` | 詳細ファイルの再試行通知。変更前後の設定を含む |
| `final` | 最終解の状態とソルバの終了理由 |

試行・探索終了・`iteration_failed` の `iteration` は、その探索が目指す更新番号です。
初回の探索は `1`、失敗した場合の最終状態は更新前の `0` になります。
`trial` は各探索内で 1 から始まります。探索を無効にした場合、探索イベントは出ません。

`iteration_end` と `iteration_failed` には `line_search_trials`（試行数）、
`step_scale`（採用倍率）、`line_search_status`（探索結果）を残します。
探索失敗では倍率は `null`、探索無効では試行数は `0`・結果は `disabled` です。
詳細ファイルとは `iteration` で対応づけます。探索を行わなければ詳細ファイルは空です。

line search が失敗した場合は探索開始点に戻し、その点で `Jᵀr` を再評価します。
`max(abs(Jᵀr)) <= tol_grad` なら `converged`、それ以外は下記の再試行を行います。
前者の最終理由は `gradient_tolerance_after_line_search_failure` です。
`iteration_failed` は探索・更新の失敗を表し、最適化全体の収束状態とは区別します。
失敗による更新量ゼロだけでは収束と判定しません。

## 採否と収束判定

収束は `max(abs(Jᵀr)) <= tol_grad` を必須にします。残差が `tol_r` より小さいだけ、
または更新量が `tol_dx` より小さいだけでは収束・終了としません。
初期点、各反復、最後に許された更新後の点でも勾配を確認します。

通常の探索は Armijo 条件（`c_armijo = 1e-4`）で十分な目的関数の減少を要求します。
数値分解能の目安 `8 * eps(float64) * max(abs(objective), abs(trial_objective))`
以下の変化は、目的関数の差だけでは採用しません。
この領域では候補点の勾配を追加評価し、次の両方を満たす場合だけ
`gradient_progress` として採用します。

- 目的関数の増加が上記の数値分解能以内である。
- 勾配の無限大ノルムが半分以下になる、または `tol_grad` を満たす。

したがって、丸め誤差範囲で `Δobj` がわずかに正でも、勾配が改善した更新は採用できます。
目的関数の単調減少より、数値分解能付近での確認可能な勾配改善を優先する処理です。
評価した勾配・数値分解能・必要減少量を詳細履歴に残します。
勾配が改善せず更新量も `tol_dx` 以下なら、候補を棄却して元の点から再調整します。
有意な減少がある小ステップは採用して反復を続けます。
表現できない更新や再試行上限による停止は引き続き `stalled` とし、
達成不可能な勾配精度を保証するものではありません。

## line search の再試行

未収束の探索失敗に対して、同じ探索開始点から次の順に再試行します。

1. 試行数上限による失敗なら、上限を一度だけ2倍にして再探索します。
2. 拡張しても失敗する場合、または最小倍率まで縮めても失敗する場合は、
   ダンピングを増やして更新方向を再計算します。
   ただし、数値分解能付近で更新量が小さすぎる場合は、反復内で一度だけ
   ダンピングを下げて方向を再計算することを優先します。
3. 毎回、失敗後の勾配を確認します。再試行上限またはダンピング上限に達し、
   未収束のまま改善点が見つからなければ `stalled` で終了します。

```python
options = {
    "ls_max_retries": 3,        # 各反復で最初の探索に追加できる再試行数
    "damping_increase": 10.0,   # ダンピングを増やす倍率
    "damping_max": 1e12,       # 増加後のダンピングの上限
    "damping_decrease": 0.1,   # ダンピングを下げる倍率
    "damping_min_factor": 100.0, # λ の相対下限の係数
    "c_armijo": 1e-4,         # 通常の探索に必要な減少の係数
}
```

`ls_max_retries = 0` で再試行を無効にできます。探索自体を無効にした場合も再試行しません。
各線形化点で `λ_min = damping_min_factor * eps * max(diag(JᵀJ))` を計算します。
初期値、増加後の値、減少後の値のすべてをこの下限以上に保つため、`damping=0` でも
実質的に無減衰な値にはなりません。増加時は
`min(max(damping_max, λ_min), max(λ_min, damping * damping_increase))` です。
したがって、絶対上限 `damping_max` が数値下限より小さい場合は、数値下限を優先します。
フルステップを採用した場合、または改善を確認した小ステップを採用した場合は、次の
反復でダンピングを `damping_decrease` 倍に下げますが、下限未満にはしません。
反復履歴の `damping` は実際に使った値、`damping_min` はその点の相対下限、
`next_damping` は次回の値です。
`ls_min_step` と `ls_beta` は変更しません。各再探索は倍率 `1` から始まり、
拡張した試行数上限はその反復内だけで使います。

再試行で変数を更新したり反復番号を増やしたりはしません。
詳細は `(iteration, retry, trial)` で識別します。最初の探索は `retry = 0`、
再試行は `1` から始まり、`trial` は各探索内で `1` に戻ります。
`line_search_trials` は同じ反復での全試行数の合計です。
`iteration_retry` / `line_search_retry` に変更前後のダンピング・試行数上限を残し、
各試行にも実際に使用した設定を記録します。表の再試行行では `λ` がダンピング、
`ls` が試行数上限です。

最終理由は再試行上限で `line_search_retry_limit`、ダンピング上限で
`line_search_damping_limit` です。評価例外は再試行せず記録して再送出します。

- `objective`：重み付き残差の二乗和 `||r||²`。半分にはしません。
- `delta_objective`：現在の目的関数値 − 直前の採用状態の値。初期ステップは `0.0`。
  改善時は負です。探索試行は探索開始時の値との差で、棄却試行同士の差ではありません。
  更新のない失敗行・最終行は通常 `0.0`、評価不能な場合は `null` です。
- `residual_norm`：`||r||₂`。
- `jt_r_inf_norm`：`max(abs(Jᵀr))`。目的関数の勾配そのものは `2 Jᵀr` です。
- `step_norm`：実際の更新量（試行では候補の更新量）の L2 ノルム。
  初期状態では `null`。最終行だけは従来の `SolveStats.step_norm` と一致させます。
  更新前に収束した場合、最終行は最後に採用した更新量ではなく、終了判定時の値です。
- `step_scale`：更新方向に掛けた倍率。`1` はフルステップ。
- `elapsed_seconds`：記録開始からの経過秒数。
- `reason`：採用・棄却・終了の理由。`accepted`、`gradient_progress`、
  `insufficient_decrease`、`roundoff_no_gradient_progress`、`non_finite`、
  `max_trials`、`step_too_small`（倍率下限）、`tiny_update`（更新量の停滞）など。

各状態行の目的関数・残差・勾配は同じ点で評価します。通常の試行では残差のみ評価し、
数値分解能付近の試行では勾配も評価します。
非有限の数値は JSON の `null` とし、試行の `reason = "non_finite"` で区別します。
評価で例外が発生した試行は `evaluation_error` を記録し、点を探索開始時に戻して
例外を再送出します。異常中断時は `final` 行がない場合があります。

## 保存量と座標

```python
options = {
    "history": True,           # メモリへの保存（既定）
    "history_vectors": False,  # 全成分は省略（既定）
    "history_path": "history.jsonl",  # ファイル保存は任意
    "line_search_history_path": "line_search.jsonl",  # 詳細の保存先（省略時は自動命名）
}
```

`history_vectors = True` では、状態行に `variables` と `jt_r` の配列を追加します。
試行行には候補の `variables` を追加します。成分はソルバへ渡した変数の順序です。
nullspace 縮約を使う場合は縮約後の座標であり、元の軌道パラメータではありません。
この勾配は、元の制約付き問題全体の KKT 診断を置き換えるものではありません。

`history = False` は両方のメモリ保存を無効にしますが、指定されたファイルには保存します。
`line_search_history_path` だけを指定して探索詳細のみファイル保存することもできます。
メモリ保存・両方のパス・verbose を無効にすれば記録処理を行いません。既存のヤコビアン評価は再利用します。
反復上限など、最終状態の勾配が未評価の場合に限り、その評価が追加されます。

既存の `on_iter` は互換性のため従来の更新前通知を維持します。
正式な反復履歴には `outcome.history` を使用してください。
`format_solver_text_log` も反復と探索要約を表示し、試行の詳細は混ぜません。

## 軌道最適化の例

```bash
PYTHONPATH=. .venv/bin/python examples/robokots_trajectory_dynamics.py \
  --solver gauss_newton --show-history --history history.jsonl --line-search-history line_search.jsonl
```

Pinocchio の例も同じ引数を受け付けます。
反復は既定で実行中に表示されます。`--quiet` で表示を止められます。
探索詳細の表表示には `--show-line-search` を追加してください。
全成分も保存する場合は `--history-vectors` を追加してください。
