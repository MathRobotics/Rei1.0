# Jv・JᵀvによるGauss–Newton

`solver="gauss_newton_operator"` は、従来の `gauss_newton` と別の実装です。
既定のLMおよび従来の密行列版GNのアルゴリズムは変更していません。

```python
from rei import solve

out = solve(runtime, solver="gauss_newton_operator", options={
    "max_iters": 200,
    "tol_grad": 1e-10,
    "damping": 1e-8,
    "line_search": True,
    "inner_tol": 1e-8,
    "inner_max_iters": 100,
    "history_vectors": True,
    "history_path": "gn_operator.jsonl",
    "verbose": False,
})
```

直接呼ぶ場合は `solve_gauss_newton_operator(problem, ...)` を使います。
`solve()` 経由の既定の反復上限は200、直接呼び出しでは既存GNと同じ20です。
密行列版との比較では、それぞれ同じ `x0` を渡してください。

## 既存版と共通の機能

- Armijoラインサーチ、探索回数の拡張、ダンピングを変えた再試行。
- ダンピングの増減・上限・スケールに応じた数値的下限。
- `tol_grad` による収束判定。小ステップだけでは収束としません。
- 目的関数の丸め誤差付近での勾配改善による試行採用。
- `x0`、`required`、runtimeの `weighted`・`term_indices`。
- `history`、`history_vectors`、`history_path`、`line_search_history_path`。
- `verbose`、3引数・4引数の `on_iter`、`profiler`。
- ラインサーチ棄却・評価例外時の点の復元。

目的関数の履歴は `||r||²`、勾配診断は従来通り `Jᵀr` です。
外側の反復上限は既存版と同様、更新の回数を制限します。

## ステップ計算

拡大最小二乗問題 `min_h ||Jh+r||² + λ||h||²` を、JvとJᵀvを用いた
CGLSで近似的に解きます。ソルバーは `J`、`JᵀJ`、拡大行列を生成しません。
NumPyだけで利用できます。

`inner_tol` は線形部分問題の正規残差
`||Jᵀ(Jh+r)+λh||₂ / ||Jᵀr||₂` の許容値です。
既定値は `1e-8`、`0 < inner_tol < 1` とします。
`inner_max_iters` は正の整数で、省略時は `max(20, 2*n_total)`。
内側の反復上限に達した場合は得られた近似ステップを外側のラインサーチに渡します。
内側の収束は、非線形問題全体の収束を意味しません。
積の形状不正・非有限値・CGLSの曲率計算の破綻は例外として報告します。

`out.meta["inner_solves"]` と状態履歴の `linear_solve` イベントに、各線形求解の
反復回数、`converged` / `max_iters`、正規残差ノルムと相対値を記録します。
`history=False` でもmetaの診断は残ります。

## 必要な問題インターフェース

汎用問題は `n_total`、`get_point()`、`set_point(x)`、`required_list(required)`、
`eval(required=...)`、`jvp(v, required=...)`、`vjp(w, required=...)` を提供します。
`linearize()` は不要です。積は現在の点における同じ残差の微分であり、互いに
転置の関係を満たす必要があります。`required` はキーワード引数です。
汎用問題に重みや項選択を指定する機能はありません。

runtimeでは `residual_jvp(direction, weighted=True, required=None, term_indices=None)`
と既存の `residual_vjp` を使用します。独自の式は
`jvp(ctx, tangents)` と `eval_value(ctx)` を提供できます。
`tangents` は `expr.vars` と同じ順序のベクトルのリスト、返り値は未加重残差の方向微分です。
Jᵀv側は既存の `vjp(ctx, rhs)` を使用します。

## メモリと実行時間の留意点

- 変数・定数・差・積み重ね・軌道変数の式にはJVPを用意しています。
  時間差分・成分選択・Hingeにも専用のJVPがあります。
  JVPのない式は `expr.eval(ctx)` の局所ヤコビアンブロックで補います。
  バックエンドも含めた完全な行列非生成は、その経路全体に専用の積がある場合に限ります。
- `DirectVectorExpr` はコールバックが返す局所ブロックでJVP/VJPを計算します。
  任意の独自式・コストにVJPが自動追加されるわけではありません。
- nullspace削減は `J_reduced v = J_full (N v)`、
  `J_reducedᵀ w = Nᵀ (J_fullᵀ w)` として積を伝播します。
  RoboKotsの対応する集約dynamics状態では、軌道上の方向をまとめて
  `jacobian_mul` に渡します。ヤコビアン全列ではなく1方向のバッチです。
  既定の状態依存は値だけを要求し、非対応バックエンドは必要時に局所微分を取得します。
- 既存GNと同じ下限 `damping_min_factor * eps * max(diag(JᵀJ))` を使います。
  通常は列ごとのJVPと行ごとのVJPのうち、積の回数が少ない方を使います。
  全列は保存しません。汎用問題が `jacobian_column_squared_norms(required=...)` を
  提供すると、この計算を省略できます。返り値は長さ `n_total` の有限・非負ベクトルです。
- 反復解法のため、密行列版と反復数や試行採否が完全一致する保証はありません。
  大規模問題での速度改善は未測定です。積の計算費用・条件数・内側の反復数に依存します。

RoboKotsの2関節・7自由度DOCを実測したところ、専用の積への接続後も密行列版の方が高速でした。
比較条件・時間・検証値は [DOCの実測報告](../developer/benchmarks/kots_doc_operator.md) を参照してください。

実装は `rei/optimize/solvers/gauss_newton_operator.py`、積とCGLSは
`rei/optimize/solvers/_jacobian_operator.py` に分離しています。
外側のループは密行列版を基にしており、共通の挙動を修正する際は両方の回帰確認が必要です。
