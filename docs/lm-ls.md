# LM + line search

`lm-ls` は LM の更新方向に Armijo バックトラックを適用するハイブリッドです。
古典的 LM の `levenberg_marquardt` とは別に選択します。

```python
out = solve(runtime, solver="lm-ls", options={
    "tol_grad": 1e-8,       # 既定値。収束判定は ||Jᵀr||∞ のみ
    "max_iters": 200,      # 採用された更新の上限
    "history_path": "history.jsonl",
})
```

方向は `(JᵀJ + λI) h = -Jᵀr` を拡大最小二乗で解いて求めます。
`f = ||r||²` に対し、`α = 1, β, β², …` と縮め、厳密な減少と
`f(x + αh) <= f(x) + 2 c α (Jᵀr)ᵀh` を満たす点を採用します。
実装では浮動小数点で実際に表現された変位を使用します。

- `α=1` で採用：λ を `damping_decrease=0.1` 倍に減らす。
- 短縮ステップで採用：λ を維持する。
- 探索失敗：同じ点で λ を `damping_increase=10` 倍にし、方向を再計算する。
  再試行は `ls_max_retries=3` 回まで。

初期 λ は `tau=1e-3` と `max(diag(JᵀJ))` の積です。`damping` で直接指定もできます。
各線形化点で `λ_min = damping_min_factor * eps * max(diag(JᵀJ))` を適用し、
`damping_min_factor=100` を既定とします。増加上限は `damping_max=1e12` で、
数値下限がそれより大きければ数値下限を優先します。
探索設定は `ls_beta=0.5`、`ls_min_step=1e-12`、`ls_max_iters=40`、`c_armijo=1e-4` です。

`tol_dx` は受け付けません。勾配未達の探索失敗・表現不能な更新は `stalled`、
更新数上限は `max_iters` です。`out.meta["reason"]` と
`out.meta["gradient_converged"]` で確認できます。有限精度や不正確なヤコビアンによって、
目標勾配に達する前に停滞する可能性はあります。丸め誤差内の目的関数変化を
勾配改善だけで採用する処理や、最急降下方向への切り替えはありません。

採用状態は `out.history` と `history.jsonl`、各探索試行は
`out.line_search_history` と `history.line_search.jsonl` に保存します。
`line_search_history_path` で探索履歴の出力先を指定できます。
`history_vectors=True` で変数・状態勾配も記録し、`verbose=False` で標準出力を無効にできます。
