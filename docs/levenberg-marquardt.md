# LM 法の基準実装

`solve()` の既定は `solver="levenberg_marquardt"` です。
Madsen・Nielsen・Tingleff の [Algorithm 3.16](https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf)
を基準にした、ゲイン比と Nielsen のダンピング更新による標準的な LM 法です。
LM には複数の更新則があります。「唯一の古典的 LM」という意味ではなく、
今後の改造を比較するために、この明示した方式を基準に固定します。

## 使い方と旧方式の再現

```python
# 新しい基準：古典的 LM
out = solve(runtime, solver="levenberg_marquardt", options={
    "max_iters": 200,
    "tol_grad": 1e-8,
    "tol_dx": 1e-12,
    "tau": 1e-3,
})

# 従来の動作：同じ問題・初期点・オプションで再現
old = solve(runtime, solver="gauss_newton", options=previous_options)
```

比較時は別の runtime を用意するか、両方に同じ `x0` を渡してください。
ソルバは runtime の点を変更するため、連続実行だけでは同じ初期条件になりません。
`nls()` の動作は変えていません。`solve_gauss_newton()` は利用可能なままですが、
数値的に無減衰となることを避ける相対ダンピング下限を持ちます。
既存コードで `solver="gauss_newton"` を明示している場合は、自動で LM に変わりません。
LM へ移行する際は `line_search`・`ls_*`・`c_armijo`・`tol_r`・旧ダンピング更新設定を
除去してください。LM はこれらを黙って無視せずエラーにします。

## アルゴリズム

残差を `r`、そのヤコビアンを `J`、`g = Jᵀr` とします。

1. 初期値：`λ = τ max(diag(JᵀJ))`、`ν = 2`。`damping` を指定すれば初期 `λ` を直接指定。
2. `(JᵀJ + λI)h = -g` を解く。
3. `ρ = 実際の目的関数減少量 / 線形化モデルの予測減少量` を求める。
4. `ρ > 0` なら採用し、`λ *= max(1/3, 1-(2ρ-1)³)`、`ν = 2`。
5. それ以外は点を戻し、`λ *= ν`、`ν *= 2`。同じ点の `r` と `J` を再利用する。

線形方程式は拡大最小二乗問題として `numpy.linalg.lstsq` で解き、正規方程式の
明示的な生成を避けます。目的関数と履歴は既存形式に合わせて `||r||²` です。
予測減少量も `-2gᵀh - ||Jh||²` とし、半分の係数が混在しないようにしています。

ラインサーチ、勾配改善による丸め誤差付近の採用、独自の再試行拡張は入れません。
棄却も一反復として `max_iters` に数えます。従来方式の「採用更新回数」とは異なります。
非有限な試行は棄却し、評価中の例外でも点を元に戻して例外を伝播します。

## 停止条件

- `||Jᵀr||∞ <= tol_grad`：`reason="gradient_tolerance"`。
- `||h|| <= tol_dx (||x|| + tol_dx)`：`reason="step_tolerance"`。

どちらも基準 LM では `status="converged"` ですが、後者は勾配条件の達成を保証しません。
`out.meta["gradient_converged"]` と最終履歴の同名フィールドで区別できます。
`tol_dx=0` は相対ステップによる収束判定を無効にします。ただし浮動小数点で点を
更新できない場合は `stalled`、反復上限では `max_iters` となり、無限には続けません。

## 履歴と改造箇所

`history_path="history.jsonl"` を指定すると次の二つを保存します。

- `history.jsonl` / `out.history`：初期点・各試行後に保持する点・最終点。
- `history.lm_trials.jsonl` / `out.trial_history`：候補点、採否、`damping`、
  `next_damping`、`gain_ratio`、`actual_reduction`、`predicted_reduction`。

詳細の保存先は `trial_history_path` で指定します。LM ではラインサーチ履歴は空です。
棄却後の状態行は目的関数差分と更新量がゼロで、候補の更新量は試行行に残ります。
初期点の目的関数差分はゼロ。小ステップ判定で候補評価前に終了した場合、
その反復には試行行はなく、最終行に終了理由を残します。

実装は `rei/optimize/solvers/levenberg_marquardt.py` に分離しています。
方向計算 `_lm_step` とダンピング更新 `_accepted_damping` が変更箇所の入口です。
基準実装を変える前に、別の方式として分離して比較できるようにしてください。
独立した正規方程式による参照実装との逐次比較テストを用意しています。
