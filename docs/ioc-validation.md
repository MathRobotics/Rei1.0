# IOCの妥当性検証

2026-09-22。基準 `c03e6ce` の `estimate_ioc_weights()` を、既知解と反例で検証した。
検証コードは `tests/test_ioc_validation.py`。

## 結論

1. 既定の列正規化を使うと、内部係数をそのまま元の目的関数の重みとして返していた。修正済み。
2. 制約のある観測では、目的項の停留条件だけから生成時の重みを推定できるとは限らない。既定モードはKKT乗数を推定しない。この制限を診断に明示し、2026-10-14に明示選択のKKT推定を追加した（下記）。
3. 重複項やゼロ勾配項では、重みを一意に復元できない。階数・除外項・停留残差の診断を追加した。
4. 識別可能な無制約問題では、既知重みの復元と再最適化を確認した。

## 正規化の不具合と修正

目的項を `f0 = x²/2`, `f1 = (x-2)²/2`、観測を `x=0.5` とすると、
勾配行列は `[0.5, -1.5]`。重みの合計を1とする正解は `[0.75, 0.25]`。

修正前の既定設定は `[0.5, 0.5]` を返し、元スケールの停留残差ノルムは0.5だった。
一方、正規化後の残差は0なので、内部ソルバーだけを見ると成功に見えていた。

`A_scaled = A diag(scales)` を使い、内部係数を `z` とすると、
元の係数は `scales * z`。これを合計1へ再正規化するよう変更した。

戻り値の意味:

- `weights`、`terms[*].weight`、`active_terms[*].weight`: 元の非加重目的項に対応する係数。合計1。
- `scaled_weights`: 列正規化後の単体問題で求めた内部係数。旧 `weights` に相当する表現。
- `stationarity.ikkt_residual`: `A @ weights`。
- `stationarity.ikkt_residual_scaled`: `A_scaled @ scaled_weights`。
- `simplex`: 内部の正規化後問題の求解結果。

有効項がない場合は従来どおりゼロ重みを返すが、これは合計1の推定結果ではないと診断する。
既存の呼び出し側は、正規化後係数を必要とする場合に `scaled_weights` を使用する。
ノイズがある場合は正規化方式が近似問題の解に影響する。今回の修正はその差を消すものではない。

## 制約付きの反例

上記と同じ目的項の真の重みを `[0.1, 0.9]` にすると、無制約の最適点は `x=1.8`。
等式 `x=0.5`、または不等式 `x<=0.5` を加えると、観測は `x=0.5` になる。

真の重みでの勾配は `-1.3`。制約ヤコビアンが1なので、乗数1.3でKKT停留条件を満たす。
不等式の場合も乗数は非負で、実行可能性と相補性を満たす。

既定の目的項停留条件によるIOCはこの観測から `[0.75, 0.25]` を推定する。目的項の残差がゼロになっても、
生成時の重みを復元した証拠にならない。`include_constraints=True` にしても、
制約項の `J.T @ residual` は境界でゼロになり、KKTの `J.T @ multiplier` の代わりにはならない。

今回の対応:

- `validation.constraint_term_indices` に検出した制約項を記録。
- `validation.constraint_kkt_checked=False` と診断メッセージを返す。
- 制約項がある場合、`identifiability.unique_stationary_weights` を真にしない。

追加したKKTモードでは等式制約の自由な乗数、不等式制約の非負乗数と活動集合を扱う。
ただし実装しても、一つの制約境界の観測だけでは重みは一意に決まらないことがある。
現在の診断は保守的で、非活動制約や事前に削減済みの制約を自動的に対応済みとは判定しない。

## 識別性とノイズ

- 重複目的項: 同じ目標0を持つ2項と目標2を持つ1項では、最初の2項の重みの合計しか決まらない。
- ゼロ勾配項: 観測で勾配がゼロの項を除外しても、その真の重みがゼロとは限らない。
- すべてゼロ: 重みを決める情報がなく、推定成功とは扱わない。
- 不整合な観測: 2成分の目標 `[0,0]`／`[2,4]` に対し、観測 `[0.5,1.1]` は単一の正の重み比と整合しない。停留残差が残ることを確認。

`identifiability` は、有効項の勾配行列へ合計1の条件を追加した行列の階数・特異値・零空間次元を返す。
階数不足は線形方程式だけでは一意性を示せないという診断であり、非負制約の境界を考慮した完全な一意性判定ではない。
`unique_stationary_weights` はこの観測での線形停留条件に関する保守的な判定で、非凸問題の大域最適性や真の行動原理を保証しない。

## 検証ケース

23ケースを追加:

- 4つの正規化方式で `[0.75,0.25]` を復元し、再最適化で `x=0.5` を再現。
- 2変数・3目的の識別可能な問題で `[0.2,0.3,0.5]` を復元。
- 一つの目的項を0.01／100／1e6倍しても、係数を元の単位へ戻すと同じ比率になることを確認。
- 重複項、ゼロ勾配、全ゼロ、不整合な観測を診断。
- 等式／不等式と `include_constraints` の組み合わせで、制約付き推定の制限を確認。
- 不正な正規化epsilonを拒否。
- Pinocchio／RoboKots Rustで2自由度モデルの2つの手先目標を使い、既知重みからデモを生成。
  IOC側の参照重みは1に戻して推定し、重み誤差1e-7以内、元スケール停留残差1e-8未満、再最適化の関節角誤差1e-7以内を確認。

ロボット検証は静的IKであり、動力学主体の軌道・実測データ・大規模な複数デモでの復元までは検証していない。
Rust側の再最適化は既定の勾配閾値1e-10で `stalled` となったため、
両バックエンドを `tol_grad=tol_dx=1e-8` に揃えた。重み・停留残差・再現誤差の上記許容値は維持した。

追加テストでPinocchioのモックによるモジュール上書きも判明したため、
モックを各テストのアダプタ参照だけに適用し、実ライブラリの統合テストを汚染しないよう修正した。

実行:

```sh
PYTHONPATH=. python -m pytest tests/test_ioc_validation.py -q
PYTHONPATH=. MPLBACKEND=Agg python -m pytest -q
```

## 制約付きKKT推定（2026-10-14）

`rei.equations.estimate_stationarity_weights(source)` が汎用の目的項停留条件による推定、
`estimate_kkt_weights(source, KktConstraintData(...))` が制約付き推定を担う。
いずれも `StationaritySource` を受け取り、ロボットのコンパイル処理に依存しない。
既存の `estimate_ioc_weights(compiled)` は接続とバックエンド診断を担当する。

```python
result = estimate_ioc_weights(compiled, p=demonstration, constraint_mode="kkt")
print(result["weights"])
print(result["kkt"]["equality_multipliers"])
print(result["kkt"]["inequality_multipliers"])
print(result["kkt"]["ok"], result["identifiability"]["unique_stationary_weights"])
```

KKTモードは以下を同時に満たす最小残差解を求める。

- `A @ w + Jh.T @ lambda + Jg.T @ mu = 0`
- `w >= 0`, `sum(w) = 1`、等式乗数 `lambda` は自由、`mu >= 0`
- `h = 0`, `g <= 0`、非活動行の `mu = 0`、`mu * g = 0`

等式法線をSVDで除き、非負係数と重みの合計条件を持つ最小二乗を活動集合法で解く。
各活動面は零空間の最小二乗で解き、NumPyだけで動作する。
不等式は `linearize_inequality_constraints()` の符号付き余裕とヤコビアンを使う。
境界で勾配がゼロになるHingeペナルティを使用しない。

`kkt_active_tol` は活動集合の判定、`kkt_feasibility_tol` は実行可能性・相補性の判定に使用する。
`simplex_max_iters` と `simplex_tol` はこのモードでも反復上限・停留判定を制御する。
`include_constraints=True` は拒否する。制約は重み付き目的項ではなく乗数として扱う。
`active_mode` にかかわらずゼロ勾配を含む全目的項を保持する。
`simplex_method` はKKTモードには使用しない（専用の活動集合法を使用）。

`stationarity.ikkt_residual` はKKTモードでは**乗数を含む**停留残差になる。
目的項だけの勾配は `stationarity.objective_gradient` で確認できる。
乗数の単位は重みを合計1にした元の目的項の `J.T @ r` に対応する。
内部係数 `scaled_weights` と元の係数 `weights` の変換は既定モードと同じ。
KKT求解結果は `kkt`、既定モードの単体求解結果は `simplex` に置く。

KKT適合と識別性は別の判定であり、制約が重みの違いを吸収する例では成功したKKT解でも
`unique_stationary_weights=False` を返す。等式・活動不等式の法線と重み合計条件を含む行列から、
乗数だけの零空間を差し引いた重みの自由度を `weight_affine_nullity` として返す。
非負境界を使う厳密な一意性判定ではなく、保守的な線形階数判定である。
KKT適合だけで非凸問題の最小性や生成時の行動原理を保証しない。

`tests/test_ioc_kkt.py` は等式の許容方向で既知重み `[.75, .25]` を復元し、
活動不等式の正の乗数、非活動乗数ゼロ、逆向き法線、実行不可能性、相補性違反、
従属法線、ゼロ・重複目的項、反復上限を検証する。
凸二次問題の閉形式による再最適化も一致する。
100個のランダムな非負制約付き小問題は、SciPy SLSQPとの独立比較で目的値差 `4.44e-16` 以下だった。

実行例（追加依存不要）:

```sh
uv run --locked python examples/constrained_stationarity_ioc.py
```

動力学主体の軌道や実測データでの制約付き重み復元は、今後の検証対象とする。
