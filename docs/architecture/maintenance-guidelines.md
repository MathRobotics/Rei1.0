# 変更時に維持するAPIと責務

2026-09-23の保守性レビューで、設定の継承・評価回数・入力検証・IOC診断の責務を整理した。

## 利用者が判断しやすいAPI

- アダプタを再度変換するとき、省略した設定は継承する。明示した設定は新しいビューに反映し、元のオブジェクトを変更しない。
- 適用できないフィルタや重み指定を黙って無視しない。汎用capabilityで扱えない指定はエラーにする。
- `linearize()` は同じ状態で評価した値と微分を返す。内部で値と微分を別々に再評価しない。特にgenerator形式の依存指定や状態を持つバックエンドで重要。
- Gauss–Newtonの引数は状態変更前に検証する。初期残差ゼロの場合にも検証を省略しない。
- ペナルティ残差と符号付き制約は別のAPI。`as_constraint_problem()` は従来の残差を返し、KKTは `linearize_inequality_constraints()` を使う。

## 実装を配置する場所

| 責務 | 配置 |
|---|---|
| 汎用capabilityへの変換 | `rei/problem/adapters.py`、`rei/flow/problem.py` |
| 式木の子要素と走査 | 各Exprの `children()`、`rei/core/expr/traversal.py` |
| バックエンド固有の式・微分積の一括評価 | `rei/optimize_backends/_robotics_products.py` |
| 求解ループとその引数検証 | 各ソルバー。dispatchは数値を黙って丸めない |
| Gauss–Newtonの反復・線探索・再試行・履歴 | `rei/optimize/solvers/_gauss_newton_loop.py` |
| 外部ソルバーの呼び出し・API差分の吸収 | `rei/optimize/solvers/scipy.py`、`cyipopt.py`、`liteopt.py` |
| 外部ソルバー共通の目的関数評価・初期点設定 | `rei/optimize/solvers/_external_common.py` |
| バックエンド状態・ヤコビアンの取得 | 各バックエンドのadapter/API層 |
| IOCのコンパイル・ランタイム接続 | `rei/optimize_backends/trajectory_ioc.py` |
| capabilityからのIOC推定・識別性の診断 | `rei/equations/ioc.py`、`_ioc_diagnostics.py` |
| 等式・活動不等式のKKT重み・乗数推定 | `rei/equations/ioc_kkt.py` |

IOCの診断文言は数値判定の根拠にしない。階数・残差・制約有無から判定し、
説明メッセージの追加や翻訳で `unique_stationary_weights` が変わらないようにする。
既存の辞書形式の戻り値は維持する。

汎用IOC推定は `StationaritySource` と符号付き制約データを受け取り、
コンパイル済みロボットを必要としない。既定の目的項停留条件による推定を維持し、
`constraint_mode="kkt"` で制約乗数を推定する。制約のペナルティ勾配を
乗数の代用にしない。KKT適合と重みの識別性は別々に判定する。

新しい合成Exprは `children()` で子要素を宣言する。レポート・描画・窓更新は
`walk_expr()` を利用し、式の内部属性名を独自に列挙しない。
走査はノードの同一性で重複・循環を除外する。既存のcustom Exprはdataclassまたは
従来の属性名による走査にも対応する。

汎用ランタイムは任意の `TermProductPlan` にまとめて評価できる項を委譲する。
ロボット状態名やバックエンド内部をランタイムから調べない。RoboKotsのコンパイル層が
builderを明示的に渡して実行計画を設定し、未設定・非対応の項は通常のExpr評価へ戻る。
層のimport方向は `tests/test_namespace_dependency_rules.py` で確認する。

密行列版とoperator版のGauss–Newtonは同じ反復制御を使う。密行列版は拡大最小二乗、
operator版はCGLS、Krylov版は差し替えた線形ステップを利用する。
LM・LM-LSのアルゴリズムは各専用実装に置く。

外部ソルバーの依存ライブラリは各求解関数内で遅延importする。
`dispatch.py` は設定解決・オプション検証・呼び出し先の選択を担当する。
`solve()` は設定を解決した要求を作り、選択した実装を1回呼び出す。
設定の継承・共通引数の検証・dispatch固有の既定値の補完は求解前に完了する。
数値反復や外部ライブラリのAPI差分は各ソルバーが担当する。
整数指定の回数・項番号を丸めない。liteoptのGD/GNで適用できないRei設定は
黙って無視せず、呼び出し前にエラーにする。ライブラリ固有の指定は
`backend_options` として各ソルバーへ渡す。
従来の `rei`、`rei.optimize`、`rei.optimize.solvers` および
`rei.optimize.solvers.dispatch` からのソルバー関数のimport経路を維持する。

## 回帰確認

今回追加した `tests/test_api_maintainability.py` は、設定継承・非破壊の上書き、
1回の線形化、iteratorの依存指定、不正な設定での状態保存、0反復の互換性を確認する。
IOCの計算結果は既知解を使った既存の `test_ioc_validation.py` で確認する。

```sh
uv run --locked --group dev python -m pytest tests/test_api_maintainability.py tests/test_ioc_validation.py -q -rs
MPLBACKEND=Agg uv run --locked --group dev python -m pytest -q -rs
```

既定の収集対象は `pyproject.toml` の `testpaths = ["tests"]` で指定する。
`examples/model` などのリンク先にある外部リポジトリのテストを収集しない。
任意バックエンドのモックはテストごとに導入・復元し、`sys.modules` や
親パッケージの属性に残さない。テストの収集時に代替モジュールを登録しない。

実RoboKots・Pinocchioも含めて検証する場合は、上記コマンドに
`--group kots --group pinocchio` を `uv run` のオプションとして追加する。

対応するPython環境ごとに実行する。実ロボットライブラリを使うテストのスキップを
成功と取り違えない。実行時間の改善はテスト成功だけでは主張せず、必要時に別途測定する。

今回の変更は主要API周辺の改善であり、リポジトリ全体の設計問題を網羅したものではない。
動的IOC・性能測定・配布環境の網羅などの残る課題は `docs/improvement-plan.md` で追跡する。
