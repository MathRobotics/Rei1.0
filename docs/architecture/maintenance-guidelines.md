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
| 求解ループとその引数検証 | 各ソルバー。dispatchは数値を黙って丸めない |
| バックエンド状態・ヤコビアンの取得 | 各バックエンドのadapter/API層 |
| IOCのコンパイル・推定の組み立て | `rei/optimize_backends/trajectory_ioc.py` |
| IOCの識別性・適用範囲の診断 | `rei/optimize_backends/_ioc_diagnostics.py` |

IOCの診断文言は数値判定の根拠にしない。階数・残差・制約有無から判定し、
説明メッセージの追加や翻訳で `unique_stationary_weights` が変わらないようにする。
既存の辞書形式の戻り値は維持する。

## 回帰確認

今回追加した `tests/test_api_maintainability.py` は、設定継承・非破壊の上書き、
1回の線形化、iteratorの依存指定、不正な設定での状態保存、0反復の互換性を確認する。
IOCの計算結果は既知解を使った既存の `test_ioc_validation.py` で確認する。

```sh
PYTHONPATH=. python -m pytest tests/test_api_maintainability.py tests/test_ioc_validation.py -q
PYTHONPATH=. MPLBACKEND=Agg python -m pytest -q
```

対応するPython環境ごとに実行する。実ロボットライブラリを使うテストのスキップを
成功と取り違えない。実行時間の改善はテスト成功だけでは主張せず、必要時に別途測定する。

今回の変更は主要API周辺の改善であり、リポジトリ全体の設計問題を網羅したものではない。
依存固定・CI・制約付きIOCなどの未完了項目は `docs/improvement-plan.md` で追跡する。
