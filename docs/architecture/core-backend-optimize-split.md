# Core / Backend / Optimize 分離設計メモ

最終更新: 2026-02-16
ブランチ: `arch/core-backend-optimize-split`
ステータス: Phase 3 完了（互換 wrapper / legacy alias 削除済み）

## 1. 背景と目的

現在の `eiopt` は NLS 最適化ユースケースで統一されているが、今後は「最適化以外の問題」でも DSL/Expr/状態計算を再利用したい。

このため、責務を次の 4 層に分割する。

1. `core`: 状態・時間・軌道・Expr の評価と微分収集
2. `backends.state`: ロボティクス等の状態計算実装（`StateKey` を満たす）
3. `optimize`: NLS 問題定義・コスト・ソルバ・最適化向け builder
4. `optimize_backends`: `optimize` と backend state の接続コンパイル層

## 2. 依存方向ルール

許可する依存は以下のみ。

- `core` -> (なし)
- `backends.state` -> `core`
- `optimize` -> `core`
- `optimize_backends` -> `optimize`, `core`, `backends.state`

禁止する依存:

- `core` -> `backends.state` / `optimize` / `optimize_backends`
- `backends.state` -> `optimize`

## 3. パッケージ再編（ターゲット）

```text
eiopt/
  core/
    state_cache.py
    state_schema.py
    time_grid.py
    trajectory.py
    bspline.py
    expr/                # 新設
      types.py           # Variable / VariablePack / RuntimeContext / Expr
      nodes.py
      registry.py
      stdlib.py

  backends/
    state/               # 新設
      template.py
      spatial.py
      kots.py
      pinocchio.py

  optimize/              # 新設
    builder.py
    runtime.py
    problem.py
    costs.py
    solvers/
      dispatch.py
      gauss_newton.py
      nls.py
    dsl/
      io.py
      dsl_ops.py
      variable_utils.py
      environment.py
      trajectory_compile.py
    report.py
    term_gradient_matrix.py
    simplex_weight_solver.py
    reductions/
      nullspace.py
      matrix_scaling.py

  optimize_backends/     # 新設
    trajectory_adapter.py
    kots.py
    pinocchio.py
```

補足: 旧版では中間段階として互換 wrapper を経由したが、現在は削除済み。

## 4. API 方針（破壊的変更を許容）

主要 API は新名前空間を正とする。

- `compile_problem` -> `eiopt.optimize.builder.compile_nls_problem`
- `solve_runtime` -> `eiopt.optimize.solvers.dispatch.solve`
- `ProblemRuntime` -> `eiopt.optimize.runtime.NLSRuntime`
- backend compile helper -> `eiopt.optimize_backends.*`
- backend state builder -> `eiopt.backends.state.*`

旧 API / 旧 namespace は削除済み。

## 5. 変更フェーズ

### Phase 1 (完了)

- 設計ドキュメント追加
- `optimize` / `backends.state` / `optimize_backends` の namespace を追加
- 既存実装への薄い wrapper を追加（挙動は維持）

### Phase 2 (完了)

- `model/term.py` から core 側型 (`Variable`/`Expr`) と optimize 側型 (`Cost`) を分離
- `dsl/builder.py` を optimize 専用 builder として再配置
- `expr` と `dsl` の循環的な参照を段階的に解消

### Phase 3 (完了)

- import path を新 API に全面切替
- 旧 path の削除（破壊的変更）
- README / examples / tests を新構造へ統一

## 6. optional dependency 方針

`scipy` と `cyipopt` は `optimize` 配下の optional solver とし、以下を満たす。

- import は遅延ロード
- `pyproject.toml` は optional dependency 化を行う
- `numpy` のみで core + gauss_newton 最小系が動く

## 7. 成功条件

- `core` だけで Expr 評価・微分収集・StateCache が利用できる
- backend state 実装が optimize 非依存で import 可能
- optimize 機能は core 依存で完結
- optimize + backend の結合は `optimize_backends` に隔離される

## 8. 現時点のリスク

- optional backend (`robokots`, `pinocchio`) の import タイミングに注意が必要
- 外部ユーザーが旧 import path に依存している場合は破壊的変更になる
- `optimize_backends` の dynamics field 自動推論は DSL 記述に依存するため、
  backend 固有の拡張記法を導入する場合は推論ロジックの拡張が必要

## 9. このブランチでの実施済み項目（履歴）

注記: 以下には「途中段階で互換 wrapper 化した履歴」も含む。最終状態は
section 9 の末尾にある削除項目（旧 wrapper / 旧 alias の撤去）が正。

- `eiopt.optimize` / `eiopt.backends.state` / `eiopt.optimize_backends` namespace を追加
- `Variable`/`VariablePack`/`RuntimeContext`/`Expr`/`DirectVectorExpr` の実体を
  `eiopt.core.expr.types` に追加
- `ExprRegister` と Expr node 群の実体を `eiopt.core.expr` 側へ移動し、
  `eiopt.expr.*` は互換 wrapper 化
- `eiopt.model.term` は Expr 側型を re-export する互換レイヤへ変更
- Cost 実装 (`L2`/`Scalar`/`Diagonal`/`Huber`) の実体を `eiopt.optimize.costs` に移動
- `compile_problem` の実体を `eiopt.optimize.builder.compile_nls_problem` 側へ移動し、
  `eiopt.dsl.builder` を互換 wrapper 化
- `eiopt.optimize.problem` / `eiopt.optimize.runtime` に `NLSProblem` / `NLSRuntime`
  の実体を移動し、`eiopt.model.problem` / `eiopt.model.runtime` は互換 wrapper 化
- `eiopt.optimize.solvers` (`dispatch` / `gauss_newton` / `nls`) に solver 実体を移動し、
  `eiopt.solvers.*` は互換 wrapper 化
- `report` / `term_gradient_matrix` / `simplex_weight_solver` の実体を
  `eiopt.optimize.*` に移動し、旧モジュールを互換 wrapper 化
- `matrix_scaling` / `nullspace` の実体を `eiopt.optimize.reductions.*` に移動し、
  `eiopt.model.*` は互換 wrapper 化
- `eiopt.optimize_backends.trajectory_adapter` に実体を移動し、
  `eiopt.backends.trajectory_adapter` は互換 wrapper 化
- `eiopt.optimize_backends.kots` / `eiopt.optimize_backends.pinocchio` に
  trajectory compile 実装本体を移し、`compile_nls_problem` / `NLSRuntime`
  ベースへ置換
- `eiopt.backends.state.kots` / `eiopt.backends.state.pinocchio` に
  state builder 実装本体を移動し、`backends.state -> core` 依存へ整理
- `eiopt.backends.state.template` / `eiopt.backends.state.spatial` に
  共通ユーティリティ実体を移動し、`eiopt.backends._template` /
  `eiopt.backends._spatial` は互換 wrapper 化
- `eiopt.backends.kots` / `eiopt.backends.pinocchio` は互換 wrapper 化
  （`kots` は `StateType` 上書き互換のため薄い互換 subclass を維持）
- `eiopt.optimize.dsl.*` (`dsl_ops` / `io` / `environment` / `variable_utils` /
  `trajectory` / `trajectory_compile`) に実体を移動し、`eiopt.dsl.*` は互換 wrapper 化
- `trajectory` DSL helper 群を `eiopt.core.trajectory_dsl` に抽出し、
  `core.expr.stdlib` から参照して `core -> optimize` 依存を解消
- trajectory 系 examples の import を `eiopt.optimize_backends.*` /
  `eiopt.backends.state.*` の正規経路へ更新
- README / examples の canonical import を `eiopt.optimize.*` /
  `eiopt.optimize_backends.*` 中心へ更新
- top-level 旧 alias (`eiopt.compile_problem`, `eiopt.ProblemRuntime`) と
  旧 namespace export (`eiopt.dsl`, `eiopt.model`, `eiopt.solvers`) を削除
- `tests/test_namespace_layering.py` を legacy 同一性確認から
  canonical API と legacy 削除確認のテストへ更新
- `eiopt.optimize` から legacy alias (`compile_problem`, `solve_runtime`) を削除し、
  solver/builder の正規 entrypoint を `solve` / `compile_nls_problem` に統一
- `eiopt.optimize.runtime` から `ProblemRuntime` alias を削除
- `eiopt.optimize.problem` から `Problem` alias を削除
- `eiopt.optimize.builder` から `build_problem` / `collect_required` alias を削除
- 旧 wrapper モジュール (`eiopt/dsl/*`, `eiopt/model/*`, `eiopt/solvers/*`,
  `eiopt/report.py`, `eiopt/simplex_weight_solver.py`, `eiopt/term_gradient_matrix.py`)
  を削除
- 旧 wrapper モジュール (`eiopt/expr/*`, `eiopt/backends/_template.py`,
  `eiopt/backends/_spatial.py`, `eiopt/backends/trajectory_adapter.py`,
  `eiopt/backends/kots.py`, `eiopt/backends/pinocchio.py`) を削除
- `eiopt.backends` は `backends.state.*` のみを公開する最小 namespace に整理
