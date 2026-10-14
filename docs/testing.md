# テストと依存更新

開発依存は `uv.lock` で固定する。Pythonの最低対応版は3.11。
RoboKotsは `pyproject.toml` のGitコミット指定を使い、移動するブランチを参照しない。
公開パッケージの依存範囲と、開発・CI用のロックは分けて扱う。
CIはuv 0.9.18を使用する。

## 通常のテスト

```sh
uv sync --locked --group dev
uv pip check
MPLBACKEND=Agg uv run --locked --group dev python -m pytest -q -rs
```

既定の収集対象は `tests/`。追加ライブラリが必要なテストは、未導入なら理由を表示してスキップする。
バックエンドのモックはテストごとに復元するため、実ライブラリの統合テストを置換しない。

既存環境を変更せず、Pythonごとに新規環境で再現する例:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/rei-py311 uv sync --locked --python 3.11 --group dev
MPLBACKEND=Agg /tmp/rei-py311/bin/python -m pytest -q -rs
```

外付けファイルシステムのリンク制約がある場合は、内部ディスクに環境を作り、
`uv sync` に `--link-mode copy` を加える。

## 実ライブラリの検証

RoboKotsのソースビルドにはRustツールチェーンが必要。

```sh
uv sync --locked --group dev --group kots --group pinocchio
uv pip check
MPLBACKEND=Agg uv run --locked --group dev --group kots --group pinocchio \
  python -m pytest -q -rs --fail-on-skip
```

`--fail-on-skip` はテスト内・モジュール収集時のスキップを検知し、終了コードを非ゼロにする。
既定では無効なので、最小依存環境のテストには適用しない。

[GitHub Actions](../.github/workflows/tests.yml) は次の構成。

- UbuntuのPython 3.11／3.13で、dev依存と本体の全テスト。
- UbuntuのPython 3.13で、実RoboKots RustとPinocchioの導入・インポート確認。
  DOC、URDF/JSON一致、モデル摂動、軌道更新、既知重みIOC、固定関節を
  スキップ禁止で検証する。依存一覧と実ライブラリのパスをログへ残す。

ネイティブcyipopt・liteoptの実ライブラリによる最適化は、このCI構成の対象外。
各外部ソルバーの接続はモックを使ったAPIテストで確認する。
共有ランナーの実行時間を性能の合否には使用しない。

## 依存更新

1. 通常実装とは分けた変更で、対象依存の範囲またはRoboKotsの `rev` を変更する。
2. `uv lock --upgrade-package <package>`（ソース変更なら `uv lock`）でロックを更新する。
3. ロックの差分を確認し、Python別の新規環境で上記テストと `uv pip check` を実行する。
4. 実行したOS・Python・依存バージョン・RoboKotsコミット・合格数・スキップ理由を記録してコミットする。

`--locked` は設定とロックが一致しない場合に失敗する。通常検証で再解決を暗黙に行わない。
過去の結果と今回の実行範囲は [Python互換性の記録](../developer/benchmarks/python_compatibility.md) を参照。
