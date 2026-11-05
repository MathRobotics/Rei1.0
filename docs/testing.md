# テストと依存更新

開発依存は `uv.lock` で固定する。Pythonの最低対応版は3.11。
RoboKotsは `pyproject.toml` で `main` ブランチを指定し、検証するコミットは `uv.lock` に固定する。
MathRoboは `0.0.6` を使用する。
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

## Pythonバージョン別の一括検証

uvを導入し、リポジトリの [scripts/test-python.sh](../scripts/test-python.sh) を使う。
任意のディレクトリから実行でき、既存の `.venv` を変更しない。

```sh
./scripts/test-python.sh                         # 3.11 / 3.12 / 3.13 / 3.14、本体テスト
./scripts/test-python.sh 3.11 3.13               # 指定バージョンのみ
./scripts/test-python.sh 3.15                    # 3.11以降の別バージョンも指定可能
./scripts/test-python.sh --suite robotics 3.13   # NumPy / Pinocchio、スキップ禁止
./scripts/test-python.sh --suite all             # 本体とroboticsの両方
./scripts/test-python.sh 3.13 -- tests/test_runtime_integrity.py
```

既定の `core` はdev依存だけで全テストを収集し、任意バックエンドの未導入による
スキップを許可する。`robotics` はCIと同じNumPy／Pinocchioの検証で、Rustは含まない。
Rust版の対応確認は、後述の手順でPython・OS・CPUに合うwheelを用意して別途実行する。

環境は `.venv-python-tests/<version>-<suite>/` に分けて作成し、`uv.lock` に従って
正確に同期する。この専用環境に手動で追加したパッケージは同期時に削除される。
Pythonや依存がなければuvが取得を試みる。依存の対応wheelがない場合も失敗として記録し、
残りのバージョンの検証を続ける。1件でも失敗すればスクリプト全体が非ゼロで終了する。

ログと `summary.tsv` は `.python-test-results/<run-id>/` に保存する。
`core` のPASSは、任意バックエンドまで検証済みという意味ではないので、スキップ理由も確認する。
環境・ログの保存先を変える場合:

```sh
./scripts/test-python.sh --suite all 3.11 3.13 \
  --env-dir /tmp/rei-python-tests --log-dir /tmp/rei-python-results
```

`--` 以降はpytestへ渡す（例: `-- -x`）。`core` ではテストファイルの指定も可能。
対応宣言を自動で保証するものではなく、実行したバージョン・依存・スキップを記録する検証ツール。

## 実ライブラリの検証

RoboKots本体はPythonパッケージになり、インストールにRustは不要。
Rustバックエンドは別配布の `robokots-native` wheel（native API 1）が必要。
Python・OS・CPUに合うwheelを用意し、同じ環境に追加する。
詳細は [RoboKotsの導入手順](https://github.com/MathRobotics/RoboKots/blob/main/docs/RUST_CORE_SEPARATION.md) を参照。

```sh
uv sync --locked --group dev --group kots --group pinocchio
uv pip install --no-deps "/path/to/robokots_native-<version>-<platform-tags>.whl"
uv pip check
MPLBACKEND=Agg uv run --no-sync --locked --group dev --group kots --group pinocchio \
  python -m pytest -q -rs --fail-on-skip
```

`--fail-on-skip` はテスト内・モジュール収集時のスキップを検知し、終了コードを非ゼロにする。
既定では無効なので、最小依存環境のテストには適用しない。
通常の `uv run` は既定で追加パッケージを保持する。`--no-sync` は依存の同期そのものを省く任意指定。
再同期するときは `uv sync --inexact` を使うか、同期後にwheelを再インストールする。

[GitHub Actions](../.github/workflows/tests.yml) は次の構成。

- UbuntuのPython 3.11／3.13で、dev依存と本体の全テスト。
- UbuntuのPython 3.13で、実RoboKots NumPyとPinocchioの導入・インポート確認と、
  Rustパラメータを除いたモデル読込・摂動・次数推定・固定関節・軌道写像・サンプルCLIなどをスキップ禁止で実行。
  Rust専用テストが混在するURDF/JSON一致・数値／自動微分モジュールでは、`numpy` パラメータのみ実行する。

CIではRoboKotsのRustバックエンドを使用せず、ネイティブwheelもインストールしない。
Rustの検証は、対応wheelを導入したローカル環境で上記の全テストを実行する。

ネイティブcyipopt・liteoptの実ライブラリによる最適化は、このCI構成の対象外。
各外部ソルバーの接続はモックを使ったAPIテストで確認する。
共有ランナーの実行時間を性能の合否には使用しない。

## 依存更新

1. 通常実装とは分けた変更で、対象依存の範囲を変更する。RoboKotsは `branch = "main"` を維持する。
2. `uv lock --upgrade-package <package>` でロックを更新する。RoboKotsのmain更新は `uv lock --upgrade-package robokots` を使う。
3. ロックの差分を確認し、Python別の新規環境で上記テストと `uv pip check` を実行する。
4. 実行したOS・Python・依存バージョン・RoboKotsコミット・合格数・スキップ理由を記録してコミットする。

`--locked` は設定とロックが一致しない場合に失敗する。通常検証で再解決を暗黙に行わない。
過去の結果と今回の実行範囲は [Python互換性の記録](../developer/benchmarks/python_compatibility.md) を参照。
