# RoboKots DOC: Jv・Jᵀv経路と実測

この記録は `gauss_newton_operator`（CGLS版）の比較。後続の
[信頼領域Krylov版の実装・測定](kots_doc_krylov.md)は別途参照。

2026-09-25。**RoboKotsの専用積に接続しても、この2つのDOCでは密行列版GNより遅い。**
前回追加したoperator版には接続不足があり、それは今回修正した。

## 比較条件

- モデル: `examples/models/planar2.urdf` と `7_dof_arm.urdf`。
- 設定: `examples/spec/robokots_traj_dynamics_d12.toml`。
  201評価点、50個のB-spline制御点、2秒、境界条件はnullspace削減。
  既存の重みはそのまま（速度・加速度 `1e-4`、トルク `1e-10`）。
- 両方式で同じ初期点、`max_iters=10`、`tol_grad=1e-8`、履歴有効、標準出力無効。
  operator版は `inner_tol=1e-8`、既定の内側反復上限。
- RoboKots Rust、モデルorder=3。Python 3.13.1、NumPy 2.5.3、macOS ARM64。
- モデル読み込み・問題コンパイル・nullspace構築は求解時間に含めない。
  各方式を1回ウォームアップ後、3回測定。測定順を交互に反転。
  求解後、計測区間外で密なJを使って勾配を独立検証。
- `OPENBLAS_NUM_THREADS=1`、`VECLIB_MAXIMUM_THREADS=1`。
  ネイティブAPIの呼び出し回数・時間の計装を含む壁時計時間。
  プロファイラーの実行と、報告する時間測定は分離した。

メインの比較は、Reiの依存固定と同じRoboKotsソースコミット
`825dfe29cb4cae04b118298ea9a51a8d08e9de83` のクリーンなローカルcheckoutを使用。
Rust拡張の実体パスとSHA-256もJSONに保存している。
別途、インストール済みの `3f2673d7af58cd34f7af11686136241cdf905c91` でも測定し、
同じ傾向を確認した。`direct_url.json` の配布メタデータと実際のimport先は区別して記録した。

## 測定結果

| モデル | 密行列GN 中央値 | Jv/Jᵀv版 中央値 | 密行列版に対する時間比 | 外側反復数 |
|---|---:|---:|---:|---:|
| 2関節 | 0.117秒 | 1.195秒 | 10.2倍（遅い） | 両方式2回 |
| 7自由度 | 1.444秒 | 17.044秒 | 11.8倍（遅い） | 両方式6回 |


固定コミット版の2関節では、接続修正前のoperator版の中央値は4.825秒。
修正後は1.195秒で、operator版自体は約4倍改善したが、密行列版の速度には達していない。
修正前の3回は3.412–4.863秒とばらつきがあり、この倍率は当該測定の中央値による比較である。


| 固定コミット版の検証項目 | 2関節 | 7自由度 |
|---|---:|---:|
| 変数数（削減後） | 92 | 322 |
| 残差数 | 2010 | 7035 |
| 内側反復数の合計 | 98 | 1664 |
| 内側反復上限到達数 | 0 | 1 |
| 最終目的関数 | 0.415424 | 1.52 |
| 独立に検証した勾配∞ノルム | 6.06152e-12 | 2.04755e-10 |
| 境界残差2ノルム | 1.72336e-13 | 3.16854e-13 |
| 密行列版との解の最大絶対差 | 3.78e-11 | 3.36e-12 |

各方式の最終目的関数、勾配、境界残差、反復数、各回の時間は
[`kots_doc_operator_results.json`](kots_doc_operator_results.json) に保存。
「収束したが別の解だった」「内側反復上限で打ち切っただけ」を速度向上として扱わない。
この測定では双方とも外側の勾配条件を満たし、目的関数・解・境界残差は一致した。
7自由度のoperator版では内側反復上限に1回達したが、その近似ステップから継続し、
最終的には独立に検証した外側の勾配条件を満たしている。

## 発見と修正

前回追加した版でnullspace削減を使うと、`residual_jvp` / `residual_vjp` が
削減後runtimeに存在せず、adapterが密な `linearize()` に戻っていた。
実際にRoboKotsへ渡っていたのは `(201, 6, 100)` の全列RHSで、Jᵀv呼び出しはゼロだった。
さらに、既定の `required` にヤコビアンの状態キーが含まれ、値評価時にも密な微分を要求していた。

今回の修正:

1. nullspace削減の積を `J_full(Nv)` と `Nᵀ(J_fullᵀw)` に接続。
2. 値だけの状態依存をoperator用に選択。非対応の式・バックエンドは局所微分を遅延取得。
3. StateCacheと式のStackからRoboKotsのバッチJVPまでを接続。
   パラメータ方向から各時刻の運動方向を計算し、1列RHSを渡す。
   アフィン軌道のオフセットは方向微分に含めない。
4. Hinge、時間差分、成分選択のJVPを直接伝播。

既存の `gauss_newton` の求解アルゴリズムは変更していない。
実RoboKotsでJ生成・全列RHSを禁止したテストでも、削減後のJv/Jᵀvが
密な参照値と一致し、内積の転置関係を満たして求解できることを確認した。
異なる時刻順・重複時刻・異なる方向を与えたバッチJVPも検証した。
固定コミット版のRoboKotsをimportした全体テストは **734 passed**。

## なぜまだ遅いか

Jvが1回安くても、それを何回要求するかがDOC全体の時間を決める。
密行列版は各線形化でJを一度作り、NumPyの密な最小二乗を解く。
今回のoperator版は、前処理なしのCGLSと、密行列版と同じ**厳密なダンピング下限**を使う。

- 下限のための `max(diag(JᵀJ))` を求めるだけで、変数数分のJvが必要。
  2関節では92変数×3点=276回、7自由度では322変数×7点=2254回。
- CGLSの条件数依存により内側の反復も多い。線形化数を減らさずに積だけ切り替えると、
  Jv/Jᵀvの累積費用が密な求解を超える。
- 積ごとの式・状態要求の走査など、Python側の費用も残る。
  インストール済み版でのプロファイルでは、求解時間の約48%がダンピング下限、
  約43%がCGLSだった（プロファイラー付き測定なので、絶対時間の比較には使用しない）。

## 次の高速化候補

速度を目的とした次の変更は、単にRoboKotsの積を呼ぶだけでは足りない。

- **前処理**: 軌道の速度・加速度ペナルティを使う前処理やブロック前処理でCGLS反復数を減らす。
- **ダンピング下限の計算**: 問題固有の列ノルム評価、ブロック化、または明示的な近似モード。
  現状の厳密な下限と近似モードを黙って同一視しない。
- **固定点での積の準備**: 内側反復中に変わらない式・時刻・状態参照・重みを事前に準備し、
  Pythonの同じ走査を繰り返さない。

これらは今回の測定では導入していない。現状、このサンプルの速度を優先するなら
`solver="gauss_newton"` を使う。operator版の速度優位性やピークメモリ削減率を
他の問題・自由度・重み付けに一般化するだけの測定はない。

## 再現

RoboKotsの固定コミットと対応するRust拡張が利用できるPython環境で実行する。
ローカルcheckoutを使う場合は、そのパスを `PYTHONPATH` に含める。

```sh
PYTHONPATH=/path/to/RoboKots:. OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python developer/benchmarks/kots_doc_operator.py \
  --model planar2.urdf --steps 201 --controls 50 --repeat 3 --warmup 1 \
  --max-iters 10 --output /tmp/doc-planar2.json

PYTHONPATH=/path/to/RoboKots:. OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python developer/benchmarks/kots_doc_operator.py \
  --model 7_dof_arm.urdf --steps 201 --controls 50 --repeat 3 --warmup 1 \
  --max-iters 10 --output /tmp/doc-7dof.json
```
