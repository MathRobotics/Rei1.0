# 固定関節を持つモデルでの最適化

`rei.backends.state.robotics.lock_urdf_joints(xml, positions)` は、URDFを変更せずに
指定関節を固定した新しいURDF文字列を返します。回転関節はrad、直動関節はmで指定します。
ロボットやバックエンドに固有の関節名はライブラリに含めません。

```python
from pathlib import Path
from rei.backends.state.robotics import lock_urdf_joints
from robokots.kots import Kots
from robokots.urdf_io import urdf_xml_to_model_data

source = Path("examples/model/robots/franka_fr3_duo.urdf")
reduced = lock_urdf_joints(source.read_text(), {
    "left_fr3v2_finger_joint1": 0.02,
    "right_fr3v2_finger_joint1": 0.02,
})
model = Kots.from_json_data(
    urdf_xml_to_model_data(reduced.xml), order=3, backend="rust",
)
assert model.dof() == 14
```

返り値には `xml`、URDF記載順の `active_joint_names`、mimic追従関節を含む
`locked_joints` が入ります。バックエンドの座標順はバックエンド側で確認してください。
縮約済みモデルを通常の `compile_kots_trajectory_problem` に渡すと、縮約後の
自由度から軌道変数・ヤコビアンの次元が決まります。固定関節用の最適化変数や
等式制約を追加する必要はありません。

固定変換は `T_origin @ T_motion(q_fixed)` です。すべてのリンク、慣性、重心、
visual/collision要素は残すため、固定したハンドの質量も腕の動力学に含まれます。
固定関節の反力は最適化対象の関節トルクから除外されます。mimic追従関節は
`multiplier * q_master + offset` で再帰的に固定します。未知の関節、非有限値、
関節制限外の固定位置、mimic循環・矛盾を拒否します。

未固定のmimicはそのまま保持されます。現在のRoboKotsはこれを独立自由度として
扱うため、下記サンプルでは未固定mimicを検出するとエラーにします。
相対メッシュパスは書き換えません。別のディレクトリへXMLを保存する場合は、
元ディレクトリ基準のメッシュ解決を維持してください。物理モデルの縮約であり、
ROSコントローラやtransmission設定の再構成は行いません。

## FR3 DuoでDOC → IOCを実行

Problem Specと同じTOMLに固定位置を保存できます。
`examples/spec/franka_fr3_duo_doc.toml` には次の設定が入っています。

```toml
[model.locked_joints]
left_fr3v2_finger_joint1 = 0.02
right_fr3v2_finger_joint1 = 0.02
```

回転関節はrad、直動関節はmです。`--lock-joint NAME=POSITION` を指定すると、
その関節だけTOMLの値を上書きします。セクションがなければ固定しません。
`mimic`追従関節を列挙する必要はありません。

Pythonでは `load_joint_locks_toml(path)` でこの設定を読み取り、
`lock_urdf_joints(xml, positions)` に渡します。この読み込みAPIは単独のTOMLでも
使えます。`load_problem_spec_toml()` は最適化DSLのみを作成するため、
モデルの固定はモデル読み込み時に別途適用してください。

hand付きの `examples/model/robots/franka_fr3_duo.urdf` を用意し、
リポジトリルートから実行します。
`examples/model` は `../../RobotModel` への相対シンボリックリンクです。
Reiと同じ親ディレクトリにRobotModelを配置し、RobotModel側のREADMEに従って
hand付きURDFを生成してください。リンク自体にはモデルデータを含みません。

```bash
uv run --group kots python examples/robokots_doc_noise_ioc.py \
  --model examples/model/robots/franka_fr3_duo.urdf \
  --spec examples/spec/franka_fr3_duo_doc.toml \
  --gravity 0 0 -9.81
```

両方のmimic追従指も0.02 mで固定され、左右の腕のみ14自由度になります。
このspecは左腕joint1..7、右腕joint1..7の座標順を前提にし、関節ごとの
境界値・位置制限を指定します。サンプルは21評価点、8制御点で実行します。
URDFを変更した場合は座標順と制限値も確認してください。

この例の位置制限はペナルティ項です。速度・トルク上限や自己衝突回避は
含んでいません。また、現行IOCは境界条件等の制約KKT乗数を推定しないため、
DOCの収束は真の目的関数重みの復元を保証しません。返された `validation` を
確認してください。この制約は関節縮約とは独立です。
現在のサンプルはDOCの境界条件をnullspaceで除去しますが、IOCは元の全変数空間で
勾配を評価します。IOC側でも同じnullspaceへ勾配を射影する対応は未実装です。
その射影を行えば、除去済みの等式制約については乗数を推定する必要はありません。

## 検証

`tests/test_urdf_joint_locking.py` は分岐モデルについて、回転・直動関節の
非ゼロ位置での固定、mimic処理、リンク慣性保持、縮約前後の逆動力学の一致、
NumPy/Rust上のReiヤコビアンと有限差分の一致を検証します。
