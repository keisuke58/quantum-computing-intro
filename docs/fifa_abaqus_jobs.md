# fifa の Abaqus で TSV の参照解を作る（指示書）

TSV の熱応力の参照解（FEM）を Abaqus で作ります．目的は 2 つです．

1. **答え合わせ**：自作の軸対称 FEM（`42_tsv_axisym.py`）が Abaqus と一致するかを確かめる
2. **3D の参照解**：正方配列の単位セルを 3D で解き，2 次元・軸対称の近似がどこまで正しいかを見る

寸法は MORE-Stress（arXiv:2411.12690）に合わせています：ビア直径 5 µm，高さ 50 µm，ピッチ 15 µm，ΔT = −250 K（275 → 25 °C）．
単位は µm と MPa．材料は Cu（E = 110 GPa，ν = 0.35，α = 17 ppm/K）と Si（E = 130 GPa，ν = 0.28，α = 2.6 ppm/K）．
ライナー（絶縁膜）は省略しています（PINN 側と条件をそろえるため）．

**注意**：ジョブ 1 は fifa で動作確認済み（2026-11-03）．`from caeModules import *` と `F-Output-1` の作成はリポジトリ版に反映済みなので，修正版のコピーは不要．エラーが出たら，ログの最後の 20 行ほどをチャットに貼ってください．

**検証**：ジョブ 1 のビア中心（中央面）の応力 σ_rr = σ_θθ = 642 MPa，σ_zz = 917 MPa は，平面ひずみの解析解（`43_check_axisym_fe.py` と同じ式）と一致．

## 毎回の手順

```bash
cd ~/quantum-computing-intro        # fifa 上のクローン（無ければ git clone して git checkout ccr-e0b1f2e9-h1hyaz）
git pull
mkdir -p ~/abaqus_tsv && cd ~/abaqus_tsv     # Abaqus の作業ファイル（.odb など）はリポジトリの外に出す
abaqus cae noGUI=$HOME/quantum-computing-intro/abaqus/tsv_abaqus.py -- mode=<mode> -- bc=<bc> -- size=<size> -- ncpu=<n>   # 引数ごとに -- を付ける
abaqus python $HOME/quantum-computing-intro/abaqus/extract_odb.py tsv_<mode>_<bc>.odb
```

`extract_odb.py` が `tsv_<mode>_<bc>_stress.csv`（要素の重心の座標・応力・von Mises）と `_disp.csv`（節点の変位）を出します．
CSV を gzip してリポジトリの `results/abaqus/` に置き，push してください．

```bash
mkdir -p ~/quantum-computing-intro/results/abaqus
gzip -c tsv_<mode>_<bc>_stress.csv > ~/quantum-computing-intro/results/abaqus/tsv_<mode>_<bc>_stress.csv.gz
gzip -c tsv_<mode>_<bc>_disp.csv   > ~/quantum-computing-intro/results/abaqus/tsv_<mode>_<bc>_disp.csv.gz
cd ~/quantum-computing-intro && git add results/abaqus && git commit -m "Abaqus: tsv_<mode>_<bc>" && git push
```

push できないときは CSV をチャットにアップロードしてください．終わったら下の表の「状態」を「完了（日付）」にして一緒に push してください．

## ジョブ一覧（上から順に）

| 順 | mode | bc | size | ncpu | 内容 | 目安 | 状態 |
|---|---|---|---|---|---|---|---|
| 1 | `axisym` | `clamped` | 0.05 | 4 | 軸対称・上下固定．自作 FEM の答え合わせ（まずこれだけ回して結果を見せてください） | 数分 | 完了（2026-11-03） |
| 2 | `axisym` | `free` | 0.05 | 4 | 軸対称・上面自由（ウェハ表面の応力集中） | 数分 | 未実行 |
| 3 | `3d` | `clamped` | 0.4 | 8 | 3D 単位セル（1/4）・上下固定．MORE-Stress のシナリオ 1 と同じ | 数十分 | 未実行 |
| 4 | `3d` | `free` | 0.4 | 8 | 3D 単位セル（1/4）・上面自由 | 数十分 | 未実行 |

- 1 と 2 は軽いので，ログインしたその場で回してかまいません．3 と 4 が重い場合は，fifa のジョブ投入の決まりに従ってください
- 3D で要素数が多すぎる・メモリが足りないときは `size=0.6` に上げる．逆に余裕があれば `size=0.3`

## 中身（確認用）

| 項目 | axisym | 3d |
|---|---|---|
| 解く領域 | r ∈ [0, 8.46]，z ∈ [0, 25]（同じ面積の円柱セル） | x, y ∈ [0, 7.5]，z ∈ [0, 25]（正方セルの 1/4） |
| 要素 | CAX8（構造格子の四角形 2 次） | C3D10（四面体 2 次） |
| 下面 z = 0 | 高さの中央面（u_z = 0，対称） | 同じ |
| 側面 | 軸 r = 0 と外周 r = R で u_r = 0 | x = 0, 7.5 で u_x = 0，y = 0, 7.5 で u_y = 0（対称） |
| 上面 z = 25 | clamped：全変位 0 / free：何も課さない | 同じ |
| 温度 | 初期 275 °C → ステップで 25 °C（一様） | 同じ |

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| `findAt` で領域が見つからない | 結合（Boolean merge）で面・セルの番号が変わった可能性．ログを貼ってください |
| ライセンスが取れない | 時間をおいて再実行．`abaqus licensing lmstat` で空きを確認 |
| `extract_odb.py` で `KeyError: 'cool'` | ステップ名が違う．ログを貼ってください |
