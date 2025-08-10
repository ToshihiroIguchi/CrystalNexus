# -*- coding: utf-8 -*-
"""
CHGnetを用いたBaTiO3結晶構造の緩和と物性計算
必要なライブラリ: pymatgen, chgnet, matplotlib, numpy
インストール方法: 
  pip install pymatgen chgnet matplotlib numpy
"""

from chgnet.model import CHGNet
from chgnet.model.dynamics import StructOptimizer
from pymatgen.core import Structure
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import warnings

# 警告を表示する設定
warnings.filterwarnings("always", category=UserWarning)

# 日本語フォント問題の解決
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 1. モデルの読み込み
print("CHGnetモデルを読み込み中...")
chgnet = CHGNet.load()

# 2. BaTiO3の結晶構造をCIFファイルから読み込み
cif_path = "BaTiO3.cif"
print(f"{cif_path} から構造を読み込み中...")
if not os.path.exists(cif_path):
    raise FileNotFoundError(f"エラー: {cif_path} が見つかりません。ファイルを確認してください")

structure = Structure.from_file(cif_path)
num_atoms = len(structure)
print(f"原子数: {num_atoms}")

# 3. 構造緩和の実行
print("\n構造緩和を開始...")
optimizer = StructOptimizer()
result = optimizer.relax(structure, verbose=True)

relaxed_structure = result['final_structure']
print("\n構造緩和が完了しました！")

# 4. 緩和済み構造の保存
relaxed_cif_path = "BaTiO3_relaxed.cif"
relaxed_structure.to(filename=relaxed_cif_path, fmt="cif")
print(f"緩和済み構造を {relaxed_cif_path} に保存")

# 5. 物性計算（緩和済み構造を使用）
print("\n物性計算を実行中...")
prediction = chgnet.predict_structure(relaxed_structure)

# 出力キーを確認
print(f"予測結果のキー: {list(prediction.keys())}")

# 6. 計算結果の表示
print("\n===== 計算結果 =====")

# CHGnet 0.3.0の出力キーに基づく処理
if "e" not in prediction:
    warnings.warn("警告: 全エネルギーが計算されていません", UserWarning)
else:
    energy = prediction["e"]
    print(f"全エネルギー: {energy:.6f} eV")
    print(f"原子あたりエネルギー: {energy/num_atoms:.6f} eV/atom")

if "f" not in prediction:
    warnings.warn("警告: 原子力データが存在しません", UserWarning)
else:
    forces = prediction["f"]
    print(f"\n原子力の最大値: {np.max(np.abs(forces)):.6f} eV/Å")

    # 原子ごとの力の可視化
    plt.figure(figsize=(10, 6))
    plt.title("Atomic Forces Magnitude", fontsize=14)
    plt.xlabel("Atom Index", fontsize=12)
    plt.ylabel("Force Magnitude (eV/Å)", fontsize=12)
    force_magnitudes = np.linalg.norm(forces, axis=1)
    plt.bar(range(len(force_magnitudes)), force_magnitudes)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("atomic_forces.png", dpi=300)
    print("\n原子ごとの力を atomic_forces.png に保存")

if "s" not in prediction:
    warnings.warn("警告: 応力テンソルが存在しません", UserWarning)
else:
    stress_tensor = prediction["s"]
    print("\n応力テンソル (GPa):")
    print(np.array2string(stress_tensor, precision=4, suppress_small=True))

if "m" not in prediction:
    warnings.warn("警告: 磁気モーメントが存在しません", UserWarning)
else:
    magmom = prediction["m"]
    if isinstance(magmom, (list, np.ndarray)):
        total_magmom = np.sum(magmom)
        print(f"\n全磁気モーメント: {total_magmom:.6f} μB")
    else:
        print(f"\n磁気モーメント: {magmom:.6f} μB")

# 7. 結果の要約をファイルに保存
print("\n計算結果を chgnet_results.txt に保存中...")
with open("chgnet_results.txt", "w") as f:
    f.write("===== CHGnet Calculation Summary =====\n")
    f.write(f"CHGnet version: 0.3.0\n")
    f.write(f"Structure file: {cif_path}\n")
    f.write(f"Relaxed structure: {relaxed_cif_path}\n")
    f.write(f"Number of atoms: {num_atoms}\n\n")
    
    f.write("● Energy Properties\n")
    if "e" in prediction:
        f.write(f"Total energy: {energy:.6f} eV\n")
        f.write(f"Energy per atom: {energy/num_atoms:.6f} eV/atom\n")
    else:
        f.write("Total energy: Not available\n")
    
    f.write("\n● Magnetic Properties\n")
    if "m" in prediction:
        if isinstance(magmom, (list, np.ndarray)):
            f.write(f"Total magnetic moment: {total_magmom:.6f} μB\n\n")
        else:
            f.write(f"Magnetic moment: {magmom:.6f} μB\n\n")
    else:
        f.write("Magnetic moment: Not available\n\n")
    
    f.write("● Stress Tensor (GPa)\n")
    if "s" in prediction:
        for i in range(3):
            f.write(f"{stress_tensor[i,0]:.6f} {stress_tensor[i,1]:.6f} {stress_tensor[i,2]:.6f}\n")
    else:
        f.write("Not available\n")
    
    if "f" in prediction:
        f.write("\n● Atomic Forces (eV/Å)\n")
        for i, force in enumerate(forces):
            f.write(f"Atom {i}: {force[0]:.6f} {force[1]:.6f} {force[2]:.6f}\n")

print("すべての計算が完了しました！")