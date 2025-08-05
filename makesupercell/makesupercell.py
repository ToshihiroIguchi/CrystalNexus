# -*- coding: utf-8 -*-
"""
BaTiO3.cif を読み込み、対称性解析は最初の構造のみ実施、
必要に応じて 2×2×2 スーパーセル（supercell）を構築、
構造表示および raw_structure の解放を行うスクリプト。
"""

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def print_initial_symmetry(structure, symprec=0.1, angle_tolerance=5.0):
    """
    初回構造に対してのみ対称性と格子情報を表示する関数。
    symprec と angle_tolerance は状況に応じて設定。
    """
    analyzer = SpacegroupAnalyzer(structure,
                                  symprec=symprec,
                                  angle_tolerance=angle_tolerance)
    symbol = analyzer.get_space_group_symbol()
    number = analyzer.get_space_group_number()
    lat = structure.lattice
    a, b, c = lat.a, lat.b, lat.c
    alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma

    print("=== initial (raw) structure ===")
    print(f"Space group: {symbol} (No. {number})")
    print(f"a={a:.6f} Å, b={b:.6f} Å, c={c:.6f} Å")
    print(f"alpha={alpha:.3f}°, beta={beta:.3f}°, gamma={gamma:.3f}°")
    print(f"Number of sites: {len(structure)}")
    print("==============================\n")


# ————————————————
# 1. CIF ファイルを読み込み
raw_structure = Structure.from_file("BaTiO3.cif")

# 2. 初期構造に対して symprec = 0.1, angle_tolerance = 5° で対称性を表示
print_initial_symmetry(raw_structure, symprec=0.1, angle_tolerance=5.0)

# ————————————————
# 3. super = True の場合は 2×2×2 supercell を作成
supercell = True
if supercell:
    # 元構造を直接変更せず、安全のためコピーしてから操作
    new_structure = raw_structure.copy()
    new_structure.make_supercell([2, 2, 2])
else:
    # 同様に常に copy() を使うことで安全性を確保
    new_structure = raw_structure.copy()

# ————————————————
# 4. raw_structure の参照を明示的に解除
# これは、元の構造が残っているとバグの温床になるため。
raw_structure = None

# ————————————————
# 5. new_structure の格子・サイト情報を表示
# 対称性の情報は表示しない。
lat = new_structure.lattice
a, b, c = lat.a, lat.b, lat.c
alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma

print("--- new_structure ---")
print(f"a={a:.6f} Å, b={b:.6f} Å, c={c:.6f} Å")
print(f"alpha={alpha:.3f}°, beta={beta:.3f}°, gamma={gamma:.3f}°")
print(f"Number of sites: {len(new_structure)}")
print("---------------------\n")
