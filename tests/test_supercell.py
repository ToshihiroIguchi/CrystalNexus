#!/usr/bin/env python3
"""
スーパーセルの格子定数を確認するテストスクリプト
"""

from pymatgen.io.cif import CifParser
from pymatgen.io.cif import CifWriter

def test_supercell():
    # 元のCIFファイルを読み込み
    with open('BaTiO3.cif', 'r') as f:
        cif_content = f.read()
    
    print("=== 元の結晶構造 ===")
    parser = CifParser.from_str(cif_content)
    structures = parser.get_structures()
    original = structures[0]
    
    print(f"格子定数: a={original.lattice.a:.6f}, b={original.lattice.b:.6f}, c={original.lattice.c:.6f}")
    print(f"体積: {original.lattice.volume:.6f} Ų")
    print(f"原子数: {len(original.sites)}")
    
    # 2x2x2スーパーセルを作成
    print("\n=== 2x2x2スーパーセル ===")
    supercell = original.make_supercell([2, 2, 2])
    
    print(f"格子定数: a={supercell.lattice.a:.6f}, b={supercell.lattice.b:.6f}, c={supercell.lattice.c:.6f}")
    print(f"体積: {supercell.lattice.volume:.6f} Ų")
    print(f"原子数: {len(supercell.sites)}")
    
    # 倍率を確認
    print(f"\n格子定数の倍率:")
    print(f"a: {supercell.lattice.a / original.lattice.a:.2f}倍")
    print(f"b: {supercell.lattice.b / original.lattice.b:.2f}倍")
    print(f"c: {supercell.lattice.c / original.lattice.c:.2f}倍")
    print(f"体積: {supercell.lattice.volume / original.lattice.volume:.2f}倍")
    
    # スーパーセルのCIFを生成して確認
    writer = CifWriter(supercell)
    supercell_cif = str(writer)
    
    # 生成されたCIFを再解析
    print("\n=== 生成されたスーパーセルCIFの解析 ===")
    parser2 = CifParser.from_str(supercell_cif)
    structures2 = parser2.get_structures()
    parsed_supercell = structures2[0]
    
    print(f"格子定数: a={parsed_supercell.lattice.a:.6f}, b={parsed_supercell.lattice.b:.6f}, c={parsed_supercell.lattice.c:.6f}")
    print(f"体積: {parsed_supercell.lattice.volume:.6f} Ų")
    print(f"原子数: {len(parsed_supercell.sites)}")

if __name__ == "__main__":
    test_supercell()