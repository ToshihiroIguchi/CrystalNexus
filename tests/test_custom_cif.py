#!/usr/bin/env python3
"""
カスタムCIF生成関数のテスト
"""

from pymatgen.io.cif import CifParser
from pymatgen.transformations.standard_transformations import SupercellTransformation
from io import StringIO
import sys
import os

# サーバーコードのパスを追加
sys.path.insert(0, '/home/toshihiro/CrystalNexus/cifview')

def test_custom_cif_generation():
    print("=== カスタムCIF生成関数テスト ===\n")
    
    # cif_serverから関数をインポート
    try:
        from cif_server import generate_custom_supercell_cif
        print("✅ カスタムCIF関数のインポート成功")
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    
    # 1. スーパーセル構造を作成
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    parser = CifParser(StringIO(original_cif))
    original_structure = parser.get_structures()[0]
    
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    transformation = SupercellTransformation(supercell_matrix)
    supercell = transformation.apply_transformation(original_structure)
    
    print(f"🔬 スーパーセル構造: {len(supercell.sites)}原子, a={supercell.lattice.a:.6f}")
    
    # 2. 原子置換
    from pymatgen.core.periodic_table import Element
    modified_supercell = supercell.copy()
    original_site = supercell.sites[0]
    modified_supercell.replace(0, Element('Sr'), coords=original_site.coords, coords_are_cartesian=True)
    
    print(f"🔬 置換後構造: {len(modified_supercell.sites)}原子, a={modified_supercell.lattice.a:.6f}")
    
    # 3. カスタムCIF生成
    supercell_metadata = {
        'multipliers': {'a': 2, 'b': 2, 'c': 2},
        'original_atoms': 5,
        'is_supercell': True
    }
    
    print("🔬 カスタムCIF生成中...")
    custom_cif = generate_custom_supercell_cif(modified_supercell, supercell_metadata)
    
    # 4. 生成されたCIFを解析
    test_parser = CifParser(StringIO(custom_cif))
    test_structure = test_parser.get_structures()[0]
    
    print(f"🔬 生成CIF解析結果: {len(test_structure.sites)}原子, a={test_structure.lattice.a:.6f}")
    
    # 5. 結果評価
    atoms_preserved = len(test_structure.sites) == len(modified_supercell.sites)
    lattice_preserved = abs(test_structure.lattice.a - modified_supercell.lattice.a) < 0.01
    
    print(f"\n🔍 評価結果:")
    print(f"  原子数保持: {'✅' if atoms_preserved else '❌'} ({len(test_structure.sites)}/{len(modified_supercell.sites)})")
    print(f"  格子定数保持: {'✅' if lattice_preserved else '❌'} ({test_structure.lattice.a:.6f}/{modified_supercell.lattice.a:.6f})")
    
    # 6. CIF内容を確認
    print(f"\n🔬 生成されたCIF (最初の20行):")
    lines = custom_cif.split('\n')
    for i, line in enumerate(lines[:20]):
        print(f"  {i+1:2d}: {line}")
    if len(lines) > 20:
        print(f"  ... (他{len(lines)-20}行)")
    
    success = atoms_preserved and lattice_preserved
    print(f"\n📊 テスト結果: {'✅ 成功' if success else '❌ 失敗'}")
    
    return success

if __name__ == "__main__":
    success = test_custom_cif_generation()
    exit(0 if success else 1)