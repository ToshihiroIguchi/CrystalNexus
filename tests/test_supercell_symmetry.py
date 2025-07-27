#!/usr/bin/env python3
"""
スーパーセル作成時の対称性変化の詳細分析
"""

from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import SupercellTransformation
from io import StringIO

def test_supercell_symmetry():
    print("=== スーパーセル対称性分析テスト ===\n")
    
    # BaTiO3.cifを読み込み
    with open('BaTiO3.cif', 'r') as f:
        cif_content = f.read()
    
    parser = CifParser(StringIO(cif_content))
    original_structure = parser.get_structures()[0]
    
    print("🔬 元の構造:")
    original_analyzer = SpacegroupAnalyzer(original_structure)
    original_spacegroup_info = original_structure.get_space_group_info()
    print(f"  空間群: {original_spacegroup_info[1]} (#{original_spacegroup_info[0]})")
    print(f"  結晶系: {original_analyzer.get_crystal_system()}")
    print(f"  格子定数: a={original_structure.lattice.a:.4f}Å")
    print(f"  原子数: {len(original_structure.sites)}")
    print()
    
    # 異なるサイズのスーパーセルをテスト
    supercell_sizes = [
        ([2, 1, 1], "2×1×1"),
        ([2, 2, 1], "2×2×1"), 
        ([2, 2, 2], "2×2×2"),
        ([3, 3, 3], "3×3×3")
    ]
    
    for matrix_factors, label in supercell_sizes:
        print(f"🔬 スーパーセル {label}:")
        
        # スーパーセル作成
        supercell_matrix = [[matrix_factors[0], 0, 0], [0, matrix_factors[1], 0], [0, 0, matrix_factors[2]]]
        transformation = SupercellTransformation(supercell_matrix)
        supercell = transformation.apply_transformation(original_structure)
        
        # 対称性解析
        supercell_analyzer = SpacegroupAnalyzer(supercell)
        supercell_spacegroup_info = supercell.get_space_group_info()
        
        print(f"  空間群: {supercell_spacegroup_info[1]} (#{supercell_spacegroup_info[0]})")
        print(f"  結晶系: {supercell_analyzer.get_crystal_system()}")
        print(f"  格子定数: a={supercell.lattice.a:.4f}Å")
        print(f"  原子数: {len(supercell.sites)}")
        
        # 対称性変化の分析
        if supercell_spacegroup_info == original_spacegroup_info:
            print("  ✅ 元の対称性が保持されています")
        else:
            print("  ⚠️  対称性が変化しました")
            print(f"    元: {original_spacegroup_info[1]} (#{original_spacegroup_info[0]})")
            print(f"    新: {supercell_spacegroup_info[1]} (#{supercell_spacegroup_info[0]})")
        print()
    
    print("🔍 スーパーセルでの対称性変化について:")
    print("1. スーパーセルは元の構造の周期的な繰り返しです")
    print("2. 格子定数が変わるため、対称性が低下する場合があります")
    print("3. 特に非等方的な拡張（例：2×1×1）では対称性が大幅に変化します")
    print("4. 正しい解析には、実際のスーパーセル構造の対称性を使用すべきです")
    print()
    
    print("✅ 推奨される修正:")
    print("サーバーでスーパーセル情報を表示する際は:")
    print("- supercell.get_space_group_info() を使用")
    print("- SpacegroupAnalyzer(supercell).get_crystal_system() を使用")
    print("- 元の構造の情報ではなく、実際のスーパーセル構造の情報を表示")

if __name__ == "__main__":
    test_supercell_symmetry()