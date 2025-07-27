#!/usr/bin/env python3
"""
スーパーセル作成時の空間群・結晶系分析テスト
"""

from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import SupercellTransformation
from io import StringIO

def test_spacegroup_analysis():
    print("=== スーパーセル作成時の空間群・結晶系分析テスト ===\n")
    
    # BaTiO3.cifを読み込み
    with open('BaTiO3.cif', 'r') as f:
        cif_content = f.read()
    
    # 元の構造を解析
    parser = CifParser(StringIO(cif_content))
    original_structure = parser.get_structures()[0]
    
    print("🔬 元の構造:")
    original_analyzer = SpacegroupAnalyzer(original_structure)
    original_spacegroup = original_analyzer.get_space_group_info()
    print(f"  空間群: {original_spacegroup}")
    print(f"  結晶系: {original_analyzer.get_crystal_system()}")
    print(f"  格子定数: a={original_structure.lattice.a:.4f}, b={original_structure.lattice.b:.4f}, c={original_structure.lattice.c:.4f}")
    print(f"  原子数: {len(original_structure.sites)}")
    print()
    
    # 2x2x2 スーパーセルを作成
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    transformation = SupercellTransformation(supercell_matrix)
    supercell = transformation.apply_transformation(original_structure)
    
    print("🔬 スーパーセル (2x2x2):")
    supercell_analyzer = SpacegroupAnalyzer(supercell)
    
    # 異なる対称性解析の精度設定でテスト
    tolerances = [0.1, 0.01, 0.001]
    
    for tol in tolerances:
        try:
            analyzer_with_tol = SpacegroupAnalyzer(supercell, symprec=tol)
            spacegroup_info = analyzer_with_tol.get_space_group_info()
            crystal_system = analyzer_with_tol.get_crystal_system()
            
            print(f"  精度 {tol}:")
            print(f"    空間群: {spacegroup_info}")
            print(f"    結晶系: {crystal_system}")
        except Exception as e:
            print(f"  精度 {tol}: エラー - {e}")
    
    print(f"  格子定数: a={supercell.lattice.a:.4f}, b={supercell.lattice.b:.4f}, c={supercell.lattice.c:.4f}")
    print(f"  原子数: {len(supercell.sites)}")
    print()
    
    # スーパーセルの対称性が元の構造と異なる理由を調査
    print("🔍 対称性の変化について:")
    print("スーパーセルは元の構造を拡張したものですが、以下の理由で対称性が変化する可能性があります：")
    print("1. 格子の繰り返しによる新しい対称性の出現")
    print("2. 並進対称性の変化")
    print("3. 空間群の子群への変化")
    print()
    
    # 正しいスーパーセル情報の取得方法
    print("✅ 推奨される解析方法:")
    print("スーパーセルの対称性は:")
    
    # より寛容な精度設定での解析
    try:
        refined_analyzer = SpacegroupAnalyzer(supercell, symprec=0.01, angle_tolerance=5)
        refined_spacegroup = refined_analyzer.get_space_group_info()
        refined_crystal_system = refined_analyzer.get_crystal_system()
        
        print(f"  精密解析での空間群: {refined_spacegroup}")
        print(f"  精密解析での結晶系: {refined_crystal_system}")
        
        # 元の対称性を保持しているかチェック
        if refined_spacegroup == original_spacegroup:
            print("  ✅ 元の対称性が保持されています")
        else:
            print("  ⚠️  対称性が変化しました（これは正常な場合があります）")
            
    except Exception as e:
        print(f"  エラー: {e}")

if __name__ == "__main__":
    test_spacegroup_analysis()