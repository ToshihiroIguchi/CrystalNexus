#!/usr/bin/env python3
"""
厳密対称性検出パラメータのテスト
"""

from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.core.periodic_table import Element
from io import StringIO

def test_strict_symmetry_parameters():
    print("=== 厳密対称性検出パラメータテスト ===\n")
    
    # 1. 元の構造読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    parser = CifParser(StringIO(original_cif))
    original_structure = parser.get_structures()[0]
    
    # 2. スーパーセル作成
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    transformation = SupercellTransformation(supercell_matrix)
    supercell_structure = transformation.apply_transformation(original_structure)
    
    # 3. 原子置換（1個のBa→Sr）
    modified_structure = supercell_structure.copy()
    original_site = supercell_structure.sites[0]
    new_element = Element("Sr")
    modified_structure.replace(0, new_element, coords=original_site.coords, coords_are_cartesian=True)
    
    print("🔬 基本情報:")
    print(f"  元の構造: {len(original_structure.sites)}原子")
    print(f"  スーパーセル: {len(supercell_structure.sites)}原子")
    print(f"  修正後: {len(modified_structure.sites)}原子")
    print(f"  修正内容: Ba→Sr置換")
    
    # 4. 様々なパラメータでテスト
    test_parameters = [
        {"symprec": 1e-5, "angle_tolerance": 5, "name": "pymatgen標準"},
        {"symprec": 1e-6, "angle_tolerance": 1, "name": "厳密レベル1"},
        {"symprec": 1e-7, "angle_tolerance": 0.5, "name": "厳密レベル2"},
        {"symprec": 1e-8, "angle_tolerance": 0.1, "name": "超厳密"},
        {"symprec": 1e-4, "angle_tolerance": 10, "name": "緩い設定"},
    ]
    
    print(f"\n🔬 対称性解析結果:")
    print(f"{'設定名':<15} {'symprec':<10} {'angle_tol':<10} {'空間群':<15} {'結晶系':<15}")
    print("-" * 75)
    
    # 元の構造
    orig_analyzer = SpacegroupAnalyzer(original_structure)
    print(f"{'元の構造':<15} {'-':<10} {'-':<10} {orig_analyzer.get_space_group_symbol():<15} {orig_analyzer.get_crystal_system():<15}")
    
    # スーパーセル
    super_analyzer = SpacegroupAnalyzer(supercell_structure)
    print(f"{'スーパーセル':<15} {'-':<10} {'-':<10} {super_analyzer.get_space_group_symbol():<15} {super_analyzer.get_crystal_system():<15}")
    
    # 修正後の構造を各パラメータでテスト
    for params in test_parameters:
        try:
            analyzer = SpacegroupAnalyzer(
                modified_structure, 
                symprec=params["symprec"], 
                angle_tolerance=params["angle_tolerance"]
            )
            space_group = analyzer.get_space_group_symbol()
            crystal_system = analyzer.get_crystal_system()
            
            print(f"{params['name']:<15} {params['symprec']:<10} {params['angle_tolerance']:<10} {space_group:<15} {crystal_system:<15}")
            
        except Exception as e:
            print(f"{params['name']:<15} {params['symprec']:<10} {params['angle_tolerance']:<10} {'エラー':<15} {str(e)[:15]:<15}")
    
    # 5. 複数原子置換のテスト
    print(f"\n🔬 複数原子置換テスト:")
    
    # 3個のBa原子を置換
    multi_modified = supercell_structure.copy()
    ba_indices = []
    for i, site in enumerate(supercell_structure.sites):
        if str(site.specie) == "Ba2+":
            ba_indices.append(i)
            if len(ba_indices) >= 3:
                break
    
    print(f"  置換対象Ba原子インデックス: {ba_indices}")
    
    for idx in ba_indices:
        original_site = supercell_structure.sites[idx]
        multi_modified.replace(idx, new_element, coords=original_site.coords, coords_are_cartesian=True)
    
    print(f"  3個Ba→Sr置換後:")
    for params in test_parameters:
        try:
            analyzer = SpacegroupAnalyzer(
                multi_modified, 
                symprec=params["symprec"], 
                angle_tolerance=params["angle_tolerance"]
            )
            space_group = analyzer.get_space_group_symbol()
            crystal_system = analyzer.get_crystal_system()
            
            print(f"    {params['name']:<15}: {space_group} / {crystal_system}")
            
        except Exception as e:
            print(f"    {params['name']:<15}: エラー - {str(e)[:30]}")
    
    # 6. 推奨パラメータの決定
    print(f"\n🎯 推奨パラメータ:")
    
    # 最も厳密だが実用的なパラメータを特定
    recommended = {"symprec": 1e-6, "angle_tolerance": 1}
    
    try:
        rec_analyzer = SpacegroupAnalyzer(
            modified_structure, 
            symprec=recommended["symprec"], 
            angle_tolerance=recommended["angle_tolerance"]
        )
        rec_space_group = rec_analyzer.get_space_group_symbol()
        rec_crystal_system = rec_analyzer.get_crystal_system()
        
        print(f"  symprec: {recommended['symprec']}")
        print(f"  angle_tolerance: {recommended['angle_tolerance']}")
        print(f"  結果: {rec_space_group} / {rec_crystal_system}")
        
        # 対称性変化の確認
        orig_space_group = orig_analyzer.get_space_group_symbol()
        symmetry_changed = rec_space_group != orig_space_group
        
        print(f"  対称性変化: {'✅ 検出' if symmetry_changed else '❌ 未検出'}")
        print(f"  元: {orig_space_group} → 修正後: {rec_space_group}")
        
        return recommended
        
    except Exception as e:
        print(f"  エラー: {e}")
        return {"symprec": 1e-5, "angle_tolerance": 5}  # フォールバック

if __name__ == "__main__":
    recommended_params = test_strict_symmetry_parameters()
    print(f"\n📋 実装推奨パラメータ: {recommended_params}")