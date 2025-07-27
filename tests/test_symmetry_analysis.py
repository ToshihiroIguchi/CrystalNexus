#!/usr/bin/env python3
"""
対称性解析の詳細調査 - 原子編集後の空間群・結晶系変化
"""

import requests
import json
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import SupercellTransformation
from io import StringIO

def test_symmetry_analysis():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== 対称性解析詳細調査 ===\n")
    
    # 1. BaTiO3.cifを読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    print("🔬 ステップ1: 元の構造の対称性")
    parser = CifParser(StringIO(original_cif))
    original_structure = parser.get_structures()[0]
    original_analyzer = SpacegroupAnalyzer(original_structure)
    
    print(f"  元の構造:")
    print(f"    原子数: {len(original_structure.sites)}")
    print(f"    空間群: {original_structure.get_space_group_info()}")
    print(f"    結晶系: {original_analyzer.get_crystal_system()}")
    
    # 2. スーパーセル作成
    print("\n🔬 ステップ2: スーパーセル作成API")
    supercell_response = requests.post(f'{SERVER_URL}/create_supercell', json={
        'cif_content': original_cif,
        'a_multiplier': 2,
        'b_multiplier': 2,
        'c_multiplier': 2
    })
    
    supercell_data = supercell_response.json()
    supercell_cif = supercell_data['supercell_cif']
    supercell_info = supercell_data.get('supercell_info', {})
    
    print(f"  スーパーセル:")
    print(f"    原子数: {supercell_info.get('atom_count')}")
    print(f"    空間群: {supercell_info.get('space_group')}")
    print(f"    結晶系: {supercell_info.get('crystal_system')}")
    
    # 3. スーパーセルの直接解析
    print("\n🔬 ステップ3: スーパーセル構造の直接解析")
    supercell_parser = CifParser(StringIO(supercell_cif))
    supercell_structure = supercell_parser.get_structures()[0]
    supercell_analyzer = SpacegroupAnalyzer(supercell_structure)
    
    print(f"  CIF解析結果:")
    print(f"    原子数: {len(supercell_structure.sites)}")
    print(f"    空間群: {supercell_structure.get_space_group_info()}")
    print(f"    結晶系: {supercell_analyzer.get_crystal_system()}")
    
    # 4. 原子置換実行
    print("\n🔬 ステップ4: 原子置換実行")
    
    supercell_metadata = {
        'multipliers': {'a': 2, 'b': 2, 'c': 2},
        'original_atoms': 5,
        'is_supercell': True
    }
    
    replace_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': [0],  # 最初の原子を置換
        'new_element': 'Sr',
        'supercell_metadata': supercell_metadata
    })
    
    replace_data = replace_response.json()
    modified_cif = replace_data['modified_cif']
    modified_info = replace_data.get('modified_structure_info', {})
    
    print(f"  原子置換後（API結果）:")
    print(f"    原子数: {replace_data.get('modified_atom_count')}")
    print(f"    空間群: {modified_info.get('space_group')}")
    print(f"    結晶系: {modified_info.get('crystal_system')}")
    
    # 5. 修正構造の直接解析
    print("\n🔬 ステップ5: 修正構造の直接解析")
    modified_parser = CifParser(StringIO(modified_cif))
    modified_structure = modified_parser.get_structures()[0]
    modified_analyzer = SpacegroupAnalyzer(modified_structure)
    
    print(f"  直接解析結果:")
    print(f"    原子数: {len(modified_structure.sites)}")
    print(f"    空間群: {modified_structure.get_space_group_info()}")
    print(f"    結晶系: {modified_analyzer.get_crystal_system()}")
    
    # 6. 手動でスーパーセルを作成して置換
    print("\n🔬 ステップ6: 手動スーパーセル+置換の対称性")
    
    # 手動でスーパーセル作成
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    transformation = SupercellTransformation(supercell_matrix)
    manual_supercell = transformation.apply_transformation(original_structure)
    
    print(f"  手動スーパーセル:")
    print(f"    原子数: {len(manual_supercell.sites)}")
    
    manual_analyzer = SpacegroupAnalyzer(manual_supercell)
    print(f"    空間群: {manual_supercell.get_space_group_info()}")
    print(f"    結晶系: {manual_analyzer.get_crystal_system()}")
    
    # 原子置換
    from pymatgen.core.periodic_table import Element
    manual_modified = manual_supercell.copy()
    original_site = manual_supercell.sites[0]
    new_element = Element("Sr")
    manual_modified.replace(0, new_element, coords=original_site.coords, coords_are_cartesian=True)
    
    manual_modified_analyzer = SpacegroupAnalyzer(manual_modified)
    print(f"  手動置換後:")
    print(f"    原子数: {len(manual_modified.sites)}")
    print(f"    空間群: {manual_modified.get_space_group_info()}")
    print(f"    結晶系: {manual_modified_analyzer.get_crystal_system()}")
    
    # 7. SpacegroupAnalyzerの詳細設定テスト
    print("\n🔬 ステップ7: SpacegroupAnalyzer詳細設定")
    
    # より厳密な対称性解析
    strict_analyzer = SpacegroupAnalyzer(manual_modified, symprec=1e-5, angle_tolerance=5)
    print(f"  厳密解析:")
    print(f"    空間群: {strict_analyzer.get_space_group_symbol()} (#{strict_analyzer.get_space_group_number()})")
    print(f"    結晶系: {strict_analyzer.get_crystal_system()}")
    
    # より緩い対称性解析
    loose_analyzer = SpacegroupAnalyzer(manual_modified, symprec=1e-2, angle_tolerance=1)
    print(f"  緩い解析:")
    print(f"    空間群: {loose_analyzer.get_space_group_symbol()} (#{loose_analyzer.get_space_group_number()})")
    print(f"    結晶系: {loose_analyzer.get_crystal_system()}")
    
    # 8. 評価
    print("\n🔍 対称性解析評価:")
    
    api_symmetry_changed = modified_info.get('space_group') != supercell_info.get('space_group')
    direct_symmetry_changed = modified_structure.get_space_group_info() != supercell_structure.get_space_group_info()
    manual_symmetry_changed = manual_modified.get_space_group_info() != manual_supercell.get_space_group_info()
    
    print(f"  API結果で対称性変化: {'✅' if api_symmetry_changed else '❌'}")
    print(f"  直接解析で対称性変化: {'✅' if direct_symmetry_changed else '❌'}")
    print(f"  手動解析で対称性変化: {'✅' if manual_symmetry_changed else '❌'}")
    
    # 期待値：原子置換によりP1になるべき
    expected_space_group = "P1"
    api_is_p1 = expected_space_group in str(modified_info.get('space_group', ''))
    direct_is_p1 = expected_space_group in str(modified_structure.get_space_group_info())
    manual_is_p1 = expected_space_group in str(manual_modified.get_space_group_info())
    
    print(f"  API結果がP1: {'✅' if api_is_p1 else '❌'}")
    print(f"  直接解析がP1: {'✅' if direct_is_p1 else '❌'}")
    print(f"  手動解析がP1: {'✅' if manual_is_p1 else '❌'}")
    
    return True

if __name__ == "__main__":
    test_symmetry_analysis()