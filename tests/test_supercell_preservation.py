#!/usr/bin/env python3
"""
修正後のスーパーセル保持機能のテスト
"""

import requests
import json
from pymatgen.io.cif import CifParser
from io import StringIO

def test_supercell_preservation():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== スーパーセル保持機能テスト ===\n")
    
    # 1. BaTiO3.cifを読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    # 2. 元の構造を解析
    print("🔬 ステップ1: 元の構造解析")
    original_parser = CifParser(StringIO(original_cif))
    original_structure = original_parser.get_structures()[0]
    print(f"  元の構造: {len(original_structure.sites)}原子, a={original_structure.lattice.a:.6f}")
    print()
    
    # 3. スーパーセル作成
    print("🔬 ステップ2: 2×2×2スーパーセル作成")
    supercell_response = requests.post(f'{SERVER_URL}/create_supercell', json={
        'cif_content': original_cif,
        'a_multiplier': 2,
        'b_multiplier': 2, 
        'c_multiplier': 2
    })
    
    supercell_data = supercell_response.json()
    if not supercell_data.get('success'):
        print(f"❌ スーパーセル作成失敗: {supercell_data.get('error')}")
        return False
    
    supercell_cif = supercell_data['supercell_cif']
    supercell_info = supercell_data.get('supercell_info', {})
    
    print(f"  スーパーセル: {supercell_info.get('atom_count')}原子, a={supercell_info.get('lattice_parameters', {}).get('a')}")
    print()
    
    # 4. 原子置換（修正版）
    print("🔬 ステップ3: 原子置換（スーパーセル保持版）")
    
    supercell_metadata = {
        'multipliers': supercell_data.get('multipliers', {}),
        'original_atoms': supercell_data.get('original_atoms', 0),
        'is_supercell': True
    }
    
    replace_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': [0],
        'new_element': 'Sr',
        'supercell_metadata': supercell_metadata
    })
    
    if replace_response.status_code != 200:
        print(f"❌ HTTPエラー: {replace_response.status_code}")
        return False
    
    replace_data = replace_response.json()
    if not replace_data.get('success'):
        print(f"❌ 原子置換失敗: {replace_data.get('error')}")
        return False
    
    modified_cif = replace_data['modified_cif']
    
    # 5. 修正後の構造を解析
    print("🔬 ステップ4: 修正後の構造解析")
    modified_parser = CifParser(StringIO(modified_cif))
    modified_structure = modified_parser.get_structures()[0]
    
    print(f"  修正後構造: {len(modified_structure.sites)}原子, a={modified_structure.lattice.a:.6f}")
    print()
    
    # 6. 結果の評価
    print("🔍 結果の評価:")
    
    expected_atoms = 40  # 2×2×2スーパーセル
    expected_lattice_a = 7.980758  # スーパーセルの格子定数
    
    atoms_preserved = len(modified_structure.sites) == expected_atoms
    lattice_preserved = abs(modified_structure.lattice.a - expected_lattice_a) < 0.01
    
    print(f"  原子数保持: {'✅' if atoms_preserved else '❌'} ({len(modified_structure.sites)}/{expected_atoms})")
    print(f"  格子定数保持: {'✅' if lattice_preserved else '❌'} ({modified_structure.lattice.a:.6f}/{expected_lattice_a:.6f})")
    
    # 7. 置換が正しく実行されているかチェック
    print("\n🔬 置換確認:")
    first_atom = modified_structure.sites[0]
    print(f"  最初の原子: {first_atom.specie} (期待値: Sr)")
    replacement_correct = str(first_atom.specie) == 'Sr'
    print(f"  置換成功: {'✅' if replacement_correct else '❌'}")
    
    # 8. CIFファイルの詳細確認
    print(f"\n🔬 CIF詳細:")
    cif_lines = modified_cif.split('\n')
    print(f"  CIF行数: {len(cif_lines)}")
    
    for line in cif_lines:
        if '_cell_length_a' in line:
            print(f"  格子定数a: {line.strip()}")
            break
    
    # 化学式確認
    for line in cif_lines:
        if '_chemical_formula_sum' in line:
            print(f"  化学式: {line.strip()}")
            break
    
    # 原子数カウント
    atom_lines = [line for line in cif_lines if line.strip() and 
                 any(element in line for element in ['Ba', 'Sr', 'Ti', 'O']) and 
                 len(line.split()) >= 7]
    print(f"  CIF内原子数: {len(atom_lines)}")
    
    # 総合評価
    overall_success = atoms_preserved and lattice_preserved and replacement_correct
    
    print(f"\n📊 総合評価: {'✅ 成功' if overall_success else '❌ 失敗'}")
    
    if overall_success:
        print("🎉 スーパーセル保持機能が正常に動作しています！")
    else:
        print("💡 まだ問題が残っています。さらなる調査が必要です。")
    
    return overall_success

if __name__ == "__main__":
    success = test_supercell_preservation()
    exit(0 if success else 1)