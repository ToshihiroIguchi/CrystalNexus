#!/usr/bin/env python3
"""
スーパーセル再構築のデバッグ
"""

import requests
import json
from pymatgen.io.cif import CifParser
from io import StringIO

def test_reconstruction_debug():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== スーパーセル再構築デバッグ ===\n")
    
    # 1. BaTiO3.cifを読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    # 2. スーパーセル作成
    print("🔬 ステップ1: スーパーセル作成")
    supercell_response = requests.post(f'{SERVER_URL}/create_supercell', json={
        'cif_content': original_cif,
        'a_multiplier': 2,
        'b_multiplier': 2,
        'c_multiplier': 2
    })
    
    supercell_data = supercell_response.json()
    if not supercell_data.get('success'):
        print(f"❌ スーパーセル作成失敗")
        return
    
    supercell_cif = supercell_data['supercell_cif']
    
    # 3. 送信前にCIF内容を確認
    print("🔬 送信するCIF内容の確認:")
    test_parser = CifParser(StringIO(supercell_cif))
    test_structure = test_parser.get_structures()[0]
    print(f"  送信CIF解析結果: {len(test_structure.sites)}原子, a={test_structure.lattice.a:.6f}")
    print()
    
    # 4. より簡単なテスト：原子数40→39（1個削除）
    print("🔬 ステップ2: 原子削除テスト（スーパーセル保持版）")
    
    supercell_metadata = {
        'multipliers': supercell_data.get('multipliers', {}),
        'original_atoms': supercell_data.get('original_atoms', 0),
        'is_supercell': True
    }
    
    print(f"  送信メタデータ: {supercell_metadata}")
    
    # 最後の原子を削除
    delete_response = requests.post(f'{SERVER_URL}/delete_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': [39],  # 最後の原子を削除
        'supercell_metadata': supercell_metadata
    })
    
    if delete_response.status_code != 200:
        print(f"❌ HTTPエラー: {delete_response.status_code}")
        return
    
    delete_data = delete_response.json()
    if not delete_data.get('success'):
        print(f"❌ 原子削除失敗: {delete_data.get('error')}")
        return
    
    print(f"  original_atom_count: {delete_data.get('original_atom_count')}")
    print(f"  modified_atom_count: {delete_data.get('modified_atom_count')}")
    
    # 5. 結果を解析
    modified_cif = delete_data['modified_cif']
    result_parser = CifParser(StringIO(modified_cif))
    result_structure = result_parser.get_structures()[0]
    
    print(f"  結果: {len(result_structure.sites)}原子, a={result_structure.lattice.a:.6f}")
    
    # 6. 期待値との比較
    expected_atoms = 39  # 40から1個削除
    expected_lattice_a = 7.980758
    
    atoms_correct = len(result_structure.sites) == expected_atoms
    lattice_correct = abs(result_structure.lattice.a - expected_lattice_a) < 0.01
    
    print(f"\n🔍 評価:")
    print(f"  原子数: {'✅' if atoms_correct else '❌'} ({len(result_structure.sites)}/{expected_atoms})")
    print(f"  格子定数: {'✅' if lattice_correct else '❌'} ({result_structure.lattice.a:.6f}/{expected_lattice_a:.6f})")
    
    overall_success = atoms_correct and lattice_correct
    print(f"\n📊 結果: {'✅ 成功' if overall_success else '❌ 失敗'}")
    
    return overall_success

if __name__ == "__main__":
    success = test_reconstruction_debug()
    exit(0 if success else 1)