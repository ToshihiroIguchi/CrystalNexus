#!/usr/bin/env python3
"""
原子置換のレスポンス詳細テスト
modified_structure_info の内容を詳しく調査
"""

import requests
import json

def test_atom_replacement_detail():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== 原子置換レスポンス詳細テスト ===\n")
    
    # 1. 元のCIF読み込み
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
    
    if supercell_response.status_code != 200:
        print("❌ スーパーセル作成失敗")
        return False
    
    supercell_data = supercell_response.json()
    supercell_cif = supercell_data['supercell_cif']
    
    print(f"  ✅ スーパーセル作成成功")
    print(f"  レスポンスキー: {list(supercell_data.keys())}")
    
    # 3. 原子置換テスト
    print("\n🔬 ステップ2: 原子置換実行")
    
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
    
    print(f"  Status Code: {replace_response.status_code}")
    
    if replace_response.status_code != 200:
        print(f"  ❌ 原子置換失敗")
        print(f"  レスポンス: {replace_response.text}")
        return False
    
    try:
        replace_data = replace_response.json()
    except Exception as e:
        print(f"  ❌ JSONパースエラー: {e}")
        print(f"  Raw response: {replace_response.text}")
        return False
    
    print(f"  ✅ 原子置換レスポンス受信")
    print(f"  レスポンスキー: {list(replace_data.keys())}")
    
    # 4. レスポンス詳細分析
    print("\n📊 レスポンス詳細分析:")
    
    print(f"  success: {replace_data.get('success')}")
    print(f"  original_atom_count: {replace_data.get('original_atom_count')}")
    print(f"  modified_atom_count: {replace_data.get('modified_atom_count')}")
    print(f"  replaced_indices: {replace_data.get('replaced_indices')}")
    print(f"  new_element: {replace_data.get('new_element')}")
    
    # 5. modified_structure_info 詳細分析
    modified_info = replace_data.get('modified_structure_info', {})
    print(f"\n  modified_structure_info keys: {list(modified_info.keys())}")
    
    for key, value in modified_info.items():
        if key == 'atom_info':
            print(f"  {key}: {len(value)} items")
            if len(value) > 0:
                print(f"    First item: {value[0]}")
                print(f"    Last item: {value[-1]}")
            else:
                print(f"    ❌ atom_info is empty!")
        elif key == 'lattice_parameters' and value:
            print(f"  {key}: a={value.get('a', 'N/A')}, b={value.get('b', 'N/A')}, c={value.get('c', 'N/A')}")
        else:
            print(f"  {key}: {value}")
    
    # 6. CIF内容の確認
    modified_cif = replace_data.get('modified_cif', '')
    print(f"\n  modified_cif length: {len(modified_cif)} characters")
    if len(modified_cif) > 0:
        # CIFの最初の20行を表示
        cif_lines = modified_cif.split('\n')[:20]
        print("  CIF preview (first 20 lines):")
        for i, line in enumerate(cif_lines):
            print(f"    {i+1:2d}: {line}")
    else:
        print("  ❌ modified_cif is empty!")
    
    # 7. 問題判定
    atom_info = modified_info.get('atom_info', [])
    success = replace_data.get('success', False)
    
    print(f"\n📋 問題診断:")
    print(f"  API呼び出し成功: {'✅' if replace_response.status_code == 200 else '❌'}")
    print(f"  success フラグ: {'✅' if success else '❌'}")
    print(f"  CIF生成: {'✅' if len(modified_cif) > 0 else '❌'}")
    print(f"  atom_info生成: {'✅' if len(atom_info) > 0 else '❌'}")
    print(f"  原子数一致: {'✅' if replace_data.get('modified_atom_count', 0) == len(atom_info) else '❌'}")
    
    return len(atom_info) > 0

if __name__ == "__main__":
    success = test_atom_replacement_detail()
    exit(0 if success else 1)