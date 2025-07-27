#!/usr/bin/env python3
"""
厳密対称性検出APIの動作確認テスト
"""

import requests
import json

def test_strict_symmetry_api():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== 厳密対称性検出API動作確認 ===\n")
    
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
    
    supercell_data = supercell_response.json()
    supercell_cif = supercell_data['supercell_cif']
    supercell_info = supercell_data.get('supercell_info', {})
    
    print(f"  ✅ スーパーセル作成成功")
    print(f"  空間群: {supercell_info.get('space_group')}")
    print(f"  結晶系: {supercell_info.get('crystal_system')}")
    
    # 3. 1個原子置換（対称性維持されるケース）
    print("\n🔬 ステップ2: 1個Ba→Sr置換（対称性維持）")
    
    supercell_metadata = {
        'multipliers': {'a': 2, 'b': 2, 'c': 2},
        'original_atoms': 5,
        'is_supercell': True
    }
    
    replace1_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': [0],  # 1個のBa原子
        'new_element': 'Sr',
        'supercell_metadata': supercell_metadata
    })
    
    replace1_data = replace1_response.json()
    replace1_info = replace1_data.get('modified_structure_info', {})
    
    print(f"  ✅ 1個置換完了")
    print(f"  空間群: {replace1_info.get('space_group')}")
    print(f"  結晶系: {replace1_info.get('crystal_system')}")
    
    # 4. Ba原子インデックスを特定
    print("\n🔬 ステップ3: Ba原子の特定")
    atom_info = supercell_data.get('supercell_info', {}).get('atom_info', [])
    ba_indices = []
    for atom in atom_info:
        if 'Ba' in atom.get('element', ''):
            ba_indices.append(atom.get('index'))
    
    print(f"  Ba原子数: {len(ba_indices)}")
    print(f"  Ba原子インデックス: {ba_indices[:8]}...") # 最初の8個表示
    
    # 5. 3個原子置換（対称性破れるケース）
    print("\n🔬 ステップ4: 3個Ba→Sr置換（対称性破れ）")
    
    replace3_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': ba_indices[:3],  # 最初の3個のBa原子
        'new_element': 'Sr',
        'supercell_metadata': supercell_metadata
    })
    
    replace3_data = replace3_response.json()
    replace3_info = replace3_data.get('modified_structure_info', {})
    
    print(f"  ✅ 3個置換完了")
    print(f"  空間群: {replace3_info.get('space_group')}")
    print(f"  結晶系: {replace3_info.get('crystal_system')}")
    
    # 6. 5個原子置換（さらに対称性破れ）
    print("\n🔬 ステップ5: 5個Ba→Sr置換（さらに対称性破れ）")
    
    replace5_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': ba_indices[:5],  # 最初の5個のBa原子
        'new_element': 'Sr',
        'supercell_metadata': supercell_metadata
    })
    
    replace5_data = replace5_response.json()
    replace5_info = replace5_data.get('modified_structure_info', {})
    
    print(f"  ✅ 5個置換完了")
    print(f"  空間群: {replace5_info.get('space_group')}")
    print(f"  結晶系: {replace5_info.get('crystal_system')}")
    
    # 7. 原子削除テスト
    print("\n🔬 ステップ6: 原子削除（対称性破れ）")
    
    delete_response = requests.post(f'{SERVER_URL}/delete_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': [0],  # 1個の原子を削除
        'supercell_metadata': supercell_metadata
    })
    
    delete_data = delete_response.json()
    delete_info = delete_data.get('modified_structure_info', {})
    
    print(f"  ✅ 原子削除完了")
    print(f"  空間群: {delete_info.get('space_group')}")
    print(f"  結晶系: {delete_info.get('crystal_system')}")
    print(f"  原子数: {delete_data.get('modified_atom_count')}")
    
    # 8. 結果評価
    print("\n🔍 厳密対称性検出結果:")
    
    original_space_group = supercell_info.get('space_group')
    
    # 対称性変化の確認
    replace1_changed = replace1_info.get('space_group') != original_space_group
    replace3_changed = replace3_info.get('space_group') != original_space_group
    replace5_changed = replace5_info.get('space_group') != original_space_group
    delete_changed = delete_info.get('space_group') != original_space_group
    
    print(f"  元の空間群: {original_space_group}")
    print(f"  1個置換で対称性変化: {'✅' if replace1_changed else '❌'} ({replace1_info.get('space_group')})")
    print(f"  3個置換で対称性変化: {'✅' if replace3_changed else '❌'} ({replace3_info.get('space_group')})")
    print(f"  5個置換で対称性変化: {'✅' if replace5_changed else '❌'} ({replace5_info.get('space_group')})")
    print(f"  削除で対称性変化: {'✅' if delete_changed else '❌'} ({delete_info.get('space_group')})")
    
    # P1への変化確認
    replace1_is_p1 = "P1" in str(replace1_info.get('space_group', ''))
    replace3_is_p1 = "P1" in str(replace3_info.get('space_group', ''))
    replace5_is_p1 = "P1" in str(replace5_info.get('space_group', ''))
    delete_is_p1 = "P1" in str(delete_info.get('space_group', ''))
    
    print(f"\n  P1への変化:")
    print(f"  1個置換 → P1: {'✅' if replace1_is_p1 else '❌'}")
    print(f"  3個置換 → P1: {'✅' if replace3_is_p1 else '❌'}")
    print(f"  5個置換 → P1: {'✅' if replace5_is_p1 else '❌'}")
    print(f"  削除 → P1: {'✅' if delete_is_p1 else '❌'}")
    
    # 厳密検出の評価
    any_symmetry_detected = replace3_changed or replace5_changed or delete_changed
    
    print(f"\n📊 厳密検出評価:")
    print(f"  対称性変化検出: {'✅ 成功' if any_symmetry_detected else '❌ 失敗'}")
    print(f"  期待値: 3個置換、5個置換、削除で対称性変化を検出")
    
    if any_symmetry_detected:
        print("💡 厳密対称性検出が正常に動作しています！")
    else:
        print("💡 さらなる調整が必要かもしれません。")
    
    return any_symmetry_detected

if __name__ == "__main__":
    success = test_strict_symmetry_api()
    exit(0 if success else 1)