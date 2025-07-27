#!/usr/bin/env python3
"""
スーパーセルP1強制設定の動作確認テスト
"""

import requests
import json

def test_p1_forced_supercell():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== スーパーセルP1強制設定 動作確認 ===\n")
    
    # 1. 元のCIF読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    # 2. 通常のCIF解析（スーパーセルなし）
    print("🔬 ステップ1: 通常のCIF解析（対称性保持確認）")
    
    parse_response = requests.post(f'{SERVER_URL}/parse_cif', json={
        'cif_content': original_cif
    })
    
    if parse_response.status_code == 200:
        parse_data = parse_response.json()
        print(f"  ✅ 通常解析成功")
        print(f"  空間群: {parse_data.get('space_group')}")
        print(f"  結晶系: {parse_data.get('crystal_system')}")
        print(f"  原子数: {parse_data.get('atom_count', 'N/A')}")
        
        original_space_group = parse_data.get('space_group')
        original_crystal_system = parse_data.get('crystal_system')
        
    else:
        print(f"  ❌ 通常解析失敗: {parse_response.status_code}")
        return False
    
    # 3. スーパーセル作成（P1強制設定確認）
    print("\n🔬 ステップ2: スーパーセル作成（P1強制設定確認）")
    
    supercell_response = requests.post(f'{SERVER_URL}/create_supercell', json={
        'cif_content': original_cif,
        'a_multiplier': 2,
        'b_multiplier': 2,
        'c_multiplier': 2
    })
    
    if supercell_response.status_code == 200:
        supercell_data = supercell_response.json()
        supercell_info = supercell_data.get('supercell_info', {})
        
        print(f"  ✅ スーパーセル作成成功")
        print(f"  空間群: {supercell_info.get('space_group')}")
        print(f"  結晶系: {supercell_info.get('crystal_system')}")
        print(f"  原子数: {supercell_info.get('atom_count')}")
        
        supercell_space_group = supercell_info.get('space_group')
        supercell_crystal_system = supercell_info.get('crystal_system')
        
    else:
        print(f"  ❌ スーパーセル作成失敗: {supercell_response.status_code}")
        return False
    
    # 4. 異なる倍率でのスーパーセルテスト
    print("\n🔬 ステップ3: 異なる倍率でのスーパーセルテスト")
    
    test_multipliers = [
        {"a": 1, "b": 1, "c": 2, "name": "1×1×2"},
        {"a": 3, "b": 1, "c": 1, "name": "3×1×1"},
        {"a": 2, "b": 3, "c": 2, "name": "2×3×2"}
    ]
    
    for mult in test_multipliers:
        mult_response = requests.post(f'{SERVER_URL}/create_supercell', json={
            'cif_content': original_cif,
            'a_multiplier': mult['a'],
            'b_multiplier': mult['b'],
            'c_multiplier': mult['c']
        })
        
        if mult_response.status_code == 200:
            mult_data = mult_response.json()
            mult_info = mult_data.get('supercell_info', {})
            
            print(f"  {mult['name']}: {mult_info.get('space_group')} / {mult_info.get('crystal_system')}")
        else:
            print(f"  {mult['name']}: ❌ 失敗")
    
    # 5. 原子編集後の対称性確認
    print("\n🔬 ステップ4: 原子編集後の対称性確認")
    
    supercell_cif = supercell_data['supercell_cif']
    supercell_metadata = {
        'multipliers': {'a': 2, 'b': 2, 'c': 2},
        'original_atoms': 5,
        'is_supercell': True
    }
    
    # 原子置換
    replace_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': [0],
        'new_element': 'Sr',
        'supercell_metadata': supercell_metadata
    })
    
    if replace_response.status_code == 200:
        replace_data = replace_response.json()
        replace_info = replace_data.get('modified_structure_info', {})
        
        print(f"  置換後: {replace_info.get('space_group')} / {replace_info.get('crystal_system')}")
    else:
        print(f"  置換: ❌ 失敗")
    
    # 6. 結果評価
    print("\n🔍 結果評価:")
    
    # 元の対称性保持確認
    original_preserved = original_space_group and "P4mm" in str(original_space_group) and "tetragonal" in str(original_crystal_system)
    
    # スーパーセルP1強制確認
    supercell_forced_p1 = "P1" in str(supercell_space_group) and "triclinic" in str(supercell_crystal_system)
    
    # 対称性変化確認
    symmetry_changed = original_space_group != supercell_space_group
    
    print(f"  元の対称性保持: {'✅' if original_preserved else '❌'} ({original_space_group})")
    print(f"  スーパーセルP1強制: {'✅' if supercell_forced_p1 else '❌'} ({supercell_space_group})")
    print(f"  対称性変化確認: {'✅' if symmetry_changed else '❌'}")
    
    # 総合評価
    overall_success = original_preserved and supercell_forced_p1 and symmetry_changed
    
    print(f"\n📊 総合評価: {'✅ 成功' if overall_success else '❌ 失敗'}")
    
    if overall_success:
        print("🎉 スーパーセルP1強制設定が正常に動作しています！")
        print("💡 スーパーセル作成時：自動的にP1/triclinic")
        print("💡 通常のCIF解析：元の対称性を保持")
    else:
        print("💡 設定に問題があります。詳細を確認してください。")
    
    # 7. 詳細ログ出力
    print(f"\n📋 詳細情報:")
    print(f"  元の構造:")
    print(f"    空間群: {original_space_group}")
    print(f"    結晶系: {original_crystal_system}")
    print(f"  スーパーセル (2×2×2):")
    print(f"    空間群: {supercell_space_group}")
    print(f"    結晶系: {supercell_crystal_system}")
    print(f"    原子数: {supercell_info.get('atom_count')}")
    
    return overall_success

if __name__ == "__main__":
    success = test_p1_forced_supercell()
    exit(0 if success else 1)