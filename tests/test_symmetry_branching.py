#!/usr/bin/env python3
"""
対称性解析分岐処理の動作確認テスト
通常構造 vs スーパーセルでの異なる対称性解析
"""

import requests
import json

def test_symmetry_branching():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== 対称性解析分岐処理 動作確認 ===\n")
    
    # 1. 元のCIF読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    # 2. 通常構造での原子編集（標準解析）
    print("🔬 ステップ1: 通常構造での原子編集（標準解析）")
    
    # まず通常のCIF解析
    parse_response = requests.post(f'{SERVER_URL}/parse_cif', json={
        'cif_content': original_cif
    })
    
    if parse_response.status_code == 200:
        parse_data = parse_response.json()
        original_space_group = parse_data.get('space_group')
        original_crystal_system = parse_data.get('crystal_system')
        
        print(f"  元の構造:")
        print(f"    空間群: {original_space_group}")
        print(f"    結晶系: {original_crystal_system}")
        
        # 通常構造での原子置換（スーパーセルメタデータなし）
        normal_replace_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
            'cif_content': original_cif,
            'atom_indices': [0],  # 最初の原子を置換
            'new_element': 'Sr'
            # supercell_metadata なし
        })
        
        if normal_replace_response.status_code == 200:
            normal_replace_data = normal_replace_response.json()
            normal_replace_info = normal_replace_data.get('modified_structure_info', {})
            
            print(f"  通常構造編集後:")
            print(f"    空間群: {normal_replace_info.get('space_group')}")
            print(f"    結晶系: {normal_replace_info.get('crystal_system')}")
            print(f"    原子数: {normal_replace_data.get('modified_atom_count')}")
        else:
            print(f"  ❌ 通常構造編集失敗: {normal_replace_response.status_code}")
            return False
    else:
        print(f"  ❌ 通常構造解析失敗: {parse_response.status_code}")
        return False
    
    # 3. スーパーセル作成
    print("\n🔬 ステップ2: スーパーセル作成（P1強制）")
    
    supercell_response = requests.post(f'{SERVER_URL}/create_supercell', json={
        'cif_content': original_cif,
        'a_multiplier': 2,
        'b_multiplier': 2,
        'c_multiplier': 2
    })
    
    if supercell_response.status_code == 200:
        supercell_data = supercell_response.json()
        supercell_cif = supercell_data['supercell_cif']
        supercell_info = supercell_data.get('supercell_info', {})
        
        print(f"  スーパーセル:")
        print(f"    空間群: {supercell_info.get('space_group')}")
        print(f"    結晶系: {supercell_info.get('crystal_system')}")
        print(f"    原子数: {supercell_info.get('atom_count')}")
        
        # 4. スーパーセルでの原子編集（厳密解析）
        print("\n🔬 ステップ3: スーパーセルでの原子編集（厳密解析）")
        
        supercell_metadata = {
            'multipliers': {'a': 2, 'b': 2, 'c': 2},
            'original_atoms': 5,
            'is_supercell': True
        }
        
        supercell_replace_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
            'cif_content': supercell_cif,
            'atom_indices': [0],  # 最初の原子を置換
            'new_element': 'Sr',
            'supercell_metadata': supercell_metadata
        })
        
        if supercell_replace_response.status_code == 200:
            supercell_replace_data = supercell_replace_response.json()
            supercell_replace_info = supercell_replace_data.get('modified_structure_info', {})
            
            print(f"  スーパーセル編集後:")
            print(f"    空間群: {supercell_replace_info.get('space_group')}")
            print(f"    結晶系: {supercell_replace_info.get('crystal_system')}")
            print(f"    原子数: {supercell_replace_data.get('modified_atom_count')}")
        else:
            print(f"  ❌ スーパーセル編集失敗: {supercell_replace_response.status_code}")
            return False
    else:
        print(f"  ❌ スーパーセル作成失敗: {supercell_response.status_code}")
        return False
    
    # 5. 通常構造での原子削除テスト
    print("\n🔬 ステップ4: 通常構造での原子削除（標準解析）")
    
    normal_delete_response = requests.post(f'{SERVER_URL}/delete_atoms', json={
        'cif_content': original_cif,
        'atom_indices': [0]  # 最初の原子を削除
        # supercell_metadata なし
    })
    
    if normal_delete_response.status_code == 200:
        normal_delete_data = normal_delete_response.json()
        normal_delete_info = normal_delete_data.get('modified_structure_info', {})
        
        print(f"  通常構造削除後:")
        print(f"    空間群: {normal_delete_info.get('space_group')}")
        print(f"    結晶系: {normal_delete_info.get('crystal_system')}")
        print(f"    原子数: {normal_delete_data.get('modified_atom_count')}")
    else:
        print(f"  ❌ 通常構造削除失敗: {normal_delete_response.status_code}")
        return False
    
    # 6. 結果評価
    print("\n🔍 分岐処理評価:")
    
    # 各段階での対称性
    original_sg = original_space_group
    normal_replace_sg = normal_replace_info.get('space_group')
    supercell_sg = supercell_info.get('space_group')
    supercell_replace_sg = supercell_replace_info.get('space_group')
    normal_delete_sg = normal_delete_info.get('space_group')
    
    print(f"  元の構造: {original_sg}")
    print(f"  通常置換後: {normal_replace_sg}")
    print(f"  スーパーセル: {supercell_sg}")
    print(f"  スーパーセル置換後: {supercell_replace_sg}")
    print(f"  通常削除後: {normal_delete_sg}")
    
    # 分岐動作確認
    normal_preserved_original = "P4mm" in str(normal_replace_sg)  # 通常構造は元の対称性に近い
    supercell_forced_p1 = "P1" in str(supercell_sg)  # スーパーセルはP1強制
    supercell_edited_kept_low = "P1" in str(supercell_replace_sg) or "P" in str(supercell_replace_sg)  # 編集後も低対称性
    normal_delete_proper = normal_delete_sg != original_sg  # 削除で対称性変化
    
    print(f"\n  分岐動作:")
    print(f"  通常置換で元対称性考慮: {'✅' if normal_preserved_original else '❌'}")
    print(f"  スーパーセルP1強制: {'✅' if supercell_forced_p1 else '❌'}")
    print(f"  スーパーセル編集で低対称性維持: {'✅' if supercell_edited_kept_low else '❌'}")
    print(f"  通常削除で適切な対称性変化: {'✅' if normal_delete_proper else '❌'}")
    
    # 総合評価
    branching_works = (
        supercell_forced_p1 and  # スーパーセルはP1
        (normal_replace_sg != supercell_replace_sg)  # 通常とスーパーセルで異なる解析結果
    )
    
    print(f"\n📊 総合評価: {'✅ 成功' if branching_works else '❌ 失敗'}")
    
    if branching_works:
        print("🎉 対称性解析分岐処理が正常に動作しています！")
        print("💡 効果:")
        print("   - 通常構造：元の対称性に応じた標準解析")
        print("   - スーパーセル：P1強制 + 編集時厳密解析")
        print("   - 適切な分岐：構造タイプに応じた処理")
    else:
        print("💡 分岐処理に問題があります。")
    
    # 7. 詳細ログ
    print(f"\n📋 詳細比較:")
    print(f"  通常構造の処理:")
    print(f"    元: {original_sg}")
    print(f"    置換後: {normal_replace_sg}")
    print(f"    削除後: {normal_delete_sg}")
    print(f"  スーパーセルの処理:")
    print(f"    作成時: {supercell_sg}")
    print(f"    編集後: {supercell_replace_sg}")
    
    return branching_works

if __name__ == "__main__":
    success = test_symmetry_branching()
    exit(0 if success else 1)