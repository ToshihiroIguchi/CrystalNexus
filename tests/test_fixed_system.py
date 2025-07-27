#!/usr/bin/env python3
"""
修正後のシステムの動作テスト
"""

import requests
import json

def test_fixed_system():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== 修正後のシステムテスト ===\n")
    
    # 1. BaTiO3.cifを読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    print("🔬 ステップ1: 元のCIF解析")
    response = requests.post(f'{SERVER_URL}/parse_cif', 
                           json={'cif_content': original_cif})
    original_data = response.json()
    print(f"  元の構造:")
    print(f"    空間群: {original_data.get('space_group', 'Unknown')}")
    print(f"    結晶系: {original_data.get('crystal_system', 'Unknown')}")
    print(f"    原子数: {original_data.get('atom_count', 'Unknown')}")
    print(f"    格子定数a: {original_data.get('lattice_parameters', {}).get('a', 'Unknown')}")
    print()
    
    # 2. 2×2×2スーパーセル作成
    print("🔬 ステップ2: 2×2×2スーパーセル作成")
    supercell_response = requests.post(f'{SERVER_URL}/create_supercell', json={
        'cif_content': original_cif,
        'a_multiplier': 2,
        'b_multiplier': 2,
        'c_multiplier': 2
    })
    supercell_data = supercell_response.json()
    
    if supercell_data.get('success'):
        supercell_cif = supercell_data['supercell_cif']
        supercell_info = supercell_data.get('supercell_info', {})
        
        print(f"  スーパーセル:")
        print(f"    空間群: {supercell_info.get('space_group', 'Unknown')}")
        print(f"    結晶系: {supercell_info.get('crystal_system', 'Unknown')}")
        print(f"    原子数: {supercell_info.get('atom_count', 'Unknown')}")
        print(f"    格子定数a: {supercell_info.get('lattice_parameters', {}).get('a', 'Unknown')}")
        print()
        
        # 3. スーパーセルメタデータの準備
        supercell_metadata = {
            'multipliers': supercell_data.get('multipliers', {}),
            'original_atoms': supercell_data.get('original_atoms', 0),
            'is_supercell': True
        }
        
        # 4. 原子置換（メタデータ付き）
        print("🔬 ステップ3: 原子置換実行（修正版）")
        replace_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
            'cif_content': supercell_cif,
            'atom_indices': [0],
            'new_element': 'Sr',
            'supercell_metadata': supercell_metadata
        })
        
        replace_data = replace_response.json()
        print(f"  置換APIレスポンス成功: {replace_data.get('success')}")
        
        if replace_data.get('success'):
            modified_structure_info = replace_data.get('modified_structure_info', {})
            
            print(f"  編集後の構造:")
            print(f"    スーパーセルかどうか: {modified_structure_info.get('is_supercell', False)}")
            print(f"    空間群: {modified_structure_info.get('space_group', 'Unknown')}")
            print(f"    結晶系: {modified_structure_info.get('crystal_system', 'Unknown')}")
            print(f"    原子数: {modified_structure_info.get('atom_count', 'Unknown')}")
            print(f"    格子定数a: {modified_structure_info.get('lattice_parameters', {}).get('a', 'Unknown')}")
            print(f"    スーパーセル倍率: {modified_structure_info.get('supercell_multipliers', {})}")
            print()
            
            # 結果の評価
            print("🔍 修正結果の評価:")
            original_supercell_a = supercell_info.get('lattice_parameters', {}).get('a', 0)
            modified_a = modified_structure_info.get('lattice_parameters', {}).get('a', 0)
            
            if modified_structure_info.get('is_supercell'):
                print("  ✅ スーパーセル情報が正しく保持されています")
            else:
                print("  ❌ スーパーセル情報が失われました")
                
            if abs(float(original_supercell_a) - float(modified_a)) < 0.1:
                print("  ✅ 格子定数が正しく保持されています")
                print(f"    スーパーセル: {original_supercell_a}Å")
                print(f"    編集後: {modified_a}Å")
            else:
                print("  ❌ 格子定数が変化しました")
                print(f"    スーパーセル: {original_supercell_a}Å")
                print(f"    編集後: {modified_a}Å")
                
            original_supercell_atoms = supercell_info.get('atom_count', 0)
            modified_atoms = modified_structure_info.get('atom_count', 0)
            expected_atoms = original_supercell_atoms  # 置換なので原子数は変わらない
            
            if modified_atoms == expected_atoms:
                print("  ✅ 原子数が正しく保持されています")
                print(f"    期待値: {expected_atoms}, 実際: {modified_atoms}")
            else:
                print("  ❌ 原子数が期待値と異なります")
                print(f"    期待値: {expected_atoms}, 実際: {modified_atoms}")
                
        else:
            print(f"  ❌ 原子置換エラー: {replace_data.get('error')}")
    else:
        print(f"  ❌ スーパーセル作成エラー: {supercell_data.get('error')}")

if __name__ == "__main__":
    test_fixed_system()