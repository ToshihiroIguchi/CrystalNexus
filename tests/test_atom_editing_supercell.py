#!/usr/bin/env python3
"""
原子編集時のスーパーセル情報保持テスト
"""

import requests
import json

def test_atom_editing_workflow():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== 原子編集時のスーパーセル保持テスト ===\n")
    
    # 1. BaTiO3.cifを読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    print("🔬 ステップ1: 元のCIF解析")
    response = requests.post(f'{SERVER_URL}/parse_cif', 
                           json={'cif_content': original_cif})
    original_data = response.json()
    print(f"  元の構造 - 空間群: {original_data.get('space_group', 'Unknown')}")
    print(f"  元の構造 - 結晶系: {original_data.get('crystal_system', 'Unknown')}")
    print(f"  元の構造 - 原子数: {original_data.get('atom_count', 'Unknown')}")
    print()
    
    # 2. スーパーセル作成
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
        
        print(f"  スーパーセル - 空間群: {supercell_info.get('space_group', 'Unknown')}")
        print(f"  スーパーセル - 結晶系: {supercell_info.get('crystal_system', 'Unknown')}")
        print(f"  スーパーセル - 原子数: {supercell_info.get('atom_count', 'Unknown')}")
        print(f"  スーパーセル - 格子定数a: {supercell_info.get('lattice_parameters', {}).get('a', 'Unknown')}")
        print()
        
        # 3. 原子編集（置換）
        print("🔬 ステップ3: 原子置換実行")
        # 最初の原子（インデックス0）をSrに置換
        replace_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
            'cif_content': supercell_cif,
            'atom_indices': [0],
            'new_element': 'Sr'
        })
        
        replace_data = replace_response.json()
        
        if replace_data.get('success'):
            modified_cif = replace_data['modified_cif']
            
            # 4. 編集後の構造を再解析
            print("🔬 ステップ4: 編集後の構造解析")
            analysis_response = requests.post(f'{SERVER_URL}/parse_cif', 
                                            json={'cif_content': modified_cif})
            modified_data = analysis_response.json()
            
            print(f"  編集後 - 空間群: {modified_data.get('space_group', 'Unknown')}")
            print(f"  編集後 - 結晶系: {modified_data.get('crystal_system', 'Unknown')}")
            print(f"  編集後 - 原子数: {modified_data.get('atom_count', 'Unknown')}")
            print(f"  編集後 - 格子定数a: {modified_data.get('lattice_parameters', {}).get('a', 'Unknown')}")
            print()
            
            # 問題の分析
            print("🔍 問題の分析:")
            supercell_a = supercell_info.get('lattice_parameters', {}).get('a', 0)
            modified_a = modified_data.get('lattice_parameters', {}).get('a', 0)
            
            if abs(float(supercell_a) - float(modified_a)) > 0.1:
                print("  ❌ 格子定数が元のスーパーセルサイズと異なります")
                print(f"    スーパーセル前: {supercell_a}Å")
                print(f"    編集後: {modified_a}Å")
                print("  📋 原因: CifWriterが元の単位格子情報のみ保存している可能性")
                print("  💡 解決策: 編集時にスーパーセル情報を明示的に保持する必要")
            else:
                print("  ✅ 格子定数は正しく保持されています")
                
            print()
            print("🔧 推奨修正案:")
            print("1. 原子編集時にスーパーセルのメタデータを別途保存")
            print("2. 編集後にスーパーセル情報を復元して表示")
            print("3. または、編集時にオリジナルのスーパーセル構造情報を維持")
            
        else:
            print(f"  ❌ 原子置換エラー: {replace_data.get('error')}")
    else:
        print(f"  ❌ スーパーセル作成エラー: {supercell_data.get('error')}")

if __name__ == "__main__":
    test_atom_editing_workflow()