#!/usr/bin/env python3
"""
原子編集機能のテストスクリプト
"""

import requests
import json

def test_atomic_editing():
    # BaTiO3.cifファイルを読み込み
    with open('BaTiO3.cif', 'r') as f:
        cif_content = f.read()
    
    print("=== 元の結晶構造情報 ===")
    # 元の構造を解析
    response = requests.post('http://localhost:5000/parse_cif', 
                           headers={'Content-Type': 'application/json'},
                           json={'cif_content': cif_content})
    
    if response.status_code == 200:
        original_data = response.json()
        if original_data['success']:
            print(f"原子数: {original_data['atom_count']}")
            print("原子情報:")
            for atom in original_data['atom_info']:
                print(f"  {atom['label']}: {atom['element']}")
        else:
            print(f"❌ 構造解析失敗: {original_data.get('error', 'Unknown error')}")
            return
    else:
        print(f"❌ API呼び出し失敗: {response.status_code}")
        return
    
    print("\n=== 原子削除テスト (Ba0を削除) ===")
    # Ba0原子（インデックス0）を削除
    delete_data = {
        "cif_content": cif_content,
        "atom_indices": [0]  # Ba0を削除
    }
    
    response = requests.post('http://localhost:5000/delete_atoms', 
                           headers={'Content-Type': 'application/json'},
                           json=delete_data)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("✅ 原子削除成功！")
            print(f"原子数: {result['original_atom_count']} → {result['modified_atom_count']}")
            print(f"削除されたインデックス: {result['deleted_indices']}")
            
            # 削除後の構造を解析
            modified_cif = result['modified_cif']
            response2 = requests.post('http://localhost:5000/parse_cif', 
                                   headers={'Content-Type': 'application/json'},
                                   json={'cif_content': modified_cif})
            
            if response2.status_code == 200:
                modified_data = response2.json()
                if modified_data['success']:
                    print("修正後の原子情報:")
                    for atom in modified_data['atom_info']:
                        print(f"  {atom['label']}: {atom['element']}")
        else:
            print(f"❌ 原子削除失敗: {result.get('error', 'Unknown error')}")
    else:
        print(f"❌ API呼び出し失敗: {response.status_code}")
        print(response.text)
    
    print("\n=== 原子置換テスト (Ti1をSrに置換) ===")
    # Ti1原子（インデックス1）をSrに置換
    replace_data = {
        "cif_content": cif_content,
        "atom_indices": [1],  # Ti1を置換
        "new_element": "Sr"
    }
    
    response = requests.post('http://localhost:5000/replace_atoms', 
                           headers={'Content-Type': 'application/json'},
                           json=replace_data)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("✅ 原子置換成功！")
            print(f"置換されたインデックス: {result['replaced_indices']}")
            print(f"新しい元素: {result['new_element']}")
            
            # 置換後の構造を解析
            modified_cif = result['modified_cif']
            response2 = requests.post('http://localhost:5000/parse_cif', 
                                   headers={'Content-Type': 'application/json'},
                                   json={'cif_content': modified_cif})
            
            if response2.status_code == 200:
                modified_data = response2.json()
                if modified_data['success']:
                    print("置換後の原子情報:")
                    for atom in modified_data['atom_info']:
                        print(f"  {atom['label']}: {atom['element']}")
        else:
            print(f"❌ 原子置換失敗: {result.get('error', 'Unknown error')}")
    else:
        print(f"❌ API呼び出し失敗: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_atomic_editing()