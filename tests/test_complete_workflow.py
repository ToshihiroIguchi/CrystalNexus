#!/usr/bin/env python3
"""
完全な原子編集ワークフローのテスト
"""

import requests
import json

def test_complete_workflow():
    print("=== 完全な原子編集ワークフローのテスト ===")
    
    # Step 1: BaTiO3.cifファイルを読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    print("1. 元の構造を解析...")
    response = requests.post('http://localhost:5000/parse_cif', 
                           headers={'Content-Type': 'application/json'},
                           json={'cif_content': original_cif})
    
    if response.status_code != 200:
        print(f"❌ 初期解析失敗: {response.status_code}")
        return
    
    original_data = response.json()
    if not original_data['success']:
        print(f"❌ 初期解析失敗: {original_data.get('error')}")
        return
    
    print(f"✅ 元の構造: {original_data['atom_count']}個の原子")
    print("原子ラベル一覧:")
    for atom in original_data['atom_info']:
        print(f"  {atom['label']}: {atom['element']}")
    
    # Step 2: CIFラベル別の原子リストを確認
    label_counts = {}
    for atom in original_data['atom_info']:
        label = atom['label']
        if label not in label_counts:
            label_counts[label] = []
        label_counts[label].append(atom['index'])
    
    print(f"\n2. ラベル別原子インデックス:")
    for label, indices in label_counts.items():
        print(f"  {label}: {indices}")
    
    # Step 3: Ba0原子を削除 (Ba0のすべてのインスタンス)
    print(f"\n3. Ba0原子を削除...")
    ba0_indices = label_counts.get('Ba0', [])
    if ba0_indices:
        response = requests.post('http://localhost:5000/delete_atoms', 
                               headers={'Content-Type': 'application/json'},
                               json={
                                   'cif_content': original_cif,
                                   'atom_indices': ba0_indices
                               })
        
        if response.status_code == 200:
            delete_result = response.json()
            if delete_result['success']:
                print(f"✅ Ba0削除成功: {delete_result['original_atom_count']}→{delete_result['modified_atom_count']}原子")
                
                # 削除後の構造を確認
                deleted_cif = delete_result['modified_cif']
                response2 = requests.post('http://localhost:5000/parse_cif', 
                                       headers={'Content-Type': 'application/json'},
                                       json={'cif_content': deleted_cif})
                
                if response2.status_code == 200:
                    deleted_data = response2.json()
                    if deleted_data['success']:
                        print("削除後の原子:")
                        for atom in deleted_data['atom_info']:
                            print(f"  {atom['label']}: {atom['element']}")
                    else:
                        print(f"❌ 削除後解析失敗: {deleted_data.get('error')}")
                else:
                    print(f"❌ 削除後解析API失敗: {response2.status_code}")
            else:
                print(f"❌ Ba0削除失敗: {delete_result.get('error')}")
        else:
            print(f"❌ Ba0削除API失敗: {response.status_code}")
    else:
        print("Ba0原子が見つかりません")
    
    # Step 4: Ti1原子をSrに置換
    print(f"\n4. Ti1原子をSrに置換...")
    ti1_indices = label_counts.get('Ti1', [])
    if ti1_indices:
        response = requests.post('http://localhost:5000/replace_atoms', 
                               headers={'Content-Type': 'application/json'},
                               json={
                                   'cif_content': original_cif,
                                   'atom_indices': ti1_indices,
                                   'new_element': 'Sr'
                               })
        
        if response.status_code == 200:
            replace_result = response.json()
            if replace_result['success']:
                print(f"✅ Ti1→Sr置換成功")
                
                # 置換後の構造を確認
                replaced_cif = replace_result['modified_cif']
                response2 = requests.post('http://localhost:5000/parse_cif', 
                                       headers={'Content-Type': 'application/json'},
                                       json={'cif_content': replaced_cif})
                
                if response2.status_code == 200:
                    replaced_data = response2.json()
                    if replaced_data['success']:
                        print("置換後の原子:")
                        for atom in replaced_data['atom_info']:
                            print(f"  {atom['label']}: {atom['element']}")
                        
                        # 格子定数の変化もチェック
                        orig_lp = original_data.get('lattice_parameters', {})
                        repl_lp = replaced_data.get('lattice_parameters', {})
                        if orig_lp and repl_lp:
                            print("\n格子定数の変化:")
                            print(f"  a: {orig_lp['a']:.6f} → {repl_lp['a']:.6f}")
                            print(f"  b: {orig_lp['b']:.6f} → {repl_lp['b']:.6f}")
                            print(f"  c: {orig_lp['c']:.6f} → {repl_lp['c']:.6f}")
                    else:
                        print(f"❌ 置換後解析失敗: {replaced_data.get('error')}")
                else:
                    print(f"❌ 置換後解析API失敗: {response2.status_code}")
            else:
                print(f"❌ Ti1→Sr置換失敗: {replace_result.get('error')}")
        else:
            print(f"❌ Ti1→Sr置換API失敗: {response.status_code}")
    else:
        print("Ti1原子が見つかりません")
    
    print("\n=== ワークフローテスト完了 ===")

if __name__ == "__main__":
    test_complete_workflow()