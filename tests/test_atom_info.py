#!/usr/bin/env python3
"""
原子情報のテスト - スーパーセル作成時のatom_info確認
"""

import requests
import json

def test_atom_info():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== 原子情報テスト ===\n")
    
    # 1. BaTiO3.cifを読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    # 2. スーパーセル作成API呼び出し
    print("🔬 スーパーセル作成API呼び出し")
    supercell_response = requests.post(f'{SERVER_URL}/create_supercell', json={
        'cif_content': original_cif,
        'a_multiplier': 2,
        'b_multiplier': 2,
        'c_multiplier': 2
    })
    
    supercell_data = supercell_response.json()
    
    # 3. atom_info確認
    atom_info = supercell_data.get('supercell_info', {}).get('atom_info', [])
    
    print(f"atom_info原子数: {len(atom_info)}")
    print(f"期待値: 40原子")
    
    if len(atom_info) > 0:
        print(f"\n最初の5原子:")
        for i, atom in enumerate(atom_info[:5]):
            print(f"  {i}: {atom}")
        
        print(f"\n最後の5原子:")
        for i, atom in enumerate(atom_info[-5:], len(atom_info)-5):
            print(f"  {i}: {atom}")
    
    # 4. 原子ラベルの一意性確認
    labels = [atom.get('label', '') for atom in atom_info]
    unique_labels = set(labels)
    
    print(f"\n原子ラベル:")
    print(f"  総数: {len(labels)}")
    print(f"  ユニーク数: {len(unique_labels)}")
    print(f"  ユニークラベル: {sorted(unique_labels)}")
    
    # 5. 評価
    atoms_ok = len(atom_info) == 40
    labels_unique = len(labels) == len(unique_labels)
    
    print(f"\n評価:")
    print(f"  原子数: {'✅' if atoms_ok else '❌'} ({len(atom_info)}/40)")
    print(f"  ラベル一意性: {'✅' if labels_unique else '❌'}")
    
    overall_success = atoms_ok and labels_unique
    print(f"\n総合: {'✅ 成功' if overall_success else '❌ 失敗'}")
    
    return overall_success

if __name__ == "__main__":
    success = test_atom_info()
    exit(0 if success else 1)