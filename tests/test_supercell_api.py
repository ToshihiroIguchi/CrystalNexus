#!/usr/bin/env python3
"""
修正されたスーパーセルAPIのテストスクリプト
"""

import requests
import json

def test_supercell_api():
    # 元のBaTiO3.cifファイルを読み込み
    with open('BaTiO3.cif', 'r') as f:
        cif_content = f.read()
    
    print("=== 元の結晶構造情報 ===")
    # 元の構造をテスト
    response = requests.get('http://localhost:5000/test_cif')
    if response.status_code == 200:
        original_data = response.json()
        print(f"格子定数: a={original_data['lattice_parameters']['a']:.6f}")
        print(f"格子定数: b={original_data['lattice_parameters']['b']:.6f}")
        print(f"格子定数: c={original_data['lattice_parameters']['c']:.6f}")
        print(f"体積: {original_data['volume']:.6f} Ų")
        print(f"原子数: {original_data['atom_count']}")
        print(f"結晶系: {original_data['crystal_system']}")
        print(f"空間群: {original_data['space_group']}")
    
    print("\n=== 2x2x2スーパーセル作成テスト ===")
    # 2x2x2スーパーセルを作成
    supercell_data = {
        "cif_content": cif_content,
        "a_multiplier": 2,
        "b_multiplier": 2,
        "c_multiplier": 2
    }
    
    response = requests.post('http://localhost:5000/create_supercell', 
                           headers={'Content-Type': 'application/json'},
                           json=supercell_data)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("✅ スーパーセル作成成功！")
            print(f"原子数: {result['original_atoms']} → {result['supercell_atoms']}")
            
            if 'supercell_info' in result and result['supercell_info']:
                sc_info = result['supercell_info']
                print(f"\nスーパーセル情報:")
                if 'lattice_parameters' in sc_info:
                    lp = sc_info['lattice_parameters']
                    print(f"格子定数: a={lp['a']:.6f}, b={lp['b']:.6f}, c={lp['c']:.6f}")
                print(f"体積: {sc_info.get('volume', 'N/A')} Ų")
                print(f"結晶系: {sc_info.get('crystal_system', 'N/A')}")
                print(f"空間群: {sc_info.get('space_group', 'N/A')}")
                print(f"密度: {sc_info.get('density', 'N/A')} g/cm³")
                
                if 'direct_from_pymatgen' in sc_info:
                    print("✅ 格子定数はpymatgenから直接取得")
            
            if 'lattice_comparison' in result and result['lattice_comparison']:
                lc = result['lattice_comparison']
                print(f"\n格子定数の倍率確認:")
                print(f"a: ×{lc['scaling_factors']['a']}")
                print(f"b: ×{lc['scaling_factors']['b']}")  
                print(f"c: ×{lc['scaling_factors']['c']}")
        else:
            print(f"❌ スーパーセル作成失敗: {result.get('error', 'Unknown error')}")
    else:
        print(f"❌ API呼び出し失敗: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_supercell_api()