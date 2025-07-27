#!/usr/bin/env python3
"""
最終統合テスト - 実際のAPIエンドポイントでスーパーセル保持機能を確認
"""

import requests
import json

def test_final_integration():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== 最終統合テスト（実際のAPIエンドポイント） ===\n")
    
    # 1. BaTiO3.cifを読み込み
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    print("🔬 ステップ1: 元の構造読み込み")
    print(f"  元のCIF: {len(original_cif.splitlines())}行")
    
    # 2. スーパーセル作成API呼び出し
    print("\n🔬 ステップ2: スーパーセル作成API")
    supercell_response = requests.post(f'{SERVER_URL}/create_supercell', json={
        'cif_content': original_cif,
        'a_multiplier': 2,
        'b_multiplier': 2,
        'c_multiplier': 2
    })
    
    if supercell_response.status_code != 200:
        print(f"❌ スーパーセル作成失敗: {supercell_response.status_code}")
        return False
    
    supercell_data = supercell_response.json()
    if not supercell_data.get('success'):
        print(f"❌ スーパーセル作成エラー: {supercell_data.get('error')}")
        return False
    
    supercell_cif = supercell_data['supercell_cif']
    supercell_info = supercell_data.get('supercell_info', {})
    
    print(f"  ✅ スーパーセル作成成功")
    print(f"  原子数: {supercell_info.get('atom_count', 'N/A')}")
    print(f"  格子定数a: {supercell_info.get('lattice_parameters', {}).get('a', 'N/A')}")
    print(f"  スーパーセル: {supercell_data.get('is_supercell', False)}")
    
    # 3. 原子置換API呼び出し（スーパーセル保持テスト）
    print("\n🔬 ステップ3: 原子置換API（スーパーセル保持）")
    
    supercell_metadata = {
        'multipliers': supercell_data.get('multipliers', {}),
        'original_atoms': supercell_data.get('original_atoms', 0),
        'is_supercell': True
    }
    
    print(f"  送信メタデータ: {supercell_metadata}")
    
    replace_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': [0],  # 最初の原子を置換
        'new_element': 'Sr',
        'supercell_metadata': supercell_metadata
    })
    
    if replace_response.status_code != 200:
        print(f"❌ 原子置換失敗: {replace_response.status_code}")
        return False
    
    replace_data = replace_response.json()
    if not replace_data.get('success'):
        print(f"❌ 原子置換エラー: {replace_data.get('error')}")
        return False
    
    modified_cif = replace_data['modified_cif']
    modified_info = replace_data.get('modified_structure_info', {})
    
    print(f"  ✅ 原子置換成功")
    print(f"  元の原子数: {replace_data.get('original_atom_count', 'N/A')}")
    print(f"  修正後原子数: {replace_data.get('modified_atom_count', 'N/A')}")
    print(f"  修正後格子定数a: {modified_info.get('lattice_parameters', {}).get('a', 'N/A')}")
    print(f"  スーパーセル維持: {modified_info.get('is_supercell', False)}")
    
    # 4. 結果評価
    print("\n🔍 統合テスト評価:")
    
    # 期待値
    expected_supercell_atoms = 40
    expected_supercell_lattice_a = 7.980758
    expected_modified_atoms = 40  # 置換なので原子数は変わらない
    
    # 評価
    supercell_atoms_ok = supercell_info.get('atom_count') == expected_supercell_atoms
    supercell_lattice_ok = abs(supercell_info.get('lattice_parameters', {}).get('a', 0) - expected_supercell_lattice_a) < 0.01
    modified_atoms_ok = replace_data.get('modified_atom_count') == expected_modified_atoms
    modified_lattice_ok = abs(modified_info.get('lattice_parameters', {}).get('a', 0) - expected_supercell_lattice_a) < 0.01
    supercell_maintained = modified_info.get('is_supercell', False)
    
    print(f"  スーパーセル原子数: {'✅' if supercell_atoms_ok else '❌'} ({supercell_info.get('atom_count')}/{expected_supercell_atoms})")
    print(f"  スーパーセル格子定数: {'✅' if supercell_lattice_ok else '❌'} ({supercell_info.get('lattice_parameters', {}).get('a', 'N/A'):.6f}/{expected_supercell_lattice_a:.6f})")
    print(f"  修正後原子数: {'✅' if modified_atoms_ok else '❌'} ({replace_data.get('modified_atom_count')}/{expected_modified_atoms})")
    print(f"  修正後格子定数: {'✅' if modified_lattice_ok else '❌'} ({modified_info.get('lattice_parameters', {}).get('a', 0):.6f}/{expected_supercell_lattice_a:.6f})")
    print(f"  スーパーセル維持: {'✅' if supercell_maintained else '❌'} ({supercell_maintained})")
    
    # 総合評価
    overall_success = all([
        supercell_atoms_ok,
        supercell_lattice_ok, 
        modified_atoms_ok,
        modified_lattice_ok,
        supercell_maintained
    ])
    
    print(f"\n📊 最終評価: {'✅ 完全成功' if overall_success else '❌ 部分的失敗'}")
    
    if overall_success:
        print("🎉 スーパーセル保持機能が完全に動作しています！")
        print("💡 Webアプリケーションで原子編集を行っても、スーパーセル構造が維持されます。")
    else:
        print("💡 一部の機能に問題があります。詳細な調査が必要です。")
    
    # 5. CIF出力確認
    print(f"\n📄 最終CIF確認:")
    print(f"  最終CIF行数: {len(modified_cif.splitlines())}")
    
    # カスタムCIF生成が使用されているか確認
    is_custom_cif = "supercell preserved" in modified_cif
    print(f"  カスタムCIF生成: {'✅' if is_custom_cif else '❌'}")
    
    return overall_success

if __name__ == "__main__":
    success = test_final_integration()
    exit(0 if success else 1)