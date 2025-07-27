#!/usr/bin/env python3
"""
原子編集ワークフローのテスト
二回目の編集ができない問題を調査
"""

import requests
import json
import time

def test_atom_editing_workflow():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== 原子編集ワークフローテスト ===\n")
    
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
    
    if supercell_response.status_code != 200:
        print("❌ スーパーセル作成失敗")
        return False
    
    supercell_data = supercell_response.json()
    supercell_cif = supercell_data['supercell_cif']
    supercell_info = supercell_data.get('supercell_info', {})
    atom_info_initial = supercell_info.get('atom_info', [])
    
    print(f"  ✅ スーパーセル作成成功 - {len(atom_info_initial)}個の原子")
    print(f"  初期原子ラベル: {[atom['label'] for atom in atom_info_initial[:5]]}...")
    
    # 3. 一回目の原子編集（置換）
    print("\n🔬 ステップ2: 一回目の原子編集（Ba → Sr置換）")
    
    # Ba原子のインデックスを取得
    ba_atoms = [atom for atom in atom_info_initial if 'Ba' in atom['label']]
    if not ba_atoms:
        print("❌ Ba原子が見つかりません")
        return False
    
    ba_indices = [ba_atoms[0]['index']]  # 最初のBa原子のみ
    ba_label = ba_atoms[0]['label']
    
    print(f"  編集対象: {ba_label} (index: {ba_indices[0]})")
    
    supercell_metadata = {
        'multipliers': {'a': 2, 'b': 2, 'c': 2},
        'original_atoms': 5,
        'is_supercell': True
    }
    
    first_edit_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': ba_indices,
        'new_element': 'Sr',
        'supercell_metadata': supercell_metadata
    })
    
    if first_edit_response.status_code != 200:
        print("❌ 一回目の編集失敗")
        return False
    
    first_edit_data = first_edit_response.json()
    if not first_edit_data.get('success'):
        print(f"❌ 一回目の編集エラー: {first_edit_data.get('error')}")
        return False
    
    modified_cif_1 = first_edit_data['modified_cif']
    modified_info_1 = first_edit_data.get('modified_structure_info', {})
    atom_info_after_1 = modified_info_1.get('atom_info', [])
    
    print(f"  ✅ 一回目の編集成功 - {len(atom_info_after_1)}個の原子")
    print(f"  編集後原子ラベル: {[atom['label'] for atom in atom_info_after_1[:5]]}...")
    
    # 4. 二回目の原子編集の準備
    print("\n🔬 ステップ3: 二回目の原子編集準備")
    
    # 編集後の原子情報から次の編集対象を選択
    ti_atoms = [atom for atom in atom_info_after_1 if 'Ti' in atom['label']]
    if not ti_atoms:
        print("❌ Ti原子が見つかりません")
        return False
    
    ti_indices = [ti_atoms[0]['index']]  # 最初のTi原子のみ
    ti_label = ti_atoms[0]['label']
    
    print(f"  二回目編集対象: {ti_label} (index: {ti_indices[0]})")
    
    # 5. 二回目の原子編集（置換）
    print("\n🔬 ステップ4: 二回目の原子編集（Ti → Zr置換）")
    
    second_edit_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': modified_cif_1,
        'atom_indices': ti_indices,
        'new_element': 'Zr',
        'supercell_metadata': supercell_metadata
    })
    
    if second_edit_response.status_code != 200:
        print("❌ 二回目の編集失敗")
        print(f"  Status: {second_edit_response.status_code}")
        print(f"  Response: {second_edit_response.text}")
        return False
    
    second_edit_data = second_edit_response.json()
    if not second_edit_data.get('success'):
        print(f"❌ 二回目の編集エラー: {second_edit_data.get('error')}")
        return False
    
    modified_cif_2 = second_edit_data['modified_cif']
    modified_info_2 = second_edit_data.get('modified_structure_info', {})
    atom_info_after_2 = modified_info_2.get('atom_info', [])
    
    print(f"  ✅ 二回目の編集成功 - {len(atom_info_after_2)}個の原子")
    print(f"  最終原子ラベル: {[atom['label'] for atom in atom_info_after_2[:5]]}...")
    
    # 6. 原子ラベルの変化を分析
    print("\n📊 原子ラベル変化分析:")
    
    def get_unique_labels(atom_info):
        return sorted(list(set([atom['label'] for atom in atom_info])))
    
    initial_labels = get_unique_labels(atom_info_initial)
    after_1_labels = get_unique_labels(atom_info_after_1)
    after_2_labels = get_unique_labels(atom_info_after_2)
    
    print(f"  初期ラベル: {initial_labels}")
    print(f"  1回目編集後: {after_1_labels}")
    print(f"  2回目編集後: {after_2_labels}")
    
    # 7. ワークフロー成功判定
    workflow_success = (
        len(atom_info_initial) == len(atom_info_after_1) == len(atom_info_after_2) and
        'Sr' in str(after_1_labels) and  # Ba → Sr置換の確認
        'Zr' in str(after_2_labels)      # Ti → Zr置換の確認
    )
    
    print(f"\n📋 総合評価: {'✅ 成功' if workflow_success else '❌ 失敗'}")
    
    if workflow_success:
        print("🎉 連続原子編集ワークフローが正常に動作しています！")
        print("💡 ブラウザ側の問題である可能性が高いです。")
    else:
        print("💡 サーバー側またはワークフロー全体に問題があります。")
    
    return workflow_success

if __name__ == "__main__":
    success = test_atom_editing_workflow()
    exit(0 if success else 1)