#!/usr/bin/env python3
"""
スーパーセル後のAtom Selection永続性テスト

このテストは以下の不具合を検出できます：
1. スーパーセル作成後にAtom Selectionが元の原子数に戻る
2. スーパーセル情報が適切に保持されない
3. CIF再パースによる原子情報の上書き
"""

import requests
import json
import time

# テスト設定
SERVER_BASE_URL = 'http://127.0.0.1:5000'
TIMEOUT = 15

def test_supercell_atom_selection_persistence():
    """スーパーセル後のAtom Selection永続性テスト"""
    
    print("="*80)
    print("🧪 スーパーセル後のAtom Selection永続性テスト")
    print("="*80)
    
    # サーバー接続確認
    try:
        health_response = requests.get(f'{SERVER_BASE_URL}/health', timeout=5)
        if health_response.status_code != 200:
            print("❌ サーバーに接続できません")
            return False
        print("✅ サーバー接続確認完了")
    except Exception as e:
        print(f"❌ サーバー接続エラー: {e}")
        return False
    
    # BaTiO3.cifファイル読み込み
    try:
        with open('BaTiO3.cif', 'r') as f:
            cif_content = f.read()
        print(f"✅ CIFファイル読み込み完了: {len(cif_content)}文字")
    except Exception as e:
        print(f"❌ CIFファイル読み込みエラー: {e}")
        return False
    
    print("\n" + "-"*50)
    print("📋 テスト1: 元の構造の原子情報取得")
    print("-"*50)
    
    # 1. 元の構造をパース
    parse_data = {'cif_content': cif_content}
    parse_response = requests.post(f'{SERVER_BASE_URL}/parse_cif', 
                                  json=parse_data, timeout=TIMEOUT)
    
    if parse_response.status_code != 200:
        print(f"❌ CIFパースに失敗: {parse_response.status_code}")
        return False
    
    parse_result = parse_response.json()
    if not parse_result.get('success') or 'atom_info' not in parse_result:
        print(f"❌ CIFパースエラー: {parse_result.get('error', 'Unknown error')}")
        return False
    
    original_atom_info = parse_result['atom_info']
    original_atom_count = len(original_atom_info)
    original_labels = [atom['label'] for atom in original_atom_info]
    
    print(f"✅ 元の構造: {original_atom_count}個の原子")
    print(f"   原子ラベル: {original_labels}")
    
    # 元素分布を確認
    original_elements = {}
    for atom in original_atom_info:
        elem = atom['type_symbol']
        original_elements[elem] = original_elements.get(elem, 0) + 1
    print(f"   元素分布: {original_elements}")
    
    print("\n" + "-"*50)
    print("📋 テスト2: スーパーセル作成と原子情報取得")
    print("-"*50)
    
    # 2. スーパーセル作成 (2x2x2)
    supercell_data = {
        'cif_content': cif_content,
        'a_multiplier': 2,
        'b_multiplier': 2,
        'c_multiplier': 2
    }
    
    print("🔄 スーパーセル作成中...")
    supercell_response = requests.post(f'{SERVER_BASE_URL}/create_supercell',
                                      json=supercell_data, timeout=TIMEOUT)
    
    if supercell_response.status_code != 200:
        print(f"❌ スーパーセル作成に失敗: {supercell_response.status_code}")
        print(f"   レスポンス: {supercell_response.text[:200]}")
        return False
    
    supercell_result = supercell_response.json()
    if not supercell_result.get('success') or 'supercell_info' not in supercell_result:
        print(f"❌ スーパーセル作成エラー: {supercell_result.get('error', 'Unknown error')}")
        return False
    
    supercell_info = supercell_result['supercell_info']
    if 'atom_info' not in supercell_info:
        print("❌ スーパーセル情報にatom_infoが含まれていません")
        return False
    
    supercell_atom_info = supercell_info['atom_info']
    supercell_atom_count = len(supercell_atom_info)
    supercell_labels = [atom['label'] for atom in supercell_atom_info]
    
    print(f"✅ スーパーセル: {supercell_atom_count}個の原子")
    print(f"   最初の15個のラベル: {supercell_labels[:15]}")
    
    # スーパーセルの元素分布を確認
    supercell_elements = {}
    for atom in supercell_atom_info:
        elem = atom['type_symbol']
        supercell_elements[elem] = supercell_elements.get(elem, 0) + 1
    print(f"   元素分布: {supercell_elements}")
    
    # 期待値と比較
    expected_multiplier = 8  # 2x2x2 = 8
    expected_atom_count = original_atom_count * expected_multiplier
    expected_elements = {elem: count * expected_multiplier 
                        for elem, count in original_elements.items()}
    
    print(f"\n📊 期待値との比較:")
    print(f"   期待原子数: {expected_atom_count}, 実際: {supercell_atom_count}")
    print(f"   期待元素分布: {expected_elements}")
    
    # 原子数チェック
    atom_count_correct = (supercell_atom_count == expected_atom_count)
    print(f"   原子数正確性: {'✅' if atom_count_correct else '❌'}")
    
    # 元素分布チェック  
    elements_correct = (supercell_elements == expected_elements)
    print(f"   元素分布正確性: {'✅' if elements_correct else '❌'}")
    
    print("\n" + "-"*50)
    print("📋 テスト3: スーパーセルCIF再パーステスト（不具合検出）")
    print("-"*50)
    
    # 3. スーパーセルCIFを再パースして不具合を検出
    supercell_cif = supercell_result.get('supercell_cif', '')
    if not supercell_cif:
        print("❌ スーパーセルCIFデータが取得できません")
        return False
    
    print(f"🔄 スーパーセルCIF再パース中... ({len(supercell_cif)}文字)")
    reparse_data = {'cif_content': supercell_cif}
    reparse_response = requests.post(f'{SERVER_BASE_URL}/parse_cif',
                                    json=reparse_data, timeout=TIMEOUT)
    
    if reparse_response.status_code != 200:
        print(f"❌ スーパーセルCIF再パースに失敗: {reparse_response.status_code}")
        return False
    
    reparse_result = reparse_response.json()
    if not reparse_result.get('success') or 'atom_info' not in reparse_result:
        print(f"❌ スーパーセルCIF再パースエラー: {reparse_result.get('error', 'Unknown error')}")
        return False
    
    reparsed_atom_info = reparse_result['atom_info']
    reparsed_atom_count = len(reparsed_atom_info)
    reparsed_labels = [atom['label'] for atom in reparsed_atom_info]
    
    print(f"📊 再パース結果:")
    print(f"   原子数: {reparsed_atom_count}")
    print(f"   最初の15個のラベル: {reparsed_labels[:15]}")
    
    # 不具合検出ロジック
    print(f"\n🔍 不具合検出:")
    
    # Case 1: 再パース後に原子数が元に戻ってしまう（主要な不具合）
    if reparsed_atom_count == original_atom_count:
        print(f"🚨 【不具合検出】再パース後に原子数が元の構造に戻りました")
        print(f"   スーパーセル: {supercell_atom_count}個 → 再パース: {reparsed_atom_count}個")
        print(f"   これは明らかな不具合です。フロントエンドでcurrentAtomInfoが上書きされている可能性があります。")
        return False
    
    # Case 2: 原子数は正しいが、ラベルが異なる
    elif reparsed_atom_count == supercell_atom_count:
        if reparsed_labels != supercell_labels:
            print(f"⚠️  【軽微な不具合】原子数は正しいが、ラベルが異なります")
            print(f"   これはラベル生成ロジックの違いが原因の可能性があります。")
        else:
            print(f"✅ 再パース後も正しいスーパーセル情報が保持されています")
    
    # Case 3: 予期しない原子数
    else:
        print(f"🚨 【予期しない不具合】再パース後の原子数が期待値と異なります")
        print(f"   期待: {supercell_atom_count}個, 実際: {reparsed_atom_count}個")
        return False
    
    print("\n" + "="*50)
    print("📈 テスト結果サマリー")
    print("="*50)
    print(f"元の構造: {original_atom_count}個の原子 ({original_labels})")
    print(f"スーパーセル: {supercell_atom_count}個の原子")
    print(f"再パース: {reparsed_atom_count}個の原子")
    print(f"原子数正確性: {'✅' if atom_count_correct else '❌'}")
    print(f"元素分布正確性: {'✅' if elements_correct else '❌'}")
    
    # 最終判定
    overall_success = (atom_count_correct and elements_correct and 
                      reparsed_atom_count == supercell_atom_count)
    
    if overall_success:
        print("\n🎉 全てのテストに合格しました！")
        print("   スーパーセル後のAtom Selection永続性は正常に動作しています。")
    else:
        print("\n❌ テストに失敗しました。")
        print("   スーパーセル後のAtom Selection永続性に問題があります。")
    
    return overall_success

def main():
    """メイン実行関数"""
    print("スーパーセル後のAtom Selection永続性テスト開始")
    print(f"サーバー: {SERVER_BASE_URL}")
    print(f"タイムアウト: {TIMEOUT}秒")
    
    try:
        success = test_supercell_atom_selection_persistence()
        if success:
            print("\n✅ テスト完了: 合格")
            exit(0)
        else:
            print("\n❌ テスト完了: 不具合検出")
            exit(1)
    except Exception as e:
        print(f"\n💥 テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()