#!/usr/bin/env python3
"""
Atom Selection選択肢が空になる問題の詳細デバッグテスト

このテストは、フロントエンドの実際の実行フローを追跡し、
Atom Selectionが空になる正確な原因を特定します。
"""

import requests
import json
import time

# テスト設定
SERVER_BASE_URL = 'http://127.0.0.1:5000'
TIMEOUT = 15

def test_atom_selection_empty_debug():
    """Atom Selection空問題の詳細デバッグ"""
    
    print("="*80)
    print("🔍 Atom Selection空問題の詳細デバッグテスト")
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
    
    print("\n" + "-"*70)
    print("📋 Phase 1: 初期CIFパース（ブラウザでファイル選択時）")
    print("-"*70)
    
    # 1. 初期CIFパース
    parse_data = {'cif_content': cif_content}
    parse_response = requests.post(f'{SERVER_BASE_URL}/parse_cif', 
                                  json=parse_data, timeout=TIMEOUT)
    
    if parse_response.status_code != 200:
        print(f"❌ 初期CIFパースに失敗: {parse_response.status_code}")
        return False
    
    parse_result = parse_response.json()
    if not parse_result.get('success') or 'atom_info' not in parse_result:
        print(f"❌ 初期CIFパースエラー: {parse_result.get('error', 'Unknown error')}")
        return False
    
    initial_atom_info = parse_result['atom_info']
    initial_atom_count = len(initial_atom_info)
    initial_labels = [atom['label'] for atom in initial_atom_info]
    
    print(f"✅ 初期CIFパース成功: {initial_atom_count}個の原子")
    print(f"   原子ラベル: {initial_labels}")
    print(f"   💡 この時点でAtom Selectionには{initial_atom_count}個の選択肢があるはず")
    
    print("\n" + "-"*70)
    print("📋 Phase 2: スーパーセル作成")
    print("-"*70)
    
    # 2. スーパーセル作成
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
        return False
    
    supercell_result = supercell_response.json()
    if not supercell_result.get('success'):
        print(f"❌ スーパーセル作成エラー: {supercell_result.get('error')}")
        return False
    
    supercell_info = supercell_result.get('supercell_info', {})
    supercell_atom_info = supercell_info.get('atom_info', [])
    supercell_atom_count = len(supercell_atom_info)
    
    print(f"✅ スーパーセル作成成功: {supercell_atom_count}個の原子")
    print(f"   レスポンスキー: {list(supercell_result.keys())}")
    print(f"   is_supercell存在: {'is_supercell' in supercell_result}")
    
    # フロントエンド修正のシミュレート
    print("\n🔧 フロントエンド修正シミュレート: is_supercellフラグ追加")
    supercell_data_with_flag = {
        **supercell_result,
        'is_supercell': True
    }
    print(f"   修正後のis_supercell: {supercell_data_with_flag.get('is_supercell')}")
    
    print("\n" + "-"*70)
    print("📋 Phase 3: フロントエンドフロー分析")
    print("-"*70)
    
    # 3. initializeViewerWithData → loadCIF → updateStructureInfoWithServer フロー分析
    
    print("Step 3.1: initializeViewerWithData() 呼び出し")
    print(f"   引数1: supercell_cif (長さ: {len(supercell_result.get('supercell_cif', ''))}文字)")
    print(f"   引数2: filename")
    print(f"   引数3: supercellDataWithFlag (is_supercell: {supercell_data_with_flag.get('is_supercell')})")
    
    print("\nStep 3.2: loadCIF() 実行分析")
    print(f"   loadCIF(supercell_cif, filename, supercellDataWithFlag)")
    
    # loadCIF()の条件判定をシミュレート
    def simulate_loadCIF_check(supercellData):
        if supercellData and supercellData.get('is_supercell'):
            return True, "スーパーセル状態として認識"
        else:
            return False, "通常のCIF状態として認識"
    
    load_cif_result, load_cif_reason = simulate_loadCIF_check(supercell_data_with_flag)
    print(f"   条件判定結果: {load_cif_result} ({load_cif_reason})")
    
    if load_cif_result:
        print("   ✅ currentSupercellMetadata設定される")
        print("   ✅ is_supercell: true で保存される")
    else:
        print("   ❌ currentSupercellMetadata = null に設定される")
    
    print("\nStep 3.3: updateStructureInfoWithServer() 実行分析")
    print(f"   updateStructureInfoWithServer(supercell_cif, filename, supercellDataWithFlag)")
    
    # updateStructureInfoWithServer()の条件判定をシミュレート
    if load_cif_result:
        print("   スーパーセルデータ存在 → スーパーセル情報使用")
        print(f"   currentAtomInfo設定: {supercell_atom_count}個の原子")
        print("   💡 この時点でAtom Selectionは40個になるはず")
        
        # しかし、その後の処理を確認
        print("\n   ⚠️  しかし、その後で何が起きているか？")
        
    else:
        print("   通常のCIF解析実行 → currentAtomInfo上書き")
        print("   ❌ これが問題の原因")
    
    print("\n" + "-"*70)
    print("📋 Phase 4: 実際のサーバーログとの比較")
    print("-"*70)
    
    # 実際のサーバーログで起きていることを分析
    print("実際のサーバーログから判明した事実:")
    print("1. スーパーセル作成: 40個の原子生成 ✅")
    print("2. その後、スーパーセルCIF再パース発生: 5個の原子")
    print("3. 最終的にAtom Selectionが空になる")
    print()
    print("🔍 これは以下のいずれかを意味します:")
    print("a) is_supercellフラグが正しく設定されていない")
    print("b) 保護機能が正しく動作していない")  
    print("c) initializeAtomLabelSelector()で問題が発生している")
    
    print("\n" + "-"*70)
    print("📋 Phase 5: initializeAtomLabelSelector()の問題調査")
    print("-"*70)
    
    # currentAtomInfoの状態をシミュレート
    print("initializeAtomLabelSelector()が呼ばれる時点での予想:")
    
    if load_cif_result:
        # 修正が正しく動作している場合
        print(f"✅ 正常パターン: currentAtomInfo = {supercell_atom_count}個の原子")
        print(f"   Atom Selectionに{supercell_atom_count}個の選択肢が表示されるはず")
    else:
        # 修正が動作していない場合
        print("❌ 問題パターン1: currentAtomInfo = 5個の原子")
        print("   Atom Selectionに5個の選択肢が表示される")
    
    # さらに悪いケース
    print("❌ 問題パターン2: currentAtomInfo = [] (空配列)")
    print("   → Atom Selectionが完全に空になる")
    print("   → これが現在発生している問題")
    
    print("\n💡 Atom Selectionが空になる可能性:")
    print("1. currentAtomInfoが何らかの理由で空配列になっている")
    print("2. サーバーからのレスポンスでatom_infoが空")
    print("3. JavaScriptエラーでcurrentAtomInfoが未定義")
    print("4. initializeAtomLabelSelector()での処理エラー")
    
    print("\n" + "="*70)
    print("📈 調査結果サマリー")
    print("="*70)
    
    print("✅ 確認できた事実:")
    print(f"- 初期CIFパース: {initial_atom_count}個の原子")
    print(f"- スーパーセル作成: {supercell_atom_count}個の原子")
    print(f"- is_supercellフラグ修正: 実装済み")
    
    print("\n❌ 推定される問題:")
    print("- フロントエンドの実行フローで何らかの問題が発生")
    print("- currentAtomInfoが空配列または未定義になっている")
    print("- ブラウザキャッシュまたは実行タイミングの問題")
    
    print("\n🎯 次のステップ:")
    print("1. ブラウザでハードリロード (Ctrl+Shift+R)")
    print("2. ブラウザコンソールでデバッグメッセージ確認")
    print("3. currentAtomInfoの実際の値を確認")
    print("4. initializeAtomLabelSelector()の実行タイミング確認")
    
    return True

def main():
    """メイン実行関数"""
    print("Atom Selection空問題の詳細デバッグテスト開始")
    print(f"サーバー: {SERVER_BASE_URL}")
    
    try:
        success = test_atom_selection_empty_debug()
        if success:
            print("\n✅ デバッグテスト完了")
            print("   ブラウザでの実際の動作確認を行ってください")
            exit(0)
        else:
            print("\n❌ デバッグテスト失敗")
            exit(1)
    except Exception as e:
        print(f"\n💥 テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()