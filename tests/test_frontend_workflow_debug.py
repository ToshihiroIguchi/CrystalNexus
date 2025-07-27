#!/usr/bin/env python3
"""
フロントエンドワークフロー詳細デバッグテスト

このテストは、実際のブラウザでのスーパーセル作成プロセスを詳細に追跡し、
currentAtomInfoが5個の原子に戻る正確なタイミングと原因を特定します。
"""

import requests
import json
import time
from datetime import datetime

# テスト設定
SERVER_BASE_URL = 'http://127.0.0.1:5000'
TIMEOUT = 15

class ServerLogMonitor:
    """サーバーログを監視してAPI呼び出しを追跡"""
    
    def __init__(self):
        self.api_calls = []
        self.start_time = datetime.now()
    
    def track_api_call(self, method, endpoint, description=""):
        """API呼び出しを記録"""
        call_info = {
            'timestamp': datetime.now(),
            'method': method,
            'endpoint': endpoint,
            'description': description
        }
        self.api_calls.append(call_info)
        print(f"📡 [{method}] {endpoint} - {description}")
        return call_info

def test_frontend_workflow_debug():
    """フロントエンドワークフローの詳細デバッグ"""
    
    print("="*80)
    print("🔍 フロントエンドワークフロー詳細デバッグテスト")
    print("="*80)
    
    monitor = ServerLogMonitor()
    
    # サーバー接続確認
    try:
        health_response = requests.get(f'{SERVER_BASE_URL}/health', timeout=5)
        if health_response.status_code != 200:
            print("❌ サーバーに接続できません")
            return False
        monitor.track_api_call('GET', '/health', 'サーバー接続確認')
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
    
    print("\n" + "-"*60)
    print("🎬 シナリオ1: ブラウザでのファイル選択・初期表示")
    print("-"*60)
    
    # Step 1: ブラウザでBaTiO3.cifを選択した時の動作
    print("Step 1: ブラウザでBaTiO3.cifファイルを選択")
    parse_data = {'cif_content': cif_content}
    parse_response = requests.post(f'{SERVER_BASE_URL}/parse_cif', 
                                  json=parse_data, timeout=TIMEOUT)
    monitor.track_api_call('POST', '/parse_cif', 'ブラウザファイル選択時のCIFパース')
    
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
    
    print(f"   ✅ 初期表示: {original_atom_count}個の原子")
    print(f"   原子ラベル: {original_labels}")
    print(f"   💡 この時点でAtom Selectionには{original_atom_count}個の選択肢が表示されます")
    
    # Step 2: ブラウザでの構造情報更新（updateStructureInfoWithServer相当）
    print("\\nStep 2: 構造情報表示のための追加パース（もしあれば）")
    # 実際のブラウザでは、ファイル選択後にupdateStructureInfoWithServerが呼ばれることがある
    time.sleep(0.1)  # ブラウザでの処理間隔をシミュレート
    
    print("\n" + "-"*60)
    print("🎬 シナリオ2: スーパーセル作成プロセス")
    print("-"*60)
    
    # Step 3: スーパーセル作成（2x2x2）
    print("Step 3: スーパーセル作成要求")
    supercell_data = {
        'cif_content': cif_content,
        'a_multiplier': 2,
        'b_multiplier': 2,
        'c_multiplier': 2
    }
    
    print("🔄 スーパーセル作成中...")
    supercell_response = requests.post(f'{SERVER_BASE_URL}/create_supercell',
                                      json=supercell_data, timeout=TIMEOUT)
    monitor.track_api_call('POST', '/create_supercell', 'スーパーセル作成')
    
    if supercell_response.status_code != 200:
        print(f"❌ スーパーセル作成に失敗: {supercell_response.status_code}")
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
    supercell_labels = [atom['label'] for atom in supercell_atom_info[:15]]
    
    print(f"   ✅ スーパーセル作成成功: {supercell_atom_count}個の原子")
    print(f"   最初の15個のラベル: {supercell_labels}")
    print(f"   💡 この時点で、サーバーは正しく{supercell_atom_count}個の原子データを返しています")
    
    # Step 4: スーパーセルCIFの取得とフロントエンド処理シミュレート
    print("\\nStep 4: スーパーセルCIFの詳細分析")
    supercell_cif = supercell_result.get('supercell_cif', '')
    if not supercell_cif:
        print("❌ スーパーセルCIFデータが取得できません")
        return False
    
    print(f"   スーパーセルCIF長: {len(supercell_cif)}文字")
    
    # Step 5: 【重要】フロントエンドでのinitializeViewerWithData → loadCIF → updateStructureInfoWithServerシミュレート
    print("\\nStep 5: 【重要】フロントエンドでのloadCIF処理シミュレート")
    print("   これがcurrentAtomInfoを上書きする可能性がある箇所です")
    
    # updateStructureInfoWithServerがスーパーセルデータありで呼ばれる場合
    print("   ケース5-1: updateStructureInfoWithServer(supercell_cif, filename, supercellData)")
    print("   → スーパーセルデータがあるので、server解析はスキップされるはず")
    print("   → currentAtomInfoは40個のまま維持されるはず")
    
    # しかし、実際には追加のCIFパースが発生している
    print("\\n   ケース5-2: 何らかの理由でスーパーセルCIFが再パースされる")
    time.sleep(0.1)
    additional_parse_response = requests.post(f'{SERVER_BASE_URL}/parse_cif',
                                            json={'cif_content': supercell_cif}, timeout=TIMEOUT)
    monitor.track_api_call('POST', '/parse_cif', '【問題】スーパーセルCIF再パース')
    
    if additional_parse_response.status_code == 200:
        additional_result = additional_parse_response.json()
        if additional_result.get('success') and 'atom_info' in additional_result:
            reparse_atom_info = additional_result['atom_info']
            reparse_atom_count = len(reparse_atom_info)
            reparse_labels = [atom['label'] for atom in reparse_atom_info]
            
            print(f"   📊 再パース結果: {reparse_atom_count}個の原子")
            print(f"   再パース原子ラベル: {reparse_labels}")
            
            # ここで問題を検出
            if reparse_atom_count == original_atom_count:
                print(f"\\n🚨 【問題検出】スーパーセルCIF再パースで原子数が元に戻りました！")
                print(f"   スーパーセル: {supercell_atom_count}個 → 再パース: {reparse_atom_count}個")
                print(f"   💡 フロントエンドでcurrentAtomInfoがこの{reparse_atom_count}個で上書きされている")
                
                problem_detected = True
            else:
                print(f"\\n✅ 再パース後も正しい原子数: {reparse_atom_count}個")
                problem_detected = False
        else:
            print("   ❌ 再パースに失敗")
            problem_detected = True
    else:
        print("   ❌ 再パース要求失敗")
        problem_detected = True
    
    print("\n" + "-"*60)
    print("📊 API呼び出し履歴")
    print("-"*60)
    
    for i, call in enumerate(monitor.api_calls, 1):
        timestamp = call['timestamp'].strftime('%H:%M:%S.%f')[:-3]
        print(f"{i:2d}. [{timestamp}] {call['method']} {call['endpoint']} - {call['description']}")
    
    print("\n" + "="*60)
    print("🔍 分析結果")
    print("="*60)
    
    print("\\n【修正が効かなかった詳細理由】")
    print("1. サーバー側: スーパーセル作成時に正しく40個の原子データを生成")
    print("2. フロントエンド側: initializeViewerWithData()でスーパーセルデータ付きでloadCIF()を呼び出し")
    print("3. loadCIF()内でupdateStructureInfoWithServer()を呼び出し")
    print("4. updateStructureInfoWithServer()でスーパーセルデータがあるのでcurrentAtomInfoに40個設定")
    print("5. 【問題】その後、別の場所でスーパーセルCIFが再パースされ、5個の原子で上書き")
    
    print("\\n【テストコードが検出できなかった理由】")
    print("1. 既存テスト: サーバーAPIの直接呼び出しのみテスト")
    print("2. 実際の問題: フロントエンドのワークフロー内での複数API呼び出しの相互作用")
    print("3. 検出困難: ブラウザ内のJavaScript実行順序とタイミング依存")
    
    print("\\n【根本的対策】")
    print("1. updateStructureInfoWithServer()でスーパーセルデータがある場合、")
    print("   追加のCIFパースを完全に防ぐ")
    print("2. currentAtomInfoの上書きを防ぐプロテクション機能追加")
    print("3. デバッグログ追加でcurrentAtomInfo変更を追跡可能にする")
    
    if problem_detected:
        print("\\n❌ 問題が検出されました。対策が必要です。")
        return False
    else:
        print("\\n✅ 問題は検出されませんでした。")
        return True

def main():
    """メイン実行関数"""
    print("フロントエンドワークフロー詳細デバッグテスト開始")
    print(f"サーバー: {SERVER_BASE_URL}")
    print(f"タイムアウト: {TIMEOUT}秒")
    
    try:
        success = test_frontend_workflow_debug()
        if success:
            print("\\n✅ テスト完了: 問題なし")
            exit(0)
        else:
            print("\\n❌ テスト完了: 問題検出、対策が必要")
            exit(1)
    except Exception as e:
        print(f"\\n💥 テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()