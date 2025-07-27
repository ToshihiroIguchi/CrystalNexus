#!/usr/bin/env python3
"""
ファイルアップロード時のCIF解析テスト
Enhanced parsing endpoint の動作確認
"""

import requests
import json

def test_file_upload_parsing():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== ファイルアップロードCIF解析テスト ===\n")
    
    # 1. サーバー健康状態確認
    print("🔬 ステップ1: サーバー健康状態確認")
    try:
        health_response = requests.get(f'{SERVER_URL}/health')
        if health_response.status_code == 200:
            print("  ✅ サーバー正常")
        else:
            print(f"  ❌ サーバー異常: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ 接続エラー: {e}")
        return False
    
    # 2. 元のCIF読み込み（ファイルアップロードシミュレーション）
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    print(f"📁 CIFファイル内容読み込み完了 ({len(original_cif)} chars)")
    
    # 3. 通常のparse_cifエンドポイントテスト
    print("\n🔬 ステップ2: 通常のparse_cifエンドポイント")
    parse_response = requests.post(f'{SERVER_URL}/parse_cif', json={
        'cif_content': original_cif
    })
    
    if parse_response.status_code == 200:
        parse_data = parse_response.json()
        print(f"  ✅ 通常解析成功")
        print(f"    空間群: {parse_data.get('space_group')}")
        print(f"    結晶系: {parse_data.get('crystal_system')}")
        print(f"    化学式: {parse_data.get('formula')}")
        print(f"    原子数: {parse_data.get('atom_count')}")
        print(f"    格子定数a: {parse_data.get('lattice_parameters', {}).get('a')}Å")
        
        # Unknown値チェック
        unknowns = []
        for key, value in parse_data.items():
            if value == "Unknown" or value == 0 or value is None:
                unknowns.append(key)
        
        if unknowns:
            print(f"  ⚠️  Unknown値: {unknowns}")
        else:
            print(f"  ✅ 全情報取得成功")
            
    else:
        print(f"  ❌ 通常解析失敗: {parse_response.status_code}")
        print(f"  エラー詳細: {parse_response.text}")
    
    # 4. Enhanced parse_cif_enhancedエンドポイントテスト
    print("\n🔬 ステップ3: Enhanced parse_cif_enhancedエンドポイント")
    enhanced_response = requests.post(f'{SERVER_URL}/parse_cif_enhanced', json={
        'cif_content': original_cif
    })
    
    if enhanced_response.status_code == 200:
        enhanced_data = enhanced_response.json()
        print(f"  ✅ Enhanced解析成功")
        print(f"    空間群: {enhanced_data.get('space_group')}")
        print(f"    結晶系: {enhanced_data.get('crystal_system')}")
        print(f"    化学式: {enhanced_data.get('formula')}")
        print(f"    原子数: {enhanced_data.get('atom_count')}")
        print(f"    格子定数a: {enhanced_data.get('lattice_parameters', {}).get('a')}Å")
        print(f"    解析方法: {enhanced_data.get('parsing_method')}")
        
        # Unknown値チェック
        unknowns = []
        for key, value in enhanced_data.items():
            if value == "Unknown" or value == 0 or value is None:
                unknowns.append(key)
        
        if unknowns:
            print(f"  ⚠️  Unknown値: {unknowns}")
        else:
            print(f"  ✅ 全情報取得成功（Enhanced）")
            
    else:
        print(f"  ❌ Enhanced解析失敗: {enhanced_response.status_code}")
        print(f"  エラー詳細: {enhanced_response.text}")
    
    # 5. エラーケーステスト（不正なCIF）
    print("\n🔬 ステップ4: エラーケーステスト")
    invalid_cif = "invalid cif content"
    
    error_response = requests.post(f'{SERVER_URL}/parse_cif_enhanced', json={
        'cif_content': invalid_cif
    })
    
    if error_response.status_code == 200:
        error_data = error_response.json()
        print(f"  ✅ エラーハンドリング成功")
        print(f"    解析成功: {error_data.get('success', False)}")
        print(f"    エラーメッセージ: {error_data.get('error', 'なし')}")
    else:
        print(f"  ❌ エラーハンドリング失敗: {error_response.status_code}")
    
    # 6. 結果比較
    print("\n📊 結果比較:")
    if parse_response.status_code == 200 and enhanced_response.status_code == 200:
        normal_data = parse_response.json()
        enhanced_data = enhanced_response.json()
        
        comparison_fields = ['space_group', 'crystal_system', 'formula', 'atom_count']
        
        print("  フィールド比較:")
        for field in comparison_fields:
            normal_val = normal_data.get(field)
            enhanced_val = enhanced_data.get(field)
            match = "✅" if normal_val == enhanced_val else "❌"
            print(f"    {field}: 通常={normal_val} vs Enhanced={enhanced_val} {match}")
    
    print(f"\n📋 結論:")
    if parse_response.status_code == 200 and enhanced_response.status_code == 200:
        print("✅ ファイルアップロード解析は正常に動作しています")
        print("💡 フロントエンド側でenhanced endpointを使用することを推奨")
        print("🌐 ブラウザテスト: http://127.0.0.1:5000")
        return True
    else:
        print("❌ 解析エンドポイントに問題があります")
        return False

if __name__ == "__main__":
    success = test_file_upload_parsing()
    exit(0 if success else 1)