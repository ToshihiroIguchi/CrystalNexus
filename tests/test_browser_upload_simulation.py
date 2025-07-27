#!/usr/bin/env python3
"""
ブラウザファイルアップロードシミュレーション
フロントエンドの動作をシミュレート
"""

import requests
import json

def test_browser_upload_simulation():
    SERVER_URL = 'http://localhost:5000'
    
    print("=== ブラウザファイルアップロードシミュレーションテスト ===\n")
    
    # 1. ファイル読み込み（ブラウザがやること）
    with open('BaTiO3.cif', 'r') as f:
        uploaded_file_content = f.read()
    
    print(f"📁 アップロードファイル内容: {len(uploaded_file_content)} chars")
    print(f"📄 ファイル先頭: {uploaded_file_content[:100]}...")
    
    # 2. Enhanced endpointでの解析（新しいフロントエンド動作）
    print("\n🔬 ステップ1: Enhanced endpointでの解析")
    
    try:
        response = requests.post(f'{SERVER_URL}/parse_cif_enhanced', json={
            'cif_content': uploaded_file_content
        })
        
        if response.status_code == 200:
            file_info = response.json()
            print(f"  ✅ 解析成功")
            print(f"    ファイル名: test_upload.cif (シミュレート)")
            print(f"    空間群: {file_info.get('space_group')}")
            print(f"    結晶系: {file_info.get('crystal_system')}")
            print(f"    化学式: {file_info.get('formula')}")
            print(f"    原子数: {file_info.get('atom_count')}")
            print(f"    格子定数a: {file_info.get('lattice_parameters', {}).get('a')}Å")
            print(f"    格子定数b: {file_info.get('lattice_parameters', {}).get('b')}Å")
            print(f"    格子定数c: {file_info.get('lattice_parameters', {}).get('c')}Å")
            print(f"    体積: {file_info.get('volume')}Ų")
            print(f"    解析方法: {file_info.get('parsing_method', '標準CifParser')}")
            
            # Unknown値のチェック
            unknown_fields = []
            zero_fields = []
            
            check_fields = [
                'space_group', 'crystal_system', 'formula', 'atom_count', 'volume'
            ]
            
            for field in check_fields:
                value = file_info.get(field)
                if value == "Unknown":
                    unknown_fields.append(field)
                elif value == 0:
                    zero_fields.append(field)
            
            lattice_params = file_info.get('lattice_parameters', {})
            for param in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
                value = lattice_params.get(param)
                if value == "Unknown":
                    unknown_fields.append(f'lattice.{param}')
                elif value == 0:
                    zero_fields.append(f'lattice.{param}')
            
            print(f"\n  📊 データ品質チェック:")
            if unknown_fields:
                print(f"    ❌ Unknown値フィールド: {unknown_fields}")
            else:
                print(f"    ✅ Unknown値なし")
                
            if zero_fields:
                print(f"    ⚠️  ゼロ値フィールド: {zero_fields}")
            else:
                print(f"    ✅ ゼロ値なし")
            
            success = len(unknown_fields) == 0 and len(zero_fields) == 0
            
            print(f"\n  🎯 アップロード解析結果: {'✅ 成功' if success else '❌ 問題あり'}")
            
            if success:
                print(f"  💡 ブラウザでの表示予想:")
                print(f"    - 空間群: {file_info.get('space_group')}")
                print(f"    - 結晶系: {file_info.get('crystal_system')}")  
                print(f"    - 化学式: {file_info.get('formula')}")
                print(f"    - 原子数: {file_info.get('atom_count')}")
                print(f"    - 格子定数a: {lattice_params.get('a', 'N/A')}Å")
                print(f"    - 体積: {file_info.get('volume', 'N/A')}Ų")
            
            return success
            
        else:
            print(f"  ❌ 解析失敗: HTTP {response.status_code}")
            print(f"  エラー詳細: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ 例外発生: {e}")
        return False

if __name__ == "__main__":
    success = test_browser_upload_simulation()
    if success:
        print(f"\n🎉 ファイルアップロード解析が正常に動作しています！")
        print(f"🌐 ブラウザで http://127.0.0.1:5000 にアクセスして確認してください")
    else:
        print(f"\n❌ ファイルアップロード解析に問題があります")
    exit(0 if success else 1)