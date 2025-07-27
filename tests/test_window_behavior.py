#!/usr/bin/env python3
"""
ウィンドウの動作テスト用スクリプト
"""

import requests
import json

def test_supercell_creation():
    print("=== ウィンドウ動作のテスト ===")
    
    # BaTiO3.cifファイルを読み込み
    with open('BaTiO3.cif', 'r') as f:
        cif_content = f.read()
    
    print("1. スーパーセル作成テスト (2x1x1)...")
    
    # 2x1x1スーパーセルを作成
    supercell_data = {
        "cif_content": cif_content,
        "a_multiplier": 2,
        "b_multiplier": 1,
        "c_multiplier": 1
    }
    
    response = requests.post('http://localhost:5000/create_supercell', 
                           headers={'Content-Type': 'application/json'},
                           json=supercell_data)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("✅ スーパーセル作成成功！")
            print(f"原子数: {result['original_atoms']} → {result['supercell_atoms']}")
            print("ブラウザでスーパーセル作成を実行すると、3秒後にウィンドウが自動で閉じるはずです。")
        else:
            print(f"❌ スーパーセル作成失敗: {result.get('error')}")
    else:
        print(f"❌ API呼び出し失敗: {response.status_code}")
        print(response.text)
    
    print("\n=== テスト完了 ===")
    print("ブラウザで以下の操作を確認してください：")
    print("1. サーバー起動時にブラウザが自動で開く")
    print("2. CIFファイルを選択して読み込み")
    print("3. スーパーセル作成を実行")  
    print("4. スーパーセル作成後、3秒後にウィンドウが自動で閉じる")

if __name__ == "__main__":
    test_supercell_creation()