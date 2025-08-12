#!/usr/bin/env python3
"""
簡單的 Ollama 連接測試
用於診斷網路連接問題
"""

import requests
import socket
import time

# 測試配置
OLLAMA_HOST = "ollama"  # 在 Docker 容器內使用服務名稱
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

def test_network_connectivity():
    """測試基本的網路連接"""
    print("=== 網路連接測試 ===")
    try:
        # 測試 TCP 連接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((OLLAMA_HOST, OLLAMA_PORT))
        sock.close()
        
        if result == 0:
            print(f"✓ TCP 連接到 {OLLAMA_HOST}:{OLLAMA_PORT} 成功")
            return True
        else:
            print(f"✗ TCP 連接到 {OLLAMA_HOST}:{OLLAMA_PORT} 失敗")
            return False
            
    except Exception as e:
        print(f"✗ 網路連接測試失敗: {e}")
        return False

def test_ollama_api():
    """測試 Ollama API"""
    print("\n=== Ollama API 測試 ===")
    
    # 測試 API 端點
    endpoints = [
        "/api/version",
        "/api/tags", 
        "/",
    ]
    
    for endpoint in endpoints:
        url = f"{OLLAMA_BASE_URL}{endpoint}"
        try:
            print(f"測試: {url}")
            response = requests.get(url, timeout=10)
            print(f"  狀態碼: {response.status_code}")
            if response.status_code == 200:
                print(f"  回應: {response.text[:100]}...")
            else:
                print(f"  錯誤: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"  請求失敗: {e}")
        print()

def test_model_list():
    """測試可用模型列表"""
    print("=== 可用模型測試 ===")
    url = f"{OLLAMA_BASE_URL}/api/tags"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"找到 {len(models)} 個模型:")
            for model in models:
                name = model.get('name', 'Unknown')
                size = model.get('size', 'Unknown')
                print(f"  - {name} (大小: {size})")
        else:
            print(f"獲取模型列表失敗: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"請求失敗: {e}")

def test_simple_generation():
    """測試簡單的生成請求"""
    print("\n=== 簡單生成測試 ===")
    
    # 嘗試不同的模型名稱
    model_names = [
        "gpt-oss:20b",
        "gpt-oss",
        "gemma3:12b",
        "gemma3"
    ]
    
    for model_name in model_names:
        print(f"\n測試模型: {model_name}")
        url = f"{OLLAMA_BASE_URL}/api/generate"
        
        payload = {
            "model": model_name,
            "prompt": "Hello, what is your name?",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 50
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            print(f"  狀態碼: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"  回應: {result.get('response', 'No response')[:100]}...")
                print(f"  完成: {result.get('done', False)}")
                break  # 找到可用的模型就停止
            else:
                print(f"  錯誤: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"  請求失敗: {e}")

def main():
    """主測試函數"""
    print("Ollama 連接診斷工具")
    print("=" * 40)
    print(f"目標伺服器: {OLLAMA_HOST}:{OLLAMA_PORT}")
    print()
    
    # 1. 測試網路連接
    if not test_network_connectivity():
        print("\n建議檢查:")
        print("1. Ollama 服務是否正在運行")
        print("2. IP 地址和端口是否正確")
        print("3. 防火牆設置")
        print("4. 網路連接")
        return
    
    # 2. 測試 API
    test_ollama_api()
    
    # 3. 測試模型列表
    test_model_list()
    
    # 4. 測試簡單生成
    test_simple_generation()

if __name__ == "__main__":
    main()