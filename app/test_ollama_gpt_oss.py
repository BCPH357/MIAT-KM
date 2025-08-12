#!/usr/bin/env python3
"""
測試 GPT-OSS 模型的三元組抽取功能
用於診斷為什麼無法正確抽取三元組
"""

import requests
import json
import sys

# Ollama 配置 - 在 Docker 容器內使用
OLLAMA_BASE_URL = "http://ollama:11434"  # 使用 Docker Compose 服務名稱
MODEL_NAME = "gpt-oss:20b"

# 測試文本
TEST_TEXT = "MIAT 方法論著重在系統的階層式模組化的架構設計，以及演算法的離散事件建模，最後合成可維護、可擴展的系統程式碼。"

def send_ollama_request(prompt, temperature=0.1, max_tokens=1000):
    """發送請求到 Ollama API"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        print(f"發送請求到: {url}")
        print(f"使用模型: {MODEL_NAME}")
        print(f"Prompt: {prompt[:100]}...")
        print("-" * 50)
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"請求錯誤: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 解析錯誤: {e}")
        return None

def test_basic_response():
    """測試基本回應"""
    print("=== 測試 1: 基本回應測試 ===")
    prompt = "你好，請告訴我你是什麼模型？"
    result = send_ollama_request(prompt)
    
    if result:
        print("回應內容:")
        print(result.get('response', '無回應'))
        print(f"完成狀態: {result.get('done', False)}")
        print()
    
    return result

def test_triplet_extraction_simple():
    """測試簡單的三元組抽取"""
    print("=== 測試 2: 簡單三元組抽取 ===")
    prompt = f"""
請從以下文本中抽取知識三元組，格式為 (主體, 關係, 客體)：

文本: {TEST_TEXT}

請以 JSON 格式回應，例如：
[
    {{"subject": "主體", "predicate": "關係", "object": "客體"}}
]
"""
    
    result = send_ollama_request(prompt)
    
    if result:
        print("回應內容:")
        print(result.get('response', '無回應'))
        print()
    
    return result

def test_triplet_extraction_detailed():
    """測試詳細的三元組抽取（類似原系統的 prompt）"""
    print("=== 測試 3: 詳細三元組抽取 ===")
    prompt = f"""
你是一個專業的知識抽取助手。請從給定的文本中抽取知識三元組。

任務：
1. 仔細閱讀文本
2. 識別出重要的實體（人物、地點、概念、方法等）
3. 識別實體之間的關係
4. 以 (主體, 關係, 客體) 的格式輸出三元組

要求：
- 主體和客體應該是具體的實體或概念
- 關係應該清楚表達兩者之間的聯繫
- 避免過於抽象或模糊的關係
- 確保三元組在語義上是正確的

文本：
{TEST_TEXT}

請以以下JSON格式回應：
[
    {{"subject": "主體名稱", "predicate": "關係描述", "object": "客體名稱"}},
    {{"subject": "主體名稱", "predicate": "關係描述", "object": "客體名稱"}}
]

只回應JSON格式，不要其他說明文字。
"""
    
    result = send_ollama_request(prompt, temperature=0.1, max_tokens=2000)
    
    if result:
        print("回應內容:")
        response_text = result.get('response', '無回應')
        print(response_text)
        
        # 嘗試解析 JSON
        try:
            # 尋找 JSON 部分
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                parsed_json = json.loads(json_str)
                print("\n成功解析的 JSON:")
                print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
            else:
                print("\n無法找到有效的 JSON 格式")
                
        except json.JSONDecodeError as e:
            print(f"\nJSON 解析失敗: {e}")
        
        print()
    
    return result

def test_model_info():
    """測試模型資訊"""
    print("=== 測試 4: 模型資訊 ===")
    url = f"{OLLAMA_BASE_URL}/api/show"
    
    payload = {
        "name": MODEL_NAME
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print("模型資訊:")
        print(f"名稱: {result.get('name', 'N/A')}")
        print(f"大小: {result.get('size', 'N/A')}")
        if 'details' in result:
            details = result['details']
            print(f"參數數量: {details.get('parameter_size', 'N/A')}")
            print(f"量化級別: {details.get('quantization_level', 'N/A')}")
        
        print()
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"獲取模型資訊失敗: {e}")
        return None

def main():
    """主測試函數"""
    print("開始測試 GPT-OSS 模型的三元組抽取功能...")
    print("=" * 60)
    
    # 測試模型資訊
    test_model_info()
    
    # 測試基本回應
    test_basic_response()
    
    # 測試簡單三元組抽取
    test_triplet_extraction_simple()
    
    # 測試詳細三元組抽取
    test_triplet_extraction_detailed()
    
    print("測試完成！")

if __name__ == "__main__":
    main()