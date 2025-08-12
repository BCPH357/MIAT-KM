#!/usr/bin/env python3
"""
快速檢查 GPT-OSS 模型可用性和三元組抽取功能
"""

import requests
import json
from config import OLLAMA_BASE_URL

def check_gpt_oss_available():
    """檢查 GPT-OSS 模型是否可用"""
    print("🔍 檢查 GPT-OSS 模型可用性...")
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        models = data.get('models', [])
        
        gpt_oss_models = [m for m in models if 'gpt-oss' in m.get('name', '').lower()]
        
        if gpt_oss_models:
            print(f"✅ 找到 {len(gpt_oss_models)} 個 GPT-OSS 模型:")
            for model in gpt_oss_models:
                name = model.get('name', 'Unknown')
                size = model.get('size', 'Unknown')
                print(f"   - {name} (大小: {size})")
            return gpt_oss_models[0]['name']  # 返回第一個找到的模型
        else:
            print("❌ 沒有找到 GPT-OSS 模型")
            print("📋 可用模型列表:")
            for model in models:
                name = model.get('name', 'Unknown')
                print(f"   - {name}")
            return None
            
    except Exception as e:
        print(f"❌ 檢查失敗: {e}")
        return None

def test_gpt_oss_triplet_extraction(model_name):
    """測試 GPT-OSS 的三元組抽取"""
    print(f"\n🧪 測試 {model_name} 的三元組抽取...")
    
    test_text = "MIAT 方法論著重在系統的階層式模組化的架構設計，以及演算法的離散事件建模，最後合成可維護、可擴展的系統程式碼。"
    
    prompt = f"""你是一個專業的知識抽取助手。請從給定的文本中抽取知識三元組。

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

請以以下JSON格式回應：
[
    {{"subject": "主體名稱", "predicate": "關係描述", "object": "客體名稱"}},
    {{"subject": "主體名稱", "predicate": "關係描述", "object": "客體名稱"}}
]

只回應JSON格式，不要其他說明文字。

現在處理文本：{test_text}"""

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 1000
        }
    }
    
    try:
        print(f"📤 發送測試請求...")
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '').strip()
        
        print(f"📥 模型回應:")
        print(f"回應內容: {response_text}")
        print(f"回應長度: {len(response_text)} 字符")
        
        # 嘗試解析 JSON
        try:
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                parsed_json = json.loads(json_str)
                
                print(f"\n✅ 成功解析 JSON，包含 {len(parsed_json)} 個三元組:")
                for i, item in enumerate(parsed_json, 1):
                    if isinstance(item, dict):
                        subject = item.get('subject', '')
                        predicate = item.get('predicate', '')
                        obj = item.get('object', '')
                        print(f"   {i}. ({subject}) --[{predicate}]--> ({obj})")
                
                return True
            else:
                print("❌ 無法找到有效的 JSON 格式")
                return False
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失敗: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def main():
    print("🚀 GPT-OSS 模型快速檢查工具")
    print("=" * 50)
    
    # 1. 檢查模型可用性
    model_name = check_gpt_oss_available()
    
    if not model_name:
        print("\n💡 建議解決方案:")
        print("1. 確保已下載 GPT-OSS 模型:")
        print("   sudo docker-compose exec ollama ollama pull gpt-oss:20b")
        print("2. 檢查 Ollama 服務是否正常運行:")
        print("   sudo docker-compose ps")
        return False
    
    # 2. 測試三元組抽取
    success = test_gpt_oss_triplet_extraction(model_name)
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 GPT-OSS 模型運行正常，可以正確抽取三元組！")
        print("現在您可以使用以下命令進行完整的三元組抽取:")
        print("sudo docker-compose exec app python sentence_triplet_extractor.py")
    else:
        print("❌ GPT-OSS 模型無法正確抽取三元組")
        print("請檢查模型是否正確安裝或考慮使用其他模型")
    print(f"{'='*50}")
    
    return success

if __name__ == "__main__":
    main()