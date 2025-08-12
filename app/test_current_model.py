#!/usr/bin/env python3
"""
測試當前配置的模型是否能正確抽取三元組
專門針對 GPT-OSS 模型進行驗證
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import json
from sentence_triplet_extractor import DeepSeekTripletExtractor
from model_adapter import get_model_adapter
from config import OLLAMA_MODEL, OLLAMA_BASE_URL

# 測試文本
TEST_TEXT = "MIAT 方法論著重在系統的階層式模組化的架構設計，以及演算法的離散事件建模，最後合成可維護、可擴展的系統程式碼。"

def check_model_availability():
    """檢查當前模型是否可用"""
    print(f"🔍 檢查模型可用性: {OLLAMA_MODEL}")
    print(f"🔗 Ollama 服務地址: {OLLAMA_BASE_URL}")
    
    try:
        # 檢查模型列表
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        models = data.get('models', [])
        
        print(f"📋 可用模型列表 ({len(models)} 個):")
        available_models = []
        for model in models:
            name = model.get('name', 'Unknown')
            size = model.get('size', 'Unknown')
            print(f"  - {name} (大小: {size})")
            available_models.append(name)
        
        # 檢查當前配置的模型是否在列表中
        if OLLAMA_MODEL in available_models:
            print(f"✅ 當前模型 {OLLAMA_MODEL} 可用")
            return True
        else:
            print(f"❌ 當前模型 {OLLAMA_MODEL} 不在可用列表中")
            
            # 尋找類似的模型
            similar_models = [m for m in available_models if 'gpt-oss' in m.lower()]
            if similar_models:
                print(f"💡 建議使用類似的模型: {similar_models}")
            
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 無法連接到 Ollama 服務: {e}")
        return False
    except Exception as e:
        print(f"❌ 檢查模型時發生錯誤: {e}")
        return False

def test_simple_generation():
    """測試簡單的文本生成"""
    print(f"\n🧪 測試基本生成功能")
    print("-" * 50)
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "請回答：你是什麼模型？",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 100
        }
    }
    
    try:
        print(f"發送測試請求到 {OLLAMA_MODEL}...")
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '')
        
        print(f"✅ 模型回應: {response_text[:200]}...")
        print(f"📊 回應長度: {len(response_text)} 字符")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 請求失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 處理回應時發生錯誤: {e}")
        return False

def test_triplet_extraction():
    """測試三元組抽取功能"""
    print(f"\n🎯 測試三元組抽取功能")
    print("-" * 50)
    print(f"使用模型: {OLLAMA_MODEL}")
    print(f"測試文本: {TEST_TEXT}")
    
    try:
        # 檢查適配器類型
        adapter = get_model_adapter(OLLAMA_MODEL)
        print(f"📝 適配器類型: {type(adapter).__name__}")
        
        # 創建抽取器
        extractor = DeepSeekTripletExtractor()
        
        # 執行抽取
        print(f"\n開始抽取三元組...")
        triplets = extractor.extract_triplets_from_sentence(TEST_TEXT)
        
        if triplets:
            print(f"\n✅ 成功抽取 {len(triplets)} 個三元組:")
            for i, (subject, predicate, obj) in enumerate(triplets, 1):
                print(f"   {i}. ({subject}) --[{predicate}]--> ({obj})")
            return True
        else:
            print(f"\n❌ 沒有抽取到任何三元組")
            return False
            
    except Exception as e:
        print(f"❌ 三元組抽取失敗: {e}")
        return False

def test_adapter_formats():
    """測試不同適配器的格式"""
    print(f"\n🔧 測試適配器格式")
    print("-" * 50)
    
    # 測試當前模型的適配器
    adapter = get_model_adapter(OLLAMA_MODEL)
    print(f"當前模型適配器: {type(adapter).__name__}")
    
    print(f"\n系統 Prompt 預覽:")
    prompt = adapter.get_system_prompt()
    print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    
    print(f"\nAPI 選項:")
    options = adapter.get_api_options()
    print(json.dumps(options, indent=2))

def main():
    """主測試函數"""
    print("🚀 當前模型測試程式")
    print("=" * 60)
    print(f"目標模型: {OLLAMA_MODEL}")
    print(f"Ollama 服務: {OLLAMA_BASE_URL}")
    print("=" * 60)
    
    # 1. 檢查模型可用性
    if not check_model_availability():
        print("\n❌ 模型不可用，無法繼續測試")
        return False
    
    # 2. 測試基本生成
    if not test_simple_generation():
        print("\n❌ 基本生成測試失敗，無法繼續")
        return False
    
    # 3. 測試適配器格式
    test_adapter_formats()
    
    # 4. 測試三元組抽取
    success = test_triplet_extraction()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 所有測試通過！模型運行正常")
    else:
        print("❌ 三元組抽取測試失敗")
    print(f"{'='*60}")
    
    return success

if __name__ == "__main__":
    main()