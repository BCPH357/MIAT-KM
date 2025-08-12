#!/usr/bin/env python3
"""
測試自適應三元組抽取系統
驗證不同模型使用不同的 prompt 和處理邏輯
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_triplet_extractor import DeepSeekTripletExtractor
from model_adapter import get_model_adapter
from config import OLLAMA_BASE_URL

# 測試文本
TEST_SENTENCES = [
    "MIAT 方法論著重在系統的階層式模組化的架構設計，以及演算法的離散事件建模，最後合成可維護、可擴展的系統程式碼。",
    "Python 是一種高級程式語言，廣泛應用於數據科學和機器學習領域。",
    "Neo4j 是一個圖形數據庫，專門用於存儲和查詢知識圖譜。"
]

def test_model_adapter(model_name):
    """測試指定模型的適配器"""
    print(f"\n{'='*80}")
    print(f"🧪 測試模型: {model_name}")
    print(f"{'='*80}")
    
    # 獲取適配器
    adapter = get_model_adapter(model_name)
    print(f"📝 適配器類型: {type(adapter).__name__}")
    print(f"📝 系統 Prompt 預覽: {adapter.get_system_prompt()[:100]}...")
    print(f"📝 API 選項: {adapter.get_api_options()}")
    print()
    
    # 創建抽取器
    extractor = DeepSeekTripletExtractor(
        base_url=OLLAMA_BASE_URL,
        model=model_name
    )
    
    # 測試每個句子
    total_triplets = 0
    for i, sentence in enumerate(TEST_SENTENCES, 1):
        print(f"\n🔍 測試句子 {i}: {sentence}")
        print("-" * 60)
        
        try:
            triplets = extractor.extract_triplets_from_sentence(sentence)
            total_triplets += len(triplets)
            
            if triplets:
                print(f"✅ 成功抽取 {len(triplets)} 個三元組:")
                for j, (subject, predicate, obj) in enumerate(triplets, 1):
                    print(f"   {j}. ({subject}) --[{predicate}]--> ({obj})")
            else:
                print("❌ 未抽取到任何三元組")
                
        except Exception as e:
            print(f"❌ 抽取失敗: {e}")
        
        print()
    
    print(f"📊 模型 {model_name} 總計抽取: {total_triplets} 個三元組")
    return total_triplets

def test_different_models():
    """測試不同模型的表現"""
    print("🚀 開始測試自適應三元組抽取系統")
    print("📋 測試模型列表:")
    
    # 測試模型列表
    test_models = [
        "gpt-oss:20b",    # GPT-OSS 模型（使用 JSON 格式）
        "gemma3:12b",     # Gemma 模型（使用原有格式）
        "unknown-model"   # 未知模型（應該使用 Gemma 適配器作為默認）
    ]
    
    results = {}
    
    for model in test_models:
        try:
            triplet_count = test_model_adapter(model)
            results[model] = triplet_count
        except Exception as e:
            print(f"❌ 測試模型 {model} 時發生錯誤: {e}")
            results[model] = 0
    
    # 總結結果
    print(f"\n{'='*80}")
    print("📊 測試結果總結")
    print(f"{'='*80}")
    
    for model, count in results.items():
        print(f"🔹 {model}: {count} 個三元組")
    
    print(f"\n✅ 自適應系統測試完成！")

def test_adapter_selection():
    """測試適配器選擇邏輯"""
    print("\n🔧 測試適配器選擇邏輯")
    print("-" * 40)
    
    test_cases = [
        ("gpt-oss:20b", "GPTOSSAdapter"),
        ("gpt-oss", "GPTOSSAdapter"),
        ("GPT-OSS:latest", "GPTOSSAdapter"),
        ("gemma3:12b", "GemmaAdapter"),
        ("gemma3", "GemmaAdapter"),
        ("GEMMA3:latest", "GemmaAdapter"),
        ("unknown-model", "GemmaAdapter"),
        ("", "GemmaAdapter"),
        (None, "GemmaAdapter")
    ]
    
    for model_name, expected_adapter in test_cases:
        adapter = get_model_adapter(model_name)
        actual_adapter = type(adapter).__name__
        
        status = "✅" if actual_adapter == expected_adapter else "❌"
        print(f"{status} 模型: {model_name or 'None'} -> {actual_adapter} (期望: {expected_adapter})")

if __name__ == "__main__":
    print("🔬 自適應三元組抽取測試程式")
    print("=" * 80)
    
    # 1. 測試適配器選擇
    test_adapter_selection()
    
    # 2. 測試不同模型（需要實際的 Ollama 服務）
    try:
        test_different_models()
    except Exception as e:
        print(f"\n⚠️ 無法連接到 Ollama 服務進行實際測試: {e}")
        print("💡 請確保 Ollama 服務正在運行，然後重新執行測試")