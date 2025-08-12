#!/usr/bin/env python3
"""
調試配置問題 - 檢查實際使用的模型
"""

print("🔍 檢查配置文件...")

# 導入配置
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
print(f"📄 config.py 中的 OLLAMA_MODEL: {OLLAMA_MODEL}")
print(f"📄 config.py 中的 OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")

# 檢查 sentence_triplet_extractor
print("\n🔍 檢查 sentence_triplet_extractor...")
from sentence_triplet_extractor import DeepSeekTripletExtractor
extractor = DeepSeekTripletExtractor()
print(f"📄 DeepSeekTripletExtractor 實際使用的模型: {extractor.model}")
print(f"📄 DeepSeekTripletExtractor 適配器類型: {type(extractor.adapter).__name__}")

# 檢查適配器選擇
print("\n🔍 檢查適配器選擇邏輯...")
from model_adapter import get_model_adapter
adapter = get_model_adapter(OLLAMA_MODEL)
print(f"📄 根據 config.py 選擇的適配器: {type(adapter).__name__}")

adapter2 = get_model_adapter()  # 使用默認參數
print(f"📄 使用默認參數選擇的適配器: {type(adapter2).__name__}")

# 檢查環境變數
import os
print("\n🔍 檢查環境變數...")
env_model = os.environ.get('OLLAMA_MODEL', 'Not set')
print(f"📄 環境變數 OLLAMA_MODEL: {env_model}")

print("\n🎯 總結:")
print(f"✅ Config 檔案模型: {OLLAMA_MODEL}")
print(f"✅ 實際使用模型: {extractor.model}")
print(f"✅ 選擇的適配器: {type(adapter).__name__}")

if OLLAMA_MODEL != extractor.model:
    print("❌ 發現不一致！配置文件與實際使用的模型不同")
else:
    print("✅ 配置一致")