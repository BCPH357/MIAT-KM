#!/usr/bin/env python3
"""
模型自適應系統
根據不同的 LLM 模型提供不同的 prompt 和處理邏輯
"""

import re
import json
import logging
from typing import List, Tuple, Dict, Any
from config import OLLAMA_MODEL

logger = logging.getLogger(__name__)

class ModelAdapter:
    """模型適配器基類"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def get_system_prompt(self) -> str:
        """獲取系統 prompt"""
        raise NotImplementedError
        
    def get_api_options(self) -> Dict[str, Any]:
        """獲取 API 請求選項"""
        return {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 300
        }
        
    def parse_response(self, response: str) -> List[Tuple[str, str, str]]:
        """解析模型回應"""
        raise NotImplementedError

class GemmaAdapter(ModelAdapter):
    """Gemma 模型適配器（原有邏輯）"""
    
    def get_system_prompt(self) -> str:
        return """從句子中抽取三元組，格式：<三元組>主語|謂語|賓語</三元組>

規則：
1. 只抽取句子中明確存在的關係
2. 主語和賓語必須是具體實體或概念
3. 謂語是動詞或關係詞
4. 每個三元組用 <三元組></三元組> 包圍
5. 如果沒有明確關係，輸出：<三元組>無</三元組>

例子：
句子：張三使用Python開發網站
輸出：
<三元組>張三|使用|Python</三元組>
<三元組>張三|開發|網站</三元組>

句子：GRAFCET是一種控制系統設計方法
輸出：
<三元組>GRAFCET|是|控制系統設計方法</三元組>

現在處理："""

    def get_api_options(self) -> Dict[str, Any]:
        return {
            "temperature": 0.1,  # Gemma3 對溫度更敏感
            "top_p": 0.9,
            "num_predict": 300
        }

    def parse_response(self, response: str) -> List[Tuple[str, str, str]]:
        """解析 Gemma 的回應（原有邏輯）"""
        triplets = []
        
        try:
            print("🔧 開始解析三元組（Gemma 格式）...")
            print(f"原始回應: 【{response}】")
            
            # Gemma3 不需要移除 <think> 標籤，直接處理
            cleaned_response = response.strip()
            
            # 主要格式：<三元組>主語|謂語|賓語</三元組>
            pattern = r'<三元組>(.*?)</三元組>'
            matches = re.findall(pattern, cleaned_response, re.DOTALL)
            print(f"<三元組></三元組> 格式匹配: {matches}")
            
            for match in matches:
                content = match.strip()
                print(f"處理匹配項: 【{content}】")
                
                # 跳過"無"或空內容
                if content == "無" or not content:
                    print(f"跳過空/無內容: {content}")
                    continue
                
                # 解析 主語|謂語|賓語 格式
                if '|' in content and content.count('|') == 2:
                    parts = content.split('|')
                    if len(parts) == 3:
                        subject = parts[0].strip()
                        predicate = parts[1].strip()
                        obj = parts[2].strip()
                        
                        print(f"分解三元組: 主語=【{subject}】, 謂語=【{predicate}】, 賓語=【{obj}】")
                        
                        # 基本有效性檢查
                        if (subject and predicate and obj and 
                            len(subject) > 0 and len(predicate) > 0 and len(obj) > 0 and
                            len(subject) <= 50 and len(obj) <= 100):  # 長度限制
                            triplets.append((subject, predicate, obj))
                            print(f"✅ 添加有效三元組: ({subject}, {predicate}, {obj})")
                        else:
                            print(f"❌ 跳過無效三元組: 空內容或過長")
                else:
                    print(f"❌ 格式不正確，跳過: {content}")
            
        except Exception as e:
            print(f"❌ 解析過程中發生錯誤: {e}")
            logger.error(f"解析三元組回應時發生錯誤: {e}")
            logger.debug(f"原始回應: {response}")
        
        # 去重
        unique_triplets = []
        seen = set()
        for triplet in triplets:
            if triplet not in seen:
                seen.add(triplet)
                unique_triplets.append(triplet)
        
        print(f"🎯 最終結果（Gemma）: {len(unique_triplets)} 個唯一三元組")
        return unique_triplets

class GPTOSSAdapter(ModelAdapter):
    """GPT-OSS 模型適配器"""
    
    def get_system_prompt(self) -> str:
        return """你是一個專業的知識抽取助手。請從給定的文本中抽取知識三元組。

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
    {"subject": "主體名稱", "predicate": "關係描述", "object": "客體名稱"},
    {"subject": "主體名稱", "predicate": "關係描述", "object": "客體名稱"}
]

只回應JSON格式，不要其他說明文字。

現在處理文本："""

    def get_api_options(self) -> Dict[str, Any]:
        return {
            "temperature": 0.1,  # GPT-OSS 表現良好的溫度
            "top_p": 0.9,
            "num_predict": 2000  # 增加輸出長度避免 JSON 被截斷
        }

    def parse_response(self, response: str) -> List[Tuple[str, str, str]]:
        """解析 GPT-OSS 的 JSON 回應"""
        triplets = []
        
        try:
            print("🔧 開始解析三元組（GPT-OSS JSON 格式）...")
            print(f"原始回應: 【{response}】")
            print(f"回應長度: {len(response)} 字符")
            
            # 清理回應
            cleaned_response = response.strip()
            
            # 嘗試找到 JSON 部分
            start = cleaned_response.find('[')
            end = cleaned_response.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_str = cleaned_response[start:end]
                print(f"提取的 JSON 字符串: 【{json_str}】")
                print(f"JSON 字符串長度: {len(json_str)} 字符")
                
                # 檢查 JSON 是否完整
                if not json_str.endswith(']'):
                    print("⚠️ 檢測到 JSON 可能被截斷，嘗試修復...")
                    json_str = self._repair_truncated_json(json_str)
                
                try:
                    parsed_json = json.loads(json_str)
                    print(f"✅ 成功解析 JSON，包含 {len(parsed_json)} 個項目")
                    
                    for i, item in enumerate(parsed_json):
                        if isinstance(item, dict):
                            subject = item.get('subject', '').strip()
                            predicate = item.get('predicate', '').strip()
                            obj = item.get('object', '').strip()
                            
                            print(f"項目 {i+1}: 主語=【{subject}】, 謂語=【{predicate}】, 賓語=【{obj}】")
                            
                            # 基本有效性檢查
                            if (subject and predicate and obj and 
                                len(subject) > 0 and len(predicate) > 0 and len(obj) > 0 and
                                len(subject) <= 50 and len(obj) <= 100):  # 長度限制
                                triplets.append((subject, predicate, obj))
                                print(f"✅ 添加有效三元組: ({subject}, {predicate}, {obj})")
                            else:
                                print(f"❌ 跳過無效三元組: 空內容或過長")
                        else:
                            print(f"❌ 非字典項目，跳過: {item}")
                            
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 解析失敗: {e}")
                    print(f"❌ 失敗的 JSON 字符串: {json_str}")
                    # 嘗試修復常見的 JSON 錯誤
                    return self._try_repair_json(json_str)
                    
            else:
                print("❌ 無法找到有效的 JSON 格式")
                print(f"❌ 尋找 '[' 位置: {start}, 尋找 ']' 位置: {end-1}")
                # 嘗試解析其他可能的格式
                return self._parse_alternative_format(cleaned_response)
                
        except Exception as e:
            print(f"❌ 解析過程中發生錯誤: {e}")
            logger.error(f"解析 GPT-OSS 回應時發生錯誤: {e}")
            logger.debug(f"原始回應: {response}")
        
        # 去重
        unique_triplets = []
        seen = set()
        for triplet in triplets:
            if triplet not in seen:
                seen.add(triplet)
                unique_triplets.append(triplet)
        
        print(f"🎯 最終結果（GPT-OSS）: {len(unique_triplets)} 個唯一三元組")
        return unique_triplets
    
    def _repair_truncated_json(self, json_str: str) -> str:
        """修復被截斷的 JSON"""
        print("🔧 嘗試修復被截斷的 JSON...")
        
        # 移除末尾不完整的項目
        lines = json_str.split('\n')
        repaired_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # 如果行包含完整的鍵值對，保留它
                if ('"subject"' in line and '"predicate"' in line and '"object"' in line and 
                    line.count('"') >= 6):  # 至少6個引號表示完整項目
                    repaired_lines.append(line)
                elif line in ['{', '}', '[', ']', ',']:
                    repaired_lines.append(line)
                # 跳過不完整的行
        
        # 重新組織 JSON
        if repaired_lines:
            # 確保以 [ 開始
            if repaired_lines[0] != '[':
                repaired_lines.insert(0, '[')
            
            # 確保以 ] 結束
            if repaired_lines[-1] != ']':
                # 移除最後的逗號
                if repaired_lines[-1] == ',':
                    repaired_lines.pop()
                repaired_lines.append(']')
            
            repaired = '\n'.join(repaired_lines)
            print(f"修復後的 JSON: {repaired}")
            return repaired
        
        return json_str
    
    def _try_repair_json(self, json_str: str) -> List[Tuple[str, str, str]]:
        """嘗試修復損壞的 JSON"""
        print("🔧 嘗試修復 JSON...")
        
        # 常見修復：添加缺失的引號
        repaired = json_str
        
        # 修復未加引號的鍵
        repaired = re.sub(r'(\w+):', r'"\1":', repaired)
        
        # 修復未加引號的值（但要保留已有引號的）
        repaired = re.sub(r':\s*([^",\]\}]+)', r': "\1"', repaired)
        
        try:
            parsed = json.loads(repaired)
            print(f"✅ JSON 修復成功")
            return self._extract_triplets_from_json(parsed)
        except json.JSONDecodeError:
            print("❌ JSON 修復失敗")
            return []
    
    def _parse_alternative_format(self, response: str) -> List[Tuple[str, str, str]]:
        """解析替代格式"""
        print("🔧 嘗試解析替代格式...")
        
        triplets = []
        
        # 嘗試尋找類似 "主體" "關係" "客體" 的模式
        patterns = [
            r'[""]([^""]+)[""],?\s*[""]([^""]+)[""],?\s*[""]([^""]+)[""]',
            r'主體[：:]\s*([^，,]+)[，,]\s*關係[：:]\s*([^，,]+)[，,]\s*客體[：:]\s*([^，,\n]+)',
            r'(\w+)\s*[-－→]\s*(\w+)\s*[-－→]\s*(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                print(f"找到 {len(matches)} 個匹配項")
                for match in matches:
                    if len(match) == 3:
                        subject, predicate, obj = [s.strip() for s in match]
                        if subject and predicate and obj:
                            triplets.append((subject, predicate, obj))
                            print(f"✅ 添加三元組: ({subject}, {predicate}, {obj})")
                break
        
        return triplets
    
    def _extract_triplets_from_json(self, parsed_json: List[Dict]) -> List[Tuple[str, str, str]]:
        """從解析後的 JSON 中提取三元組"""
        triplets = []
        
        for item in parsed_json:
            if isinstance(item, dict):
                subject = item.get('subject', '').strip()
                predicate = item.get('predicate', '').strip()
                obj = item.get('object', '').strip()
                
                if subject and predicate and obj:
                    triplets.append((subject, predicate, obj))
        
        return triplets

def get_model_adapter(model_name: str = None) -> ModelAdapter:
    """根據模型名稱獲取對應的適配器"""
    if model_name is None:
        model_name = OLLAMA_MODEL
    
    model_name_lower = model_name.lower()
    
    if 'gpt-oss' in model_name_lower:
        print(f"🤖 使用 GPT-OSS 適配器 (模型: {model_name})")
        return GPTOSSAdapter(model_name)
    elif 'gemma' in model_name_lower:
        print(f"🤖 使用 Gemma 適配器 (模型: {model_name})")
        return GemmaAdapter(model_name)
    else:
        # 默認使用 Gemma 適配器
        print(f"⚠️ 未知模型 {model_name}，使用 Gemma 適配器")
        return GemmaAdapter(model_name)

if __name__ == "__main__":
    # 測試適配器
    print("測試模型適配器...")
    
    # 測試 GPT-OSS
    gpt_adapter = get_model_adapter("gpt-oss:20b")
    print("GPT-OSS Prompt:")
    print(gpt_adapter.get_system_prompt())
    print()
    
    # 測試 Gemma
    gemma_adapter = get_model_adapter("gemma3:12b")
    print("Gemma Prompt:")
    print(gemma_adapter.get_system_prompt())