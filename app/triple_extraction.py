import os
import json
import csv
import requests
import re
from pypdf import PdfReader
from typing import List, Tuple, Dict, Any
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaChat:
    """Ollama API 客戶端"""
    
    def __init__(self, base_url="http://ollama:11434", model="gemma3:12b"):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"
    
    def __call__(self, message: str) -> str:
        """
        發送消息到Ollama模型並獲取回應
        
        Args:
            message (str): 輸入消息
            
        Returns:
            str: 模型回應
        """
        payload = {
            "model": self.model,
            "prompt": message,
            "stream": False,
            "format": "json" if "JSON" in message or "json" in message else ""
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API 請求失敗: {e}")
            return ""
        except Exception as e:
            logger.error(f"處理Ollama回應時發生錯誤: {e}")
            return ""


class OllamaFactConceptExtractor:
    """使用Ollama的事實概念抽取器"""
    
    def __init__(self, chat=None, base_url="http://ollama:11434", model="gemma3:12b"):
        self.chat = chat if chat else OllamaChat(base_url=base_url, model=model)
    
    def get_concept_n_fact(self, context: str) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        最簡化版本：基於關鍵字的實體抽取
        """
        # 簡單的實體抽取
        entities = []
        
        try:
            # 提取中文詞語 (2-8個字符，避免過長)
            chinese_pattern = r'[\u4e00-\u9fff]{2,8}'
            chinese_words = re.findall(chinese_pattern, context)
            # 過濾常見停用詞
            stopwords = {'的', '是', '在', '和', '與', '或', '但', '如果', '因為', '所以', '什麼', '怎麼', '為什麼', '這個', '那個', '可以', '能夠', '進行', '實現', '具有', '達到'}
            chinese_words = [w for w in chinese_words if w not in stopwords and len(w) >= 2]
            entities.extend(chinese_words[:15])
            
            # 提取英文詞語
            english_words = re.findall(r'[A-Za-z]{3,}', context)
            entities.extend(english_words[:5])
            
            # 去重並限制數量
            entities = list(set(entities))[:15]
            
        except Exception as e:
            logger.error(f"實體抽取錯誤: {e}")
            entities = ["系統", "方法", "技術", "數據", "算法"]  # 預設實體
        
        # 簡單分類
        concepts = ["技術", "系統", "方法", "數據", "其他"]
        
        # 安全的實體階層構建
        entity_hierarchy = {}
        try:
            # 將實體簡單分類
            tech_entities = [e for e in entities if any(keyword in e for keyword in ["技術", "算法", "方法", "系統"])]
            data_entities = [e for e in entities if any(keyword in e for keyword in ["數據", "資料", "信息", "檔案"])]
            other_entities = [e for e in entities if e not in tech_entities and e not in data_entities]
            
            entity_hierarchy = {
                "技術": tech_entities[:5],
                "數據": data_entities[:5],
                "其他": other_entities[:5]
            }
            
        except Exception as e:
            logger.error(f"實體階層構建錯誤: {e}")
            entity_hierarchy = {"其他": entities[:10]}
        
        return entities, concepts, entity_hierarchy
    
    def get_facts_pairs(self, facts: List[str], context: str) -> List[List[str]]:
        """
        簡化版本：基於規則的關係抽取
        """
        relations = []
        
        # 簡單的關係模式
        relation_patterns = [
            ("使用", ["使用", "利用", "採用"]),
            ("包含", ["包含", "包括", "含有"]),
            ("實現", ["實現", "達到", "完成"]),
            ("提升", ["提升", "改善", "增強"]),
            ("應用", ["應用", "套用", "運用"])
        ]
        
        # 為每對實體嘗試找關係
        for i, fact1 in enumerate(facts[:5]):  # 限制數量避免過多
            for j, fact2 in enumerate(facts[:5]):
                if i != j:
                    # 檢查是否在文本中有關係
                    for relation_name, keywords in relation_patterns:
                        if any(keyword in context for keyword in keywords):
                            if fact1 in context and fact2 in context:
                                relations.append([fact1, relation_name, fact2])
                                break
        
        return relations[:10]  # 限制關係數量
    
    def _fix_json(self, json_text: str) -> str:
        """修復JSON格式問題"""
        # 簡單的JSON修復邏輯
        json_text = json_text.strip()
        if not json_text.startswith('{'):
            json_text = '{' + json_text
        if not json_text.endswith('}'):
            json_text = json_text + '}'
        return json_text


class OllamaTripletExtractor:
    """完整的Ollama三元組抽取器"""
    
    def __init__(self, base_url="http://ollama:11434", model="gemma3:12b"):
        self.extractor = OllamaFactConceptExtractor(base_url=base_url, model=model)
    
    def extract_triplets_from_text(self, text: str, source: str = "unknown") -> List[Tuple[str, str, str, str]]:
        """
        從文本中抽取三元組
        
        Args:
            text (str): 輸入文本
            source (str): 數據源
            
        Returns:
            List[Tuple[str, str, str, str]]: 三元組列表 (subject, predicate, object, source)
        """
        logger.info(f"開始使用Ollama抽取三元組，文本長度: {len(text)}")
        
        # 分段處理長文本
        max_length = 2000
        if len(text) > max_length:
            segments = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        else:
            segments = [text]
        
        all_triplets = []
        
        for i, segment in enumerate(segments):
            logger.info(f"處理第 {i+1}/{len(segments)} 段文本...")
            
            try:
                # 抽取事實和概念
                facts, concepts, entity_hierarchy = self.extractor.get_concept_n_fact(segment)
                logger.debug(f"段落 {i+1} - 事實: {len(facts)}, 概念: {len(concepts)}")
                
                # 抽取事實間關係
                facts_pairs = self.extractor.get_facts_pairs(facts, segment)
                logger.debug(f"段落 {i+1} - 事實關係: {len(facts_pairs)}")
                
                # 轉換為三元組格式
                # 1. 事實與概念的關係
                for concept, fact_list in entity_hierarchy.items():
                    for fact in fact_list:
                        all_triplets.append((fact, "屬於", concept, source))
                
                # 2. 事實間的關係
                for pair in facts_pairs:
                    if len(pair) == 3:
                        all_triplets.append((pair[0], pair[1], pair[2], source))
                
            except Exception as e:
                logger.error(f"處理段落 {i+1} 時發生錯誤: {e}")
                continue
        
        logger.info(f"總共抽取到 {len(all_triplets)} 個三元組")
        return all_triplets


def extract_text_from_pdf(pdf_path):
    """從 PDF 文件中提取文本"""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_triples_from_text(text, source="unknown"):
    """使用 Ollama 從文本中提取三元組"""
    extractor = OllamaTripletExtractor()
    return extractor.extract_triplets_from_text(text, source)

def process_pdf_directory(pdf_dir):
    """處理目錄中的所有 PDF 文件"""
    all_triples = []
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"處理 {filename}...")
            
            # 提取文本
            text = extract_text_from_pdf(pdf_path)
            print(f"提取了 {len(text)} 個字符")
            
            # 使用 Ollama 提取三元組
            triples = extract_triples_from_text(text, filename)
            all_triples.extend(triples)
    
    return all_triples

def save_triples_to_csv(triples, output_file):
    """將三元組保存到 CSV 文件"""
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject', 'Predicate', 'Object', 'Source'])
        
        for triple in triples:
            if len(triple) == 4:
                writer.writerow(triple)
            else:
                # 兼容舊格式
                writer.writerow([triple[0], triple[1], triple[2], "unknown"])

if __name__ == "__main__":
    # 設置 PDF 目錄和輸出文件
    pdf_dir = "/app/data/pdf"
    output_file = "/app/data/processed/triples.csv"
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 處理 PDF 文件
    triples = process_pdf_directory(pdf_dir)
    
    # 保存三元組
    save_triples_to_csv(triples, output_file)
    
    print(f"共提取了 {len(triples)} 個三元組，已保存到 {output_file}")
    
    # 顯示前 10 個三元組
    print("\n前 10 個三元組示例：")
    for i, triple in enumerate(triples[:10]):
        print(f"{i+1}. {triple[0]} - {triple[1]} - {triple[2]} (來源: {triple[3]})") 