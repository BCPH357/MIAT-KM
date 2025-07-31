import os
import re
import csv
from pypdf import PdfReader
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """從 PDF 文件中提取文本"""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def simple_extract_triples(text: str, source: str = "unknown") -> List[Tuple[str, str, str, str]]:
    """
    簡單的基於規則的三元組抽取
    """
    triples = []
    
    # 簡單的中文實體抽取
    entities = []
    
    # 提取中文詞語 (2-6個字符)
    chinese_words = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
    # 過濾停用詞
    stopwords = {'的', '是', '在', '和', '與', '或', '但', '如果', '因為', '所以', '什麼', '怎麼', '為什麼', '這個', '那個', '可以', '能夠', '進行', '實現', '具有', '達到', '一個', '這樣', '那樣', '已經', '正在', '將要'}
    entities = [w for w in set(chinese_words) if w not in stopwords and len(w) >= 2][:20]
    
    # 提取英文詞語
    english_words = re.findall(r'[A-Za-z]{3,}', text)
    entities.extend([w for w in set(english_words) if len(w) > 2][:10])
    
    # 基於模式的關係抽取
    relation_patterns = [
        ("使用", ["使用", "利用", "採用", "運用"]),
        ("包含", ["包含", "包括", "含有"]),
        ("實現", ["實現", "達到", "完成"]),
        ("提升", ["提升", "改善", "增強", "改進"]),
        ("應用", ["應用", "套用", "運用"]),
        ("控制", ["控制", "管理", "操作"]),
        ("分析", ["分析", "研究", "探討"]),
        ("比較", ["比較", "對比", "比對"]),
        ("設計", ["設計", "開發", "建構"]),
        ("測試", ["測試", "驗證", "檢驗"])
    ]
    
    # 為主要實體創建概念關係
    tech_keywords = ["技術", "算法", "方法", "系統", "平台"]
    for entity in entities[:15]:
        # 檢查是否為技術相關
        for tech in tech_keywords:
            if tech in entity or any(tech in text.lower() for tech in ["algorithm", "system", "method"]):
                triples.append((entity, "屬於", "技術領域", source))
                break
        else:
            triples.append((entity, "屬於", "一般概念", source))
    
    # 基於共現的關係抽取
    for i, entity1 in enumerate(entities[:10]):
        for j, entity2 in enumerate(entities[:10]):
            if i != j and entity1 in text and entity2 in text:
                # 檢查是否在同一句子中
                sentences = re.split(r'[。！？.]', text)
                for sentence in sentences:
                    if entity1 in sentence and entity2 in sentence:
                        # 找最合適的關係
                        for relation_name, keywords in relation_patterns:
                            if any(keyword in sentence for keyword in keywords):
                                triples.append((entity1, relation_name, entity2, source))
                                break
                        else:
                            # 預設關係
                            triples.append((entity1, "相關", entity2, source))
                        break
    
    # 去重
    unique_triples = []
    seen = set()
    for triple in triples:
        key = (triple[0], triple[1], triple[2])
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)
    
    return unique_triples[:50]  # 限制數量

def process_pdf_directory(pdf_dir):
    """處理目錄中的所有 PDF 文件"""
    all_triples = []
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"處理 {filename}...")
            
            try:
                # 提取文本
                text = extract_text_from_pdf(pdf_path)
                print(f"提取了 {len(text)} 個字符")
                
                # 使用簡單方法提取三元組
                triples = simple_extract_triples(text, filename)
                all_triples.extend(triples)
                print(f"提取了 {len(triples)} 個三元組")
                
            except Exception as e:
                print(f"處理 {filename} 時發生錯誤: {e}")
                continue
    
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