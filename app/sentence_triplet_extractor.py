import os
import re
import csv
import requests
import json
from pypdf import PdfReader
from typing import List, Tuple
import logging
import time
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, PDF_DIR, MARKDOWN_DIR, PROCESSED_DIR

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepSeekTripletExtractor:
    """基於句子的三元組抽取器，支持 PDF 和 Markdown 文件"""
    
    def __init__(self, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"
        
        # 系統 prompt
        self.system_prompt = """從句子中抽取三元組，格式：<三元組>主語|謂語|賓語</三元組>

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

    def split_text_into_sentences(self, text: str) -> List[str]:
        """
        將文本分割成句子
        """
        # 使用多種標點符號進行分割
        sentence_delimiters = r'[。.!！？?；;]'
        sentences = re.split(sentence_delimiters, text)
        
        # 清理句子：去除空白、過短句子
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # 過濾太短的句子（少於5個字符）和包含過多符號的句子
            if len(sentence) >= 5 and len(re.findall(r'[\u4e00-\u9fff]', sentence)) >= 3:
                cleaned_sentences.append(sentence)
        
        logger.info(f"文本分割成 {len(cleaned_sentences)} 個有效句子")
        return cleaned_sentences

    def extract_triplets_from_sentence(self, sentence: str) -> List[Tuple[str, str, str]]:
        """
        從單個句子中抽取三元組
        """
        prompt = f"{self.system_prompt}\n\n句子：\"{sentence}\"\n\n輸出："
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # 更低溫度，Gemma3 對溫度更敏感
                "top_p": 0.9,
                "num_predict": 300   # 增加輸出長度，支援多個三元組
            }
        }
        
        # 完整的對話日誌記錄
        print("=" * 100)
        print(f"🔤 正在處理句子: 【{sentence}】")
        print(f"📏 句子長度: {len(sentence)} 字符")
        print("-" * 50)
        print("📤 完整 Prompt 內容:")
        print(prompt)
        print("-" * 50)
        print("📦 API 請求 Payload:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("-" * 50)
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            raw_response = result.get('response', '').strip()
            
            # 詳細記錄 LLM 回應
            print("📥 LLM 完整原始回應:")
            print(f"回應內容: 【{raw_response}】")
            print(f"回應長度: {len(raw_response)} 字符")
            print(f"是否包含 <think>: {'是' if '<think>' in raw_response else '否'}")
            print(f"是否包含 |: {'是' if '|' in raw_response else '否'}")
            print(f"是否包含 <: {'是' if '<' in raw_response else '否'}")
            print(f"是否包含 >: {'是' if '>' in raw_response else '否'}")
            print("-" * 50)
            
            # 解析三元組
            triplets = self.parse_triplets_response(raw_response)
            
            # 詳細記錄解析過程
            print("🔍 三元組解析結果:")
            if triplets:
                print(f"✅ 成功抽取到 {len(triplets)} 個三元組:")
                for i, (subject, predicate, obj) in enumerate(triplets, 1):
                    print(f"   {i}. 主語: 【{subject}】, 謂語: 【{predicate}】, 賓語: 【{obj}】")
            else:
                print("❌ 沒有抽取到任何有效三元組")
                
            print("=" * 100)
            print()
            
            if triplets:
                logger.debug(f"句子 '{sentence[:30]}...' 抽取到 {len(triplets)} 個三元組")
            
            return triplets
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API 請求失敗: {e}")
            logger.error(f"API 請求失敗: {e}")
            print("=" * 100)
            print()
            return []
        except Exception as e:
            print(f"❌ 處理句子時發生錯誤: {e}")
            logger.error(f"處理句子時發生錯誤: {e}")
            print("=" * 100)
            print()
            return []

    def parse_triplets_response(self, response: str) -> List[Tuple[str, str, str]]:
        """
        解析 Gemma3 的回應，提取三元組（針對 Gemma3 優化）
        """
        triplets = []
        
        try:
            print("🔧 開始解析三元組...")
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
        
        print(f"🎯 最終結果: {len(unique_triplets)} 個唯一三元組")
        return unique_triplets

    def extract_triplets_from_text(self, text: str, source: str = "unknown") -> List[Tuple[str, str, str, str]]:
        """
        從文本中抽取三元組
        """
        logger.info(f"開始處理文本，長度: {len(text)} 字符")
        
        # 1. 分割句子
        sentences = self.split_text_into_sentences(text)
        
        all_triplets = []
        processed_count = 0
        
        # 2. 對每個句子進行三元組抽取
        for i, sentence in enumerate(sentences):
            logger.info(f"處理句子 {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            try:
                # 抽取三元組
                sentence_triplets = self.extract_triplets_from_sentence(sentence)
                
                # 添加源信息
                for triplet in sentence_triplets:
                    all_triplets.append((triplet[0], triplet[1], triplet[2], source))
                
                processed_count += 1
                
                # 添加延遲避免API過載
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"處理句子 {i+1} 時發生錯誤: {e}")
                continue
        
        logger.info(f"完成處理，共從 {processed_count} 個句子中抽取到 {len(all_triplets)} 個三元組")
        return all_triplets

def extract_text_from_pdf(pdf_path):
    """從 PDF 文件中提取文本"""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_markdown(md_path):
    """從 Markdown 文件中提取文本"""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 移除 Markdown 語法
    # 移除代碼塊
    content = re.sub(r'```[\s\S]*?```', '', content)
    content = re.sub(r'`[^`]*`', '', content)
    
    # 移除標題符號
    content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
    
    # 移除鏈接語法
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    
    # 移除圖片語法
    content = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', content)
    
    # 移除粗體和斜體標記
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    content = re.sub(r'\*([^*]+)\*', r'\1', content)
    content = re.sub(r'__([^_]+)__', r'\1', content)
    content = re.sub(r'_([^_]+)_', r'\1', content)
    
    # 移除列表符號
    content = re.sub(r'^[\s]*[-*+]\s+', '', content, flags=re.MULTILINE)
    content = re.sub(r'^[\s]*\d+\.\s+', '', content, flags=re.MULTILINE)
    
    # 移除水平線
    content = re.sub(r'^[-*_]{3,}$', '', content, flags=re.MULTILINE)
    
    # 清理多餘的空白行
    content = re.sub(r'\n\s*\n', '\n\n', content)
    
    return content.strip()

def process_files_directory(input_dir, file_extensions=['.pdf', '.md']):
    """處理目錄中的所有支持文件（PDF 和 Markdown）"""
    all_triplets = []
    extractor = DeepSeekTripletExtractor()
    
    if not os.path.exists(input_dir):
        logger.warning(f"目錄不存在: {input_dir}")
        return all_triplets
    
    for filename in os.listdir(input_dir):
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension in file_extensions:
            file_path = os.path.join(input_dir, filename)
            logger.info(f"開始處理文件: {filename}")
            
            try:
                # 根據文件類型提取文本
                if file_extension == '.pdf':
                    text = extract_text_from_pdf(file_path)
                elif file_extension == '.md':
                    text = extract_text_from_markdown(file_path)
                else:
                    continue
                    
                logger.info(f"從 {filename} 提取了 {len(text)} 個字符")
                
                # 限制文本長度避免處理時間過長
                if len(text) > 500000:
                    text = text[:500000]
                    logger.warning(f"文本過長，已截取前 500000 字符")
                
                # 抽取三元組
                triplets = extractor.extract_triplets_from_text(text, filename)
                all_triplets.extend(triplets)
                
                logger.info(f"從 {filename} 抽取到 {len(triplets)} 個三元組")
                
            except Exception as e:
                logger.error(f"處理文件 {filename} 時發生錯誤: {e}")
                continue
    
    return all_triplets

def process_pdf_directory(pdf_dir):
    """處理目錄中的所有 PDF 文件 (向後兼容)"""
    return process_files_directory(pdf_dir, ['.pdf'])

def save_triplets_to_csv(triplets, output_file):
    """將三元組保存到 CSV 文件"""
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject', 'Predicate', 'Object', 'Source'])
        
        for triplet in triplets:
            if len(triplet) == 4:
                writer.writerow(triplet)
            else:
                # 兼容性處理
                writer.writerow([triplet[0], triplet[1], triplet[2], "unknown"])

if __name__ == "__main__":
    # 使用全域配置的路徑
    output_file = os.path.join(PROCESSED_DIR, "triples.csv")
    
    # 確保目錄存在
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    
    logger.info("開始基於句子的三元組抽取...")
    
    all_triplets = []
    
    # 處理 PDF 文件
    if os.path.exists(PDF_DIR):
        logger.info("處理 PDF 文件...")
        pdf_triplets = process_files_directory(PDF_DIR, ['.pdf'])
        all_triplets.extend(pdf_triplets)
        logger.info(f"從 PDF 文件抽取到 {len(pdf_triplets)} 個三元組")
    
    # 處理 Markdown 文件
    if os.path.exists(MARKDOWN_DIR):
        logger.info("處理 Markdown 文件...")
        md_triplets = process_files_directory(MARKDOWN_DIR, ['.md'])
        all_triplets.extend(md_triplets)
        logger.info(f"從 Markdown 文件抽取到 {len(md_triplets)} 個三元組")
    
    # 保存結果
    save_triplets_to_csv(all_triplets, output_file)
    
    logger.info(f"抽取完成！共得到 {len(all_triplets)} 個三元組，已保存到 {output_file}")
    
    # 顯示前 10 個三元組示例
    print("\n前 10 個三元組示例：")
    for i, triplet in enumerate(all_triplets[:10]):
        print(f"{i+1}. {triplet[0]} - {triplet[1]} - {triplet[2]} (來源: {triplet[3]})") 