import re
import os
from typing import List, Dict, Tuple
from pypdf import PdfReader
import logging
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    智能文檔分塊器
    支持PDF和Markdown文件的語義感知分塊
    """
    
    def __init__(self, 
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 min_chunk_size: int = MIN_CHUNK_SIZE):
        """
        初始化文檔分塊器
        
        Args:
            chunk_size: 每個chunk的最大字符數
            chunk_overlap: chunk之間的重疊字符數
            min_chunk_size: 最小chunk大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        logger.info(f"初始化文檔分塊器 - 塊大小: {chunk_size}, 重疊: {chunk_overlap}")
    
    def read_pdf(self, file_path: str) -> str:
        """
        讀取PDF文件內容
        
        Args:
            file_path: PDF文件路徑
            
        Returns:
            提取的文本內容
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            logger.info(f"成功讀取PDF文件: {file_path}, 共 {len(reader.pages)} 頁")
            return text.strip()
            
        except Exception as e:
            logger.error(f"讀取PDF文件失敗 {file_path}: {e}")
            return ""
    
    def read_markdown(self, file_path: str) -> str:
        """
        讀取Markdown文件內容
        
        Args:
            file_path: Markdown文件路徑
            
        Returns:
            清理後的文本內容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 清理Markdown語法
            cleaned_content = self._clean_markdown(content)
            
            logger.info(f"成功讀取Markdown文件: {file_path}")
            return cleaned_content
            
        except Exception as e:
            logger.error(f"讀取Markdown文件失敗 {file_path}: {e}")
            return ""
    
    def _clean_markdown(self, text: str) -> str:
        """
        清理Markdown語法，保留純文本內容
        
        Args:
            text: 原始Markdown文本
            
        Returns:
            清理後的文本
        """
        # 移除代碼塊
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]*`', '', text)
        
        # 移除標題符號，但保留標題文字
        text = re.sub(r'#{1,6}\s*', '', text)
        
        # 移除鏈接，保留鏈接文字
        text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
        
        # 移除圖片鏈接
        text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', '', text)
        
        # 移除粗體和斜體符號
        text = re.sub(r'\*\*([^*]*)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]*)\*', r'\1', text)
        text = re.sub(r'__([^_]*)__', r'\1', text)
        text = re.sub(r'_([^_]*)_', r'\1', text)
        
        # 移除列表符號
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # 移除引用符號
        text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
        
        # 移除水平線
        text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\*{3,}$', '', text, flags=re.MULTILINE)
        
        # 清理多餘的空白
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _split_by_semantic_boundaries(self, text: str) -> List[str]:
        """
        根據語義邊界分割文本
        優先在段落、句子邊界進行分割
        
        Args:
            text: 輸入文本
            
        Returns:
            分割後的文本段落列表
        """
        # 首先按段落分割
        paragraphs = text.split('\n\n')
        
        segments = []
        current_segment = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果當前段落加上新段落超過chunk_size，則處理當前segment
            if len(current_segment) + len(paragraph) > self.chunk_size:
                if current_segment:
                    segments.append(current_segment.strip())
                    current_segment = ""
                
                # 如果單個段落太長，按句子分割
                if len(paragraph) > self.chunk_size:
                    sentences = self._split_into_sentences(paragraph)
                    temp_segment = ""
                    
                    for sentence in sentences:
                        if len(temp_segment) + len(sentence) > self.chunk_size:
                            if temp_segment:
                                segments.append(temp_segment.strip())
                            temp_segment = sentence
                        else:
                            temp_segment += " " + sentence if temp_segment else sentence
                    
                    if temp_segment:
                        current_segment = temp_segment
                else:
                    current_segment = paragraph
            else:
                current_segment += "\n\n" + paragraph if current_segment else paragraph
        
        # 添加最後一個segment
        if current_segment and len(current_segment.strip()) >= self.min_chunk_size:
            segments.append(current_segment.strip())
        
        return segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        將文本分割為句子
        
        Args:
            text: 輸入文本
            
        Returns:
            句子列表
        """
        # 中英文句子邊界
        sentence_endings = r'[。.!！？?；;]'
        sentences = re.split(sentence_endings, text)
        
        # 清理和過濾句子
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) >= 10:  # 過濾太短的句子
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_overlapping_chunks(self, segments: List[str]) -> List[str]:
        """
        創建帶重疊的chunks
        
        Args:
            segments: 文本段落列表
            
        Returns:
            帶重疊的chunk列表
        """
        if not segments:
            return []
        
        chunks = []
        current_chunk = ""
        
        for i, segment in enumerate(segments):
            # 如果當前chunk為空，直接添加segment
            if not current_chunk:
                current_chunk = segment
                continue
            
            # 如果添加新segment會超過chunk_size
            if len(current_chunk) + len(segment) > self.chunk_size:
                chunks.append(current_chunk)
                
                # 創建重疊：從當前chunk的後部開始新chunk
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    # 找到重疊文本的合適起始點（詞邊界）
                    overlap_start = overlap_text.find(' ')
                    if overlap_start > 0:
                        overlap_text = overlap_text[overlap_start+1:]
                    current_chunk = overlap_text + " " + segment
                else:
                    current_chunk = segment
            else:
                current_chunk += " " + segment
        
        # 添加最後一個chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        將文本分塊並返回帶元數據的chunk列表
        
        Args:
            text: 輸入文本
            metadata: 文檔元數據
            
        Returns:
            chunk字典列表，包含text和metadata
        """
        if not text or len(text) < self.min_chunk_size:
            return []
        
        # 語義分割
        segments = self._split_by_semantic_boundaries(text)
        
        # 創建重疊chunks
        chunks = self._create_overlapping_chunks(segments)
        
        # 創建帶元數據的chunk列表
        chunk_list = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                'text': chunk_text,
                'chunk_id': i,
                'chunk_size': len(chunk_text),
                'metadata': metadata or {}
            }
            chunk_list.append(chunk_dict)
        
        logger.info(f"文本分塊完成: {len(chunks)} 個chunks")
        return chunk_list
    
    def chunk_file(self, file_path: str) -> List[Dict]:
        """
        分塊處理單個文件
        
        Args:
            file_path: 文件路徑
            
        Returns:
            chunk字典列表
        """
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # 創建基本元數據
        metadata = {
            'source_file': file_name,
            'file_path': file_path,
            'file_type': file_ext
        }
        
        # 根據文件類型讀取內容
        if file_ext == '.pdf':
            text = self.read_pdf(file_path)
        elif file_ext in ['.md', '.markdown']:
            text = self.read_markdown(file_path)
        else:
            logger.warning(f"不支持的文件類型: {file_ext}")
            return []
        
        if not text:
            logger.warning(f"文件內容為空: {file_path}")
            return []
        
        # 分塊處理
        chunks = self.chunk_text(text, metadata)
        
        logger.info(f"文件 {file_name} 分塊完成: {len(chunks)} 個chunks")
        return chunks
    
    def chunk_directory(self, directory_path: str, file_extensions: List[str] = None) -> List[Dict]:
        """
        批量處理目錄中的文件
        
        Args:
            directory_path: 目錄路徑
            file_extensions: 支持的文件擴展名列表
            
        Returns:
            所有文件的chunk列表
        """
        if file_extensions is None:
            file_extensions = ['.pdf', '.md', '.markdown']
        
        all_chunks = []
        
        if not os.path.exists(directory_path):
            logger.error(f"目錄不存在: {directory_path}")
            return all_chunks
        
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            
            # 檢查文件擴展名
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in file_extensions and os.path.isfile(file_path):
                try:
                    file_chunks = self.chunk_file(file_path)
                    all_chunks.extend(file_chunks)
                except Exception as e:
                    logger.error(f"處理文件失敗 {file_path}: {e}")
        
        logger.info(f"目錄處理完成: {len(all_chunks)} 個chunks from {directory_path}")
        return all_chunks