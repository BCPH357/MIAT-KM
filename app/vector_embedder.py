import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import logging
from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_LENGTH

logger = logging.getLogger(__name__)

class BGEEmbedder:
    """
    BGE-M3 Embedding模型封裝類
    使用Hugging Face transformers實現BGE-M3模型的embedding功能
    """
    
    def __init__(self, 
                 model_name: str = EMBEDDING_MODEL,
                 device: str = EMBEDDING_DEVICE,
                 batch_size: int = EMBEDDING_BATCH_SIZE,
                 max_length: int = EMBEDDING_MAX_LENGTH):
        """
        初始化BGE-M3 Embedding模型
        
        Args:
            model_name: 模型名稱
            device: 設備 (cuda/cpu)
            batch_size: 批次大小
            max_length: 最大序列長度
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        
        logger.info(f"初始化BGE-M3 Embedding模型: {model_name}")
        logger.info(f"使用設備: {self.device}")
        
        try:
            # 載入tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # 移動模型到指定設備
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("BGE-M3 Embedding模型載入成功")
            
        except Exception as e:
            logger.error(f"載入BGE-M3模型失敗: {e}")
            logger.info("嘗試使用備用embedding模型...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """載入備用embedding模型"""
        try:
            # 清理GPU記憶體
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("已清理GPU記憶體緩存")
            
            # 使用sentence-transformers作為備用，強制CPU
            from sentence_transformers import SentenceTransformer
            self.fallback_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device='cpu'  # 強制使用CPU
            )
            self.use_fallback = True
            logger.info("成功載入備用embedding模型(CPU)")
        except Exception as e:
            logger.error(f"備用模型載入也失敗: {e}")
            
            # 最後備用方案：使用更小的模型
            try:
                from sentence_transformers import SentenceTransformer
                self.fallback_model = SentenceTransformer(
                    'all-MiniLM-L6-v2',  # 更小的英文模型
                    device='cpu'
                )
                self.use_fallback = True
                logger.info("使用最小備用模型(all-MiniLM-L6-v2)")
            except Exception as e2:
                logger.error(f"最小備用模型也失敗: {e2}")
                raise RuntimeError("無法載入任何embedding模型")
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        使用mean pooling策略
        這是BGE-M3的推薦pooling方法
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        編碼單個文本
        
        Args:
            text: 輸入文本
            
        Returns:
            embedding向量
        """
        if hasattr(self, 'use_fallback') and self.use_fallback:
            return self.fallback_model.encode([text])[0]
        
        try:
            with torch.no_grad():
                # tokenize文本
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # 移動到設備
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 獲取模型輸出
                outputs = self.model(**inputs)
                
                # 使用mean pooling
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                # 正規化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                return embeddings.cpu().numpy()[0]
                
        except Exception as e:
            logger.error(f"編碼文本失敗: {e}")
            return np.zeros(768)  # 返回零向量作為備用
    
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        批次編碼多個文本
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding向量列表
        """
        if hasattr(self, 'use_fallback') and self.use_fallback:
            return [emb for emb in self.fallback_model.encode(texts)]
        
        embeddings = []
        
        # 分批處理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                with torch.no_grad():
                    # tokenize批次文本
                    inputs = self.tokenizer(
                        batch_texts,
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    # 移動到設備
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # 獲取模型輸出
                    outputs = self.model(**inputs)
                    
                    # 使用mean pooling
                    batch_embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                    
                    # 正規化
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    
                    # 轉換為numpy並添加到結果
                    batch_embeddings_np = batch_embeddings.cpu().numpy()
                    embeddings.extend([emb for emb in batch_embeddings_np])
                    
            except Exception as e:
                logger.error(f"批次編碼失敗: {e}")
                # 為失敗的批次添加零向量
                zero_embedding = np.zeros(768)
                embeddings.extend([zero_embedding] * len(batch_texts))
        
        return embeddings
    
    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        編碼文本（支持單個文本或文本列表）
        
        Args:
            texts: 單個文本或文本列表
            
        Returns:
            embedding向量或向量列表
        """
        if isinstance(texts, str):
            return self.encode_single(texts)
        else:
            return self.encode_batch(texts)
    
    def get_embedding_dimension(self) -> int:
        """獲取embedding向量維度"""
        if hasattr(self, 'use_fallback') and self.use_fallback:
            return self.fallback_model.get_sentence_embedding_dimension()
        
        # QWEN模型的hidden size通常是4096，但我們可以動態獲取
        try:
            sample_embedding = self.encode_single("測試文本")
            return len(sample_embedding)
        except:
            return 768  # 備用維度
    
    def __del__(self):
        """清理資源"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass