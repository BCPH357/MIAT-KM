import requests
import json
import re
from typing import Optional, Dict, Any, Tuple
from config import OLLAMA_BASE_URL, MODEL_TEMPERATURE, MODEL_NUM_PREDICT, RAG_COT_PROMPT

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
    
    def generate(self, 
                 model: str,
                 prompt: str,
                 stream: bool = False,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        使用 Ollama 生成回應
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": -1  # 預設為無限制
            }
        }
        
        if max_tokens and max_tokens > 0:
            payload["options"]["num_predict"] = max_tokens
        else:
            # 不設置num_predict限制，讓模型自由生成
            payload["options"]["num_predict"] = -1
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=600  # 10分鐘超時，給長文本生成更多時間
            )
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {
                "error": f"請求錯誤: {str(e)}",
                "response": None
            }
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON 解析錯誤: {str(e)}",
                "response": None
            }
    
    def chat(self, 
             model: str,
             messages: list,
             stream: bool = False,
             temperature: float = 0.7) -> Dict[str, Any]:
        """
        使用 Ollama 聊天 API
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": -1  # 預設為無限制
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=600  # 10分鐘超時，給長文本生成更多時間
            )
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {
                "error": f"請求錯誤: {str(e)}",
                "response": None
            }
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON 解析錯誤: {str(e)}",
                "response": None
            }
    
    def list_models(self) -> Dict[str, Any]:
        """
        列出可用的模型
        """
        try:
            response = requests.get(f"{self.api_url}/tags")
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {
                "error": f"請求錯誤: {str(e)}",
                "models": []
            }
    
    def check_model_available(self, model_name: str) -> bool:
        """
        檢查指定模型是否可用
        """
        models_info = self.list_models()
        if "error" in models_info:
            return False
        
        available_models = [model["name"] for model in models_info.get("models", [])]
        return model_name in available_models
    
    def _parse_cot_response(self, response_text: str) -> Tuple[str, str]:
        """
        解析CoT回應，分離思考過程和最終答案
        返回: (thinking, answer)
        """
        # 嘗試匹配 <thinking>...</thinking> 和 <answer>...</answer> 格式
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response_text, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        
        thinking = ""
        answer = ""
        
        if thinking_match:
            thinking = thinking_match.group(1).strip()
        
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # 如果沒有找到標籤格式，嘗試其他格式或使用整個回應作為答案
            if thinking:
                # 如果有thinking但沒有answer標籤，取thinking後面的內容作為答案
                answer_start = response_text.find('</thinking>') + len('</thinking>')
                if answer_start < len(response_text):
                    answer = response_text[answer_start:].strip()
            else:
                # 完全沒有標籤格式，整個回應作為答案
                answer = response_text.strip()
        
        return thinking, answer
    
    def rag_generate(self,
                     model: str,
                     user_query: str,
                     knowledge_context: str,
                     temperature: float = 0.7) -> Dict[str, str]:
        """
        基於 RAG 的生成：結合用戶查詢和知識上下文
        返回包含thinking和answer的字典
        """
        # 構建帶有CoT的RAG prompt，使用config中的模板
        rag_prompt = RAG_COT_PROMPT.format(
            knowledge_context=knowledge_context,
            user_query=user_query
        )

        result = self.generate(
            model=model,
            prompt=rag_prompt,
            temperature=temperature,
            stream=False
        )
        
        if "error" in result:
            return {
                "thinking": "生成過程中發生錯誤",
                "answer": f"生成回答時發生錯誤: {result['error']}"
            }
        
        response_text = result.get("response", "無法生成回答")
        thinking, answer = self._parse_cot_response(response_text)
        
        # 在終端log思考過程
        if thinking:
            print(f"\n🤔 AI思考過程:")
            print(f"{'='*50}")
            print(thinking)
            print(f"{'='*50}")
        
        return {
            "thinking": thinking,
            "answer": answer if answer else response_text
        }
    
    def simple_generate(self, 
                       model: str,
                       user_query: str,
                       temperature: float = 0.7) -> str:
        """
        簡單生成（不使用 RAG）
        """
        result = self.generate(
            model=model,
            prompt=user_query,
            temperature=temperature,
            stream=False
        )
        
        if "error" in result:
            return f"生成回答時發生錯誤: {result['error']}"
        
        return result.get("response", "無法生成回答") 