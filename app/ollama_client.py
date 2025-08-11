import requests
import json
from typing import Optional, Dict, Any
from config import OLLAMA_BASE_URL, MODEL_TEMPERATURE, MODEL_NUM_PREDICT

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
                "temperature": temperature
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2分鐘超時
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
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
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
    
    def rag_generate(self, 
                     model: str,
                     user_query: str,
                     knowledge_context: str,
                     temperature: float = 0.7) -> str:
        """
        基於 RAG 的生成：結合用戶查詢和知識上下文
        """
        # 構建 RAG prompt
        rag_prompt = f"""你是一個知識問答助手。請根據以下提供的知識上下文來回答用戶的問題。

知識上下文：
{knowledge_context}

用戶問題：{user_query}

請根據上述知識上下文回答問題。如果知識上下文中沒有相關信息，請明確說明，並基於你的一般知識提供幫助。回答要準確、詳細且有條理。"""

        result = self.generate(
            model=model,
            prompt=rag_prompt,
            temperature=temperature,
            stream=False
        )
        
        if "error" in result:
            return f"生成回答時發生錯誤: {result['error']}"
        
        return result.get("response", "無法生成回答")
    
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