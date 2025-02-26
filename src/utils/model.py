import os
import time
from typing import Any, Mapping

from dotenv import load_dotenv
from langchain.llms.base import LLM
import google.generativeai as genai

# 載入環境變數
load_dotenv()

class llm_model(LLM):

    model: str = "gemini"
    api_key: str = None  # API 金鑰

    def __init__(self, model: str):
        super().__init__(model=model)

        # 設定 API 金鑰
        self.api_key = os.getenv("GEMINI_API_KEY")  # 讀取 Gemini API Key

        if not self.api_key:
            raise ValueError("環境變數`GEMINI_API_KEY`未設定，請確認`.env`")

    @property
    def _llm_type(self) -> str:
        return self.model

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """回傳 API 端點資訊"""
        return {"api_key": self.api_key}

    def _call(self, prompt: str, stop=None) -> str:
        """呼叫 Gemini API"""
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")  #使用 Gemini 2.0 Flash
            response = model.generate_content(prompt)

            return response.text if response.text else "Gemini API 回應為空"
        
        except Exception as e:
            print(f"Gemini API 錯誤: {e}")
            return "Gemini API 失敗，請稍後再試"


if __name__ == "__main__":
    comment = "台灣國鳥是?"

    print("\n測試 Gemini")
    start_time = time.time()
    model = llm_model("gemini")
    print(model.invoke(comment))

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds.")
    

