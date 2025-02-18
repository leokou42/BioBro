import os
import json

def load_api_key(key_name = "gemini"):
    # 讀取 JSON 檔案
    with open("api_keys.json", "r", encoding="utf-8") as file:
        data = json.load(file)  # 解析 JSON

    # 存取特定的值
    api_key = data.get(key_name)
    
    # 輸出結果
    if api_key:
        return api_key
    else:
        return None