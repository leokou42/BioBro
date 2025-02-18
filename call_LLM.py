import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import transformers
import torch
import time

load_dotenv()

# DeepSeek-R1-Distill-Llama-70B -> free and working!
def DeepSeek_R1_Distill_Llama_70B(comment):
    print("DeepSeek: R1 Distill Llama 70B")

    clinet = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = os.getenv("OPENROUTER_API_KEY"),
    )

    completion = clinet.chat.completions.create(
        model = "deepseek/deepseek-r1-distill-llama-70b:free",
        messages = [
            {"role": "user", 
             "content": comment
            }
        ]
    )
    print(completion.choices[0].message.content)

# DeepSeek: R1 Distill Qwen 32B
def DeepSeek_R1(comment):
    print("DeepSeek: R1")

    clinet = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = os.getenv("OPENROUTER_API_KEY"),
    )

    completion = clinet.chat.completions.create(
        model = "deepseek/deepseek-r1:free",
        messages = [
            {"role": "user", 
             "content": comment
            }
        ]
    )
    print(completion.choices[0].message.content)

# Llama -> model too big
def Llama(comment):
    print("meta-llama: Llama-3.3-70B-Instruct")
    model_id = "meta-llama/Llama-3.3-70B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])

# Gemini -> free and working!
def Gemini(comment, model = "gemini-2.0-flash"):
    print("Gemini: {}".format(model))
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # 設定 API Key
    genai.configure(api_key=GEMINI_API_KEY)
    # 指定模型
    model = genai.GenerativeModel(model)  # 確保模型名稱正確
    # 生成回應
    response = model.generate_content(comment)

    print(response.text)

# comment = "which is bigger, Mars or Earth? Please explain why."
# comment = "生命的意義是什麼?"
comment = "請你介紹台灣黑熊"

if __name__ == "__main__":
    print("Start testing...")
    start_time = time.time()

    # major models
    DeepSeek_R1_Distill_Llama_70B(comment)

    # minor models
    # Gemini(comment)

    # backup models
    # DeepSeek_R1(comment)
    # Llama()

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds.")
    