from utils import *
from openai import OpenAI
import google.generativeai as genai
import transformers
import torch
import time

# DeepSeek-R1-Distill-Llama-70B -> free and working!
def DeepSeek_R1_Distill_Llama_70B(comment):
    clinet = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = load_api_key("openrouter"),
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
    clinet = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = load_api_key("openrouter"),
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
def Gemini(comment):
    GEMINI_API_KEY = load_api_key("gemini")
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model = "gemini-2.0-flash",
        contents = comment,
    )

    print(response.text)

# comment = "which is bigger, Mars or Earth? Please explain why."
# comment = "生命的意義是什麼?"
comment = "請你介紹台灣黑熊"

if __name__ == "__main__":
    print("Start testing...")
    start_time = time.time()

    # DeepSeek_R1(comment)
    DeepSeek_R1_Distill_Llama_70B(comment)
    # Gemini(comment)
    # Llama()

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds.")
    