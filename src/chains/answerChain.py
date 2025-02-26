from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.model import llm_model  # 引入 LLM

def answerChain(model):
    """使用 LLM 生成最終回答"""
    template = """你是一位生物學專家，請根據以下資訊回答問題：
    1. 確保資訊正確且完整
    2. 使用繁體中文回答
    3. 若有學名，請提供
    4. 若查無資料，請誠實回答

    問題: {question}
    相關資訊:
    {context}

    回答:
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = llm_model(model)  # 使用 LLM 生成回答

    chain = prompt | model | StrOutputParser()
    return chain


if __name__ == "__main__":
    print("測試 answerChain.py")
    llm = llm_model("gemini")
    answer_chain = answerChain(model="gemini")
    # print(answer_chain)

    test_question = "台灣有哪些特有種鳥類？"
    test_context = "台灣藍鵲、帝雉、黑嘴端鳳頭燕鷗、麻雀"  # 模擬 RAG 查詢結果

    response = answer_chain.invoke({"question": test_question, "context": test_context})
    print(f"問題: {test_question}")
    print(f"資訊: {test_context}")
    print(f"回答: {response}")