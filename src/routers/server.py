from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.utils.model import llm_model
from src.chains.answerChain import answerChain  # ✅ 只保留 answerChain

# 載入環境變數
load_dotenv()

# 創建 FastAPI 應用
app = FastAPI()

# 初始化 LLM
try:
    llm = llm_model("gemini")
except ValueError as e:
    llm = None
    print(f"⚠️ LLM 初始化失敗: {e}")

# 定義請求格式
class QueryRequest(BaseModel):
    query: str

# 健康檢查 API
@app.get("/health")
def health_check():
    """檢查伺服器與 LLM 是否正常運行"""
    if llm:
        return {"status": "ok", "message": "BioBro 伺服器正常運行"}
    else:
        return {"status": "error", "message": "LLM 初始化失敗"}

# AI 問答 API（只使用 answerChain）
@app.post("/api/v1/ask")
def ask_biobro(request: QueryRequest):
    """讓使用者詢問 BioBro 並取得回答"""
    if not llm:
        raise HTTPException(status_code=500, detail="LLM 尚未正確初始化")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="問題不能為空")

    try:
        # ✅ 直接使用 answerChain 產生回答
        answerchain = answerChain(model="gemini")
        answer = answerchain.invoke({"question": request.query, "context": ""})  # 目前沒有 context

        return {"query": request.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BioBro 失敗: {e}")

# 啟動 API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
