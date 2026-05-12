# LLM Data-to-Chart 代理系統 (FastAPI + Uvicorn)

這是一個基於 FastAPI 打造的前後端分離系統，結合本地端 [Ollama](https://ollama.com/) 運行的大型語言模型 (LLM)，專門協助使用者進行**資料分析**與**圖表視覺化 (Data-to-Chart)**。

相較於 Streamlit，此版本提供了更原生的 HTML/JS 輕量前端，並在後端實際安全執行大模型所產生的圖表程式碼來擷取並回傳圖片。

## ✨ 機能說明
- **執行出圖**：後端會自動讀取大模型產生的 Python Matplotlib 程式碼並執行、擷取該圖表畫面傳送至前端渲染。
- **純本地運行**：支援透過 Ollama 服務。

## 🚀 快速開始

### 環境安裝
```bash
uv pip install -r requirements.txt
uv pip install fastapi uvicorn python-multipart matplotlib
```

### 啟動服務
```bash
uv run uvicorn main:app --reload
```
然後打開瀏覽器前往：`http://127.0.0.1:8000`
