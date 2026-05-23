import os
import sys
import re
import io
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load env
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.LLM import LLM

# Set default env settings if not explicitly set
os.environ['OLLAMA_BASE_URL'] = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')

app = FastAPI()

# CORS：讓內網穿透/反向代理轉發後的前端請求能正常呼叫 API
# 若你是同網域（例如反代後一起走同網域），也不會造成問題。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static for frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

import json

@app.post("/api/chat")
async def chat(message: str = Form(...), file: UploadFile = File(None), model: str = Form("granite4.1:3b"), history: str = Form("[]")):
    df = None
    system_context = ""
    
    try:
        past_messages = json.loads(history)
    except Exception:
        past_messages = []
    
    # Process uploaded file
    if file and file.filename:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(content))
            
        if df is not None:
            system_context = f"Assume a pandas DataFrame `df` is already loaded with the following columns: {list(df.columns)}. Data sample:\n{df.head().to_string()}\n"
            instructions = (
                "\nWrite Python code using `matplotlib.pyplot as plt` to plot the requested chart based on the existing `df` variable. DO NOT redefine or mock `df`.\n"
                "If the user is asking for a new chart, recommend TWO DIFFERENT types of charts that might be better suited. Provide exactly THREE separate ```python ... ``` blocks (1. requested, 2. first recommendation, 3. second recommendation).\n"
                "If the user is asking to apply improvements to a previous chart, just provide the improved chart in a SINGLE ```python ... ``` block.\n"
                "DO NOT print out matplotlib warnings and do not call plt.show() inside the code.\n\n"
                "Finally, you MUST include the following markdown headings in your text response:\n"
                "### Recommendation Reasons\n"
                "(Explain exactly WHY you recommended these two alternative chart types)\n"
                "### Key Insights Summary\n"
                "(Provide 2-3 key insights based on the data and the generated charts)\n"
                "### Chart Evaluation & Suggestions\n"
                "(Critique the chart design, such as color contrast, labeling, or chart type suitability, and provide concrete suggestions for improvement)\n"
            )
            system_context += instructions

    prompt = system_context + "\nUser Request: " + message
    
    # Connect to Ollama
    llm = LLM(model_name=f"ollama:{model}", api_key=None)
    
    # Run the model
    try:
        response = llm.run(prompt=prompt, past_messages=past_messages)
    except Exception as e:
        return JSONResponse({"text": f"Error connecting to LLM: {str(e)}", "image": None})
    
    # Try to execute code if there is any to generate images
    img_b64_list = []
    # Extract all python fenced code blocks more robustly
    code_matches = re.findall(r"```[pP]ython[ \t]*\n(.*?)```", response, re.DOTALL)
    if not code_matches:
        # Fallback if the LLM forgot the word 'python'
        code_matches = re.findall(r"```[ \t]*\n(.*?)```", response, re.DOTALL)
    
    for idx, code in enumerate(code_matches):
        try:
            # ---- Fix: avoid "寫死"/不準的生圖 ----
            # 每次請求都用獨立的 figure/axes，並且在執行前清掉先前狀態，
            # 讓模型只能操作我們提供的 fig/ax。
            plt.close('all')
            fig = plt.figure(figsize=(8, 5), dpi=160)
            ax = fig.add_subplot(111)

            # Provide an isolated environment mostly focused on plotting.
            # Also pass `ax` and `fig` to encourage the model to draw on the
            # provided axes rather than relying on global state.
            exec_globals = {
                "pd": pd,
                "plt": plt,
                "df": df,
                "os": os,
                "fig": fig,
                "ax": ax,
                # 常見操作：避免模型自己呼叫 show / 顯示
                "__builtins__": __builtins__,
            }

            # Wrap the code so that plt.show() becomes a no-op.
            # Many models會寫死 show() 或另起 figure 導致輸出與預期不一致。
            wrapped_code = (
                "import matplotlib.pyplot as _plt\n"
                "def _noop(*args, **kwargs):\n"
                "    return None\n"
                "_plt.show = _noop\n"
                "_plt.ioff()\n"
                f"{code}\n"
            )

            exec(wrapped_code, exec_globals)

            # ---- Fix: 僅輸出我們提供的 figure（避免模型另開 figure 造成偏差） ----
            # 如果模型確實使用了 ax/fig，我們就直接輸出 fig。
            target_fig = fig

            # 如果模型完全沒畫到 ax（例如它寫死 plt.plot 但沒指定 axes），
            # 仍嘗試把最後一次有內容的 axes 畫面取回來。
            # 這裡用畫線/patch數量做簡單判斷。
            try:
                ax_has_artists = (len(getattr(ax, "lines", [])) > 0) or (len(getattr(ax, "patches", [])) > 0)
                if not ax_has_artists:
                    # 退一步：使用 plt.gcf() 但仍保持 close/隔離
                    target_fig = plt.gcf()
            except Exception:
                target_fig = fig

            target_fig.tight_layout()
            buf = io.BytesIO()
            target_fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_b64_list.append(base64.b64encode(buf.read()).decode('utf-8'))
        except Exception as e:
            response += f"\n\n> ⚠️ **Code Execution Error in block {idx+1}:** {str(e)}"
            
    return JSONResponse({"text": response, "images": img_b64_list})

if __name__ == "__main__":
    # 內網穿透/反代環境通常需要對外綁定 host
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
