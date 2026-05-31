import os
import sys
import re
import io
import base64
import csv
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd
import requests

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Load env
load_dotenv(os.path.join(BASE_DIR, ".env"))
sys.path.append(BASE_DIR)
from src.LLM import LLM

# Set default env settings if not explicitly set
os.environ['OLLAMA_BASE_URL'] = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_DEFAULT_MODEL", "granite4.1:3b")

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
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

import json

@app.get("/api/models")
def list_models():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        models = [
            model.get("name")
            for model in data.get("models", [])
            if model.get("name")
        ]
        return JSONResponse({
            "models": sorted(models),
            "default_model": DEFAULT_OLLAMA_MODEL,
            "source": f"{base_url}/api/tags"
        })
    except Exception as e:
        return JSONResponse({
            "models": [DEFAULT_OLLAMA_MODEL],
            "default_model": DEFAULT_OLLAMA_MODEL,
            "error": str(e),
            "source": f"{base_url}/api/tags"
        })

def read_csv_upload(content: bytes) -> pd.DataFrame:
    """Read CSV files with common real-world encodings and delimiters."""
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin1")
    last_error = None

    for encoding in encodings:
        try:
            sample = content[:8192].decode(encoding)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = None

            read_options = {"encoding": encoding}
            if delimiter:
                read_options["sep"] = delimiter
            else:
                read_options.update({"sep": None, "engine": "python"})

            df = pd.read_csv(io.BytesIO(content), **read_options)

            if len(df.columns) == 1:
                column_name = str(df.columns[0])
                if any(separator in column_name for separator in (";", "\t", "|")):
                    raise ValueError(
                        f"CSV appears to use another delimiter but was parsed as one column: {column_name[:80]}"
                    )

            return df
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Unable to read CSV file. Last error: {last_error}")

def fig_to_base64(fig) -> str:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def choose_numeric_column(df: pd.DataFrame):
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        return None

    lower_to_column = {column.lower(): column for column in numeric_columns}
    for keyword in (
        "sales", "profit", "income", "avg_score", "score", "response",
        "quantity", "math", "reading", "writing"
    ):
        for lower_name, column in lower_to_column.items():
            if keyword in lower_name:
                return column

    for column in numeric_columns:
        lower_name = column.lower()
        if "id" not in lower_name and "postal" not in lower_name and df[column].nunique(dropna=True) > 1:
            return column

    return numeric_columns[0]

def choose_categorical_column(df: pd.DataFrame):
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
    if not categorical_columns:
        return None

    lower_to_column = {column.lower(): column for column in categorical_columns}
    for keyword in ("category", "segment", "region", "education", "group", "sex", "state"):
        for lower_name, column in lower_to_column.items():
            if keyword in lower_name:
                return column

    usable_columns = [
        column for column in categorical_columns
        if 1 < df[column].nunique(dropna=True) <= 30
    ]
    return usable_columns[0] if usable_columns else categorical_columns[0]

def create_fallback_chart(df: pd.DataFrame, message: str) -> str:
    message_lower = message.lower()
    numeric_column = choose_numeric_column(df)
    categorical_column = choose_categorical_column(df)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)

    if numeric_column is None:
        if categorical_column is None:
            raise ValueError("No usable columns found for fallback chart.")
        counts = df[categorical_column].astype(str).value_counts().head(12).sort_values()
        counts.plot(kind="barh", ax=ax, color="#4f8cff")
        ax.set_title(f"Top {categorical_column} Counts")
        ax.set_xlabel("Count")
        ax.set_ylabel(categorical_column)
        return fig_to_base64(fig)

    if "scatter" in message_lower and len(df.select_dtypes(include="number").columns) >= 2:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        x_column = numeric_column
        y_column = next((column for column in numeric_columns if column != x_column), numeric_columns[0])
        ax.scatter(df[x_column], df[y_column], alpha=0.65, color="#4f8cff")
        ax.set_title(f"{y_column} vs {x_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
    elif "hist" in message_lower or "直方" in message:
        df[numeric_column].dropna().plot(kind="hist", bins=20, ax=ax, color="#4f8cff", edgecolor="white")
        ax.set_title(f"Distribution of {numeric_column}")
        ax.set_xlabel(numeric_column)
        ax.set_ylabel("Frequency")
    elif "box" in message_lower or "箱" in message:
        df[numeric_column].dropna().plot(kind="box", ax=ax)
        ax.set_title(f"Box Plot of {numeric_column}")
        ax.set_ylabel(numeric_column)
    elif "pie" in message_lower or "圓餅" in message:
        if categorical_column:
            counts = df[categorical_column].astype(str).value_counts().head(8)
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"{categorical_column} Share")
        else:
            values = df.select_dtypes(include="number").sum().sort_values(ascending=False).head(8)
            ax.pie(values.values, labels=values.index, autopct="%1.1f%%", startangle=90)
            ax.set_title("Numeric Measures Share")
    elif ("line" in message_lower or "折線" in message) and categorical_column:
        grouped = df.groupby(categorical_column)[numeric_column].mean().head(20)
        grouped.plot(kind="line", marker="o", ax=ax, color="#4f8cff")
        ax.set_title(f"Average {numeric_column} by {categorical_column}")
        ax.set_xlabel(categorical_column)
        ax.set_ylabel(f"Average {numeric_column}")
        ax.tick_params(axis="x", rotation=35)
    elif categorical_column:
        aggregation = "sum" if any(key in numeric_column.lower() for key in ("sales", "profit", "quantity")) else "mean"
        grouped = getattr(df.groupby(categorical_column)[numeric_column], aggregation)()
        grouped = grouped.sort_values(ascending=False).head(12).sort_values()
        grouped.plot(kind="barh", ax=ax, color="#4f8cff")
        ax.set_title(f"{aggregation.title()} {numeric_column} by {categorical_column}")
        ax.set_xlabel(f"{aggregation.title()} {numeric_column}")
        ax.set_ylabel(categorical_column)
    else:
        df[numeric_column].dropna().head(50).plot(kind="line", ax=ax, color="#4f8cff")
        ax.set_title(f"{numeric_column} Values")
        ax.set_xlabel("Row")
        ax.set_ylabel(numeric_column)

    ax.grid(True, alpha=0.25)
    return fig_to_base64(fig)

@app.post("/api/chat")
async def chat(message: str = Form(...), file: UploadFile = File(None), model: str = Form(DEFAULT_OLLAMA_MODEL), history: str = Form("[]")):
    df = None
    system_context = ""
    
    try:
        past_messages = json.loads(history)
    except Exception:
        past_messages = []
    
    # Process uploaded file
    if file and file.filename:
        content = await file.read()
        try:
            filename = file.filename.lower()
            if filename.endswith('.csv'):
                df = read_csv_upload(content)
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            return JSONResponse({
                "text": f"Error reading uploaded file `{file.filename}`: {str(e)}",
                "images": []
            })
            
        if df is not None:
            numeric_columns = df.select_dtypes(include="number").columns.tolist()
            categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
            dtype_summary = {column: str(dtype) for column, dtype in df.dtypes.items()}
            system_context = (
                "Assume a pandas DataFrame `df` is already loaded.\n"
                f"Columns: {list(df.columns)}\n"
                f"Column dtypes: {dtype_summary}\n"
                f"Numeric columns safe for mean/sum/min/max aggregation: {numeric_columns}\n"
                f"Categorical/text columns for grouping, labels, or filters only: {categorical_columns}\n"
                f"Data sample:\n{df.head().to_string()}\n"
            )
            instructions = (
                "\nYou are an expert data visualization assistant. Infer the user's analysis goal from their request, even when they do not name a specific chart type. Choose the most suitable chart type based on the data columns, the audience, and the comparison/relationship/trend/distribution the user wants to understand.\n"
                "Write Python code using `matplotlib.pyplot as plt` to plot the chosen chart based on the existing `df` variable. DO NOT redefine or mock `df`.\n"
                "If the user explicitly asks for a specific chart type, create that requested chart first. If the user does NOT specify a chart type, treat your best chart choice as the requested chart and briefly explain why it is appropriate.\n"
                "When using aggregation methods such as mean(), sum(), median(), min(), or max(), explicitly select only numeric columns first. NEVER call df.mean(), df.groupby(...).mean(), or similar aggregation on the whole DataFrame when text columns are present.\n"
                "Use categorical/text columns only as group-by keys, axis labels, legends, or filters. If the user asks for a comparison by category, group by the category and aggregate one or more numeric columns.\n"
                "If the user asks to compare multiple numeric measures with no category, aggregate each numeric measure directly and use a clear comparison chart such as a bar chart, dot plot, or lollipop chart.\n"
                "If the user is asking for a new chart, provide exactly THREE separate ```python ... ``` blocks: (1. your selected/requested chart, 2. a meaningful alternative chart, 3. another meaningful alternative chart). The two alternatives must be genuinely different chart designs suited to the same analysis goal.\n"
                "If the user is asking to apply improvements to a previous chart, just provide the improved chart in a SINGLE ```python ... ``` block.\n"
                "Your chart code must include clear title, axis labels, readable tick labels, and annotations or visual highlights for the strongest and weakest values when the request asks for highlights.\n"
                "DO NOT print out matplotlib warnings and do not call plt.show() inside the code.\n\n"
                "Finally, you MUST include the following markdown headings in your text response:\n"
                "### Selected Chart Rationale\n"
                "(Explain which chart you selected and why it fits the user's goal and audience)\n"
                "### Recommendation Reasons\n"
                "(Explain exactly WHY you recommended these two alternative chart types)\n"
                "### Key Insights Summary\n"
                "(Provide 3 key insights based on the data and the generated charts. If relevant, identify strongest and weakest categories or measures)\n"
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

            img_b64_list.append(fig_to_base64(target_fig))
        except Exception as e:
            response += f"\n\n> ⚠️ **Code Execution Error in block {idx+1}:** {str(e)}"

    if df is not None and not img_b64_list:
        try:
            img_b64_list.append(create_fallback_chart(df, message))
            response += (
                "\n\n> ℹ️ The model response did not produce executable chart code, "
                "so the server generated a fallback chart directly from the uploaded data."
            )
        except Exception as e:
            response += f"\n\n> ⚠️ **Fallback Chart Error:** {str(e)}"
            
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
