import os
import json
import base64
import openai
import requests
import anthropic
from groq import Groq

class LLM:
    """Unified wrapper for multiple LLM providers (OpenAI, Anthropic, Groq, DeepInfra, Ollama).

    Ollama-specific environment variables:
        - OLLAMA_BASE_URL:   例如 http://localhost:11434，可改以連線其他主機。
        - OLLAMA_API_URL:    可直接指定完整 API endpoint，預設為 {OLLAMA_BASE_URL}/api/chat。
        - OLLAMA_DEFAULT_MODEL: 無明確模型後綴時使用，預設為 "llama3"。
    """

    def __init__(self, model_name, api_key):
        self.raw_model_name = model_name
        self.model_name = model_name.lower()
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.ollama_model = None
        ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        # 若 OLLAMA_API_URL 未指定，則以 Base URL 拼接 /api/chat。
        self.ollama_api_url = os.environ.get("OLLAMA_API_URL", f"{ollama_base}/api/chat")

        if "gpt-4o" in self.model_name:
            self.client = None  # OpenAI client will use requests
        elif "claude" in self.model_name:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif "llama" in self.model_name and "ollama" not in self.model_name:
            # 只有當使用者真的要走 Groq 的 llama 系列時才需要 api_key
            # 若 api_key 為 None，且模型名稱看起來是 ollama（例如 'ollama:xxx'）請走 ollama 分支。
            # 為了避免誤判，這裡若 api_key 不存在就直接拒絕（讓錯誤更明確）。
            if not self.api_key:
                raise ValueError(
                    "Detected a Groq 'llama' backend but GROQ_API_KEY/api_key is not set. "
                    "For Ollama, set d2c_model to include 'ollama' (e.g. 'ollama' or 'ollama:MODEL')."
                )
            self.client = Groq(api_key=self.api_key)
        elif "gemma" in self.model_name:
            self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepinfra.com/v1/openai")
        elif "ollama" in self.model_name:
            self.client = None
            self.ollama_model = self._resolve_ollama_model_name(self.raw_model_name)
        else:
            raise ValueError("Unsupported model name.")


    def run(self, prompt, imgs=None, past_messages=None):
        if imgs is None:
            imgs = []
        if past_messages is None:
            past_messages = []

        if "gpt-4o" in self.model_name:
            return self.run_gpt4o(prompt, imgs, past_messages)
        elif "claude" in self.model_name:
            return self.run_claude(prompt, imgs, past_messages)
        elif "llama" in self.model_name and "ollama" not in self.model_name:
            return self.run_llama(prompt, past_messages)
        elif "gemma" in self.model_name:
            return self.run_gemma2(prompt, past_messages)
        elif "ollama" in self.model_name:
            return self.run_ollama(prompt, imgs, past_messages)
        else:
            raise ValueError("Unsupported model name.")

    def run_gpt4o(self, prompt, imgs, past_messages):
        messages = past_messages.copy()
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        })

        for base64_image in imgs:
            messages[-1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })

        payload = {"model": "gpt-4o", "messages": messages}
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        # print(response.json())
        usage = response.json()['usage']
        return response.json()['choices'][0].get('message', {}).get('content', '')#, usage.get('total_tokens', -1), usage.get('prompt_tokens', -1), usage.get('completion_tokens', -1)

    def run_claude(self, prompt, imgs, past_messages):
        messages = past_messages.copy()
        content = [{"type": "text", "text": prompt}]
        for base64_image in imgs:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image,
                }
            })

        messages.append({"role": "user", "content": content})
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            messages=messages
        )
        usage = response.model_dump()['usage']
        return response.content[0].text#, usage.get('input_tokens', -1) + usage.get('output_tokens', -1), usage.get('input_tokens', -1), usage.get('output_tokens', -1)

    def run_llama(self, prompt, past_messages):
        llama_messages = past_messages.copy()
        llama_messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            messages=llama_messages,
            model="llama-3.1-70b-versatile",
        )
        usage = response.usage
        return response.choices[0].message.content#, usage.total_tokens, usage.prompt_tokens, usage.completion_tokens

    def run_gemma2(self, prompt, past_messages):
        gemma_messages = past_messages.copy()
        gemma_messages.append({
            "role": "user",
            "content": prompt
        })

        chat_completion = self.client.chat.completions.create(
            model="google/gemma-2-27b-it",
            messages=gemma_messages,
            max_tokens=2048
        )

        usage = chat_completion.usage
        return chat_completion.choices[0].message.content#, usage.total_tokens, usage.prompt_tokens, usage.completion_tokens

    def run_ollama(self, prompt, imgs, past_messages):
        ollama_messages = []
        for message in past_messages:
            role = message.get("role", "user")
            content = self._normalize_message_content(message.get("content", ""))
            if content:
                ollama_messages.append({"role": role, "content": content})

        user_message = {"role": "user", "content": prompt}
        if imgs:
            user_message["images"] = imgs

        ollama_messages.append(user_message)

        payload = {
            "model": self.ollama_model,
            "messages": ollama_messages,
            "stream": False
        }

        response = requests.post(self.ollama_api_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        if "message" in data:
            return data["message"].get("content", "")
        if "response" in data:  # Legacy key returned by some Ollama versions
            return data.get("response", "")

        raise ValueError(f"Unexpected response from Ollama: {data}")

    @staticmethod
    def _normalize_message_content(content):
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text") or block.get("content", ""))
                else:
                    parts.append(str(block))
            return "\n".join(filter(None, parts))
        if isinstance(content, dict):
            return content.get("text", "")
        return str(content)

    def _resolve_ollama_model_name(self, raw_name):
        if not raw_name:
            return os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3")

        for separator in ("/", ":", "|"):
            if separator in raw_name:
                suffix = raw_name.split(separator, 1)[1].strip()
                if suffix:
                    return suffix

        suffix = raw_name.lower().split("ollama", 1)[-1].strip("/:|-_")
        if suffix:
            return suffix

        return os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3")
