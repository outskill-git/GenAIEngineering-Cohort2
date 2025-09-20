import os
import requests
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = OPENROUTER_API_KEY or OPENAI_API_KEY
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def stream_openrouter(messages: List[Dict[str, str]], model: str = "openai/gpt-4o", temperature: float = 1.0, max_tokens: int = None):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    try:
        with requests.post(API_URL, headers=headers, json=payload, stream=True) as resp:
            if resp.status_code != 200:
                yield "", {"error": f"OpenRouter API error: {resp.status_code} {resp.text}"}
                return
            buffer = ""
            extra_params = {}
            got_any = False
            for line in resp.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        data = line[6:]
                        if data == b"[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            buffer += delta
                            got_any = True
                            yield delta, chunk  # yield both text and full chunk for extra params
                            extra_params.update(chunk)
                        except Exception as e:
                            yield "", {"error": f"Streaming decode error: {e}"}
                            continue
            if not got_any:
                yield "", {"error": "No response from LLM. Check your API key and model."}
            return
    except Exception as e:
        yield "", {"error": f"Request failed: {e}"}
        return 

def chat_openrouter(messages: List[Dict[str, str]], model: str = "openai/gpt-4o", temperature: float = 1.0, max_tokens: int = None, tools: list = None):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            return None, {"error": f"OpenRouter API error: {resp.status_code} {resp.text}"}
        data = resp.json()
        assistant_message = data["choices"][0]["message"]
        assistant_content = assistant_message.get("content", "")
        return assistant_content, data
    except Exception as e:
        return None, {"error": f"Request failed: {e}"} 