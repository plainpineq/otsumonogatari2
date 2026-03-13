import logging
import google.generativeai as genai
import openai
from typing import Dict, Any, Optional


def generate_draft(prompt: str, llm_config: Dict[str, Any]) -> str:
    """
    LLMを使用して小説の下書き（ドラフト）を直接生成します。
    JSON解析を行わず、生のテキストとして返却します。
    """
    provider = llm_config.get("provider", "chatgpt")
    model_name = llm_config.get("model_name") or llm_config.get("model") or "gpt-4o"
    api_key = llm_config.get("api_key", "")
    base_url = llm_config.get("base_url")

    # provider の正規化
    if provider == "openai":
        provider = "chatgpt"

    logging.info(
        f"[LLM Router] Generating RAW draft with provider: {provider}, model: {model_name}"
    )

    try:
        if provider == "gemini":
            if not api_key:
                raise ValueError("Gemini API Key is not configured.")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            # Gemini has its own internal timeout, but we can wrap it if needed.
            # For now, we rely on the library's default.
            response = model.generate_content(prompt)
            return response.text

        elif provider == "chatgpt" or provider == "other":
            # OpenAI互換機（Ollama含む）の処理
            # タイムアウトを設定 (600秒 = 10分)
            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=600.0)
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported LLM provider for drafting: {provider}")

    except Exception as e:
        logging.error(f"[LLM Router] Error during raw draft generation: {e}")
        raise RuntimeError(f"下書き生成に失敗しました: {e}")
