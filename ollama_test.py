import openai
import json

base_url = "http://192.168.0.26:11434/v1"  # ダッシュボードで設定しているURLと同じか確認
model_name = "gpt-oss:20b"  # <-- あなたがOllamaで使っているモデル名に置き換えてください

client = openai.OpenAI(base_url=base_url, api_key="sk-not-needed") # APIキーは不要なので適当な文字列でOK

try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        stream=False, # ストリーミングでない応答
    )
    print("Ollama Test successful!")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print(f"Ollama Test failed with error: {e}")