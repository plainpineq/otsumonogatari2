import google.generativeai as genai
genai.configure(api_key="AIzaSyAL5otjCNGnql0J0cKnJAIiZ2b7V4wfblA")

model_id = 'models/gemini-2.5-flash' # または 'gemini-3-flash'

try:
    model = genai.GenerativeModel(model_id)
    response = model.generate_content("Hi, what model are you?")
    print(f"成功: {model_id} は使用可能です。")
    print(f"回答内容: {response.text}")
except Exception as e:
    print(f"失敗: {model_id} は現在の環境/キーでは使用できません。")
    print(f"エラー詳細: {e}")