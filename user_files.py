import json, os

BASE_DIR = "user_data"


def get_user_data_path(user_id):
    """ユーザーのデータディレクトリのパスを返す"""
    return os.path.join(BASE_DIR, str(user_id))


def save_user_data(user_id, data):
    """ユーザーデータを保存する。ディレクトリがなければ作成する"""
    user_path = get_user_data_path(user_id)
    os.makedirs(user_path, exist_ok=True)
    with open(os.path.join(user_path, "working.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()  # Explicitly flush buffers
        os.fsync(f.fileno())  # Force write to disk


def load_user_data(user_id):
    """ユーザーデータを読み込む"""
    path = os.path.join(get_user_data_path(user_id), "working.json")
    if not os.path.exists(path):
        return {"documents": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
