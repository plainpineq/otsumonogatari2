# services/domain_bridge.py

def document_to_domain(document: dict) -> dict:
    """
    既存 document(JSON) → Domainモデル
    （保存形式は一切変更しない）
    """
    print("### domain_bridge loaded ###")

    # --- Intent ---
    intent_fields = []

    intent_src = document.get("intent") or {}

    # 既存キーはすべて fields に変換
    for key, value in intent_src.items():
        intent_fields.append({
            "key": key,
            "label": key,   # ← 最初は key をそのまま表示名に
            "value": value
        })

    print("DEBUG intent_src:", document.get("intent"))
    print("DEBUG intent_fields:", intent_fields)

    # intent が完全に空の場合の初期テンプレ
    if not intent_fields:
        for key, label in [
            ("genre", "ジャンル"),
            ("theme_or_claim", "テーマ・主張"),
            ("values", "価値観"),
        ]:
            intent_fields.append({
                "key": key,
                "label": label,
                "value": ""
            })

    # Unit = シーン分類としてラップ
    scene_category = {
        "id": "cat-scene",
        "name": "シーン",
        "elements": []
    }

    for unit in document.get("units", []):
        scene_category["elements"].append({
            "id": f"type-{unit['title']}",
            "name": unit["title"],
            "instances": [
                {
                    "id": unit.get("id"),
                    "content": unit.get("content", ""),
                    "_raw_unit": unit   # 逆変換用参照
                }
            ]
        })

    return {
        "id": document["id"],
        "title": document["title"],
        "intent": {
            "fields": intent_fields
        },
        "categories": [scene_category]
    }

def domain_to_document(domain: dict, document: dict) -> dict:
    """
    Domainモデル → 既存 document(JSON)
    """

    # Intent を戻す
    if "intent" in domain:
        for field in domain["intent"].get("fields", []):
            document["intent"][field["key"]] = field["value"]

    # Unit（シーン）を戻す
    for category in domain.get("categories", []):
        if category["name"] != "シーン":
            continue

        for element in category.get("elements", []):
            instance = element["instances"][0]
            raw = instance.get("_raw_unit")
            if raw is not None:
                raw["content"] = instance["content"]

    return document
