# intent_service.py

from intent_templates import COMMON_INTENTS, DOC_TYPE_INTENTS

def generate_intent(doc_type: str) -> dict:
    """
    doc_type 選択時に Intent を自動生成
    """
    fields = {}

    # 共通 Intent
    for key, label in COMMON_INTENTS:
        fields[key] = {
            "label": label,
            "value": ""
        }

    # doc_type 固有 Intent
    for key, label in DOC_TYPE_INTENTS.get(doc_type, []):
        fields[key] = {
            "label": label,
            "value": ""
        }

    return {
        "fields": fields
    }


def normalize_intent(document: dict) -> None:
    """
    Intent が未生成なら自動生成
    """
    if "intent" not in document or "fields" not in document["intent"]:
        document["intent"] = generate_intent(document["doc_type"])
