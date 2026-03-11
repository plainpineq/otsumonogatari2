# domain_mapper.py
from models import Intent


def json_to_intent(intent_json: dict | None) -> Intent:
    """
    JSON(dict) から Domain(Intent) を生成する
    JSONが未定義・欠損していても安全に生成される
    """

    if not intent_json:
        return Intent(
            genre="",
            theme_or_claim="",
            core_values="",
            constraints=[]
        )

    # 互換対応（旧キー values）
    core_values = (
        intent_json.get("core_values")
        if "core_values" in intent_json
        else intent_json.get("values", "")
    )

    constraints = intent_json.get("constraints", [])

    # 型安全化
    if not isinstance(constraints, list):
        constraints = []

    return Intent(
        genre=intent_json.get("genre", ""),
        theme_or_claim=intent_json.get("theme_or_claim", ""),
        core_values=core_values or "",
        constraints=[str(c) for c in constraints if c]
    )
