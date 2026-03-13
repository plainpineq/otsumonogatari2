from typing import Dict, Any, Optional


def build_draft_prompt(
    structure_snapshot: Dict[str, Any], additional_info: Optional[str] = None
) -> str:
    """
    小説のシーンや章の下書きを生成するためのプロンプトを構築します。
    structure_snapshot には、シーン構成、キャラクター、伏線、場所、時間、天候、
    視点、文体、文字数、補足指示などの情報が含まれることを想定しています。
    """

    scene_structure = structure_snapshot.get("scene_structure", "指定なし")
    characters = structure_snapshot.get("characters", [])
    foreshadowing = structure_snapshot.get("foreshadowing", [])
    location = structure_snapshot.get("location", "指定なし")
    time_setting = structure_snapshot.get("time", "指定なし")
    weather = structure_snapshot.get("weather", "指定なし")
    viewpoint = structure_snapshot.get("viewpoint", "三人称多視点")
    style = structure_snapshot.get("style", "標準的な小説体")
    length = structure_snapshot.get("length", "指定なし")
    instructions = structure_snapshot.get("instructions", "")

    # キャラクター情報の整形
    char_info = ""
    for char in characters:
        name = char.get("name", "不明")
        role = char.get("role", "")
        desc = char.get("description", "")
        char_info += f"- {name} ({role}): {desc}\n"
    if not char_info:
        char_info = "特になし"

    # 伏線情報の整形
    fores_info = ""
    for f in foreshadowing:
        fores_info += f"- {f}\n"
    if not fores_info:
        fores_info = "特になし"

    prompt = f"""あなたはプロの小説家として、以下の指示に基づき、物語の具体的な下書き（ドラフト）を執筆してください。

### 指示：
- 日本語で執筆してください。
- 小説としての情緒、情景描写、キャラクターの心情を重視してください。
- 指定された視点と文体を厳守してください。

### シーン設定：
- **場所**: {location}
- **時間**: {time_setting}
- **天候**: {weather}
- **視点**: {viewpoint}
- **文体**: {style}
- **目標文字数**: {length}

### 登場キャラクター：
{char_info}

### 伏線・要素：
{fores_info}

### シーン構成・展開：
{scene_structure}

### 補足指示：
{instructions}
{additional_info if additional_info else ""}

---

それでは、上記の内容を反映した、高品質な小説本文を執筆してください。
"""
    return prompt.strip()
