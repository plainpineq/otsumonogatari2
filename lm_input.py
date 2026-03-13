# llm_input.py
import json
import os
from user_files import get_user_data_path
from typing import Optional
from services.services import (
    DEFAULT_COMPOSITION_META,
)  # DEFAULT_COMPOSITION_META をインポート
from ui_labels import UI_LABELS

# ... existing code ...


def _get_formatted_intent_text(document: dict) -> str:
    """
    Format intent fields and genre_config into a readable string for prompts.
    """
    intent_fields = document.get("intent", {}).get("fields", {})
    genre_cfg = document.get("genre_config", {})

    formatted_intent = ""

    # 1. Genre Configuration
    main_genre = genre_cfg.get("main")
    sub_genres = genre_cfg.get("sub", [])
    if main_genre:
        formatted_intent += f"- 主ジャンル: {main_genre}\n"
    if sub_genres:
        formatted_intent += f"- 副ジャンル: {', '.join(sub_genres)}\n"

    # 2. Other Intent Fields
    for key, field in intent_fields.items():
        if field.get("label") and field.get("value"):
            formatted_intent += f"- {field['label']}: {field['value']}\n"

    if not formatted_intent:
        formatted_intent = "（基本設定・作者の意図は特に指定されていません）"
    return formatted_intent


def build_composition_ideas_prompt(
    document: dict,
    composition_meta: dict,
    user_id: str,
    target_category_label: Optional[str] = None,
    suffix: str = "",
    suggestion_count: int = 3,
) -> str:
    """
    Builds a prompt for the LLM to generate composition element suggestions.
    If target_category_label is provided, the prompt will be specific to that category.
    The suffix is used for naming generated files.
    """
    doc_type_label = document.get("doc_type", "不明")
    document_title = document.get("title", "不明なドキュメント")

    # Map doc_type_label (e.g., "小説") to its internal ID (e.g., "novel")
    doc_type_mapping = {
        meta["label"]: doc_id for doc_id, meta in composition_meta["doc_types"].items()
    }
    doc_type_id = doc_type_mapping.get(
        doc_type_label, "default"
    )  # Use "default" as a fallback ID if not found

    # Dynamically determine template file path
    template_file_name = f"{doc_type_id}.md"
    template_file_path = os.path.join("prompt_templates", template_file_name)

    # --- DEBUG PRINTS ---
    print(f"--- Debugging build_composition_ideas_prompt ---")
    print(f"Received target_category_label: '{target_category_label}'")
    # --- END DEBUG PRINTS ---

    # Fallback to default.md if doc_type specific template does not exist
    if not os.path.exists(template_file_path):
        template_file_path = os.path.join("prompt_templates", "default.md")

    with open(template_file_path, "r", encoding="utf-8") as f:
        template_content = f.read()

    formatted_intent = _get_formatted_intent_text(document)

    # Extract composition elements based on all defined categories and format them for the prompt
    elements_text = ""
    has_elements = False

    # Find the target category within the flattened document["composition_elements"]["categories"]
    target_category_data = None
    all_document_categories = document.get("composition_elements", {}).get(
        "categories", []
    )  # 新しいパス

    for category in all_document_categories:
        if category.get("label") == target_category_label:
            target_category_data = category
            break

    if target_category_data and target_category_data.get("elements"):
        elements_text += f"- 分類名: {target_category_data['label']}\n"
        for element in target_category_data["elements"]:
            if element.get("label"):
                elements_text += f"  - 要素名: {element['label']}\n"
                has_elements = True

    if not has_elements:
        elements_text = "（構成要素は定義されていません）\n"

    # Generate the dynamic JSON example for the prompt
    # Pass target_category_label so example is also specific to the category
    dynamic_json_example = _build_dynamic_json_example(
        document,
        target_category_label=target_category_label,
        suggestion_count=suggestion_count,
    )

    # Fill template placeholders
    prompt = template_content.format(
        document_title=document_title,
        doc_type=doc_type_label,
        intent_text=formatted_intent,
        elements_text=elements_text,
        dynamic_json_example=dynamic_json_example,
        suggestion_count=suggestion_count,
        confirmed_plot=f"確定済みの題名: {document.get('selected_title', '未設定')}\n確定済みのプロット: {document.get('selected_plot', '未設定')}",
    )

    # Output the generated prompt to a file for debugging/verification
    user_data_dir = get_user_data_path(user_id)
    os.makedirs(user_data_dir, exist_ok=True)  # Ensure the directory exists
    output_file_path = os.path.join(user_data_dir, f"generated_prompt{suffix}.md")
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Generated prompt written to: {output_file_path}")
    except Exception as e:
        print(f"Error writing prompt to file: {e}")

    return prompt


def _get_composition_elements(doc_type_id: str, composition_meta: dict) -> list[str]:
    """
    Extracts a list of composition element labels for a given document type
    from the composition_meta dictionary.
    """
    elements = []
    doc_type_data = composition_meta.get("doc_types", {}).get(doc_type_id)
    if doc_type_data and doc_type_data.get("categories"):
        for category in doc_type_data["categories"]:
            if category.get("elements"):
                for element in category["elements"]:
                    if element.get("label"):
                        elements.append(element["label"])
    return elements


def mock_llm_call(prompt: str, suggestion_count: int = 3) -> dict:
    """
    Mocks an LLM call, returning hardcoded suggestions based on parsed elements.
    In a real scenario, this would call an actual LLM API.
    """
    print(f"Mock LLM called with prompt: {prompt[:200]}...")  # Log part of the prompt

    mock_suggestions = {"suggestions": {}}

    # Extract element labels from the prompt
    elements_section_start = prompt.find("以下に挙げる各構成要素について")
    if elements_section_start != -1:
        elements_section = prompt[elements_section_start:]
        for line in elements_section.split("\n"):
            if line.strip().startswith("- "):
                element_label = line.strip()[2:].strip()
                if element_label:
                    mock_suggestions["suggestions"][element_label] = [
                        f"{element_label}の候補{i+1}" for i in range(suggestion_count)
                    ]

        # If no specific elements are found or parsed
        mock_suggestions["suggestions"]["汎用構成要素"] = [
            f"汎用アイデア{i+1}" for i in range(suggestion_count)
        ]

    return mock_suggestions


def _build_dynamic_json_example(
    document: dict,
    target_category_label: Optional[str] = None,
    suggestion_count: int = 3,
) -> str:
    """
    Generates a dynamic JSON example string based on the document's composition elements.
    If target_category_label is provided, the example will be specific to that category.
    This example serves as a strong few-shot example for the LLM.
    """
    dynamic_suggestions_list = []

    all_document_categories = document.get("composition_elements", {}).get(
        "categories", []
    )  # 新しいパス

    for category_obj in all_document_categories:
        category_name = category_obj.get("label")
        if not category_name:
            continue

        # Filter by target_category_label if provided
        if target_category_label and category_name != target_category_label:
            continue

        elements_dict = {}
        elements_in_category = category_obj.get("elements")
        if elements_in_category:
            for element_obj in elements_in_category:
                element_label = element_obj.get("label")
                if element_label:
                    # Use generic suggestion placeholders
                    elements_dict[element_label] = [
                        f"提案{i+1}" for i in range(suggestion_count)
                    ]

        # Only add category if it has elements
        if elements_dict:
            dynamic_suggestions_list.append(
                {"category": category_name, "elements": elements_dict}
            )

    # Wrap in the final "suggestions" structure
    final_json_structure = {"suggestions": dynamic_suggestions_list}

    # Generate JSON string with proper indentation and Japanese character handling
    return json.dumps(final_json_structure, indent=2, ensure_ascii=False)


def build_title_plot_proposals_prompt(
    document: dict,
    composition_meta: dict,
    user_id: str,
    suffix: str = "",
    suggestion_count: int = 3,
) -> str:
    """
    Builds a prompt for the LLM to generate initial title and plot proposals based on
    the "base" category elements in document["composition_elements"].
    """
    template_file_path = os.path.join(
        "prompt_templates", "novel_title_plot_proposals.md"
    )

    with open(template_file_path, "r", encoding="utf-8") as f:
        template_content = f.read()

    intent_fields = document.get("intent", {}).get("fields", {})
    intent_dict = {
        field.get("label"): field.get("value")
        for key, field in intent_fields.items()
        if field.get("label") and field.get("value")
    }

    # Add genre config to the dict
    genre_cfg = document.get("genre_config", {})
    if genre_cfg.get("main"):
        intent_dict["主ジャンル"] = genre_cfg["main"]
    if genre_cfg.get("sub"):
        intent_dict["副ジャンル"] = ", ".join(genre_cfg["sub"])

    formatted_intent_json = json.dumps(intent_dict, indent=2, ensure_ascii=False)

    # Current Composition Elements for context
    all_document_categories = document.get("composition_elements", {}).get(
        "categories", []
    )

    # Additional Instruction for the "基本設定" category
    additional_instruction = ""
    base_category_label = "基本項目"
    for category_obj in all_document_categories:
        if category_obj.get("label") == base_category_label:
            additional_instruction = category_obj.get(
                "additional_instruction", ""
            ).strip()
            break

    formatted_additional_instruction = ""
    if additional_instruction:
        lines = [
            line.strip() for line in additional_instruction.split("\n") if line.strip()
        ]
        formatted_additional_instruction = (
            "\n".join([f"- {line}" for line in lines]) + "\n"
        )

    # --- Extract "base" category elements from document["composition_elements"] for the prompt ---
    elements_text = ""
    dynamic_json_example_for_base = {}
    base_category_label = "基本項目"  # composition_meta.json で定義されているラベル (変更なし)

    # document["composition_elements"]["categories"] から "base" カテゴリを探す
    base_category_data = None
    if (
        "composition_elements" in document
        and "categories" in document["composition_elements"]
    ):
        for category in document["composition_elements"]["categories"]:
            # id が "base" または label が "基本設定" のカテゴリを探す
            if (
                category.get("id") == "base"
                or category.get("label") == base_category_label
            ):
                base_category_data = category
                break

    if base_category_data and base_category_data.get("elements"):
        elements_text += f"- 分類名: {base_category_data['label']}\n"
        elements_dict = {}
        for element in base_category_data["elements"]:
            if element.get("label"):
                elements_text += f"  - 要素名: {element['label']}\n"
                elements_dict[element["label"]] = [
                    f"提案{i+1}" for i in range(suggestion_count)
                ]

        dynamic_json_example_for_base = {
            "suggestions": [
                {"category": base_category_label, "elements": elements_dict}
            ]
        }

    if not elements_text:
        elements_text = "（基本項目は定義されていません）\n"
        dynamic_json_example_for_base = {
            "suggestions": [{"category": base_category_label, "elements": {}}]
        }

    # Fill template placeholders
    prompt = template_content.format(
        document_title=document.get("title", "無題"),
        intent=formatted_intent_json,
        suggestion_count=str(suggestion_count),
        elements_text=elements_text,
        dynamic_json_example=json.dumps(
            dynamic_json_example_for_base, indent=2, ensure_ascii=False
        ),
        additional_instruction_text=formatted_additional_instruction,
    )

    return prompt


def _build_dynamic_category_json_example(
    document: dict, category_label: str, suggestion_count: int
) -> str:
    """
    Generates a dynamic JSON example string for a specific category composition prompt.
    This example serves as a strong few-shot example for the LLM.
    """
    elements_dict = {}
    all_document_categories = document.get("composition_elements", {}).get(
        "categories", []
    )

    for category_obj in all_document_categories:
        if category_obj.get("label") == category_label:
            elements_in_category = category_obj.get("elements")
            if elements_in_category:
                for element_obj in elements_in_category:
                    element_label = element_obj.get("label")
                    if element_label:
                        # Generate suggestion_count number of example suggestions
                        elements_dict[element_label] = [
                            f"項目1の提案内容{i+1}" for i in range(suggestion_count)
                        ]
            break  # Found the category, exit loop

    final_json_structure = {"category": category_label, "elements": elements_dict}

    # Generate JSON string with proper indentation and Japanese character handling
    return json.dumps(final_json_structure, indent=2, ensure_ascii=False)


def build_category_composition_prompt(
    document: dict,
    composition_meta: dict,
    user_id: str,
    category_label: str,
    suffix: str = "",
    suggestion_count: int = 3,
) -> str:
    """
    Builds a prompt for the LLM to generate content for a specific composition category.
    """
    template_file_path = os.path.join(
        "prompt_templates", "novel_category_composition_template.md"
    )

    with open(template_file_path, "r", encoding="utf-8") as f:
        template_content = f.read()

    document_title = document.get("title", "不明なドキュメント")
    doc_type_label = document.get("doc_type", "不明")

    # Current Composition Elements for context (moved to top for scope)
    all_document_categories = document.get("composition_elements", {}).get(
        "categories", []
    )

    # Author's Intent / Basic Settings
    formatted_intent = _get_formatted_intent_text(document)

    # Selected Basic Elements
    basic_settings_text = ""
    selected_basic_elements = document.get("selected_basic_elements", {})

    # Get the label map for basic settings from composition_elements
    basic_settings_label_map = {}
    base_category_label_id = "base"  # Assuming "base" is the ID for "基本設定"

    # Additional Instruction for the current category
    additional_instruction = ""
    for category_obj in all_document_categories:
        if category_obj.get("label") == category_label:
            additional_instruction = category_obj.get(
                "additional_instruction", ""
            ).strip()
            break

    formatted_additional_instruction = ""
    if additional_instruction:
        lines = [
            line.strip() for line in additional_instruction.split("\n") if line.strip()
        ]
        formatted_additional_instruction = (
            "\n".join([f"- {line}" for line in lines]) + "\n"
        )

    for category_obj in all_document_categories:
        if (
            category_obj.get("id") == base_category_label_id
            or category_obj.get("label") == "基本設定"
        ):
            if category_obj.get("elements"):
                for element in category_obj["elements"]:
                    if element.get("id") and element.get("label"):
                        basic_settings_label_map[element["id"]] = element["label"]
            break  # Found the base category, no need to continue

    if selected_basic_elements:
        for label_id, value in selected_basic_elements.items():
            # Use the mapped label, or fallback to the original id if not found
            display_label = basic_settings_label_map.get(label_id, label_id)
            basic_settings_text += f"- {display_label}: {value}\n"
    if not basic_settings_text:
        basic_settings_text = "（確定済みの基本設定は特にありません）"

    elements_text = ""  # Initialize elements_text here
    for category_obj in all_document_categories:
        if category_obj.get("label") == category_label:
            elements_text += f"- 分類名: {category_obj['label']}\n"
            if category_obj.get("elements"):
                for element in category_obj["elements"]:
                    if element.get("label"):
                        elements_text += f"  - 要素名: {element['label']}\n"
            break
    if not elements_text:
        elements_text = f"（{category_label}に属する構成要素は定義されていません）"
    # Generate dynamic JSON example based on suggestion_count
    dynamic_json_example = _build_dynamic_category_json_example(
        document, category_label, suggestion_count
    )

    # Fill template placeholders
    prompt = template_content.format(
        document_title=document_title,
        doc_type=doc_type_label,
        intent_text=formatted_intent,
        basic_settings_text=basic_settings_text,
        elements_text=elements_text,
        category_label=category_label,  # Pass the specific category label to the prompt
        suggestion_count=suggestion_count,  # Add suggestion_count
        dynamic_json_example=dynamic_json_example,  # Add dynamic JSON example
        additional_instruction_text=formatted_additional_instruction,  # Add additional instruction
    )

    # Output the generated prompt to a file for debugging/verification
    user_data_dir = get_user_data_path(user_id)
    os.makedirs(user_data_dir, exist_ok=True)
    output_file_path = os.path.join(
        user_data_dir,
        f"generated_prompt_{category_label.replace(' ', '_')}_{suffix}.md",
    )
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Generated category composition prompt written to: {output_file_path}")
    except Exception as e:
        print(f"Error writing category composition prompt to file: {e}")

    return prompt
