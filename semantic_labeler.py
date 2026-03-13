import json
import time
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime

from user_files import get_user_data_path
from services.llm_client import call_llm


class SemanticLabeler:
    """
    LLMを使用してテキストに意味ラベルを付与し、検証するクラス。
    LLMからは英語キーで受け取り、内部で日本語ラベルに変換する。
    評価範囲: 0-4
    """

    def __init__(self, config_path="prompt_templates/semantic_label_schema.json"):
        try:
            with open(
                "prompt_templates/classification_batch_evaluation.md",
                "r",
                encoding="utf-8",
            ) as f:
                self.batch_evaluation_prompt_template = f.read()
        except FileNotFoundError:
            logging.error("バッチ評価プロンプトファイルが見つかりません。")
            self.batch_evaluation_prompt_template = ""

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"設定ファイル '{config_path}' の読み込みに失敗しました: {e}")

        self._initialize_validator()

    def _initialize_validator(self):
        """
        設定ファイルからバリデーション情報とEn->Jaマッピングを生成する。
        """
        self.classification_specs = {}

        for classification_name, labels_config in self.config.items():
            if classification_name == "scale":
                continue

            required_keys = set()
            evaluation_items_definitions = []
            label_mappings = {}

            for en_key, spec in labels_config.items():
                ja_label = spec.get("ja_label", en_key)
                label_mappings[en_key] = ja_label
                required_keys.add(en_key)

                description = spec.get("description", "")
                evaluation_items_definitions.append(
                    f"- `{en_key}` ({ja_label} - {description}): [0-4 の整数値]"
                )

            self.classification_specs[classification_name] = {
                "required_keys": required_keys,
                "label_mappings": label_mappings,
                "evaluation_items_definitions": "\n".join(evaluation_items_definitions),
            }

    def _validate_labels(
        self, logger: logging.Logger, classification_name: str, labels: Dict[str, Any]
    ) -> bool:
        """
        LLMからの応答(英語キー)を検証し、必要に応じて補正する。
        """
        if classification_name not in self.classification_specs:
            return False

        specs = self.classification_specs[classification_name]
        required_keys = specs["required_keys"]

        # 1. 欠損ラベルの補正 (欠損時は 0 を設定)
        for en_key in required_keys:
            if en_key not in labels:
                logger.warning(f"ラベル欠損補正: '{en_key}' が見つかりません。0 を設定します。")
                labels[en_key] = 0

        # 2. 値の検証と外れ値の補正
        for en_key in list(labels.keys()):
            if en_key not in required_keys:
                continue  # 定義外のキーは無視

            try:
                val_int = int(labels[en_key])
                if val_int > 4:
                    logger.warning(f"外れ値補正: '{en_key}' の値 {val_int} を 4 に丸めます。")
                    labels[en_key] = 4
                elif val_int < 0:
                    logger.warning(f"外れ値補正: '{en_key}' の値 {val_int} を 0 に丸めます。")
                    labels[en_key] = 0
                else:
                    labels[en_key] = val_int
            except (ValueError, TypeError):
                logger.warning(
                    f"型エラー補正: '{en_key}' の値 '{labels[en_key]}' が数値ではありません。0 を設定します。"
                )
                labels[en_key] = 0

        return True

    def _evaluate_classification_candidates_with_llm(
        self,
        logger: logging.Logger,
        classification_name: str,
        candidates: Dict[str, List[str]],
        llm_config: Dict[str, Any],
        user_id: str,
        max_retries: int = 3,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        if classification_name not in self.classification_specs:
            logger.error(f"未知の分類名 '{classification_name}' の評価はできません。")
            return None
        if not self.batch_evaluation_prompt_template:
            logger.error("プロンプトテンプレートがありません。")
            return None

        specs = self.classification_specs[classification_name]
        evaluation_items_definitions = specs["evaluation_items_definitions"]

        output_json_example_dict = {}
        first_el = next(iter(candidates.keys()), "要素1")
        num_proposals = len(candidates.get(first_el, [])) or 1
        example_labels = {en_key: 2 for en_key in specs["required_keys"]}
        output_json_example_dict[first_el] = [example_labels] * num_proposals

        full_prompt = self.batch_evaluation_prompt_template.format(
            classification_name=classification_name,
            evaluation_items_definitions=evaluation_items_definitions,
            candidates_json=json.dumps(candidates, indent=2, ensure_ascii=False),
            output_json_example=json.dumps(
                output_json_example_dict, indent=2, ensure_ascii=False
            ),
        )

        user_data_dir = get_user_data_path(user_id)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        prompt_file = os.path.join(
            user_data_dir,
            f"generated_evaluation_prompt_{classification_name.replace(' ', '_')}_{timestamp}.md",
        )
        try:
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(full_prompt)
            logger.info(f"Generated evaluation prompt written to: {prompt_file}")
        except Exception as e:
            logger.error(f"Error writing evaluation prompt: {e}")

        logger.info(f"--- LLMに送信中 ({classification_name}の候補群) ---")
        logger.info(f"[PROMPT]\n{full_prompt}")

        for attempt in range(max_retries):
            try:
                _, parsed_res = call_llm(
                    api_key=llm_config["api_key"],
                    model_name=llm_config["model_name"],
                    prompt=full_prompt,
                    llm_provider=llm_config["provider"],
                    base_url=llm_config["base_url"],
                )

                logger.info(
                    f"[LLM RESPONSE] (Attempt {attempt+1})\n{json.dumps(parsed_res, indent=2, ensure_ascii=False)}"
                )

                if not isinstance(parsed_res, dict):
                    logger.warning("検証エラー: LLM応答が辞書形式ではありません。")
                    continue

                all_valid = True
                validated = {}
                for el_name, evals in parsed_res.items():
                    if el_name not in candidates:
                        logger.warning(f"検証エラー: 未知の要素名 '{el_name}' が含まれています。")
                        continue
                    if not isinstance(evals, list):
                        logger.warning(f"検証エラー: 要素 '{el_name}' の評価がリスト形式ではありません。")
                        all_valid = False
                        break
                    for i, labels_obj in enumerate(evals):
                        if not self._validate_labels(
                            logger, classification_name, labels_obj
                        ):
                            logger.warning(f"検証エラー: 要素 '{el_name}' 案 {i+1} の検証に失敗しました。")
                            all_valid = False
                            break
                    if not all_valid:
                        break
                    validated[el_name] = evals

                if all_valid and validated:
                    logger.info(f"✅ 分類 '{classification_name}' のバッチラベル取得成功")
                    return validated
                else:
                    logger.warning(
                        f"⚠️ 分類 '{classification_name}' の検証失敗 (試行 {attempt + 1}/{max_retries})"
                    )

            except Exception as e:
                logger.error(f"❌ エラー (試行 {attempt + 1}/{max_retries}): {e}")
            time.sleep(1)

        logger.error(f"❌ {max_retries}回のリトライに失敗しました。この分類の処理をスキップします。")
        return None


def label_suggestions(
    input_data: Dict[str, Any],
    llm_config: Dict[str, Any],
    user_id: str,
    log_file_path: Optional[str] = None,
):
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fh = None
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    logger.info(f"\n==================================================")
    logger.info(f" 評価処理開始: {datetime.now().isoformat()}")
    logger.info(
        f" LLM設定: {llm_config.get('provider')} / {llm_config.get('model_name')}"
    )
    logger.info(f"==================================================")

    try:
        labeler = SemanticLabeler()
    except ValueError as e:
        logger.error(f"初期化エラー: {e}")
        yield {"event": "error", "message": str(e)}
        return

    suggestions = input_data.get("llm_suggestions", [])
    if not suggestions:
        logger.warning("llm_suggestions が空です。")
        return

    all_batches = []
    total_to_process = 0
    for group in suggestions:
        cat = group.get("category", "不明な分類")
        if cat not in labeler.classification_specs:
            logger.warning(f"未知の分類 '{cat}' をスキップします。")
            continue
        els = group.get("elements", {})
        batch_candidates = {}
        for name, texts in els.items():
            texts_list = texts if isinstance(texts, list) else [texts]
            batch_candidates[name] = texts_list
            total_to_process += len(texts_list)
        if batch_candidates:
            all_batches.append({"name": cat, "candidates": batch_candidates})

    yield {"event": "total_items", "count": total_to_process}
    current_count = 0

    for batch in all_batches:
        cat = batch["name"]
        logger.info(
            f"\n--- 処理開始: 分類 '{cat}' の候補群 (要素数: {len(batch['candidates'])}) ---"
        )
        results = labeler._evaluate_classification_candidates_with_llm(
            logger, cat, batch["candidates"], llm_config, user_id
        )

        label_map = labeler.classification_specs[cat]["label_mappings"]

        if results:
            for el_name, evals_en in results.items():
                original = []
                for group in suggestions:
                    if group.get("category") == cat:
                        original = group.get("elements", {}).get(el_name, [])
                        if isinstance(original, str):
                            original = [original]
                        break

                # Convert English keys to Japanese labels
                transformed_evals_ja = []
                for labels_en in evals_en:
                    transformed_evals_ja.append(
                        {label_map.get(k, k): v for k, v in labels_en.items()}
                    )

                labeled = {
                    "category": cat,
                    "element": el_name,
                    "text": "\n".join(original),
                    "labels": transformed_evals_ja,
                }
                yield {"event": "semantic_label", "data": labeled}
                current_count += 1
                yield {
                    "event": "progress",
                    "progress_current": current_count,
                    "progress_total": total_to_process,
                    "category_label": cat,
                    "current_element": el_name,
                }
        else:
            logger.error(f"分類 '{cat}' の評価結果取得に失敗しました。ダミー結果を生成します。")
            dummy_ja = {ja: 0 for ja in label_map.values()}
            for el_name in batch["candidates"].keys():
                yield {
                    "event": "semantic_label",
                    "data": {
                        "category": cat,
                        "element": el_name,
                        "labels": [dummy_ja],
                        "status": "failed",
                    },
                }
                current_count += 1
                yield {
                    "event": "progress",
                    "progress_current": current_count,
                    "progress_total": total_to_process,
                    "category_label": cat,
                    "current_element": el_name,
                }

    logger.info(f"\n==================================================")
    logger.info(f" 全ての意味ラベル付け処理が完了しました。")
    logger.info(f" 完了時刻: {datetime.now().isoformat()}")
    logger.info(f"==================================================\n")

    if fh:
        fh.close()
        logger.removeHandler(fh)
