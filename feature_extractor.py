import json
from typing import Dict, Any


class FeatureExtractor:
    """
    既存 classification を変更せず、
    5段階評価（0-4）へ対応した安全版
    """

    def __init__(
        self, schema_path: str = "prompt_templates/semantic_label_schema.json"
    ):
        self.schema_path = schema_path
        self.schema = self._load_schema()
        self._build_global_label_index()

    # --------------------------
    # スキーマ読み込み
    # --------------------------
    def _load_schema(self):
        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return schema

    def _build_global_label_index(self):
        self.global_label_order = []
        self.global_index_map = {}
        index = 0

        # schema ルートの 'scale' 等を除外して分類のみをループ
        for classification in self.schema.keys():
            if classification == "scale":
                continue

            # 各分類のラベルを取得（日本語ラベル ja_label ではなく、キー名 dramatic, causal 等を使用）
            labels = list(self.schema[classification].keys())
            for label in labels:
                if label == "scale":
                    continue  # 分類内個別スケールがあれば除外

                key = (classification, label)
                self.global_label_order.append(key)
                self.global_index_map[key] = index
                index += 1

    def to_global_vector(self, feature_data: Dict[str, Any]) -> list:
        """
        全分類・全ラベルを含む固定次元のベクトルを返す。
        """
        vector = [0] * len(self.global_label_order)

        classification = feature_data.get("classification")
        scalar_features = feature_data.get("scalar_features", {})

        if not classification:
            return vector

        for label, value in scalar_features.items():
            key = (classification, label)
            if key in self.global_index_map:
                index = self.global_index_map[key]
                vector[index] = value

        return vector

    def get_global_dimension(self) -> int:
        """
        グローバルベクトルの次元数を返す。
        """
        return len(self.global_label_order)

    # --------------------------
    # ラベル取得（既存分類のみ）
    # --------------------------
    def get_labels(self, classification: str):
        if classification not in self.schema:
            raise KeyError(f"schemaに存在しないclassificationです: {classification}")

        return list(self.schema[classification].keys())

    # --------------------------
    # スケール取得（0-4固定）
    # --------------------------
    def get_scale(self, classification: str = None):
        # グローバルスケールまたは分類別スケールを返す
        if (
            classification
            and classification in self.schema
            and "scale" in self.schema[classification]
        ):
            return self.schema[classification]["scale"]
        return self.schema.get("scale", {"min": 0, "max": 4})

    # --------------------------
    # 値バリデーション（0-4）
    # --------------------------
    def _validate_value(self, value: Any, classification: str) -> int:
        scale = self.get_scale(classification)
        min_v = scale["min"]
        max_v = scale["max"]

        try:
            value = int(value)
        except Exception:
            return min_v

        return max(min_v, min(max_v, value))

    # --------------------------
    # 特徴抽出（分類構造はそのまま）
    # --------------------------
    def extract_features(self, element_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        element_data 例:

        {
            "element": "村の掟",
            "classification": "世界観",
            "evaluation": {
                "緊張感": 3,
                "謎の強度": 2
            }
        }
        """

        classification = element_data.get("classification")

        if not classification:
            raise ValueError("classification が指定されていません")

        labels = self.get_labels(classification)
        raw_eval = element_data.get("evaluation", {})

        scalar_features = {}

        for label in labels:
            raw_value = raw_eval.get(label, 0)
            scalar_features[label] = self._validate_value(raw_value, classification)

        return {
            "element": element_data.get("element"),
            "classification": classification,
            "scalar_features": scalar_features,
        }

    # --------------------------
    # ベクトル化（分類内のみ）
    # --------------------------
    def to_vector(self, feature_data: Dict[str, Any]) -> list:
        classification = feature_data["classification"]
        labels = self.get_labels(classification)

        return [feature_data["scalar_features"].get(label, 0) for label in labels]
