from typing import List, Dict, Any


def expand_weights(
    global_label_order: List[tuple], category_weights: Dict[str, float]
) -> List[float]:
    """
    (classification, label) のタプル形式の順序リストに基づき、
    各次元に対する重みを生成する。
    """
    expanded = []
    for classification, _ in global_label_order:
        expanded.append(category_weights.get(classification, 1.0))
    return expanded


def weighted_l2(candidate: List[int], target: List[int], weights: List[float]) -> float:
    """
    重み付きL2距離（2乗和）を計算する。
    """
    return sum(w * (c - t) ** 2 for c, t, w in zip(candidate, target, weights))


def apply_tolerance(distance: float, tolerance: float) -> float:
    """
    許容範囲（tolerance）を適用する。
    """
    if distance <= tolerance:
        return 0.0
    return distance - tolerance


def evaluate(
    candidate_vec: List[int],
    target_vec: List[int],
    category_weights: Dict[str, float],
    label_order: List[tuple],
    tolerance: float = 0.0,
):
    """
    統合評価関数。
    """
    weights = expand_weights(label_order, category_weights)
    raw_distance = weighted_l2(candidate_vec, target_vec, weights)
    adjusted = apply_tolerance(raw_distance, tolerance)

    return {"raw_distance": raw_distance, "adjusted_distance": adjusted}


def calculate_energy_detail(
    selected_items: List[Dict[str, Any]],
    evaluation_config: Dict[str, Any],
    schema: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    一次項（E1）と二次項（E2）を分解したエネルギー計算。
    selected_items: [ {category, element, labels: {ja_label: val}}, ... ]
    """
    # 0. スキーマを用いた日本語ラベル <-> 英語キーの相互変換マップ作成
    reverse_schema = {}
    forward_schema = {}  # 英語 -> 日本語用
    if schema:
        for cat_name, labels_spec in schema.items():
            if cat_name == "scale":
                continue
            reverse_schema[cat_name] = {}
            forward_schema[cat_name] = {}
            for en_key, spec in labels_spec.items():
                ja_label = spec.get("ja_label")
                if ja_label:
                    reverse_schema[cat_name][ja_label] = en_key
                    forward_schema[cat_name][en_key] = ja_label

    # 1. 相互作用計算用のフラットラベル作成 (最新の選択状態を反映)
    # E2計算では「どの要素が選ばれているか」の全体像が必要
    flat_labels_for_e2 = {}
    for item in selected_items:
        category = item["category"]
        labels = item["labels"]
        for label, value in labels.items():
            en_key = label
            if category in reverse_schema and label in reverse_schema[category]:
                en_key = reverse_schema[category][label]

            # E2用のキー: category::en_key
            flat_labels_for_e2[f"{category}::{en_key}"] = value

    # 設定の取得
    target_values = evaluation_config.get(
        "target_values", evaluation_config.get("targets", {})
    )
    weights = evaluation_config.get("weights", {})
    category_weights = evaluation_config.get("category_weights", {})
    interactions = evaluation_config.get("interactions", [])

    # 2. & 3. 一次項計算 (E1) - 渡されたアイテムの順序を維持
    E1 = 0.0
    E1_details = []

    for item in selected_items:
        category = item["category"]
        element = item["element"]
        labels = item["labels"]

        item_e1 = 0.0
        for ja_label, value in labels.items():
            # 英語キーに変換してターゲットを探す
            en_key = ja_label
            if category in reverse_schema and ja_label in reverse_schema[category]:
                en_key = reverse_schema[category][ja_label]

            full_key = f"{category}::{en_key}"
            if full_key not in target_values:
                continue

            target = target_values[full_key]

            # 重みの決定
            weight = weights.get(full_key)
            if weight is None:
                weight = category_weights.get(category, 1.0)

            # 正規化
            x_norm = value / 4.0
            t_norm = target / 4.0

            contribution = weight * ((x_norm - t_norm) ** 2)
            item_e1 += contribution

        E1 += item_e1
        E1_details.append(
            {"key": f"{category} > {element}", "value": round(item_e1, 4)}
        )

    # 4. 二次項計算 (E2)
    E2 = 0.0
    E2_details = []

    INTERACTION_WEIGHT = 0.1  # バランス補正係数 (案C)

    for inter in interactions:
        key_a = inter.get("key_a")
        key_b = inter.get("key_b")
        strength = inter.get("strength", 0.0)

        if key_a in flat_labels_for_e2 and key_b in flat_labels_for_e2:
            val_a = flat_labels_for_e2[key_a]
            val_b = flat_labels_for_e2[key_b]

            # 正規化
            x_norm_a = val_a / 4.0
            x_norm_b = val_b / 4.0

            contribution = strength * x_norm_a * x_norm_b
            # 補正係数を適用
            final_contribution = contribution * INTERACTION_WEIGHT
            E2 += final_contribution

            # キーを日本語ラベルに変換 (表示用)
            display_a = key_a
            display_b = key_b

            if "::" in key_a:
                cat_a, en_a = key_a.split("::", 1)
                if cat_a in forward_schema and en_a in forward_schema[cat_a]:
                    display_a = f"{cat_a}::{forward_schema[cat_a][en_a]}"

            if "::" in key_b:
                cat_b, en_b = key_b.split("::", 1)
                if cat_b in forward_schema and en_b in forward_schema[cat_b]:
                    display_b = f"{cat_b}::{forward_schema[cat_b][en_b]}"

            E2_details.append(
                {
                    "key_a": display_a,
                    "key_b": display_b,
                    "value": round(final_contribution, 4),
                }
            )

    # 5. 総エネルギー
    total = E1 + E2

    # 6. 割合計算
    if total != 0:
        ratio1 = (E1 / total) * 100
        ratio2 = (E2 / total) * 100
    else:
        ratio1 = 0.0
        ratio2 = 0.0

    # 7. 詳細割合計算
    # 一次項
    for item in E1_details:
        if E1 != 0:
            item["ratio"] = round((item["value"] / E1) * 100, 2)
        else:
            item["ratio"] = 0.0

    # 二次項
    abs_sum_e2 = sum(abs(item["value"]) for item in E2_details)
    for item in E2_details:
        if abs_sum_e2 > 0:
            item["ratio"] = round((abs(item["value"]) / abs_sum_e2) * 100, 2)
        else:
            item["ratio"] = 0.0

    # 9. ソート (E1, E2 ともに入力順・定義順を維持するためソートしない)
    # E1_details.sort(key=lambda x: x["value"], reverse=True)
    # E2_details.sort(key=lambda x: abs(x["value"]), reverse=True) # 削除：定義順を維持する

    # 10. 戻り値
    return {
        "total": round(total, 4),
        "E1": round(E1, 4),
        "E2": round(E2, 4),
        "ratio1": round(ratio1, 2),
        "ratio2": round(ratio2, 2),
        "E1_details": E1_details,
        "E2_details": E2_details,
    }
