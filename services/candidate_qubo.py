from typing import List, Dict, Any, Tuple
import logging

def generate_candidate_selection_qubo(
    semantic_labels: List[Dict[str, Any]],
    evaluation_config: Dict[str, Any],
    label_mapping: Dict[str, Dict[str, str]]
) -> Tuple[Dict[Tuple[int, int], float], List[Dict[str, Any]]]:
    """
    候補選択型QUBOの構築。
    全構成要素を横断して、各要素から1つずつ候補を選択する組み合わせを最適化する。
    
    Q辞書: {(i, j): value} (i <= j)
    """
    # 1. 変数マップの作成
    # 構成要素 (element) ごとに候補 (labels) をフラットに並べる
    variables = [] # List of {cat, el, cand_idx, label_values, qubo_energy}
    element_ranges = [] # List of (start_idx, end_idx)
    
    current_idx = 0
    
    for item in semantic_labels:
        cat = item["category"]
        el = item["element"]
        candidates = item["labels"]
        
        start = current_idx
        for k, cand in enumerate(candidates):
            qe = cand.get("qubo_energy", 0.0)
            
            variables.append({
                "category": cat,
                "element": el,
                "cand_idx": k,
                "label_values": cand,
                "qubo_energy": qe
            })
            current_idx += 1
        end = current_idx
        element_ranges.append((start, end))

    Q = {}
    
    # 2. 一次項 (E1): ユーザー設定（Targets/Weights）に基づき動的に計算
    target_values = evaluation_config.get("target_values", evaluation_config.get("targets", {}))
    weights = evaluation_config.get("weights", {})
    category_weights = evaluation_config.get("category_weights", {})

    print(f"DEBUG: QUBO Generation - Targets count: {len(target_values)}")

    for i, var in enumerate(variables):
        cat = var["category"]
        cand = var["label_values"]
        
        # この候補のE1エネルギーを再計算
        e1_sum = 0.0
        
        # カテゴリごとのラベル定義をループ
        if cat in label_mapping:
            for en_key, ja_label in label_mapping[cat].items():
                full_key = f"{cat}::{en_key}"
                if full_key in target_values:
                    target = target_values[full_key]
                    val = cand.get(ja_label, 0) # 日本語ラベルで値を取得
                    
                    # 重み
                    weight = weights.get(full_key)
                    if weight is None:
                        weight = category_weights.get(cat, 1.0)
                    
                    # 正規化計算
                    x_norm = val / 4.0
                    t_norm = target / 4.0
                    contribution = weight * ((x_norm - t_norm) ** 2)
                    e1_sum += contribution
        
        # 計算したE1を一次項としてセット
        Q[(i, i)] = Q.get((i, i), 0.0) + e1_sum
        # デバッグ用：最初の数件だけログ
        if i < 3:
            print(f"DEBUG: Variable {i} ({cat}) Dynamic E1: {e1_sum:.4f}")

    # 3. 二次項 (E2): ラベル間相互作用 (J_ab * l_a * l_b)
    interactions = evaluation_config.get("interactions", [])
    
    INTERACTION_WEIGHT = 0.1 # バランス補正係数 (案C)

    # 異なる構成要素間のみ計算
    for e_idx1, (s1, e1) in enumerate(element_ranges):
        for e_idx2 in range(e_idx1 + 1, len(element_ranges)):
            s2, e2 = element_ranges[e_idx2]
            
            # 要素1の候補 i と 要素2の候補 j のペア
            for i in range(s1, e1):
                for j in range(s2, e2):
                    var_i = variables[i]
                    var_j = variables[j]
                    
                    cat_i = var_i["category"]
                    cat_j = var_j["category"]
                    cand_i = var_i["label_values"]
                    cand_j = var_j["label_values"]
                    
                    for inter in interactions:
                        ka = inter["key_a"]
                        kb = inter["key_b"]
                        strength = inter["strength"]
                        if strength == 0: continue
                        
                        parts_a = ka.split("::")
                        parts_b = kb.split("::")
                        if len(parts_a) != 2 or len(parts_b) != 2: continue
                        
                        cat_a, en_a = parts_a
                        cat_b, en_b = parts_b
                        
                        val_a = None
                        val_b = None
                        
                        if cat_a == cat_i and cat_b == cat_j:
                            ja_a = label_mapping.get(cat_i, {}).get(en_a)
                            ja_b = label_mapping.get(cat_j, {}).get(en_b)
                            if ja_a in cand_i and ja_b in cand_j:
                                val_a = cand_i[ja_a]
                                val_b = cand_j[ja_b]
                        elif cat_a == cat_j and cat_b == cat_i:
                            ja_a = label_mapping.get(cat_j, {}).get(en_a)
                            ja_b = label_mapping.get(cat_i, {}).get(en_b)
                            if ja_a in cand_j and ja_b in cand_i:
                                val_a = cand_j[ja_a]
                                val_b = cand_i[ja_b]
                        
                        if val_a is not None and val_b is not None:
                            contribution = strength * (val_a / 4.0) * (val_b / 4.0)
                            Q[(i, j)] = Q.get((i, j), 0.0) + (contribution * INTERACTION_WEIGHT)

    # 4. 制約ペナルティ係数 P の自動決定
    if Q:
        max_abs_q = max(abs(v) for v in Q.values())
        P = max_abs_q * 10.0
    else:
        P = 10.0
    if P == 0: P = 10.0

    # 5. 制約追加: 各要素から必ず1つ選択 (Σ_k x_ik - 1)^2
    # 展開: Σ x_k + 2Σx_i x_j - 2Σx_k + 1  => -Σ x_k + 2Σx_i x_j
    for start, end in element_ranges:
        # 対角項 (Linear): -P * x
        for i in range(start, end):
            Q[(i, i)] = Q.get((i, i), 0.0) - P
        # 二次項 (Interaction): 2P * x_i * x_j
        for i in range(start, end):
            for j in range(i + 1, end):
                Q[(i, j)] = Q.get((i, j), 0.0) + 2.0 * P

    return Q, variables
