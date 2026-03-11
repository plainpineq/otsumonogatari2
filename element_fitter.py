import json
from typing import List, Dict, Any



def _compute_single_fit(candidate_features: Dict[str, Any], 
                        ideal_features: Dict[str, Any],
                        feature_extractor: Any) -> float: # FeatureExtractor インスタンスを受け取る
    """
    単一の候補と単一の理想テンプレートとの間のfitスコアを計算する。
    スコアが小さいほど、理想に近いことを示す。
    """
    fit_score = 0.0
    
    # 1. スカラー特徴量の差分を計算
    for key in feature_extractor.scalar_label_maps.keys():
        candidate_value = candidate_features.get("scalar_features", {}).get(key, 0)
        ideal_value = ideal_features.get("scalar_features", {}).get(key, 0)
        fit_score += abs(candidate_value - ideal_value)

    # 2. ベクトル特徴量のペナルティを計算
    for key in feature_extractor.vector_label_orders.keys():
        ideal_vector = ideal_features.get("vector_features", {}).get(key, [])
        candidate_vector = candidate_features.get("vector_features", {}).get(key, [])
        
        # どの項目が理想に含まれているか、候補に含まれているかをセットで管理
        ideal_present_values = set()
        for i, score in enumerate(ideal_vector):
            if i < len(feature_extractor.vector_label_orders[key]) and score > 0:
                ideal_present_values.add(feature_extractor.vector_label_orders[key][i])
        
        candidate_present_values = set()
        for i, score in enumerate(candidate_vector):
            if i < len(feature_extractor.vector_label_orders[key]) and score > 0:
                candidate_present_values.add(feature_extractor.vector_label_orders[key][i])

        # 理想が持つべきエフェクトが候補に存在しない場合にペナルティ (1.0) を加算
        missing_values = ideal_present_values - candidate_present_values
        fit_score += len(missing_values)

    return fit_score

def apply_fit_to_candidates(candidates: List[Dict[str, Any]], 
                            ideal_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    各候補に対して、最もフィットする理想要素を見つけ、fit情報を付与する。

    Args:
        candidates (List[Dict]): 評価対象となる構成要素候補のリスト。
        ideal_elements (List[Dict]): ユーザーが定義した理想的な構成要素のリスト。

    Returns:
        List[Dict]: 各候補に"fit"辞書が追加されたリスト。
                     例: {"best_fit_element": "転機", "score": 2.5}
    """
    if not candidates or not ideal_elements:
        return candidates

    # FeatureExtractor を初期化してラベル情報を取得
    feature_extractor = FeatureExtractor()
    
    # feature_extractorが初期化できない場合はスキップ
    if not feature_extractor.scalar_label_maps and not feature_extractor.vector_label_orders:
        print("警告: FeatureExtractorがラベル情報をロードできませんでした。fit計算をスキップします。")
        for candidate in candidates:
            candidate["fit"] = {"best_fit_element": "N/A", "score": float('inf')}
        return candidates

    processed_candidates = []
    for candidate in candidates:
        candidate_features = candidate.get("features")
        if not candidate_features:
            candidate["fit"] = {"best_fit_element": "No Features", "score": float('inf')}
            processed_candidates.append(candidate)
            continue

        best_fit_score = float('inf')
        best_fit_element_name = "N/A"

        for ideal in ideal_elements:
            ideal_features = ideal.get("features")
            ideal_name = ideal.get("element", "Unknown")
            if not ideal_features:
                continue

            score = _compute_single_fit(candidate_features, ideal_features, feature_extractor)
            
            if score < best_fit_score:
                best_fit_score = score
                best_fit_element_name = ideal_name

        candidate["fit"] = {
            "best_fit_element": best_fit_element_name,
            "score": best_fit_score
        }
        processed_candidates.append(candidate)
        
    return processed_candidates
