# scoring.py
from typing import List, Dict, Tuple
from models import Intent


def _keyword_overlap_score(text: str, keywords: List[str]) -> float:
    """
    text に keywords がどれだけ含まれるか（0.0〜1.0）
    """
    if not text or not keywords:
        return 0.0

    text_lower = text.lower()
    hits = sum(1 for k in keywords if k.lower() in text_lower)

    return hits / len(keywords)


def _split_keywords(text: str) -> List[str]:
    """
    日本語対応を考慮した簡易分割
    （将来 MeCab / LLM に差し替え可能）
    """
    if not text:
        return []

    separators = ["、", "。", ",", ".", "・", "\n"]
    for sep in separators:
        text = text.replace(sep, " ")

    return [t.strip() for t in text.split(" ") if t.strip()]


def score_intent_unit_alignment(intent: Intent, unit_text: str) -> float:
    """
    Intent と Unit（文章）の整合性スコア
    0.0（不整合）〜 1.0（非常に整合）
    """

    if not unit_text:
        return 0.0

    # --- keyword 準備 ---
    genre_keywords = _split_keywords(intent.genre)
    theme_keywords = _split_keywords(intent.theme_or_claim)
    value_keywords = _split_keywords(intent.core_values)

    # --- 各スコア ---
    genre_score = _keyword_overlap_score(unit_text, genre_keywords)
    theme_score = _keyword_overlap_score(unit_text, theme_keywords)
    values_score = _keyword_overlap_score(unit_text, value_keywords)

    # --- 制約ペナルティ ---
    penalty = 0.0
    for constraint in intent.constraints:
        if constraint and constraint.lower() in unit_text.lower():
            penalty += 0.2  # 1違反あたりのペナルティ

    # --- 重み付き合成 ---
    raw_score = 0.25 * genre_score + 0.35 * theme_score + 0.40 * values_score

    final_score = max(0.0, raw_score - penalty)

    return round(min(final_score, 1.0), 3)


def compute_qubo_energy(
    Q: Dict[Tuple[int, int], float], binary_vector: List[int]
) -> float:
    """
    QUBOエネルギー E = x^T Q x を計算
    """
    energy = 0.0
    for (i, j), value in Q.items():
        energy += value * binary_vector[i] * binary_vector[j]
    return energy


def to_onehot(values: List[int]) -> List[int]:
    """
    0-5整数リストをOne-hot72次元に変換
    """
    binary = []
    for v in values:
        for k in range(6):
            binary.append(1 if k == v else 0)
    return binary
