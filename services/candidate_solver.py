import random
import logging
from typing import Dict, Tuple, List, Any
from services.fixstars_solver import solve_with_fixstars

def calculate_energy_components(selection_indices: List[int], Q: Dict[Tuple[int, int], float], variables: List[Dict[str, Any]]):
    """
    選択されたインデックスに基づき、一次項(E1)と二次項(E2)のエネルギーを計算する。
    """
    e1 = 0.0
    e2 = 0.0
    
    for i_idx, i in enumerate(selection_indices):
        # 一次項 (もともとの qubo_energy 分を抽出)
        e1 += variables[i]["qubo_energy"]
        
        # 二次項 (他の選択された候補との相互作用)
        for j_idx in range(i_idx + 1, len(selection_indices)):
            j = selection_indices[j_idx]
            pair = tuple(sorted((i, j)))
            # Q内の非対角項（制約P以外の寄与分）がE2に相当
            # i, j が異なる要素に属している限り、Q[(i, j)] は純粋に相互作用 J_ab 由来。
            e2 += Q.get(pair, 0.0)
            
    return e1, e2

def solve_candidate_selection_qubo(
    Q: Dict[Tuple[int, int], float],
    variables: List[Dict[str, Any]],
    element_ranges: List[Tuple[int, int]],
    api_key: str = None
) -> Dict[str, Any]:
    """
    候補選択型QUBOのソルバー（Amplify / Heuristic 分岐）。
    """
    num_vars = len(variables)
    num_elements = len(element_ranges)
    
    if num_elements == 0:
        return {"best_selection_indices": [], "total_energy": 0, "e1": 0, "e2": 0, "solver": "none"}

    # 1. Fixstars Amplify 実行 (APIキーがある場合)
    if api_key and api_key != "YOUR_API_KEY_HERE":
        try:
            fs_result = solve_with_fixstars(Q, api_key)
            sel_indices = fs_result["selected_indices"]
            
            # E1, E2 を再計算して整合性をとる
            e1, e2 = calculate_energy_components(sel_indices, Q, variables)
            
            return {
                "best_selection_indices": sel_indices,
                "total_energy": round(e1 + e2, 4),
                "e1": round(e1, 4),
                "e2": round(e2, 4),
                "solver": "fixstars"
            }
        except Exception as e:
            logging.error(f"Fixstars solver failed, falling back to heuristic: {e}")
            # Fallback to heuristic

    # 2. ヒューリスティックソルバー
    # 初期解の構築: 各要素から qubo_energy が最小の候補を選ぶ
    current_selection = []
    for start, end in element_ranges:
        best_i = start
        min_cost = float('inf')
        for i in range(start, end):
            # i == j の項を抽出 (ここでの Q[(i,i)] は E1 - P)
            cost = Q.get((i, i), 0.0)
            if cost < min_cost:
                min_cost = cost
                best_i = i
        current_selection.append(best_i)

    cur_e1, cur_e2 = calculate_energy_components(current_selection, Q, variables)
    best_e1, best_e2 = cur_e1, cur_e2
    best_selection = list(current_selection)

    # 局所探索（ランダムスワップ）
    max_iter = 2000
    for _ in range(max_iter):
        el_idx = random.randint(0, num_elements - 1)
        start, end = element_ranges[el_idx]
        
        if (end - start) <= 1:
            continue
            
        old_var_idx = current_selection[el_idx]
        new_var_idx = random.randint(start, end - 1)
        if old_var_idx == new_var_idx:
            continue
            
        current_selection[el_idx] = new_var_idx
        new_e1, new_e2 = calculate_energy_components(current_selection, Q, variables)
        
        if (new_e1 + new_e2) < (best_e1 + best_e2):
            best_e1, best_e2 = new_e1, new_e2
            best_selection = list(current_selection)
        else:
            current_selection[el_idx] = old_var_idx

    return {
        "best_selection_indices": best_selection,
        "total_energy": round(best_e1 + best_e2, 4),
        "e1": round(best_e1, 4),
        "e2": round(best_e2, 4),
        "solver": "heuristic"
    }
