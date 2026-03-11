from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash, Response
import json
import os
from datetime import datetime
import time 


from user_files import load_user_data, save_user_data, get_user_data_path
import evaluation_engine
from services.services import find_document
from feature_extractor import FeatureExtractor

evaluation_bp = Blueprint('evaluation_bp', __name__, template_folder='templates')

def _synthesize_interactions(document: dict):
    """
    ジャンル設定に基づいて相互作用を自動合成する
    """
    config = document.setdefault("evaluation_config", {})
    # 既に設定がある場合はスキップ
    if config.get("interactions"):
        return

    genre_cfg = document.get("genre_config", {"main": "", "sub": []})
    main_genre = genre_cfg.get("main")
    sub_genres = genre_cfg.get("sub", [])

    try:
        with open("default_interactions.json", "r", encoding="utf-8") as f:
            defaults = json.load(f)
        with open("prompt_templates/semantic_label_schema.json", "r", encoding="utf-8") as f:
            schema = json.load(f)
    except FileNotFoundError:
        return

    # スキーマに存在する有効なラベルセットを作成
    # 形式: "分類::英語キー"
    valid_labels = set()
    label_map_ja_to_en = {} # "分類::日本語ラベル" -> "分類::英語キー"
    
    for cat_name, labels_config in schema.items():
        if cat_name == "scale": continue
        for en_key, spec in labels_config.items():
            full_en_key = f"{cat_name}::{en_key}"
            valid_labels.add(full_en_key)
            ja_label = spec.get("ja_label")
            if ja_label:
                label_map_ja_to_en[f"{cat_name}::{ja_label}"] = full_en_key
    
    # カテゴリ名の揺らぎ吸収用マップ (default_interactions.json のカテゴリ名 -> schema のカテゴリ名)
    cat_alias = {
        "シーン案": "シーン案",
        "キャラクター案": "キャラ案",
        "キャラ案": "キャラ案",
        "伏線案": "伏線案"
    }

    # ラベル名の揺らぎ吸収用マップ (default_interactions.json のラベル名 -> schema の日本語ラベル名)
    label_alias = {
        "内面葛藤": "内的葛藤深度",
        "成長度": "成長可能性",
        "伏線密度": "伏線強度",
        "回収確度": "回収明瞭性"
    }

    interaction_map = {} # (key_a, key_b) -> strength

    def normalize_label(label_str):
        if "::" not in label_str: return None
        cat, lab = label_str.split("::")
        mapped_cat = cat_alias.get(cat, cat)
        mapped_lab = label_alias.get(lab, lab)
        
        # 直接英語キーの場合
        en_style = f"{mapped_cat}::{mapped_lab}"
        if en_style in valid_labels:
            return en_style
        
        # 日本語ラベルから英語キーへ変換
        ja_style = f"{mapped_cat}::{mapped_lab}"
        return label_map_ja_to_en.get(ja_style)

    def add_to_map(inter_list, weight=1.0):
        for inter in inter_list:
            key_a = normalize_label(inter["key_a"])
            key_b = normalize_label(inter["key_b"])
            
            if key_a and key_b:
                pair = tuple(sorted([key_a, key_b]))
                interaction_map[pair] = interaction_map.get(pair, 0.0) + (inter["strength"] * weight)

    # 1. generic を 1.0倍
    add_to_map(defaults.get("generic", []))

    # 2. main genre を 1.0倍
    if main_genre and main_genre in defaults.get("genres", {}):
        add_to_map(defaults["genres"][main_genre])

    # 3. sub genre を 0.4倍
    ALPHA = 0.4
    for sub in sub_genres:
        if sub and sub != main_genre and sub in defaults.get("genres", {}):
            add_to_map(defaults["genres"][sub], weight=ALPHA)

    # マップから配列に変換
    new_interactions = []
    for (ka, kb), strength in interaction_map.items():
        new_interactions.append({
            "key_a": ka,
            "key_b": kb,
            "strength": round(strength, 1)
        })

    config["interactions"] = new_interactions


@evaluation_bp.route("/document/<doc_id>/auto_generate_interactions", methods=["POST"])
def auto_generate_interactions(doc_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)
    if document is None:
        return jsonify({"error": "Document not found"}), 404

    # 強制再生成フラグがあれば既存を消す（今回は空の場合のみという制約だが、念のため）
    # payload = request.get_json() or {}
    # if payload.get("force"):
    #     document.setdefault("evaluation_config", {})["interactions"] = []

    _synthesize_interactions(document)
    save_user_data(session["user_id"], data)
    
    return jsonify({
        "success": True, 
        "interactions": document.get("evaluation_config", {}).get("interactions", [])
    })


@evaluation_bp.route("/document/<doc_id>/save_criteria", methods=["POST"])
def save_criteria(doc_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)
    if document is None:
        return jsonify({"error": "Document not found"}), 404

    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No criteria data provided"}), 400

    if "evaluation_config" in payload:
        # 保存前に全ての相互作用の強度を0.1単位で丸める
        config = payload["evaluation_config"]
        if "interactions" in config:
            for inter in config["interactions"]:
                if "strength" in inter:
                    inter["strength"] = round(float(inter["strength"]), 1)
        document["evaluation_config"] = config
    else:
        document["evaluation_criteria"] = payload
        
    save_user_data(session["user_id"], data)
    
    return jsonify({"success": True})


@evaluation_bp.route("/api/evaluation/apply-genre", methods=["POST"])
def apply_genre_presets():
    # 1. 認証チェック
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    # 2. パラメータ取得
    params = request.get_json()
    if not params:
        return jsonify({"success": False, "error": "No data provided"}), 400
        
    doc_id = params.get("doc_id")
    if not doc_id:
        return jsonify({"success": False, "error": "Required fields: doc_id"}), 400

    # 3. ユーザデータとドキュメントの読み込み
    try:
        data = load_user_data(user_id)
    except FileNotFoundError:
        return jsonify({"success": False, "error": "User data folder not found"}), 404

    document = find_document(data, doc_id)
    if document is None:
        return jsonify({"success": False, "error": "Document not found"}), 404

    # 4. ジャンル設定の取得 (保存されている最新値を優先)
    genre_cfg = document.get("genre_config", {})
    main_genre = genre_cfg.get("main")
    sub_genres = genre_cfg.get("sub", []) # リスト全体を取得

    # 5. プリセットファイルの読み込み
    preset_path = "genre_targets_presets.json"
    if not os.path.exists(preset_path):
        return jsonify({"success": False, "error": "Genre presets file not found"}), 500
    
    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            presets = json.load(f)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to load presets: {str(e)}"}), 500

    if not main_genre or main_genre not in presets:
        return jsonify({"success": False, "error": f"Valid main genre not set in document"}), 400

    # 6. 補正ロジックの適用 (複数副ジャンル対応)
    ALPHA = 0.5
    m_p = presets[main_genre]
    
    # 有効な副ジャンルのプリセットのみを抽出
    valid_sub_presets = [presets[s] for s in sub_genres if s in presets and s != main_genre]

    def blend_and_clamp(key, m_val, preset_type="targets"):
        """主ジャンルと、全副ジャンルの平均値をブレンドする"""
        if not valid_sub_presets:
            return int(max(0, min(4, round(m_val))))

        # 全副ジャンルの値を合計して平均を出す
        sub_sum = 0
        for s_p in valid_sub_presets:
            sub_sum += s_p.get(preset_type, {}).get(key, m_val)
        s_val_avg = sub_sum / len(valid_sub_presets)

        diff = s_val_avg - m_val
        res = m_val + (ALPHA * diff)
        return int(max(0, min(4, round(res + (0.001 if diff > 0 else -0.001)))))

    # 既存の interactions をバックアップ
    eval_cfg = document.setdefault("evaluation_config", {})
    preserved_interactions = eval_cfg.get("interactions", [])

    # targets の生成
    new_targets = {}
    m_targets = m_p.get("targets", {})
    
    # 全てのターゲットキー（主ジャンルにあるもの全て）を走査
    for full_key, m_v in m_targets.items():
        new_targets[full_key] = blend_and_clamp(full_key, m_v, "targets")

    # category_weights の生成
    new_weights = {}
    m_weights = m_p.get("category_weights", {})
    cat_map = {"伏線案": "伏線案", "キャラ案": "キャラ案", "シーン案": "シーン案"}

    for p_cat, m_v in m_weights.items():
        ui_cat = cat_map.get(p_cat, p_cat)
        # 重み（0-5）もブレンド（blend_and_clampは0-4用だが、重みも同様のロジックで適用）
        # 便宜上 blend_and_clamp を流用（round結果は整数になる）
        new_weights[ui_cat] = blend_and_clamp(p_cat, m_v, "category_weights")
    
    # 7. 保存データの更新
    document["evaluation_config"] = {
        "targets": new_targets,
        "category_weights": new_weights,
        "interactions": preserved_interactions
    }

    save_user_data(user_id, data)

    return jsonify({
        "success": True,
        "genre_applied": {"main": main_genre, "sub": sub_genres},
        "evaluation_config": document["evaluation_config"]
    })


@evaluation_bp.route("/document/<doc_id>/calculate_fit_stream", methods=["GET"])
def calculate_fit_stream_route(doc_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    current_user_id = session["user_id"]

    def generate(user_id_arg):
        data = load_user_data(user_id_arg)
        document = find_document(data, doc_id)

        if document is None:
            yield f"data: {json.dumps({'error': 'Document not found'})}\n\n"
            return
        
        candidates = document.get("numerical_features")
        if not candidates:
            yield f"data: {json.dumps({'error': 'No candidates', 'message': '先に意味ラベルを付与・数値化してください'})}\n\n"
            return

        # 評価設定の取得 (新旧対応)
        config = document.get("evaluation_config", {})
        criteria = document.get("evaluation_criteria", {})
        
        targets_dict = config.get("targets", {})
        category_weights = config.get("category_weights", {})
        
        # fallback to old format if new one is empty
        if not targets_dict and "global_target" in criteria:
            targets_dict = criteria.get("global_target", {})
        if not category_weights and "category_weights" in criteria:
            category_weights = criteria.get("category_weights", {})

        try:
            extractor = FeatureExtractor()
            label_order = extractor.global_label_order
            
            # --- 保存された基準からグローバルターゲットベクトルを構築 ---
            target_vec = []
            for classification, label_key in label_order:
                # 新フォーマットは "分類::ラベル" キー
                new_key = f"{classification}::{label_key}"
                if new_key in targets_dict:
                    val = targets_dict[new_key]
                else:
                    # 旧フォーマットは 階層構造
                    val = targets_dict.get(classification, {}).get(label_key, 0)
                target_vec.append(val)
            # -----------------------------------------------------------

            system_tolerance = 0.0 # システム固定
            processed_candidates = []
            total = len(candidates)
            
            yield f"data: {json.dumps({'progress': 0, 'message': '適合度計算を開始しました'})}\n\n"

            for i, cand in enumerate(candidates):
                cand_features = cand.get("features", {})
                cand_vec = extractor.to_global_vector(cand_features)
                
                eval_result = evaluation_engine.evaluate(
                    cand_vec, target_vec, category_weights, label_order, system_tolerance
                )
                
                cand["fit"] = {
                    "best_fit_element": "Global Target",
                    "score": eval_result["adjusted_distance"],
                    "raw_distance": eval_result["raw_distance"],
                    "global_vector": cand_vec
                }
                
                processed_candidates.append(cand)
                
                progress = int((i + 1) / total * 100)
                yield f"data: {json.dumps({'progress': progress, 'message': f'{i+1}/{total} 候補を評価中...'})}\n\n"
                time.sleep(0.01)

            document["fit_results"] = processed_candidates
            save_user_data(user_id_arg, data)
            
            yield f"data: {json.dumps({'progress': 100, 'message': '適合度計算が完了しました', 'complete': True, 'fit_results': processed_candidates})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'message': f'エラー: {str(e)}'})}\n\n"

    return Response(generate(current_user_id), mimetype='text/event-stream')
