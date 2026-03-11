from typing import Dict, Any # NEW: Import Dict and Any for type hinting

from flask import Flask, render_template, request, redirect, session, send_file, make_response, flash, jsonify
from datetime import timedelta, datetime
import os
import json
import io
import glob
import pandas as pd
import logging

import shutil
from db import init_user_db, get_user_conn
from auth import login
from security import hash_password
from user_files import load_user_data, save_user_data, get_user_data_path
import semantic_labeler # NEW: Import the entire semantic_labeler module
import evaluation_engine # NEW: Import evaluation_engine
from evaluation_blueprint import evaluation_bp, _synthesize_interactions # NEW: Import the new blueprint and synthesis helper

def _reset_generation_counter(user_id: str):
    session[f"generation_counter_{user_id}"] = 0

def _get_next_generation_number(user_id: str) -> int:
    if f"generation_counter_{user_id}" not in session:
        _reset_generation_counter(user_id)
    session[f"generation_counter_{user_id}"] += 1
    return session[f"generation_counter_{user_id}"]


from ui_labels import UI_LABELS
from intent_templates import COMMON_INTENTS, DOC_TYPE_INTENTS
from lm_input import build_composition_ideas_prompt, mock_llm_call, build_title_plot_proposals_prompt, build_category_composition_prompt

# Helper function for cleaning up old generated files
def _cleanup_old_generated_files(user_id: str):
    user_data_dir = get_user_data_path(user_id)
    if not os.path.exists(user_data_dir):
        return

    # Patterns for files to delete
    patterns = [
        os.path.join(user_data_dir, "generated_prompt*.md"), # `_` を削除し、より広い範囲にマッチ
        os.path.join(user_data_dir, "generated_llm*.txt"),  # `_` を削除し、より広い範囲にマッチ
        os.path.join(user_data_dir, "generated_llm*.json") # `_` を削除し、より広い範囲にマッチ
    ]

    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"Cleaned up old generated file: {file_path}")
            except PermissionError as e:
                print(f"ERROR: Permission denied when deleting {file_path}: {e}")
                # ここではflashはしない。バックグラウンド処理のため。
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

def strip_api_keys(data: Any) -> Any:
    """
    Recursively remove all keys that contain 'api_key' from a dictionary or list.
    """
    if isinstance(data, dict):
        return {
            k: strip_api_keys(v) 
            for k, v in data.items() 
            if "api_key" not in k.lower()
        }
    elif isinstance(data, list):
        return [strip_api_keys(v) for v in data]
    return data

from services.services import (
    create_document,
    delete_document,
    rename_document,
    find_document,
    update_units_content,
    update_intent,
    normalize_composition_elements,
    update_composition_elements,
    DEFAULT_COMPOSITION_META
)
from services.domain_bridge import (
    document_to_domain,
    domain_to_document
)
from intent_service import normalize_intent as normalize_intent_service
from semantic_labeler import label_suggestions
from feature_extractor import FeatureExtractor
from element_fitter import apply_fit_to_candidates

from draft_manager import DraftManager
from draft_context_builder import build_draft_prompt
from llm_router import generate_draft

app = Flask(__name__)
app.secret_key = "storyforge-secret"
app.permanent_session_lifetime = timedelta(hours=2)

init_user_db()
app.register_blueprint(evaluation_bp) # NEW: Register the evaluation blueprint

# ---------- 認証 ----------

@app.route("/")
def root():
    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login_view():
    return login()


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = hash_password(request.form["password"])

        with get_user_conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (email, password_hash, created_at) "
                    "VALUES (?, ?, datetime('now'))",
                    (email, password)
                )
                conn.commit()
                flash("登録が完了しました。ログインしてください。", "success")
                return redirect("/login")
            except conn.IntegrityError:
                flash("このメールアドレスは既に使用されています。", "error")
                return render_template("register.html"), 400

    return render_template("register.html")


@app.route("/logout")
def logout():
    user_id = session.get("user_id")
    if user_id:
        user_data_dir = get_user_data_path(user_id)
        if os.path.exists(user_data_dir):
            try:
                shutil.rmtree(user_data_dir)
                print(f"Deleted user data directory for user: {user_id}")
            except Exception as e:
                print(f"Error deleting user data directory: {e}")
    
    session.clear()
    return redirect("/login")

# ---------- ダッシュボード ----------

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    data = load_user_data(session["user_id"])
    documents = data.get("documents", [])
    
    # Get server configurations from session or fallback to working.json
    llm_servers = session.get("llm_servers")
    quantum_server = session.get("quantum_server")
    
    if not llm_servers or not quantum_server:
        saved_settings = data.get("settings", {})
        if not llm_servers:
            llm_servers = saved_settings.get("llm_servers", {})
            session["llm_servers"] = llm_servers
        if not quantum_server:
            quantum_server = saved_settings.get("quantum_server", {})
            session["quantum_server"] = quantum_server

    # Ensure all roles have a default dictionary to prevent template errors
    for role in ["generation", "evaluation", "drafting"]:
        llm_servers.setdefault(role, {})

    user_config = {
        "llm_servers": llm_servers,
        "quantum_server": quantum_server,
        "suggestion_count": session.get("suggestion_count", 3)
    }

    # Load LLM models config
    llm_models_config = {}
    try:
        with open("llm_models_config.json", "r", encoding="utf-8") as f:
            llm_models_config = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load llm_models_config.json: {e}")

    # Load API keys config for demo
    api_keys_config = {"api_keys_enabled": False, "llm_providers": [], "analysis_servers": []}
    try:
        if os.path.exists("api_keys_config.json"):
            with open("api_keys_config.json", "r", encoding="utf-8") as f:
                api_keys_config = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load api_keys_config.json: {e}")

    return render_template(
        "dashboard.html",
        documents=documents,
        doc_types=DOC_TYPE_INTENTS.keys(),
        user_config=user_config,
        llm_models_config=llm_models_config,
        api_keys_config=api_keys_config
    )

@app.route("/help/gemini-api")
def help_gemini_api():
    return render_template("help_gemini_api.html")

@app.route("/help/amplify-api")
def help_amplify_api():
    return render_template("help_amplify_api.html")

@app.route("/usage")
def usage():
    return render_template("usage.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "user_id" not in session: # Removed data_loaded check
        return redirect("/dashboard")

    if 'file' not in request.files:
        flash("ファイルがありません")
        return redirect("/dashboard")

    file = request.files["file"]

    if file.filename == '':
        flash("ファイルが選択されていません")
        return redirect("/dashboard")

    # Assuming 'file' is not empty at this point, if it passes previous checks.
    # The 'if not file:' check is now less critical given the above.

    # Load the uploaded JSON data
    try:
        file_content_bytes = file.read()
        try:
            uploaded_json_content = json.loads(file_content_bytes.decode('utf-8'))
        except UnicodeDecodeError:
            # Fallback for common non-UTF-8 encodings, e.g., Shift-JIS for Japanese
            uploaded_json_content = json.loads(file_content_bytes.decode('shift_jis'))
    except json.JSONDecodeError:
        flash("無効なJSONファイルです")
        return redirect("/dashboard")
    except UnicodeDecodeError:
        flash("ファイルのエンコーディングを認識できませんでした。UTF-8またはShift-JISで保存されていることを確認してください。")
        return redirect("/dashboard")

    # Validate that the uploaded content is a dictionary and looks like a document
    # (e.g., has a 'title' key)
    if not isinstance(uploaded_json_content, dict) or "title" not in uploaded_json_content:
        flash("アップロードされたファイルは有効なドキュメント形式ではありません。'title'キーが見つかりません。")
        return redirect("/dashboard")

    # Load existing user data
    existing_data = load_user_data(session["user_id"])
    
    # Ensure 'documents' key exists and is a list
    if "documents" not in existing_data or not isinstance(existing_data["documents"], list):
        existing_data["documents"] = []

    # Append the new document to the existing list
    existing_data["documents"].append(uploaded_json_content)

    # Save the updated user data
    save_user_data(session["user_id"], existing_data)
    flash("ファイルをアップロードしました。", "success")
    return redirect("/dashboard")

@app.route("/update_suggestion_count", methods=["POST"])
def update_suggestion_count():
    if "user_id" not in session:
        flash("ログインしてください。", "error")
        return redirect("/login")

    try:
        suggestion_count = int(request.form["suggestion_count"])
        if not (1 <= suggestion_count <= 5):
            raise ValueError("提案個数は1から5の範囲で指定してください。")
        session["suggestion_count"] = suggestion_count
        flash("提案個数を保存しました。", "success")
    except ValueError as e:
        flash(f"エラー: {e}", "error")
    except Exception as e:
        flash(f"予期せぬエラーが発生しました: {e}", "error")
    
    return redirect("/dashboard")


@app.route("/save_servers_config", methods=["POST"])
def save_servers_config():
    if "user_id" not in session:
        flash("ログインしてください。", "error")
        return redirect("/login")

    data = load_user_data(session["user_id"])
    
    # Load API keys config for auto-filling disabled fields
    api_keys_config = {"api_keys_enabled": False, "llm_providers": [], "analysis_servers": []}
    try:
        if os.path.exists("api_keys_config.json"):
            with open("api_keys_config.json", "r", encoding="utf-8") as f:
                api_keys_config = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load api_keys_config.json for saving: {e}")

    # --- LLMサーバー設定の解析 ---
    llm_servers = {}
    roles = ["generation", "evaluation", "drafting"]
    for role in roles:
        provider = request.form.get(f"llm_servers[{role}][provider]", "gemini")
        api_key = request.form.get(f"llm_servers[{role}][api_key]", "")
        
        # If API key is empty and demo mode is enabled, try to find the pre-configured key
        if api_keys_config.get("api_keys_enabled") and not api_key:
            for p_config in api_keys_config.get("llm_providers", []):
                if p_config.get("provider") == provider:
                    # Check for multiple keys structure
                    if "keys" in p_config and p_config["keys"]:
                        api_key = p_config["keys"][0].get("api_key", "") # Default to first key if none provided
                    else:
                        api_key = p_config.get("api_key", "")
                    break

        role_config = {
            "provider": provider,
            "api_key": api_key,
            "model_name": request.form.get(f"llm_servers[{role}][model_name]", ""),
            "base_url": request.form.get(f"llm_servers[{role}][base_url]", ""),
            "temperature": request.form.get(f"llm_servers[{role}][temperature]", "0.7"),
            "max_tokens": request.form.get(f"llm_servers[{role}][max_tokens]", "2048"),
            "huggingface_api_key": request.form.get(f"llm_servers[{role}][huggingface_api_key]", ""),
            "huggingface_model_base_endpoint": request.form.get(f"llm_servers[{role}][huggingface_model_base_endpoint]", ""),
            "huggingface_model_id": request.form.get(f"llm_servers[{role}][huggingface_model_id]", ""),
        }
        # providerに応じたデフォルトモデル名の設定
        if role_config["provider"] == "gemini" and not role_config["model_name"]:
            role_config["model_name"] = "gemini-pro"
        elif role_config["provider"] == "chatgpt" and not role_config["model_name"]:
            role_config["model_name"] = "gpt-4o-mini"
        # No default for Hugging Face model endpoint/id, they must be provided by the user.

        # APIキーのトリム
        if role_config.get("api_key"):
            role_config["api_key"] = role_config["api_key"].strip()
        if role_config.get("huggingface_api_key"):
            role_config["huggingface_api_key"] = role_config["huggingface_api_key"].strip()
            
        llm_servers[role] = role_config

    # --- 量子サーバー設定の解析 ---
    api_key = request.form.get("quantum_server[api_key]", "").strip()
    if api_keys_config.get("api_keys_enabled") and not api_key:
        if api_keys_config.get("analysis_servers"):
            api_key = api_keys_config["analysis_servers"][0].get("api_key", "")

    quantum_server = {
        "api_key": api_key
    }

    # --- 保存 ---
    session["llm_servers"] = llm_servers
    session["quantum_server"] = quantum_server
    
    # working.json にも永続化 (APIキーは除外して保存)
    if "settings" not in data:
        data["settings"] = {}
    data["settings"]["llm_servers"] = strip_api_keys(llm_servers)
    data["settings"]["quantum_server"] = strip_api_keys(quantum_server)
    save_user_data(session["user_id"], data)
    
    flash("サーバー設定を保存しました。", "success")
    return redirect("/dashboard")


@app.route("/api/save_servers_config", methods=["POST"])
def api_save_servers_config():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        settings = request.get_json()
        if not settings:
            return jsonify({"success": False, "message": "No data received"}), 400

        llm_servers = settings.get("llm_servers", {})
        quantum_server = settings.get("quantum_server", {})

        # APIキーのトリム (全ロール分)
        for role in llm_servers:
            if "api_key" in llm_servers[role]:
                llm_servers[role]["api_key"] = llm_servers[role]["api_key"].strip()
            if "huggingface_api_key" in llm_servers[role]:
                llm_servers[role]["huggingface_api_key"] = llm_servers[role]["huggingface_api_key"].strip()

        if "api_key" in quantum_server:
            quantum_server["api_key"] = quantum_server["api_key"].strip()

        # セッションの更新 (APIキーを含む全データ)
        session["llm_servers"] = llm_servers
        session["quantum_server"] = quantum_server

        # working.json の更新 (APIキーを除外)
        data = load_user_data(session["user_id"])
        if "settings" not in data:
            data["settings"] = {}
        data["settings"]["llm_servers"] = strip_api_keys(llm_servers)
        data["settings"]["quantum_server"] = strip_api_keys(quantum_server)
        save_user_data(session["user_id"], data)

        return jsonify({"success": True})
    except Exception as e:
        logging.error(f"Error in api_save_servers_config: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/document/<doc_id>/save", methods=["POST"])
def api_save_document(doc_id):
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        data = load_user_data(session["user_id"])
        document = find_document(data, doc_id)
        if not document:
            return jsonify({"success": False, "message": "Document not found"}), 404

        req_data = request.get_json()
        
        # Determine what to update based on the keys in req_data
        updated = False
        
        # 1. Intent (Basic Settings)
        if "intent" in req_data:
            document["intent"] = req_data["intent"]
            updated = True
            
        # 2. Genre Config
        if "genre_config" in req_data:
            document["genre_config"] = req_data["genre_config"]
            updated = True

        # 3. Composition Elements
        if "composition_elements" in req_data:
            document["composition_elements"] = req_data["composition_elements"]
            # Re-normalize to ensure consistency (e.g. updating document["units"])
            normalize_composition_elements(document)
            updated = True

        # 4. Manuscript Full Text
        if "manuscript_full_text" in req_data:
            if "manuscript" not in document:
                document["manuscript"] = {"chapters": [], "version": 0, "full_text": ""}
            document["manuscript"]["full_text"] = req_data["manuscript_full_text"]
            document["manuscript"]["last_saved"] = datetime.now().isoformat()
            updated = True

        # 5. Drafting Configs (Blueprint/Directing)
        if "drafting_configs" in req_data:
            document["drafting_configs"] = req_data["drafting_configs"]
            updated = True

        if updated:
            save_user_data(session["user_id"], data)
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": "No valid update data provided"}), 400

    except Exception as e:
        logging.error(f"Error in api_save_document: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


# ---------- ドキュメント ----------

@app.route("/document/create", methods=["POST"])
def document_create():
    if "user_id" not in session: # Removed data_loaded check
        return redirect("/dashboard")

    data = load_user_data(session["user_id"])

    document = create_document(
        data=data,
        title=request.form["title"],
        doc_type=request.form["doc_type"]
    )

    save_user_data(session["user_id"], data)
    return redirect(f"/document/{document['id']}")


@app.route("/document/<doc_id>/delete", methods=["POST"])
def document_delete_route(doc_id):
    if "user_id" not in session:
        return redirect("/login")

    data = load_user_data(session["user_id"])
    if delete_document(data, doc_id):
        save_user_data(session["user_id"], data)
        flash("作品を削除しました。")
    else:
        flash("作品の削除に失敗しました。", "error")
    
    return redirect("/dashboard")


@app.route("/document/<doc_id>/rename", methods=["POST"])
def document_rename_route(doc_id):
    if "user_id" not in session:
        return redirect("/login")

    new_title = request.form.get("title")
    if not new_title:
        flash("タイトルを入力してください。", "error")
        return redirect("/dashboard")

    data = load_user_data(session["user_id"])
    if rename_document(data, doc_id, new_title):
        save_user_data(session["user_id"], data)
        flash("作品名を変更しました。")
    else:
        flash("作品名の変更に失敗しました。", "error")
    
    return redirect("/dashboard")


from services.scoring import compute_qubo_energy, to_onehot



def calculate_qubo_energy_for_item(document, category, labels_ja, schema):
    """
    特定項目のエネルギー貢献を計算する。
    E = weight * (value - target)^2 の貢献分を返す。
    ※候補選択型への移行に伴い、単体項のエネルギーとして再定義。
    """
    try:
        config = document.get("evaluation_config", {})
        targets = config.get("targets", {})
        category_weights = config.get("category_weights", {})
        
        energy = 0.0
        
        if category not in schema:
            return None
            
        weight = category_weights.get(category, 1.0)
        
        for en_key, spec in schema[category].items():
            ja_label = spec.get("ja_label", en_key)
            val = labels_ja.get(ja_label, 0)
            
            # 目標値の取得
            target_key = f"{category}::{en_key}"
            target_val = targets.get(target_key, 2) # デフォルト2
            
            # 正規化 (0-4 -> 0-1)
            val_norm = val / 4.0
            target_norm = target_val / 4.0
            
            energy += weight * ((val_norm - target_norm) ** 2)
            
        return round(energy, 4)
        
    except Exception as e:
        logging.error(f"Energy calculation failed: {e}")
        return None


def _migrate_semantic_labels(document: Dict[str, Any]) -> bool:
    """
    旧構造: { category, element, text: "A\nB", labels: [{eval1}, {eval2}] }
    新構造: { category, element, labels: [{text: "A", eval1}, {text: "B", eval2}] }
    """
    modified = False
    semantic_labels = document.get("semantic_labels", [])
    if not isinstance(semantic_labels, list):
        return False

    for item in semantic_labels:
        # labels 内に text が欠けている、かつ root.text が存在する場合に移行
        labels = item.get("labels", [])
        root_text = item.get("text")

        if root_text is not None and any("text" not in lbl for lbl in labels):
            texts = root_text.split("\n")
            for i, lbl in enumerate(labels):
                if "text" not in lbl:
                    lbl["text"] = texts[i] if i < len(texts) else ""
            
            # root.text を削除
            item.pop("text", None)
            modified = True
            
    return modified


@app.route("/document/<doc_id>", methods=["GET", "POST"])
def view_document(doc_id):
    if "user_id" not in session: # Removed data_loaded check
        return redirect("/dashboard")

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)

    if document is None:
        return redirect("/dashboard")

    # データ構造の自動マイグレーション
    if _migrate_semantic_labels(document):
        save_user_data(session["user_id"], data)

    if request.method == "POST":
        # composition_elements の更新を処理
        if request.form.get("update_composition_elements"):
            update_composition_elements(document, request.form)
        else: # 既存の unit content 更新も残しておく
            update_units_content(document, request.form)
        
        # Ensure normalization happens AFTER updates/deletions are processed
        normalize_intent_service(document) # Re-normalize intent after any potential changes
        normalize_composition_elements(document) # Re-normalize composition elements after any potential changes

        save_user_data(session["user_id"], data)
        if request.form.get("update_composition_elements"):
            flash("構成要素を保存しました。", "success")
        else:
            flash("原稿を保存しました。", "success")
        return redirect(f"/document/{doc_id}#composition") # 常に構成要素タブにリダイレクト

    # GETリクエストの場合も、テンプレートに渡す前にcomposition_elementsを正規化
    normalize_composition_elements(document)

    labels = UI_LABELS[document["doc_type"]]

    # 日本語の doc_type を英語のキーにマッピング
    doc_type_mapping = {meta["label"]: doc_id for doc_id, meta in DEFAULT_COMPOSITION_META["doc_types"].items()}
    mapped_doc_type_id = doc_type_mapping.get(document["doc_type"])

    # Ensure all expected document keys are present for the template to avoid Undefined errors
    document.setdefault("llm_suggestions", [])
    document.setdefault("semantic_labels", [])
    document.setdefault("numerical_features", [])
    document.setdefault("fit_results", [])
    document.setdefault("composition_elements", {})
    document.setdefault("intent", {"fields": {}}) # Add default for intent
    
    # evaluation_config の初期化と interactions の取得
    eval_config = document.setdefault("evaluation_config", {})
    interactions = eval_config.get("interactions", [])
    
    # Load semantic label schema for dynamic UI generation
    semantic_label_schema = {}
    try:
        with open("prompt_templates/semantic_label_schema.json", "r", encoding="utf-8") as f:
            semantic_label_schema = json.load(f)
    except FileNotFoundError:
        logging.warning("Warning: semantic_label_schema.json not found.")
    except json.JSONDecodeError:
        logging.warning("Warning: semantic_label_schema.json is invalid JSON.")

    # Load available genres from default_interactions.json
    available_genres = []
    try:
        with open("default_interactions.json", "r", encoding="utf-8") as f:
            defaults = json.load(f)
            available_genres = list(defaults.get("genres", {}).keys())
    except Exception as e:
        logging.warning(f"Warning: Failed to load genres from default_interactions.json: {e}")

    response = make_response(render_template(
        "document.html",
        document=document,
        labels=labels,
        mapped_doc_type_id=mapped_doc_type_id,
        global_composition_meta=DEFAULT_COMPOSITION_META, # NEW: Pass current global meta
        semantic_label_schema=semantic_label_schema, # NEW: Pass full schema to template
        available_genres=available_genres # NEW: Pass genre list
    ))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/document/<doc_id>/intent", methods=["POST"])
def edit_intent(doc_id):
    if "user_id" not in session: # Removed data_loaded check
        return redirect("/dashboard")

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)

    if document is None:
        return redirect("/dashboard")

    # ジャンル設定の更新
    if "main_genre" in request.form:
        document["genre_config"] = {
            "main": request.form.get("main_genre"),
            "sub": request.form.getlist("sub_genres")
        }
        # ジャンル変更時に相互作用が空なら自動生成 (Requirement 9)
        _synthesize_interactions(document)

    # ★ ここで正規化（重要）
    normalize_intent_service(document) # services.py の normalize_intent との衝突を避けるためリネーム

    # ★ Intent更新（削除・追加・保存すべて）
    update_intent(document, request.form)

    save_user_data(session["user_id"], data)
    flash("基本設定を保存しました。", "success")
    return redirect(f"/document/{doc_id}#intent")


from services.llm_client import call_llm # Import the generic LLM client


# ====================================================================
# NEW STAGED GENERATION FLOW (STEP 1, 2, 3)
# ====================================================================

@app.route("/document/<doc_id>/generate_proposals", methods=["POST"])
def generate_proposals(doc_id):
    """
    STEP 1: Generate initial Title and Plot proposals.
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)
    if document is None:
        return jsonify({"error": "Document not found"}), 404

    # Cleanup old files and reset counter for a new full generation cycle
    _cleanup_old_generated_files(session["user_id"])
    _reset_generation_counter(session["user_id"])
    session["generation_cleanup_done"] = True
    
    current_generation_number = _get_next_generation_number(session["user_id"])
    suffix = f"_{current_generation_number}_proposals"

    # Get LLM config
    llm_servers = session.get("llm_servers", {})
    generation_config = llm_servers.get("generation", {})
    llm_provider = generation_config.get("provider")
    llm_api_key = generation_config.get("api_key")
    llm_model_name = generation_config.get("model_name")
    llm_base_url = generation_config.get("base_url")
    llm_huggingface_api_key = generation_config.get("huggingface_api_key")
    llm_huggingface_model_base_endpoint = generation_config.get("huggingface_model_base_endpoint")
    llm_huggingface_model_id = generation_config.get("huggingface_model_id")
    suggestion_count = session.get("suggestion_count", 3)

    is_config_incomplete = False
    if not llm_provider:
        is_config_incomplete = True
    elif llm_provider in ["gemini", "chatgpt"]:
        if not llm_model_name or not llm_api_key:
            is_config_incomplete = True
    elif llm_provider == "other":
        if not llm_base_url or not llm_model_name:
            is_config_incomplete = True
    elif llm_provider == "huggingface":
        if not llm_huggingface_model_base_endpoint or not llm_huggingface_model_id: # API Key can be optional for public models
            is_config_incomplete = True

    if is_config_incomplete:
        mock_titles = [document.get("title", "無題")] + [f"模擬タイトル案{i+1}" for i in range(suggestion_count - 1)]
        return jsonify({"suggestions": [{"category": "基本項目", "elements": {"題名": mock_titles}}, {"category": "基本項目", "elements": {"あらすじ": [f"模擬プロット案{i+1}: これはモックデータです。" for i in range(suggestion_count)]}}], "message": "LLM設定が不完全なため、モックデータを使用しました。"}), 200

    try:
        prompt = build_title_plot_proposals_prompt(document, DEFAULT_COMPOSITION_META, session["user_id"], suffix=suffix, suggestion_count=suggestion_count)

        # Save the prompt to a file
        user_data_dir = get_user_data_path(session["user_id"])
        prompt_file_path = os.path.join(user_data_dir, f"generated_prompt{suffix}.md")
        with open(prompt_file_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Saved prompt to: {prompt_file_path}")

        raw_text, suggestions_dict = call_llm(
            api_key=llm_api_key,
            model_name=llm_model_name,
            prompt=prompt,
            llm_provider=llm_provider,
            base_url=llm_base_url,
            huggingface_api_key=llm_huggingface_api_key,
            huggingface_model_base_endpoint=llm_huggingface_model_base_endpoint,
            huggingface_model_id=llm_huggingface_model_id
        )

        # Ensure one of the title suggestions is exactly the document title
        doc_title = document.get("title", "無題")
        for sug in suggestions_dict.get("suggestions", []):
            if sug.get("category") in ["基本項目", "基本設定"]:
                elements = sug.get("elements", {})
                if "題名" in elements and isinstance(elements["題名"], list):
                    # Check if the title is already there, if not replace the first one
                    if doc_title not in elements["題名"]:
                        elements["題名"][0] = doc_title

        # Save responses        user_data_dir = get_user_data_path(session["user_id"])
        with open(os.path.join(user_data_dir, f"generated_llm{suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)
        with open(os.path.join(user_data_dir, f"generated_llm{suffix}.json"), "w", encoding="utf-8") as f:
            json.dump(suggestions_dict, f, ensure_ascii=False, indent=2)

        # Update suggestions in the main document object
        if "llm_suggestions" not in document or not isinstance(document["llm_suggestions"], list):
            document["llm_suggestions"] = []

        # Update existing categories or append new ones
        new_suggestions = suggestions_dict.get("suggestions", [])
        for new_sug in new_suggestions:
            category_name = new_sug.get("category")
            if not category_name:
                continue
            
            found = False
            for i, existing_sug in enumerate(document["llm_suggestions"]):
                if existing_sug.get("category") == category_name:
                    document["llm_suggestions"][i] = new_sug
                    found = True
                    break
            
            if not found:
                document["llm_suggestions"].append(new_sug)
        
        save_user_data(session["user_id"], data)

        return jsonify(suggestions_dict)

    except (ValueError, RuntimeError, Exception) as e:
        return jsonify({"error": f"LLM呼び出し中にエラーが発生しました: {str(e)}"}), 500

@app.route("/document/<doc_id>/save_selection", methods=["POST"])
def save_selection(doc_id):
    """
    STEP 2: Save the selected basic elements from the generation tab.
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)
    if document is None:
        return jsonify({"error": "Document not found"}), 404

    try:
        selected_elements = request.get_json()
        if not isinstance(selected_elements, dict):
            return jsonify({"error": "Invalid JSON data"}), 400

        # Store the selected basic elements in the document
        document["selected_basic_elements"] = selected_elements
        
        # Optionally, update the document's title if 'title' is in selected_elements
        if "title" in selected_elements:
            document["title"] = selected_elements["title"]

        save_user_data(session["user_id"], data)
        return jsonify({"success": True, "message": "Selection saved successfully."})

    except Exception as e:
        return jsonify({"error": f"Failed to save selection: {str(e)}"}), 500

@app.route("/document/<doc_id>/generate_composition", methods=["POST"])
def generate_composition(doc_id):
    """
    STEP 3: Generate full composition elements for a specific category using LLM.
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)
    if document is None:
        return jsonify({"error": "Document not found"}), 404

    # Perform cleanup and counter reset only once per generation session
    if not session.get("generation_cleanup_done", False):
        _cleanup_old_generated_files(session["user_id"])
        _reset_generation_counter(session["user_id"])
        session["generation_cleanup_done"] = True
    
    try:
        request_data = request.get_json()
        category_label = request_data.get("category_label")
        if not category_label:
            return jsonify({"error": "Category label is required"}), 400

        # Get LLM config
        llm_servers = session.get("llm_servers", {})
        generation_config = llm_servers.get("generation", {})
        llm_provider = generation_config.get("provider")
        llm_api_key = generation_config.get("api_key")
        llm_model_name = generation_config.get("model_name")
        llm_base_url = generation_config.get("base_url")
        llm_huggingface_api_key = generation_config.get("huggingface_api_key")
        llm_huggingface_model_base_endpoint = generation_config.get("huggingface_model_base_endpoint")
        llm_huggingface_model_id = generation_config.get("huggingface_model_id")

        is_config_incomplete = False
        if not llm_provider:
            is_config_incomplete = True
        elif llm_provider in ["gemini", "chatgpt"]:
            if not llm_model_name or not llm_api_key:
                is_config_incomplete = True
        elif llm_provider == "other":
            if not llm_base_url or not llm_model_name:
                is_config_incomplete = True
        elif llm_provider == "huggingface":
            if not llm_huggingface_model_base_endpoint or not llm_huggingface_model_id: # API Key can be optional for public models
                is_config_incomplete = True

        if is_config_incomplete:
            # Fallback to mock data if LLM config is incomplete
            mock_suggestions = {"category": category_label, "elements": {}}
            # Find the category in document's composition_elements to get element labels
            for category_obj in document.get("composition_elements", {}).get("categories", []):
                if category_obj.get("label") == category_label:
                    if category_obj.get("elements"):
                        for element in category_obj["elements"]:
                            if element.get("label"):
                                mock_suggestions["elements"][element["label"]] = [f"模擬提案: {element['label']}の内容"]
                    break
            return jsonify({"suggestions": [mock_suggestions], "message": "LLM設定が不完全なため、モックデータを使用しました。"}), 200

        # Build prompt
        # Need to generate a unique suffix for saving the prompt file
        current_generation_number = _get_next_generation_number(session["user_id"])
        suffix = f"_{current_generation_number}_{category_label.replace(' ', '_')}"
        suggestion_count = session.get("suggestion_count", 3) # Retrieve suggestion_count from session

        prompt = build_category_composition_prompt(
            document, DEFAULT_COMPOSITION_META, session["user_id"],
            category_label=category_label, suffix=suffix,
            suggestion_count=suggestion_count # Pass the suggestion_count
        )
        
        raw_text, suggestions_dict = call_llm(
            api_key=llm_api_key,
            model_name=llm_model_name,
            prompt=prompt,
            llm_provider=llm_provider,
            base_url=llm_base_url,
            huggingface_api_key=llm_huggingface_api_key,
            huggingface_model_base_endpoint=llm_huggingface_model_base_endpoint,
            huggingface_model_id=llm_huggingface_model_id
        )

        # Save LLM raw response and parsed JSON response for debugging
        user_data_dir = get_user_data_path(session["user_id"])
        with open(os.path.join(user_data_dir, f"generated_llm_full_composition{suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)
        with open(os.path.join(user_data_dir, f"generated_llm_full_composition{suffix}.json"), "w", encoding="utf-8") as f:
            json.dump(suggestions_dict, f, ensure_ascii=False, indent=2)

        # Update document with new suggestions
        if "llm_suggestions" not in document or not isinstance(document["llm_suggestions"], list):
            document["llm_suggestions"] = []
        
        # Ensure the suggestions_dict contains the expected structure
        if isinstance(suggestions_dict, dict) and "category" in suggestions_dict and "elements" in suggestions_dict:
            # Check if this category already exists in llm_suggestions and update it
            found = False
            for i, existing_suggestion in enumerate(document["llm_suggestions"]):
                if existing_suggestion.get("category") == suggestions_dict["category"]:
                    document["llm_suggestions"][i] = suggestions_dict # Replace
                    found = True
                    break
            if not found:
                document["llm_suggestions"].append(suggestions_dict) # Add new

        save_user_data(session["user_id"], data)

        return jsonify({"suggestions": [suggestions_dict]}) # Wrap in a list as the frontend expects

    except (ValueError, RuntimeError, Exception) as e:
        return jsonify({"error": f"全体構成の生成中にエラーが発生しました: {str(e)}"}), 500

@app.route("/document/<doc_id>/add_composition_element", methods=["POST"])
def add_composition_element(doc_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)
    if document is None:
        return jsonify({"error": "Document not found"}), 404

    request_data = request.get_json()
    new_element_label = request_data.get("label")

    if not new_element_label:
        return jsonify({"error": "Label is required"}), 400

    # Find or create a user-editable category to add the suggestion to
    elements_data = document["composition_elements"]
    doc_type_specific_categories = elements_data["doc_type_specific"].setdefault("categories", [])
    common_categories = elements_data["common"].setdefault("categories", [])
    
    # Try to add to an editable doc_type_specific category first
    target_category = next((cat for cat in doc_type_specific_categories if cat.get("editable")), None)
    
    # If no editable doc_type_specific category, try editable common category
    if not target_category:
        target_category = next((cat for cat in common_categories if cat.get("editable")), None)

    # If still no editable category, create a new "AI提案" category under common
    if not target_category:
        ai_suggestions_category_id = "ai_suggestions_cat"
        target_category = next((cat for cat in common_categories if cat["id"] == ai_suggestions_category_id), None)
        if not target_category:
            target_category = {
                "id": ai_suggestions_category_id,
                "label": "AI提案",
                "editable": True,
                "elements": []
            }
            common_categories.append(target_category)
    
    target_elements = target_category.setdefault("elements", [])
    target_elements.append({
        "id": os.urandom(4).hex(), # Generate a unique ID for the new element
        "label": new_element_label,
        "value": "",
        "editable": True
    })

    save_user_data(session["user_id"], data)
    return jsonify({"message": "Composition element added successfully"})


@app.route("/document/<doc_id>/download", methods=["GET"])
def download_document(doc_id):
    if "user_id" not in session:
        return redirect("/login")

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)

    if document is None:
        return redirect("/dashboard")

    # ダウンロード用にdocumentのコピーを作成し、不要なキーを削除
    download_document_data = document.copy()
    keys_to_remove = [
        # "llm_suggestions", # ユーザーの要望によりダウンロード対象に含める
        # "semantic_labels", # ユーザーの要望によりダウンロード対象に含める
        # "numerical_features",
        # "fit_results",
        "units" # ユーザー提供の例にないため削除
    ]
    for key in keys_to_remove:
        download_document_data.pop(key, None) # キーが存在しない場合はエラーにならないようにpop(key, None)を使用

    document_json = json.dumps(download_document_data, ensure_ascii=False, indent=2)
    
    # Use io.BytesIO to create an in-memory file
    file_data = io.BytesIO(document_json.encode('utf-8'))
    
    # Use send_file for robust downloading
    response = send_file(
        file_data,
        mimetype='application/json',
        as_attachment=True,
        download_name=f"{document['title']}.json"
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/document/<doc_id>/save_generated_category", methods=["POST"])
def save_generated_category(doc_id):
    """
    Save edited LLM-generated suggestions for a specific category.
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)
    if document is None:
        return jsonify({"error": "Document not found"}), 404

    try:
        request_data = request.get_json()
        category_label = request_data.get("category")
        updated_elements = request_data.get("elements")

        if not category_label or updated_elements is None:
            return jsonify({"error": "Missing category or elements data"}), 400

        if "llm_suggestions" not in document:
            document["llm_suggestions"] = []

        found = False
        for i, suggestion in enumerate(document["llm_suggestions"]):
            if suggestion.get("category") == category_label:
                document["llm_suggestions"][i]["elements"] = updated_elements
                found = True
                break
        
        if not found:
            document["llm_suggestions"].append({
                "category": category_label,
                "elements": updated_elements
            })

        save_user_data(session["user_id"], data)
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": f"Failed to save generated category: {str(e)}"}), 500


@app.route("/document/<doc_id>/calculate_energy", methods=["POST"])
def calculate_energy(doc_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)
    if not document:
        return jsonify({"error": "Document not found"}), 404

    try:
        req_data = request.get_json()
        selected_items = req_data.get("selected_items", []) # リスト形式で受け取る (順序保持のため)
        
        evaluation_config = document.get("evaluation_config", {})
        
        # Load schema for label mapping
        semantic_label_schema = {}
        try:
            with open("prompt_templates/semantic_label_schema.json", "r", encoding="utf-8") as f:
                semantic_label_schema = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load semantic_label_schema for energy calculation: {e}")

        results = evaluation_engine.calculate_energy_detail(selected_items, evaluation_config, semantic_label_schema)
        return jsonify(results)
    except Exception as e:
        logging.error(f"Energy calculation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/document/<doc_id>/save_manual_selection", methods=["POST"])
def save_manual_selection(doc_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)
    if not document:
        return jsonify({"error": "Document not found"}), 404

    try:
        req_data = request.get_json()
        category = req_data.get("category")
        element = req_data.get("element")
        index = req_data.get("index")

        if "best_selection" not in document:
            document["best_selection"] = {}
        
        if category not in document["best_selection"]:
            document["best_selection"][category] = {}
        
        document["best_selection"][category][element] = index
        
        save_user_data(session["user_id"], data)
        return jsonify({"success": True})
    except Exception as e:
        logging.error(f"Manual selection save failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/document/<doc_id>/evaluate", methods=["POST"])
def evaluate_document(doc_id):
    """
    評価ページにリダイレクトする。実際の処理はクライアントサイドのJavaScriptが
    /evaluate/stream エンドポイントを呼び出すことで開始される。
    """
    if "user_id" not in session:
        return redirect("/login")

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)

    # LLM設定が不完全な場合はエラーを表示して中断
    evaluation_config = session.get("llm_servers", {}).get("evaluation", {})
    llm_provider = evaluation_config.get("provider")
    is_config_incomplete = not llm_provider or not evaluation_config.get("model_name") or \
                           (llm_provider in ["gemini", "chatgpt"] and not evaluation_config.get("api_key")) or \
                           (llm_provider == "other" and not evaluation_config.get("base_url"))

    if is_config_incomplete:
        return jsonify({"status": "error", "message": "評価を実行するには、まずダッシュボードで「構成要素評価 用 LLM」設定を完了してください。"}), 400

    # interactionsが設定されていない場合は自動生成を試みる (フォールバック)
    interactions = document.get("evaluation_config", {}).get("interactions", [])
    if not interactions:
        _synthesize_interactions(document)
        save_user_data(session["user_id"], data)
        interactions = document.get("evaluation_config", {}).get("interactions", [])

    if not interactions:
        return jsonify({"status": "error", "message": "評価を実行するには、「評価基準」タブで「ジャンルから自動設定」を行うか、手動で「要素間の相乗効果」を設定してください。"}), 400

    # ページをリロードし、クライアント側でストリーミングを開始させる
    return jsonify({"status": "success", "message": "評価処理を開始します。結果はリアルタイムで表示されます..."}), 200

# Moved evaluate_stream to top-level


@app.route('/document/<doc_id>/evaluate/stream', methods=['GET', 'POST'])
def evaluate_stream(doc_id):
    """
    Server-Sent Events (SSE) を使用して、意味ラベル評価の結果をストリーミングする。
    classification_filter が指定された場合、その分類のみを評価対象とする。
    """
    from flask import Response # Keep local import for Response, as it's used only here.
    # logging is imported at module level, no need here
    
    logger = logging.getLogger(__name__)

    if "user_id" not in session:
        return Response("Unauthorized", status=401)
    
    llm_config = session.get("llm_servers", {}).get("evaluation", {})
    classification_filter = request.args.get("classification_filter")

    def generate_labels_stream(user_id_arg, llm_config_arg, classification_filter_arg):
        data = load_user_data(user_id_arg)
        document = find_document(data, doc_id)

        if not document:
            yield f"data: {json.dumps({'error': 'ドキュメントが見つかりません。'})}\n\n"
            return
        
        final_evaluation_suggestions = []
        if document.get("llm_suggestions") and isinstance(document["llm_suggestions"], list):
            for suggestion_group in document["llm_suggestions"]:
                if suggestion_group.get("category") != "基本設定":
                    if classification_filter_arg and suggestion_group.get("category") != classification_filter_arg:
                        continue
                    final_evaluation_suggestions.append(suggestion_group)

        evaluation_input = {"llm_suggestions": final_evaluation_suggestions}
        if not evaluation_input["llm_suggestions"]:
            yield f"data: {json.dumps({'error': '評価対象の構成要素候補が見つかりません。'})}\n\n"
            return

        user_data_dir = get_user_data_path(user_id_arg)
        log_file_path = os.path.join(user_data_dir, "labeler.log")
        
        all_results = []
        logger_to_cleanup = logging.getLogger("semantic_labeler")
        total_items_count = 0 

        try:
            for event_data in semantic_labeler.label_suggestions(evaluation_input, llm_config=llm_config_arg, user_id=user_id_arg, log_file_path=log_file_path):
                event_type = event_data.get("event")

                if event_type == "total_items":
                    total_items_count = event_data.get("count", 0)
                    yield f"data: {json.dumps({'progress_total': total_items_count})}\n\n"
                elif event_type == "progress":
                    yield f"data: {json.dumps({
                        'progress_current': event_data['progress_current'],
                        'progress_total': event_data['progress_total'],
                        'category_label': event_data['category_label'],
                        'current_element': event_data['current_element']
                    })}\n\n"
                elif event_type == "semantic_label":
                    labeled_result = event_data.get("data")
                    if labeled_result:
                        # Load semantic label schema for dynamic energy calculation
                        semantic_label_schema = {}
                        try:
                            with open("prompt_templates/semantic_label_schema.json", "r", encoding="utf-8") as f:
                                semantic_label_schema = json.load(f)
                        except:
                            pass
                        
                        # 各提案に対してエネルギーと量子最適解を計算
                        if semantic_label_schema:
                            cat = labeled_result.get("category")
                            
                            config = document.get("evaluation_config", {})
                            criteria = document.get("evaluation_criteria", {})
                            
                            targets = config.get("targets", {})
                            weights = config.get("category_weights", {})
                            
                            # fallback
                            if not targets and "global_target" in criteria:
                                targets = criteria.get("global_target", {})
                            if not weights and "category_weights" in criteria:
                                weights = criteria.get("category_weights", {})

                            for labels_ja in labeled_result.get("labels", []):
                                energy = calculate_qubo_energy_for_item(document, cat, labels_ja, semantic_label_schema)
                                labels_ja["qubo_energy"] = energy

                                # --- 乖離スコア (diff_score) の計算を追加 ---
                                current_values = []
                                for cat_name, labels_config in semantic_label_schema.items():
                                    if cat_name == "scale": continue
                                    for en_key, spec in labels_config.items():
                                        ja_label = spec.get("ja_label", en_key)
                                        if cat_name == cat:
                                            current_values.append(labels_ja.get(ja_label, 0))
                                        else:
                                            target_key = f"{cat_name}::{en_key}"
                                            if target_key in targets:
                                                current_values.append(targets[target_key])
                                            else:
                                                current_values.append(targets.get(cat_name, {}).get(en_key, 2))

                                target_vector = []
                                weights_vector = []
                                for cn, lc in semantic_label_schema.items():
                                    if cn == "scale": continue
                                    w = weights.get(cn, 1.0)
                                    for ek, sp in lc.items():
                                        target_key = f"{cn}::{ek}"
                                        if target_key in targets:
                                            target_vector.append(targets[target_key])
                                        else:
                                            target_vector.append(targets.get(cn, {}).get(ek, 2))
                                        weights_vector.append(w)
                                
                                diff_score = 0.0
                                for i in range(len(current_values)):
                                    diff_score += abs(current_values[i] - target_vector[i]) * weights_vector[i]
                                labels_ja["diff_score"] = round(diff_score, 2)
                                # -----------------------------------------

                        all_results.append(labeled_result)
                        # labeled_result contains labels: [{text, ...}, ...] already from label_suggestions
                        yield f"data: {json.dumps({'semantic_label': labeled_result}, ensure_ascii=False)}\n\n"
                else:
                    logging.warning(f"Received unexpected event type: {event_type} with data: {event_data}")
            
            # --- Ensure migration-like check for new items (though labeler is updated) ---
            for res in all_results:
                if "text" in res:
                    res.pop("text", None)
            # ---------------------------------------------------------------------------
            
            # Save all results to document["semantic_labels"]
            if classification_filter_arg:
                document["semantic_labels"] = [
                    label for label in document.get("semantic_labels", []) 
                    if label.get("category") != classification_filter_arg
                ]
                document["semantic_labels"].extend(all_results)
            else:
                document["semantic_labels"] = all_results # No filter, replace all

            # --- 自動数値化処理の追加 (案1) ---
            try:
                extractor = FeatureExtractor()
                document["numerical_features"] = extractor.create_numerical_features(document["semantic_labels"])
                logging.info("Evaluation completed: Numerical features automatically generated.")
            except Exception as fe_err:
                logging.error(f"Automatic vectorization failed: {fe_err}")
            # ---------------------------------

            save_user_data(user_id_arg, data)
            yield "event: end_stream\ndata: {}\n\n"
            
        except GeneratorExit:
            # Client disconnected, just stop the generator
            return
        except Exception as e:
            logger_to_cleanup.error(f"Error during streaming semantic labels: {e}")
            yield f"data: {json.dumps({'error': f'評価処理中にサーバーエラーが発生しました: {str(e)}'}, ensure_ascii=False)}\n\n"
            return
        finally:
            # Ensure handlers are closed to prevent PermissionError (Errno 13) on subsequent calls
            for handler in logger_to_cleanup.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger_to_cleanup.removeHandler(handler)

    return Response(generate_labels_stream(session["user_id"], llm_config, classification_filter), mimetype='text/event-stream')


@app.route("/document/<doc_id>/download_evaluation")
def download_evaluation(doc_id):
    if "user_id" not in session:
        return redirect("/login")

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)

    if not document or "semantic_labels" not in document or not document["semantic_labels"]:
        flash("ダウンロードする評価データがありません。", "warning")
        return redirect(f"/document/{doc_id}#evaluation")

    try:
        # Flatten the data
        df = pd.json_normalize(document["semantic_labels"], sep='_')
        
        # Convert to CSV
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        
        # Create response
        response = make_response(csv_data)
        response.headers["Content-Disposition"] = "attachment; filename=evaluation_results.csv"
        response.headers["Content-Type"] = "text/csv; charset=utf-8-sig"
        return response

    except Exception as e:
        flash(f"CSVの生成中にエラーが発生しました: {e}", "error")
        return redirect(f"/document/{doc_id}#evaluation")


@app.route("/save_server_settings_from_ui", methods=["POST"])
def save_server_settings_from_ui():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        # JavaScriptから送信されたJSONデータを受け取る
        settings_to_save = request.get_json()
        if not settings_to_save:
            return jsonify({"success": False, "message": "Invalid JSON data received"}), 400

        # 受け取った設定をセッションにも反映させる（次回アクセス時の初期表示のため）
        session["llm_servers"] = settings_to_save.get("llm_servers", {})
        session["quantum_server"] = settings_to_save.get("quantum_server", {})

        settings_json = json.dumps(settings_to_save, ensure_ascii=False, indent=2)
        
        file_data = io.BytesIO(settings_json.encode('utf-8'))
        
        response = send_file(
            file_data,
            mimetype='application/json',
            as_attachment=True,
            download_name='server_settings.json'
        )
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        logging.error(f"Error in save_server_settings_from_ui: {e}")
        return jsonify({"success": False, "message": f"設定のダウンロード中にエラーが発生しました: {str(e)}"}), 500


@app.route("/load_server_settings", methods=["POST"])
def load_server_settings():
    if "user_id" not in session:
        flash("ログインしてください。", "error")
        return redirect("/login")

    if 'file' not in request.files:
        flash("ファイルがありません", "error")
        return redirect("/dashboard")

    file = request.files['file']
    if file.filename == '':
        flash("ファイルが選択されていません", "error")
        return redirect("/dashboard")

    try:
        file_content = file.read().decode('utf-8')
        settings = json.loads(file_content)

        if "llm_servers" in settings:
            session["llm_servers"] = settings["llm_servers"]
        if "quantum_server" in settings:
            session["quantum_server"] = settings["quantum_server"]
        
        flash("サーバー設定をロードしました。", "success")
    except json.JSONDecodeError:
        flash("無効なJSONファイルです", "error")
    except Exception as e:
        flash(f"設定のロード中にエラーが発生しました: {str(e)}", "error")
    
    return redirect("/dashboard")


from services.candidate_qubo import generate_candidate_selection_qubo
from services.candidate_solver import solve_candidate_selection_qubo

@app.route("/document/<doc_id>/quantum_optimize", methods=["POST"])
def quantum_optimize(doc_id):
    print(f"\n--- Received Quantum Optimization Request for Doc: {doc_id} ---")
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = load_user_data(session["user_id"])
    document = find_document(data, doc_id)
    if not document:
        return jsonify({"error": "Document not found"}), 404

    try:
        config = document.get("evaluation_config", {})
        
        # Load semantic label schema for mappings
        semantic_label_schema = {}
        try:
            with open("prompt_templates/semantic_label_schema.json", "r", encoding="utf-8") as f:
                semantic_label_schema = json.load(f)
        except:
            return jsonify({"error": "Schema not found"}), 500

        label_mapping = {}
        for cat, labels in semantic_label_schema.items():
            if cat == "scale": continue
            label_mapping[cat] = {en: spec["ja_label"] for en, spec in labels.items()}

        # 1. データの構築
        semantic_labels = document.get("semantic_labels", [])
        if not semantic_labels:
            print("Warning: No semantic_labels found in document.")
            return jsonify({"error": "No evaluation data found."}), 400

        # APIキーの取得
        quantum_server = session.get("quantum_server", {})
        api_key = quantum_server.get("api_key")
        if api_key:
            api_key = api_key.strip()

        # interactionsが設定されていない場合は自動生成を試みる (フォールバック)
        interactions = config.get("interactions", [])
        if not interactions:
            _synthesize_interactions(document)
            save_user_data(session["user_id"], data)
            config = document.get("evaluation_config", {}) # 再取得
            interactions = config.get("interactions", [])

        if not interactions:
             return jsonify({"error": "最適化を実行するには、「評価基準」タブで「ジャンルから自動設定」を行うか、手動で「要素間の相乗効果」を設定してください。"}), 400

        # 2. QUBO生成
        Q, variables = generate_candidate_selection_qubo(semantic_labels, config, label_mapping)
        
        # 3. 構造情報の抽出
        element_ranges = []
        current_idx = 0
        for item in semantic_labels:
            start = current_idx
            current_idx += len(item["labels"])
            element_ranges.append((start, current_idx))

        # 4. 解決 (Amplify or Heuristic)
        result = solve_candidate_selection_qubo(Q, variables, element_ranges, api_key=api_key)

        # 5. 結果の構築 ( { category: { element: candidate_index } } )
        best_selection_map = {}
        for var_idx in result["best_selection_indices"]:
            var = variables[var_idx]
            cat = var["category"]
            el = var["element"]
            cand_idx = var["cand_idx"]
            
            if cat not in best_selection_map:
                best_selection_map[cat] = {}
            best_selection_map[cat][el] = cand_idx

        # 6. ドキュメントに保存 (永続化)
        document["best_selection"] = best_selection_map
        document["optimized_selection"] = best_selection_map # 推奨構成として記録
        document["best_selection_energy"] = {
            "total": result["total_energy"],
            "e1": result["e1"],
            "e2": result["e2"],
            "solver": result["solver"]
        }
        save_user_data(session["user_id"], data)

        # ログ出力 (サーバーコンソールに確実に表示するため print を使用)
        unique_cats = len(set(item["category"] for item in semantic_labels))
        print("\n=== Candidate Selection QUBO Solve ===")
        print(f"Solver: {result['solver']}")
        print(f"Categories: {unique_cats}")
        print(f"Elements: {len(semantic_labels)}")
        # 候補数の平均（または範囲）を表示
        avg_cands = len(variables) / len(semantic_labels) if semantic_labels else 0
        print(f"Candidates per element: {avg_cands:.1f} (avg)")
        print(f"Binary variables: {len(variables)}")
        print(f"Best total energy: {result['total_energy']}")
        print(f"E1: {result['e1']}")
        print(f"E2: {result['e2']}")
        print("=======================================\n")

        return jsonify({
            "success": True,
            "best_selection": best_selection_map,
            "total_energy": result["total_energy"],
            "e1": result["e1"],
            "e2": result["e2"],
            "solver": result["solver"]
        })

    except Exception as e:
        logging.error(f"Quantum optimization failed: {e}")
        return jsonify({"error": str(e)}), 500

# ---------- 下書き管理 (Draft Management) API ----------

@app.route("/api/evaluation/current_structure", methods=["GET"])
def get_current_structure():
    """評価タブで選択された現在の構成テキストセットを取得する（順序維持リスト形式）"""
    user_id = session.get("user_id")
    doc_id = request.args.get("doc_id")
    if not user_id or not doc_id:
        return jsonify({"error": "Unauthorized or missing doc_id"}), 401

    data = load_user_data(user_id)
    document = find_document(data, doc_id)
    if not document:
        return jsonify({"error": "Document not found"}), 404

    best_selection = document.get("best_selection", {})
    suggestions = document.get("llm_suggestions", [])
    
    if not best_selection:
        return jsonify({"error": "評価タブで構成が選択されていません。"}), 400

    # 順序を維持するためのリスト形式: [ { category: "...", elements: [ { label: "...", text: "..." }, ... ] } ]
    ordered_structure = []
    try:
        # llm_suggestions の並び順（評価タブの表示順）でループ
        for cat_sug in suggestions:
            category_name = cat_sug["category"]
            if category_name not in best_selection:
                continue
            
            category_elements = []
            # そのカテゴリ内の要素を取得（llm_suggestions内での並び順を維持）
            selected_indices = best_selection[category_name]
            
            for element_name, proposals in cat_sug.get("elements", {}).items():
                if element_name in selected_indices:
                    idx = selected_indices[element_name]
                    if 0 <= idx < len(proposals):
                        category_elements.append({
                            "label": element_name,
                            "text": proposals[idx]
                        })
            
            if category_elements:
                ordered_structure.append({
                    "category": category_name,
                    "elements": category_elements
                })

        return jsonify({
            "success": True,
            "structure": ordered_structure
        })
    except Exception as e:
        logging.error(f"Failed to restore ordered structure: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/draft/start_from_evaluation", methods=["POST"])
def start_from_evaluation():
    """評価結果の選択を基に執筆プロセスを開始する"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    params = request.get_json()
    doc_id = params.get("doc_id")
    # selected_elements: [{"category": "...", "element": "...", "text": "..."}, ...]
    selected_elements = params.get("selected_elements", [])
    additional_info = params.get("additional_info", {})
    mode = params.get("mode", "generate") # "generate" or "prompt_only"
    
    target_type = params.get("target_type")
    chapter_id = params.get("chapter_id")
    scene_id = params.get("scene_id")

    if not selected_elements:
        return jsonify({"error": "構成要素が選択されていません。"}), 400

    dm = DraftManager(user_id, doc_id)
    document = dm.document
    if not document:
        return jsonify({"error": "Document not found"}), 404

    try:
        # 1. structure_snapshotの構築
        scene_structure = "\n\n".join([f"【{e['category']} > {e['element']}】\n{e['text']}" for e in selected_elements])
        
        structure_snapshot = {
            "scene_structure": scene_structure,
            "location": additional_info.get("location"),
            "time": additional_info.get("time"),
            "weather": additional_info.get("weather"),
            "viewpoint": additional_info.get("viewpoint"),
            "style": additional_info.get("style"),
            "length": additional_info.get("length"),
            "instructions": additional_info.get("instructions")
        }

        # 2. プロンプト生成
        prompt = build_draft_prompt(structure_snapshot)

        # 3. モード別処理
        if mode == "prompt_only":
            return jsonify({
                "success": True,
                "mode": "prompt_only",
                "prompt": prompt
            })

        elif mode == "generate":
            # 1. セッションから設定を取得 (Dashboardで保存されたもの)
            llm_servers = session.get("llm_servers", {})
            llm_config = llm_servers.get("drafting", {}) # Dashboardでは 'drafting' というキー名
            
            # 2. セッションになければ working.json から取得
            if not llm_config or not llm_config.get("api_key"):
                settings = dm.data.get("settings", {}).get("llm_servers", {})
                llm_config = settings.get("drafting", {}) # 'draft' から 'drafting' に統一
            
            # 3. それでもなければ環境変数 (最終手段)
            if not llm_config or not llm_config.get("api_key"):
                llm_config = {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": os.environ.get("OPENAI_API_KEY", "")
                }

            if not llm_config.get("api_key") and llm_config.get("provider") in ["openai", "chatgpt"]:
                 return jsonify({"error": "LLMのAPIキーが設定されていません。ダッシュボードで設定してください。"}), 400

            content = generate_draft(prompt, llm_config)

            # 保存処理
            now_str = datetime.now().isoformat()
            if target_type == "scene":
                draft = dm.add_scene_draft(chapter_id, scene_id, content, prompt, structure_snapshot)
                if not draft:
                    return jsonify({"error": "指定されたシーンが見つかりませんでした。"}), 404
                # 追加後に時刻を上書き（DraftManagerの既存仕様を最小限の変更で補完）
                draft["created_at"] = now_str
            else:
                draft = dm.add_chapter_draft(chapter_id, content, prompt)
                if not draft:
                    return jsonify({"error": "指定された章が見つかりませんでした。"}), 404
                draft["created_at"] = now_str
            
            dm.save()
            return jsonify({
                "success": True,
                "mode": "generate",
                "draft_id": draft["draft_id"],
                "content": content
            })

        return jsonify({"error": "Invalid mode"}), 400

    except Exception as e:
        logging.error(f"Start from evaluation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/draft/generate", methods=["POST"])
def draft_generate():
    """下書き生成API"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    params = request.get_json()
    target_type = params.get("target_type") # "scene" or "chapter"
    chapter_id = params.get("chapter_id")
    scene_id = params.get("scene_id")
    structure_override = params.get("structure_override")
    additional_info = params.get("additional_info")
    
    dm = DraftManager(user_id, params.get("doc_id"))
    
    # LLM設定の取得 (working.json の settings.llm_servers.drafting を参照)
    llm_config = dm.data.get("settings", {}).get("llm_servers", {}).get("drafting", {})
    if not llm_config or not llm_config.get("provider"):
        # デフォルト設定がなければ作成
        llm_config = {
            "provider": "openai", # デフォルト
            "model": "gpt-4o",
            "api_key": os.environ.get("OPENAI_API_KEY", "")
        }
        
    try:
        if target_type == "scene":
            scene = dm.find_scene(chapter_id, scene_id)
            if not scene:
                return jsonify({"error": "Scene not found"}), 404
            
            # スナップショット優先、なければシーンの設定から構築
            structure = structure_override if structure_override else scene.get("structure_snapshot", {})
            prompt = build_draft_prompt(structure, additional_info)
            content = generate_draft(prompt, llm_config)
            
            draft = dm.add_scene_draft(chapter_id, scene_id, content, prompt, structure)
            dm.save()
            return jsonify({"success": True, "draft": draft})
            
        elif target_type == "chapter":
            chapter = dm.find_chapter(chapter_id)
            if not chapter:
                return jsonify({"error": "Chapter not found"}), 404
            
            structure = structure_override if structure_override else {}
            prompt = build_draft_prompt(structure, additional_info)
            content = generate_draft(prompt, llm_config)
            
            draft = dm.add_chapter_draft(chapter_id, content, prompt)
            dm.save()
            return jsonify({"success": True, "draft": draft})
        
        else:
            return jsonify({"error": "Invalid target_type"}), 400
            
    except Exception as e:
        logging.error(f"Draft generation API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/draft/save_configs", methods=["POST"])
def draft_save_configs():
    """下書きタブの設定（設計図と演出設定）を保存するAPI"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    params = request.get_json()
    doc_id = params.get("doc_id")
    story_blueprint = params.get("story_blueprint")
    directing_settings = params.get("directing_settings")

    if not doc_id:
        return jsonify({"error": "Missing doc_id"}), 400

    dm = DraftManager(user_id, doc_id)
    if dm.save_drafting_configs(story_blueprint, directing_settings):
        dm.save()
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Document not found"}), 404

@app.route("/api/draft/list", methods=["GET"])
def draft_list():
    """章・シーンの一覧取得"""
    user_id = session.get("user_id")
    doc_id = request.args.get("doc_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    dm = DraftManager(user_id, doc_id)
    
    response_data = {
        "success": True, 
        "manuscript": dm.manuscript,
    }
    # drafting_configsがあれば含める
    if dm.document and "drafting_configs" in dm.document:
        response_data["drafting_configs"] = dm.document["drafting_configs"]
        
    return jsonify(response_data)

@app.route("/api/draft/get", methods=["GET"])
def draft_get():
    """詳細取得"""
    user_id = session.get("user_id")
    doc_id = request.args.get("doc_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    chapter_id = request.args.get("chapter_id")
    scene_id = request.args.get("scene_id")
    
    dm = DraftManager(user_id, doc_id)
    if scene_id:
        scene = dm.find_scene(chapter_id, scene_id)
        return jsonify({"success": True, "scene": scene})
    else:
        chapter = dm.find_chapter(chapter_id)
        return jsonify({"success": True, "chapter": chapter})

@app.route("/api/draft/update", methods=["POST"])
def draft_update():
    """章・シーンの更新または追加"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    params = request.get_json()
    doc_id = params.get("doc_id")
    chapter_id = params.get("chapter_id")
    scene_id = params.get("scene_id")
    title = params.get("title")
    order = params.get("order")
    manuscript_content = params.get("manuscript_content")
    
    dm = DraftManager(user_id, doc_id)

    # 全文保存の場合
    if manuscript_content is not None:
        dm.manuscript["full_text"] = manuscript_content
        # バージョン更新
        current_version = dm.manuscript.get("version", 0)
        dm.manuscript["version"] = current_version + 1
        dm.manuscript["last_saved"] = datetime.now().isoformat()
        dm.save()
        return jsonify({"success": True, "version": dm.manuscript["version"]})

    if chapter_id and scene_id:
        if scene_id == 'new':
            dm.add_scene(chapter_id, title or "新しいシーン")
        else:
            scene = dm.find_scene(chapter_id, scene_id)
            if scene:
                if title is not None: scene["title"] = title
                if order is not None: scene["order"] = order
    elif chapter_id:
        if chapter_id == 'new':
            dm.add_chapter(title or "新しい章")
        else:
            chapter = dm.find_chapter(chapter_id)
            if chapter:
                if title is not None: chapter["title"] = title
                if order is not None: chapter["order"] = order
    else:
        # 両方ない場合は新規章
        dm.add_chapter(title or "新しい章", order)
            
    dm.save()
    return jsonify({"success": True})

@app.route("/api/draft/status", methods=["POST"])
def draft_status():
    """生成ステータス確認 (今回は同期的なので常に idle)"""
    return jsonify({"success": True, "status": "idle"})

@app.route("/api/draft/delete", methods=["POST"])
def draft_delete():
    """原稿案または章・シーンの削除API"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    params = request.get_json()
    doc_id = params.get("doc_id")
    draft_id = params.get("draft_id")
    chapter_id = params.get("chapter_id")
    scene_id = params.get("scene_id")
    
    dm = DraftManager(user_id, doc_id)
    deleted = False

    # 章またはシーンの削除
    if chapter_id and scene_id:
        deleted = dm.delete_scene(chapter_id, scene_id)
    elif chapter_id:
        deleted = dm.delete_chapter(chapter_id)
    # 原稿案（ドラフト）の削除
    elif draft_id:
        # 章レベルのドラフトから探す
        for chap in dm.manuscript.get("chapters", []):
            drafts = chap.get("chapter_level_drafts", [])
            new_drafts = [d for d in drafts if d["draft_id"] != draft_id]
            if len(new_drafts) < len(drafts):
                chap["chapter_level_drafts"] = new_drafts
                deleted = True
                break
            
            # シーンレベルのドラフトから探す
            for scene in chap.get("scenes", []):
                s_drafts = scene.get("drafts", [])
                new_s_drafts = [d for d in s_drafts if d["draft_id"] != draft_id]
                if len(new_s_drafts) < len(s_drafts):
                    scene["drafts"] = new_s_drafts
                    deleted = True
                    break
            if deleted: break

    if deleted:
        dm.save()
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Item not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
