import json
import os
import logging
import uuid
import time
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse, Http404
from django.views.decorators.http import require_POST
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.utils import timezone
from django.contrib import messages

# from .models import Document, Unit, Entity, Intent, CompositionMeta
from user_files import load_user_data, save_user_data
from intent_templates import COMMON_INTENTS, DOC_TYPE_INTENTS

# 文書タイプ（ラベルから内部キーへのマッピング）
DOC_TYPE_MAP = {
    '小説': 'novel',
    '脚本': 'script',
    '論文': 'thesis',
    '記事': 'article',
    '随筆': 'essay',
    'novel': 'novel',
    'script': 'script',
    'thesis': 'thesis',
    'article': 'article',
    'essay': 'essay'
}

def get_meta_for_doc(user, doc_id):
    """working.jsonから特定のドキュメントを取得する"""
    user_data = load_user_data(user.email)
    if "documents" not in user_data:
        user_data["documents"] = []
    
    for doc in user_data["documents"]:
        if doc.get("id") == doc_id:
            if "data" not in doc:
                doc["data"] = {}
            return doc, user_data
    return None, user_data

def save_meta_for_doc(user, user_data):
    """working.jsonを保存する"""
    save_user_data(user.email, user_data)

def get_doc_or_404(user, doc_id):
    """ドキュメントをJSONから取得し、見つからない場合はHttp404を返す"""
    doc, user_data = get_meta_for_doc(user, doc_id)
    if not doc:
        raise Http404(f"Document {doc_id} not found in user data.")
    return doc, user_data

from services.services import (
    generate_proposals as srv_generate_proposals,
    generate_composition as srv_generate_composition,
    DEFAULT_COMPOSITION_META,
)
from evaluation_engine import calculate_energy_detail
from services.candidate_qubo import generate_candidate_selection_qubo
from services.candidate_solver import solve_candidate_selection_qubo
from semantic_labeler import label_suggestions
from draft_context_builder import build_draft_prompt
from llm_router import generate_draft as llm_generate_draft

@login_required
def dashboard(request):
    """ユーザーのドキュメント一覧とサーバー設定を表示する"""
    user_data = load_user_data(request.user.email)
    documents = user_data.get("documents", [])

    llm_servers = request.session.get("llm_servers", {})
    quantum_server = request.session.get("quantum_server", {})

    for role in ["generation", "evaluation", "drafting"]:
        llm_servers.setdefault(role, {})

    user_config = {
        "llm_servers": llm_servers,
        "quantum_server": quantum_server,
        "suggestion_count": request.session.get("suggestion_count", 3)
    }

    llm_models_config = {}
    try:
        config_path = os.path.join(settings.BASE_DIR, "llm_models_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                llm_models_config = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load llm_models_config.json: {e}")

    api_keys_config = {"api_keys_enabled": False, "llm_providers": [], "analysis_servers": []}
    try:
        keys_path = os.path.join(settings.BASE_DIR, "api_keys_config.json")
        if os.path.exists(keys_path):
            with open(keys_path, "r", encoding="utf-8") as f:
                api_keys_config = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load api_keys_config.json: {e}")

    return render(
        request,
        "dashboard.html",
        {
            "documents": documents,
            "doc_types": DOC_TYPE_INTENTS.keys(),
            "user_config": user_config,
            "llm_models_config": llm_models_config,
            "api_keys_config": api_keys_config,
        }
    )

@login_required
def document_detail(request, doc_id):
    """ドキュメントの詳細（編集・生成画面）を表示する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    meta = doc # alias for existing code

    if request.method == 'POST' and request.POST.get('update_composition_elements'):
        # 構成要素の更新処理
        categories = meta['data'].get('composition_elements', {}).get('categories', [])
        
        # カテゴリ情報の更新
        for cat in categories:
            # ラベルの更新
            new_cat_label = request.POST.get(f'category_{cat["id"]}_label')
            if new_cat_label:
                cat['label'] = new_cat_label
            
            # 追加指示の更新
            new_instruction = request.POST.get(f'category_{cat["id"]}_additional_instruction')
            if new_instruction is not None:
                cat['additional_instruction'] = new_instruction
            
            # 各要素ラベルの更新
            if 'elements' in cat:
                for el in cat['elements']:
                    new_el_label = request.POST.get(f'element_label_{cat["id"]}_{el["id"]}')
                    if new_el_label:
                        el['label'] = new_el_label
        
        # 新しいカテゴリの追加
        if 'add_doc_type_category' in request.POST:
            new_cat_label = request.POST.get('new_doc_type_category_label')
            if new_cat_label:
                new_cat_id = str(uuid.uuid4())[:8]
                categories.append({
                    "id": new_cat_id,
                    "label": new_cat_label,
                    "editable": True,
                    "elements": []
                })
        
        # カテゴリの削除
        if 'remove_doc_type_category' in request.POST:
            remove_cat_id = request.POST.get('remove_doc_type_category')
            categories = [c for c in categories if c['id'] != remove_cat_id]
        
        # 各カテゴリ内での要素追加
        for cat in categories:
            if f'add_element_{cat["id"]}' in request.POST:
                new_el_id = str(uuid.uuid4())[:8]
                if 'elements' not in cat: cat['elements'] = []
                cat['elements'].append({
                    "id": new_el_id,
                    "label": f"新しい{cat['label']}",
                    "editable": True
                })
            
            # 要素の削除
            if 'elements' in cat:
                for el in list(cat['elements']):
                    if f'remove_element_{cat["id"]}_{el["id"]}' in request.POST:
                        cat['elements'] = [e for e in cat['elements'] if e['id'] != el['id']]

        meta['data']['composition_elements'] = {"categories": categories}
        save_meta_for_doc(request.user, user_data)
        messages.success(request, "構成要素を更新しました。")
        return redirect('document_detail', doc_id=doc_id)

    schema_path = os.path.join(settings.BASE_DIR, "prompt_templates", "semantic_label_schema.json")
    semantic_label_schema = {}
    try:
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                semantic_label_schema = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load semantic_label_schema.json: {e}")

    # デフォルトのインテント項目を補完
    intent_data = meta['data'].get('intent', {})
    if 'fields' not in intent_data:
        intent_data['fields'] = {}
    
    # 共通項目
    for key, label in COMMON_INTENTS:
        if key not in intent_data['fields']:
            intent_data['fields'][key] = {"label": label, "value": ""}
    
    # 文書タイプ別項目
    doc_type_label = doc.get('doc_type', 'novel')
    if doc_type_label == 'novel': doc_type_label = '小説'
    elif doc_type_label == 'script': doc_type_label = '脚本'
    elif doc_type_label == 'thesis': doc_type_label = '論文'
    elif doc_type_label == 'article': doc_type_label = '記事'
    elif doc_type_label == 'essay': doc_type_label = '随筆'

    if doc_type_label in DOC_TYPE_INTENTS:
        for key, label in DOC_TYPE_INTENTS[doc_type_label]:
            if key not in intent_data['fields']:
                intent_data['fields'][key] = {"label": label, "value": ""}

    # 構成要素の補完 (composition_meta.json からのデフォルト)
    composition_elements = meta['data'].get('composition_elements', {})
    if not composition_elements or not composition_elements.get('categories'):
        doc_type_raw = doc.get('doc_type', 'novel')
        doc_type_key = DOC_TYPE_MAP.get(doc_type_raw, doc_type_raw)
        doc_type_meta = DEFAULT_COMPOSITION_META.get('doc_types', {}).get(doc_type_key, {})
        if doc_type_meta:
            composition_elements = {
                "categories": doc_type_meta.get('categories', [])
            }
            # ファイルに保存して次回以降も確実に読み込めるようにする
            meta['data']['composition_elements'] = composition_elements
            save_meta_for_doc(request.user, user_data)

    doc_context = {
        "id": doc['id'],
        "title": doc.get('title', '無題'),
        "synopsis": doc.get('synopsis', ''),
        "doc_type": doc.get('doc_type', 'novel'),
        "intent": intent_data,
        "genre_config": meta['data'].get('genre_config', {}),
        "composition_elements": composition_elements,
        "llm_suggestions": meta['data'].get('llm_suggestions', []),
        "selected_basic_elements": meta['data'].get('selected_basic_elements', {}),
        "semantic_labels": meta['data'].get('semantic_labels', []),
        "best_selection": meta['data'].get('best_selection', {}),
        "optimized_selection": meta['data'].get('optimized_selection', {}),
        "best_selection_energy": meta['data'].get('best_selection_energy', {}),
        "evaluation_config": meta['data'].get('evaluation_config', {}),
    }

    available_genres = ["ファンタジー", "SF", "ミステリー", "恋愛", "ホラー", "歴史"]
    genres_path = os.path.join(settings.BASE_DIR, "genre_targets_presets.json")
    try:
        if os.path.exists(genres_path):
            with open(genres_path, "r", encoding="utf-8") as f:
                genres_data = json.load(f)
                available_genres = list(genres_data.keys())
    except:
        pass

    doc_type_raw = doc.get('doc_type', 'novel')
    mapped_doc_type_id = DOC_TYPE_MAP.get(doc_type_raw, doc_type_raw)
    doc_type_meta_data = DEFAULT_COMPOSITION_META.get('doc_types', {}).get(mapped_doc_type_id, {})

    context = {
        'document': doc_context,
        'doc_type_meta_data': doc_type_meta_data,
        'semantic_label_schema': semantic_label_schema,
        'semantic_label_keys': list(semantic_label_schema.keys()),
        'available_genres': available_genres,
        'mapped_doc_type_id': mapped_doc_type_id,
        'global_composition_meta': DEFAULT_COMPOSITION_META
    }

    return render(request, 'document.html', context)

@login_required
@require_POST
def document_create(request):
    """新しいドキュメントを作成する"""
    title = request.POST.get('title')
    doc_type = request.POST.get('doc_type', 'novel')
    
    user_data = load_user_data(request.user.email)
    if "documents" not in user_data:
        user_data["documents"] = []
    
    doc_id = str(uuid.uuid4())
    new_doc = {
        "id": doc_id,
        "title": title,
        "doc_type": doc_type,
        "synopsis": "",
        "data": {}
    }
    
    # デフォルトのインテントと構成要素を初期化
    intent_fields = {}
    # 共通インテント
    for key, label in COMMON_INTENTS:
        intent_fields[key] = {"label": label, "value": ""}
    
    # 文書タイプ別インテント
    doc_type_label = doc_type
    if doc_type == 'novel': doc_type_label = '小説'
    elif doc_type == 'script': doc_type_label = '脚本'
    elif doc_type == 'thesis': doc_type_label = '論文'
    elif doc_type == 'article': doc_type_label = '記事'
    elif doc_type == 'essay': doc_type_label = '随筆'

    if doc_type_label in DOC_TYPE_INTENTS:
        for key, label in DOC_TYPE_INTENTS[doc_type_label]:
            intent_fields[key] = {"label": label, "value": ""}
            
    new_doc['data']['intent'] = {"fields": intent_fields}
    
    # 構成要素の初期化 (composition_meta.json から)
    internal_doc_type = DOC_TYPE_MAP.get(doc_type, doc_type)
    doc_type_meta = DEFAULT_COMPOSITION_META.get('doc_types', {}).get(internal_doc_type, {})
    if doc_type_meta:
        new_doc['data']['composition_elements'] = {
            "categories": doc_type_meta.get('categories', [])
        }
    
    user_data["documents"].append(new_doc)
    save_user_data(request.user.email, user_data)
    
    return redirect('document_detail', doc_id=doc_id)

@login_required
@require_POST
def save_servers_config(request):
    """サーバー設定（LLM, 量子）をセッションに保存する"""
    try:
        data = json.loads(request.body)
        request.session['llm_servers'] = data.get('llm_servers', {})
        request.session['quantum_server'] = data.get('quantum_server', {})
        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)

@login_required
@require_POST
def generate_proposals(request, doc_id):
    """ステップ1: 物語の案を生成する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    try:
        intent_data = doc['data'].get('intent', {})
        fields = intent_data.get('fields', {})
        
        intent_dict = {k: v.get('value', '') for k, v in fields.items()}
        # サービス側が期待する特定のキーへのマッピング (互換性のため)
        if 'theme' in intent_dict and 'theme_or_claim' not in intent_dict:
            intent_dict['theme_or_claim'] = intent_dict['theme']
        if 'genre' not in intent_dict:
            intent_dict['genre'] = doc['data'].get('genre_config', {}).get('main', '')
        
        llm_servers = request.session.get("llm_servers", {})
        suggestion_count = request.session.get("suggestion_count", 3)

        suggestions = srv_generate_proposals(doc_id, intent_dict, llm_servers, suggestion_count)
        
        doc['data']['llm_suggestions'] = suggestions
        save_meta_for_doc(request.user, user_data)

        return JsonResponse({"success": True, "suggestions": suggestions})
    except Exception as e:
        logging.exception("Failed to generate proposals")
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
@require_POST
def save_selection(request, doc_id):
    """ステップ2: 選択された案を保存する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    try:
        data = json.loads(request.body)
        if 'title' in data:
            doc['title'] = data['title']
        if 'plot' in data:
            doc['synopsis'] = data['plot']
        
        doc['data']['selected_basic_elements'] = data
        save_meta_for_doc(request.user, user_data)
        
        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
@require_POST
def generate_composition(request, doc_id):
    """ステップ3: カテゴリ別の構成要素を生成する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    try:
        data = json.loads(request.body)
        category_label = data.get('category_label')
        
        llm_servers = request.session.get("llm_servers", {})
        suggestion_count = request.session.get("suggestion_count", 3)
        
        intent_data = doc['data'].get('intent', {})
        fields = intent_data.get('fields', {})
        intent_dict = {k: v.get('value', '') for k, v in fields.items()}
        if 'theme' in intent_dict and 'theme_or_claim' not in intent_dict:
            intent_dict['theme_or_claim'] = intent_dict['theme']
        if 'genre' not in intent_dict:
            intent_dict['genre'] = doc['data'].get('genre_config', {}).get('main', '')
        
        selected_elements = doc['data'].get('selected_basic_elements', {})
        
        suggestions = srv_generate_composition(
            doc_id, category_label, intent_dict, selected_elements, llm_servers, suggestion_count
        )
        
        if 'llm_suggestions' not in doc['data']:
            doc['data']['llm_suggestions'] = []
        
        found = False
        for i, s in enumerate(doc['data']['llm_suggestions']):
            if s.get('category') == category_label:
                doc['data']['llm_suggestions'][i] = suggestions[0]
                found = True
                break
        if not found:
            doc['data']['llm_suggestions'].extend(suggestions)
            
        save_meta_for_doc(request.user, user_data)
        return JsonResponse({"success": True, "suggestions": suggestions})
    except Exception as e:
        logging.exception("Failed to generate composition")
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
def download_document(request, doc_id):
    """ドキュメントをJSONとしてダウンロードする"""
    doc, _ = get_doc_or_404(request.user, doc_id)
    
    data = {
        "id": doc['id'],
        "title": doc.get('title', '無題'),
        "synopsis": doc.get('synopsis', ''),
        "doc_type": doc.get('doc_type', 'novel'),
        "composition": doc.get('data', {})
    }
    
    response = HttpResponse(json.dumps(data, ensure_ascii=False, indent=2), content_type='application/json')
    response['Content-Disposition'] = f'attachment; filename="{doc.get("title", "document")}.json"'
    return response

@login_required
@require_POST
def api_save_document(request, doc_id):
    """各タブからの設定・データを保存する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    try:
        data = json.loads(request.body)
        
        if 'intent' in data:
            doc['data']['intent'] = data['intent']
        if 'manuscript_full_text' in data:
            doc['data']['manuscript_full_text'] = data['manuscript_full_text']
        if 'evaluation_config' in data:
            doc['data']['evaluation_config'] = data['evaluation_config']
            
        save_meta_for_doc(request.user, user_data)
        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
@require_POST
def evaluate_start(request, doc_id):
    """評価処理の開始（クライアント側のSSE接続を促す）"""
    return JsonResponse({"status": "ok"})

@login_required
def evaluate_stream(request, doc_id):
    """SSEを使用して評価進捗と結果をストリーミング返信する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    
    def event_stream():
        llm_servers = request.session.get("llm_servers", {})
        llm_config = llm_servers.get('evaluation', {})
        
        input_data = {"llm_suggestions": doc['data'].get('llm_suggestions', [])}
        all_labels = doc['data'].get('semantic_labels', [])

        for event in label_suggestions(input_data, llm_config, str(request.user.id)):
            if event['event'] == 'semantic_label':
                result_item = event['data']
                found = False
                for i, old in enumerate(all_labels):
                    if old['category'] == result_item['category'] and old['element'] == result_item['element']:
                        all_labels[i] = result_item
                        found = True
                        break
                if not found:
                    all_labels.append(result_item)
                
                yield f"data: {json.dumps({'semantic_label': result_item}, ensure_ascii=False)}\n\n"
            elif event['event'] == 'progress':
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            elif event['event'] == 'error':
                yield f"data: {json.dumps({'error': event['message']}, ensure_ascii=False)}\n\n"

        doc['data']['semantic_labels'] = all_labels
        save_meta_for_doc(request.user, user_data)
        yield "event: end_stream\ndata: {}\n\n"

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

@login_required
@require_POST
def calculate_energy(request, doc_id):
    """現在の選択に基づいた構造エネルギー（スコア）を計算する"""
    doc, _ = get_doc_or_404(request.user, doc_id)
    try:
        data = json.loads(request.body)
        selected_items = data.get('selected_items', [])
        
        schema_path = os.path.join(settings.BASE_DIR, "prompt_templates", "semantic_label_schema.json")
        schema = {}
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)

        eval_config = doc['data'].get('evaluation_config', {})
        
        energy_results = calculate_energy_detail(selected_items, eval_config, schema)
        return JsonResponse(energy_results)
    except Exception as e:
        logging.exception("Energy calculation failed")
        return JsonResponse({"error": str(e)}, status=500)

@login_required
@require_POST
def quantum_optimize(request, doc_id):
    """量子アニーリングまたはヒューリスティックで最適な組み合わせを探索する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    try:
        if not doc['data'] or 'semantic_labels' not in doc['data']:
            return JsonResponse({"success": False, "error": "評価結果がありません。"})

        semantic_labels = doc['data']['semantic_labels']
        eval_config = doc['data'].get('evaluation_config', {})
        quantum_config = request.session.get("quantum_server", {})
        api_key = quantum_config.get("api_key")

        schema_path = os.path.join(settings.BASE_DIR, "prompt_templates", "semantic_label_schema.json")
        label_mapping = {}
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
                for cat, labs in schema.items():
                    if cat == "scale": continue
                    label_mapping[cat] = {en: spec["ja_label"] for en, spec in labs.items()}

        Q, variables, element_ranges = generate_candidate_selection_qubo(semantic_labels, eval_config, label_mapping)
        result = solve_candidate_selection_qubo(Q, variables, element_ranges, api_key)

        if result.get('best_selection_indices'):
            best_selection = {}
            for idx in result['best_selection_indices']:
                var = variables[idx]
                cat = var["category"]
                el = var["element"]
                cand_idx = var["cand_idx"]
                if cat not in best_selection: best_selection[cat] = {}
                best_selection[cat][el] = cand_idx
            
            doc['data']['optimized_selection'] = best_selection
            doc['data']['best_selection'] = best_selection
            doc['data']['best_selection_energy'] = {
                "total": result['total_energy'],
                "e1": result['e1'],
                "e2": result['e2'],
                "solver": result['solver']
            }
            save_meta_for_doc(request.user, user_data)
            return JsonResponse({"success": True, "best_selection": best_selection, **result})
        
        return JsonResponse({"success": False, "error": "最適化に失敗しました。"})
    except Exception as e:
        logging.exception("Quantum optimization failed")
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
@require_POST
def save_manual_selection(request, doc_id):
    """手動での候補選択を保存する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    try:
        data = json.loads(request.body)
        category = data.get('category')
        element = data.get('element')
        index = data.get('index')

        if 'best_selection' not in doc['data']: doc['data']['best_selection'] = {}
        if category not in doc['data']['best_selection']: doc['data']['best_selection'][category] = {}

        doc['data']['best_selection'][category][element] = index
        save_meta_for_doc(request.user, user_data)
        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
def manuscript_data(request, doc_id):
    """原稿データを取得する"""
    doc, _ = get_doc_or_404(request.user, doc_id)
    
    manuscript = doc['data'].get('manuscript', {"chapters": []})
    drafts = doc['data'].get('drafts', [])
    manuscript_full_text = doc['data'].get('manuscript_full_text', "")
    
    return JsonResponse({
        "success": True,
        "data": manuscript,
        "drafts": drafts,
        "manuscript_full_text": manuscript_full_text
    })

@login_required
@require_POST
def save_manuscript(request, doc_id):
    """原稿全文を保存する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    try:
        data = json.loads(request.body)
        text = data.get('text', "")
        doc['data']['manuscript_full_text'] = text
        save_meta_for_doc(request.user, user_data)
        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
@require_POST
def generate_draft(request, doc_id):
    """下書きを生成する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    try:
        data = json.loads(request.body)
        target_type = data.get('target_type')
        target_id = data.get('target_id')
        mode = data.get('mode', 'generate')
        structure_override = data.get('structure')
        
        prompt = build_draft_prompt(structure_override, structure_override.get('additional_info', ""))
        
        if mode == 'prompt_only':
            return JsonResponse({"success": True, "prompt": prompt})
            
        llm_servers = request.session.get("llm_servers", {})
        llm_config = llm_servers.get("drafting", {})
        
        content = llm_generate_draft(prompt, llm_config)
        
        if 'drafts' not in doc['data']: doc['data']['drafts'] = []
            
        target_name = "不明な対象"
        manuscript = doc['data'].get('manuscript', {"chapters": []})
        if target_type == 'chapter':
            for chap in manuscript.get('chapters', []):
                if chap.get('chapter_id') == target_id: target_name = chap.get('title'); break
        elif target_type == 'scene':
            for chap in manuscript.get('chapters', []):
                for sc in chap.get('scenes', []):
                    if sc.get('scene_id') == target_id: target_name = sc.get('title'); break

        new_draft = {
            "id": str(uuid.uuid4())[:8],
            "target_type": target_type,
            "target_id": target_id,
            "target_name": target_name,
            "content": content,
            "prompt": prompt,
            "created_at": timezone.now().isoformat()
        }
        doc['data']['drafts'].append(new_draft)
        save_meta_for_doc(request.user, user_data)
        
        return JsonResponse({"success": True, "draft_id": new_draft["id"]})
    except Exception as e:
        logging.exception("Draft generation failed")
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
@require_POST
def save_draft_configs(request, doc_id):
    """下書き設定を保存する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    try:
        data = json.loads(request.body)
        doc['data']['drafting_configs'] = data
        save_meta_for_doc(request.user, user_data)
        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
@require_POST
def upload_document(request):
    """JSONファイルからドキュメントをインポートする"""
    if 'file' not in request.FILES:
        return JsonResponse({"success": False, "error": "ファイルがありません"}, status=400)
    
    f = request.FILES['file']
    try:
        content = f.read()
        try:
            data = json.loads(content.decode('utf-8'))
        except UnicodeDecodeError:
            data = json.loads(content.decode('shift_jis'))
            
        user_data = load_user_data(request.user.email)
        if "documents" not in user_data:
            user_data["documents"] = []
            
        new_doc = {
            "id": data.get('id', str(uuid.uuid4())),
            "title": data.get('title', f.name),
            "synopsis": data.get('synopsis', ""),
            "doc_type": data.get('doc_type', 'novel'),
            "data": data.get('composition', data)
        }
        user_data["documents"].append(new_doc)
        save_user_data(request.user.email, user_data)
        
        messages.success(request, f"作品「{new_doc['title']}」をインポートしました。")
        return redirect('dashboard')
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
@require_POST
def delete_document(request, doc_id):
    """ドキュメントを削除する"""
    user_data = load_user_data(request.user.email)
    documents = user_data.get("documents", [])
    
    new_documents = []
    found_title = "Unknown"
    found = False
    for doc in documents:
        if doc.get("id") == doc_id:
            found_title = doc.get("title", "Unknown")
            found = True
        else:
            new_documents.append(doc)
    
    if found:
        user_data["documents"] = new_documents
        save_user_data(request.user.email, user_data)
        messages.success(request, f"作品「{found_title}」を削除しました。")
    else:
        messages.error(request, "ドキュメントが見つかりませんでした。")
        
    return redirect('dashboard')

@login_required
@require_POST
def draft_update(request):
    """章・シーンの更新または追加"""
    try:
        params = json.loads(request.body)
        doc_id = params.get("doc_id")
        chapter_id = params.get("chapter_id")
        scene_id = params.get("scene_id")
        title = params.get("title")
        order = params.get("order")
        
        doc, user_data = get_doc_or_404(request.user, doc_id)
        
        if 'manuscript' not in doc['data']:
            doc['data']['manuscript'] = {"chapters": [], "full_text": ""}
        
        manuscript = doc['data']['manuscript']
        
        if scene_id:
            if scene_id == 'new':
                for chap in manuscript.get('chapters', []):
                    if chap.get('id') == chapter_id or chap.get('chapter_id') == chapter_id:
                        new_scene = {
                            "id": str(uuid.uuid4())[:8],
                            "scene_id": str(uuid.uuid4())[:8],
                            "title": title or "新しいシーン",
                            "content": "",
                            "order": len(chap.get('scenes', [])) + 1
                        }
                        chap.setdefault('scenes', []).append(new_scene)
                        break
            else:
                for chap in manuscript.get('chapters', []):
                    for sc in chap.get('scenes', []):
                        if sc.get('id') == scene_id or sc.get('scene_id') == scene_id:
                            if title is not None: sc["title"] = title
                            if order is not None: sc["order"] = order
                            break
        elif chapter_id:
            if chapter_id == 'new':
                new_chap = {
                    "id": str(uuid.uuid4())[:8],
                    "chapter_id": str(uuid.uuid4())[:8],
                    "title": title or "新しい章",
                    "scenes": [],
                    "order": len(manuscript.get('chapters', [])) + 1
                }
                manuscript.setdefault('chapters', []).append(new_chap)
            else:
                for chap in manuscript.get('chapters', []):
                    if chap.get('id') == chapter_id or chap.get('chapter_id') == chapter_id:
                        if title is not None: chap["title"] = title
                        if order is not None: chap["order"] = order
                        break
        else:
            new_chap = {
                "id": str(uuid.uuid4())[:8],
                "chapter_id": str(uuid.uuid4())[:8],
                "title": title or "新しい章",
                "scenes": [],
                "order": len(manuscript.get('chapters', [])) + 1
            }
            manuscript.setdefault('chapters', []).append(new_chap)

        save_meta_for_doc(request.user, user_data)
        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
@require_POST
def draft_delete(request):
    """章・シーンまたは下書きの削除"""
    try:
        params = json.loads(request.body)
        doc_id = params.get("doc_id")
        chapter_id = params.get("chapter_id")
        scene_id = params.get("scene_id")
        draft_id = params.get("draft_id")
        
        doc, user_data = get_doc_or_404(request.user, doc_id)
        
        if draft_id:
            drafts = doc['data'].get('drafts', [])
            doc['data']['drafts'] = [d for d in drafts if d.get('id') != draft_id]
        elif chapter_id and scene_id:
            manuscript = doc['data'].get('manuscript', {})
            for chap in manuscript.get('chapters', []):
                if chap.get('id') == chapter_id or chap.get('chapter_id') == chapter_id:
                    chap['scenes'] = [s for s in chap.get('scenes', []) if s.get('id') != scene_id and s.get('scene_id') != scene_id]
                    break
        elif chapter_id:
            manuscript = doc['data'].get('manuscript', {})
            manuscript['chapters'] = [c for c in manuscript.get('chapters', []) if c.get('id') != chapter_id and c.get('chapter_id') != chapter_id]

        save_meta_for_doc(request.user, user_data)
        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
def draft_list(request):
    """ドラフト一覧と原稿構成、設定を取得する"""
    doc_id = request.GET.get("doc_id")
    doc, _ = get_doc_or_404(request.user, doc_id)
    
    manuscript = doc['data'].get('manuscript', {"chapters": []})
    drafts = doc['data'].get('drafts', [])
    
    # 章・シーンに紐づくドラフトをマッピング (JSの期待する形式)
    for chap in manuscript.get('chapters', []):
        chap_id = chap.get('chapter_id')
        chap['chapter_level_drafts'] = [d for d in drafts if d.get('target_type') == 'chapter' and d.get('target_id') == chap_id]
        for sc in chap.get('scenes', []):
            sc_id = sc.get('scene_id')
            sc['drafts'] = [d for d in drafts if d.get('target_type') == 'scene' and d.get('target_id') == sc_id]

    return JsonResponse({
        "success": True,
        "manuscript": manuscript,
        "drafts": drafts,
        "drafting_configs": doc['data'].get('drafting_configs', {})
    })

@login_required
def draft_get(request):
    """章・シーンの詳細取得"""
    doc_id = request.GET.get("doc_id")
    chapter_id = request.GET.get("chapter_id")
    scene_id = request.GET.get("scene_id")
    
    doc, _ = get_doc_or_404(request.user, doc_id)
    manuscript = doc['data'].get('manuscript', {"chapters": []})
    
    if scene_id:
        for chap in manuscript.get('chapters', []):
            for sc in chap.get('scenes', []):
                if sc.get('id') == scene_id or sc.get('scene_id') == scene_id:
                    return JsonResponse({"success": True, "scene": sc})
    else:
        for chap in manuscript.get('chapters', []):
            if chap.get('id') == chapter_id or chap.get('chapter_id') == chapter_id:
                return JsonResponse({"success": True, "chapter": chap})
                
    return JsonResponse({"success": False, "error": "Not found"}, status=404)

@login_required
@require_POST
def intent_update(request, doc_id):
    """基本設定（インテント）を更新または項目を追加・削除する"""
    doc, user_data = get_doc_or_404(request.user, doc_id)
    
    if 'intent' not in doc['data']:
        doc['data']['intent'] = {"fields": {}}
    
    # ジャンル設定の更新
    main_genre = request.POST.get('main_genre')
    sub_genres = request.POST.getlist('sub_genres')
    doc['data']['genre_config'] = {
        "main": main_genre,
        "sub": sub_genres
    }
    
    # 既存項目の更新
    fields = doc['data']['intent'].get('fields', {})
    for key in list(fields.keys()):
        val = request.POST.get(f'intent_value_{key}')
        if val is not None:
            fields[key]['value'] = val
            
    # 項目の追加
    if 'add_intent' in request.POST:
        new_label = request.POST.get('new_intent_label')
        new_value = request.POST.get('new_intent_value')
        if new_label:
            new_key = str(uuid.uuid4())[:8]
            fields[new_key] = {"label": new_label, "value": new_value}
            
    # 項目の削除
    if 'remove_intent' in request.POST:
        remove_key = request.POST.get('remove_intent')
        if remove_key in fields:
            del fields[remove_key]
            
    doc['data']['intent']['fields'] = fields
    save_meta_for_doc(request.user, user_data)
    
    messages.success(request, "基本設定を保存しました。")
    return redirect('document_detail', doc_id=doc_id)

def help_gemini_api(request):
    return render(request, "help_gemini_api.html")

def help_amplify_api(request):
    return render(request, "help_amplify_api.html")

def usage(request):
    return render(request, "usage.html")
