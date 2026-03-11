import json
import os
import uuid
from typing import Dict, Any, List, Optional
from user_files import get_user_data_path, load_user_data, save_user_data

class DraftManager:
    def __init__(self, user_id: str, doc_id: Optional[str] = None):
        self.user_id = user_id
        self.doc_id = doc_id
        self.data = load_user_data(user_id)
        self.document = self._get_document(doc_id) if doc_id else None
        self._init_structures()

    def _get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        for doc in self.data.get("documents", []):
            if doc.get("id") == doc_id:
                return doc
        return None

    def _init_structures(self):
        """データを初期化し、必要に応じて移行を行う"""
        # グローバルなmanuscriptが存在し、かつ特定のドキュメントが指定されている場合の移行
        if "manuscript" in self.data and self.document:
            if "manuscript" not in self.document:
                self.document["manuscript"] = self.data.pop("manuscript")
        
        # ドキュメント固有のmanuscript初期化
        if self.document:
            if "manuscript" not in self.document:
                self.document["manuscript"] = {"chapters": []}
            if "drafting_configs" not in self.document:
                self.document["drafting_configs"] = {
                    "story_blueprint": [],
                    "directing_settings": {}
                }
        else:
            # フォールバック: 指定がない場合はトップレベル（旧仕様互換）
            if "manuscript" not in self.data:
                self.data["manuscript"] = {"chapters": []}

        # settingsの初期化
        if "settings" not in self.data:
            self.data["settings"] = {"llm_servers": {}}
        
        draft_cfg = self.data["settings"]["llm_servers"].setdefault("drafting", {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": ""
        })

    @property
    def manuscript(self):
        if self.document:
            return self.document["manuscript"]
        return self.data["manuscript"]

    def save(self):
        """現在のデータを保存する"""
        save_user_data(self.user_id, self.data)

    def generate_id(self, prefix: str = "") -> str:
        """ユニークなIDを生成する"""
        return f"{prefix}{uuid.uuid4().hex[:8]}"

    def find_chapter(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        """指定されたIDの章を探す"""
        for chap in self.manuscript.get("chapters", []):
            if chap["chapter_id"] == chapter_id:
                return chap
        return None

    def find_scene(self, chapter_id: str, scene_id: str) -> Optional[Dict[str, Any]]:
        """指定されたIDのシーンを探す"""
        chapter = self.find_chapter(chapter_id)
        if not chapter:
            return None
        for scene in chapter.get("scenes", []):
            if scene["scene_id"] == scene_id:
                return scene
        return None

    def add_chapter(self, title: str, order: Optional[int] = None) -> Dict[str, Any]:
        """新しい章を追加する"""
        chapters = self.manuscript.setdefault("chapters", [])
        if order is None:
            order = len(chapters) + 1
        
        new_chapter = {
            "chapter_id": self.generate_id("chap_"),
            "title": title,
            "order": order,
            "chapter_level_drafts": [],
            "scenes": []
        }
        chapters.append(new_chapter)
        chapters.sort(key=lambda x: x["order"])
        return new_chapter

    def add_scene(self, chapter_id: str, title: str, order: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """章に新しいシーンを追加する"""
        chapter = self.find_chapter(chapter_id)
        if not chapter:
            return None
        
        scenes = chapter.setdefault("scenes", [])
        if order is None:
            order = len(scenes) + 1
            
        new_scene = {
            "scene_id": self.generate_id("scene_"),
            "title": title,
            "order": order,
            "structure_snapshot": {},
            "drafts": []
        }
        scenes.append(new_scene)
        scenes.sort(key=lambda x: x["order"])
        return new_scene

    def add_chapter_draft(self, chapter_id: str, content: str, prompt_used: str):
        """章レベルの下書きを追加する"""
        chapter = self.find_chapter(chapter_id)
        if not chapter:
            return None
        
        draft = {
            "draft_id": self.generate_id("drft_"),
            "content": content,
            "prompt_used": prompt_used,
            "created_at": uuid.uuid4().hex
        }
        chapter.setdefault("chapter_level_drafts", []).append(draft)
        return draft

    def add_scene_draft(self, chapter_id: str, scene_id: str, content: str, prompt_used: str, structure_snapshot: Dict[str, Any]):
        """シーンの下書きを追加する"""
        scene = self.find_scene(chapter_id, scene_id)
        if not scene:
            return None
        
        draft = {
            "draft_id": self.generate_id("drft_"),
            "content": content,
            "prompt_used": prompt_used,
            "structure_snapshot": structure_snapshot,
            "created_at": uuid.uuid4().hex
        }
        scene.setdefault("drafts", []).append(draft)
        scene["structure_snapshot"] = structure_snapshot
        return draft

    def delete_chapter(self, chapter_id: str) -> bool:
        """章を削除する"""
        chapters = self.manuscript.get("chapters", [])
        initial_count = len(chapters)
        self.manuscript["chapters"] = [c for c in chapters if c["chapter_id"] != chapter_id]
        return len(self.manuscript["chapters"]) < initial_count

    def delete_scene(self, chapter_id: str, scene_id: str) -> bool:
        """シーンを削除する"""
        chapter = self.find_chapter(chapter_id)
        if not chapter:
            return False
        
        scenes = chapter.get("scenes", [])
        initial_count = len(scenes)
        chapter["scenes"] = [s for s in scenes if s["scene_id"] != scene_id]
        return len(chapter["scenes"]) < initial_count

    def save_drafting_configs(self, story_blueprint: List[Dict[str, Any]], directing_settings: Dict[str, Any]):
        """下書きタブの設定（設計図と演出設定）を保存する"""
        if not self.document:
            return False
        self.document["drafting_configs"] = {
            "story_blueprint": story_blueprint,
            "directing_settings": directing_settings
        }
        return True
