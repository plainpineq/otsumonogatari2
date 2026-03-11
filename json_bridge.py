# json_bridge.py
from typing import Dict

from models import Document, Unit, Entity, Intent
from services.services import (
    get_document,
    list_units,
    list_entities,
    get_intent,
    create_unit,
    create_entity,
    save_intent
)
from db import get_conn
import uuid
import json


# =========================
# JSON → Domain
# =========================

def import_document_from_json(json_doc: Dict) -> str:
    """
    JSON形式の document を Domain(DB) に取り込む。
    すでに存在する場合は何もしない。
    戻り値: document_id
    """

    document_id = json_doc["id"]

    # ---- Document ----
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM story WHERE id=?",
            (document_id,)
        ).fetchone()

        if not row:
            conn.execute("""
                INSERT INTO story (id, title, synopsis, doc_type)
                VALUES (?, ?, ?, ?)
            """, (
                document_id,
                json_doc.get("title", ""),
                "",
                json_doc.get("doc_type", "novel")
            ))
            conn.commit()

    # ---- Units ----
    existing_units = {
        u.title for u in list_units(document_id)
    }

    for order, unit in enumerate(json_doc.get("units", [])):
        if unit.get("title") in existing_units:
            continue

        create_unit(
            document_id=document_id,
            title=unit.get("title", ""),
            summary=unit.get("content", ""),
            order_no=order
        )

    # ---- Entities ----
    existing_entities = {
        e.name for e in list_entities(document_id)
    }

    for ent in json_doc.get("entities", []):
        if ent.get("name") in existing_entities:
            continue

        create_entity(
            document_id=document_id,
            name=ent.get("name", ""),
            role=ent.get("role", ""),
            description=ent.get("description", "")
        )

    # ---- Intent ----
    intent_json = json_doc.get("intent", {})
    if intent_json:
        intent = Intent(
            document_id=document_id,
            genre=intent_json.get("genre", ""),
            theme_or_claim=intent_json.get("theme_or_claim", ""),
            values=intent_json.get("values", ""),
            constraints=intent_json.get("constraints", [])
        )
        save_intent(intent)

    return document_id
