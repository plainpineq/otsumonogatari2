from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from core.models import Document, Unit, Entity, Intent, CompositionMeta
import sqlite3
import json
import os
import uuid
from pathlib import Path

User = get_user_model()

class Command(BaseCommand):
    help = 'Migrate data from legacy JSON files and users.db'

    def handle(self, *args, **options):
        self.migrate_users()
        self.migrate_json_data()

    def migrate_users(self):
        db_path = "users.db"
        if not os.path.exists(db_path):
            self.stdout.write(self.style.WARNING(f"{db_path} not found. Skipping user migration."))
            return

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM users")
            users = cursor.fetchall()
            for row in users:
                email = row['email']
                username = email.split('@')[0]
                if User.objects.filter(username=username).exists():
                    username = email 
                
                if User.objects.filter(email=email).exists():
                    self.stdout.write(f"User {email} already exists. Skipping.")
                    continue

                user = User(
                    username=username,
                    email=email,
                    password=row['password_hash'], 
                    date_joined=row['created_at'] if row['created_at'] else "2024-01-01 00:00:00"
                )
                user.save()
                self.stdout.write(self.style.SUCCESS(f"Migrated user: {email}"))
        except Exception as e:
             self.stdout.write(self.style.ERROR(f"Error migrating users: {e}"))
        finally:
            conn.close()

    def migrate_json_data(self):
        base_dir = Path("user_data")
        self.stdout.write(f"Scanning for user data in: {base_dir.absolute()}")
        if not base_dir.exists():
            self.stdout.write(self.style.WARNING("user_data directory not found. Skipping data migration."))
            return

        for user_dir in base_dir.iterdir():
            if not user_dir.is_dir():
                continue
            
            user_email = user_dir.name 
            self.stdout.write(f"Processing user directory: {user_email}")
            
            try:
                user = User.objects.get(email=user_email)
            except User.DoesNotExist:
                self.stdout.write(self.style.WARNING(f"User {user_email} not found in Django DB. Skipping data."))
                continue

            json_path = user_dir / "working.json"
            self.stdout.write(f"Checking for JSON file at: {json_path}")
            if not json_path.exists():
                self.stdout.write(self.style.WARNING(f"File not found: {json_path}"))
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    self.stdout.write(f"Successfully loaded JSON for {user_email}")
                except json.JSONDecodeError as e:
                    self.stdout.write(self.style.ERROR(f"JSON Decode Error for {user_email}: {e}"))
                    continue
            
            documents = data.get("documents", [])
            self.stdout.write(f"Found {len(documents)} documents for {user_email}")
            for doc_data in documents:
                doc_id = doc_data.get("id")
                if Document.objects.filter(id=doc_id).exists():
                    self.stdout.write(f"Document {doc_id} already exists. Skipping.")
                    continue

                # Create Document
                doc = Document.objects.create(
                    id=doc_id,
                    title=doc_data.get("title", "Untitled"),
                    synopsis=doc_data.get("synopsis", ""),
                    doc_type=doc_data.get("doc_type", "novel"),
                    owner=user,
                )
                self.stdout.write(f"Migrated Document: {doc.title} ({doc_id})")

                # Create Units
                for order, unit_data in enumerate(doc_data.get("units", [])):
                    unit_id = unit_data.get("id")
                    if not unit_id:
                        unit_id = str(uuid.uuid4())
                        
                    Unit.objects.create(
                        id=unit_id, 
                        document=doc,
                        title=unit_data.get("title", ""),
                        summary=unit_data.get("content", ""), 
                        order_no=order
                    )

                # Create Entities
                for ent_data in doc_data.get("entities", []):
                    ent_id = ent_data.get("id")
                    if not ent_id:
                        ent_id = str(uuid.uuid4())

                    Entity.objects.create(
                        id=ent_id,
                        document=doc,
                        name=ent_data.get("name", ""),
                        role=ent_data.get("role", ""),
                        description=ent_data.get("description", "")
                    )

                # Create Intent
                intent_data = doc_data.get("intent", {})
                if intent_data:
                    Intent.objects.create(
                        document=doc,
                        genre=intent_data.get("genre", ""),
                        theme_or_claim=intent_data.get("theme_or_claim", ""),
                        core_values=intent_data.get("values", ""),
                        constraints=intent_data.get("constraints", [])
                    )

                # Create CompositionMeta
                if "composition" in doc_data:
                     CompositionMeta.objects.create(
                        document=doc,
                        data=doc_data["composition"]
                     )
