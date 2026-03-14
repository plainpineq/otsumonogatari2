from django.db import models
from django.conf import settings
import uuid

class Document(models.Model):
    """
    1つの文章成果物を表す。
    小説・論文・ブログなどを doc_type で区別する。
    """
    id = models.CharField(primary_key=True, max_length=50, default=uuid.uuid4)
    title = models.CharField(max_length=200)
    synopsis = models.TextField(blank=True, default="")
    doc_type = models.CharField(max_length=50, default="novel")
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

class Unit(models.Model):
    """
    Documentを構成する最小の構造単位。
    小説: シーン
    論文: 節 / サブセクション
    """
    id = models.CharField(primary_key=True, max_length=50, default=uuid.uuid4)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='units')
    title = models.CharField(max_length=200)
    summary = models.TextField(blank=True, default="")
    order_no = models.IntegerField()
    time_start = models.BigIntegerField(null=True, blank=True)
    time_end = models.BigIntegerField(null=True, blank=True)
    location = models.CharField(max_length=200, blank=True, default="")

    class Meta:
        ordering = ['order_no']

    def __str__(self):
        return self.title

class Entity(models.Model):
    """
    Document内で参照・登場する要素。
    小説: キャラクター
    """
    id = models.CharField(primary_key=True, max_length=50, default=uuid.uuid4)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='entities')
    name = models.CharField(max_length=200)
    role = models.CharField(max_length=100, blank=True, default="")
    description = models.TextField(blank=True, default="")

    def __str__(self):
        return self.name

class Intent(models.Model):
    """
    作者の判断・価値観を表すドメインモデル
    """
    # SQLiteのauthor_contextテーブルの主キーはstory_idなので、OneToOneで対応
    document = models.OneToOneField(Document, on_delete=models.CASCADE, related_name='intent', primary_key=True)
    genre = models.CharField(max_length=100, blank=True, default="")
    theme_or_claim = models.TextField(blank=True, default="")
    core_values = models.TextField(blank=True, default="") # values in db
    constraints = models.JSONField(default=list, blank=True) # JSON list of strings

    def __str__(self):
        return f"Intent for {self.document.title}"

class CompositionMeta(models.Model):
    """
    JSONファイルベースで管理されていた詳細な構成情報を格納する。
    """
    document = models.OneToOneField(Document, on_delete=models.CASCADE, related_name='composition_meta')
    data = models.JSONField(default=dict)

    def __str__(self):
        return f"Composition for {self.document.title}"
