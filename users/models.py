from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    """
    カスタムユーザーモデル。
    Emailをユーザー名（ログイン用）として使用する。
    """
    email = models.EmailField('メールアドレス', unique=True)
    username = models.CharField(
        'ユーザー名',
        max_length=150,
        unique=True,
        help_text='150文字以内の英数字・記号（@/./+/-/_）が使用可能です。',
        validators=[AbstractUser.username_validator],
        error_messages={
            'unique': "そのユーザー名は既に使用されています。",
        },
        null=True, blank=True
    )

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def __str__(self):
        return self.email
