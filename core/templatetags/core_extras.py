import json
from django import template
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models.query import QuerySet

register = template.Library()

@register.filter(name='tojson')
def tojson(value):
    try:
        if isinstance(value, QuerySet):
            return json.dumps(list(value.values()), cls=DjangoJSONEncoder)
        # Django Model インスタンスの場合
        if hasattr(value, '__dict__'):
            data = value.__dict__.copy()
            data.pop('_state', None)
            # Remove non-serializable fields if any
            return json.dumps(data, cls=DjangoJSONEncoder)
        return json.dumps(value, cls=DjangoJSONEncoder)
    except Exception as e:
        return json.dumps({"error": str(e)})

@register.filter(name='to_json')
def to_json(value):
    return tojson(value)
