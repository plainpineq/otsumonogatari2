import os
import json

def _load_intent_templates():
    """Loads the intent templates from an external JSON file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'intent_templates.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert lists of lists back to lists of tuples
    common_intents = [tuple(item) for item in data['common_intents']]
    doc_type_intents = {k: [tuple(item) for item in v] for k, v in data['doc_type_intents'].items()}
    
    return common_intents, doc_type_intents

COMMON_INTENTS, DOC_TYPE_INTENTS = _load_intent_templates()