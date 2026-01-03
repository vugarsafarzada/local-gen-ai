import json
import os
from typing import List, Dict, Any

OUTPUTS_DIR = "outputs"
HISTORY_FILE = os.path.join(OUTPUTS_DIR, "history.json")

def ensure_outputs_dir():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

def load_history() -> List[Dict[str, Any]]:
    ensure_outputs_dir()
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_history(history: List[Dict[str, Any]]):
    ensure_outputs_dir()
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def add_history_item(item: Dict[str, Any]):
    history = load_history()
    history.append(item)
    save_history(history)

def delete_history_item(item_id: str) -> bool:
    history = load_history()
    
    # Find item to get filename
    item = next((x for x in history if x['id'] == item_id), None)
    
    if item:
        # Delete image file
        image_path = os.path.join(OUTPUTS_DIR, item['filename'])
        if os.path.exists(image_path):
            os.remove(image_path)
            
        # Remove from list
        history = [x for x in history if x['id'] != item_id]
        save_history(history)
        return True
    return False

def clear_history():
    history = load_history()
    for item in history:
        image_path = os.path.join(OUTPUTS_DIR, item['filename'])
        if os.path.exists(image_path):
            os.remove(image_path)
    save_history([])