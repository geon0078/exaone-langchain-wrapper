# save_chat.py

import json
import os

def save_chat_history(new_history, filename):
    """대화 기록을 파일에 추가 저장 (기존 데이터 유지)"""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    # 기존 기록 + 새 기록 합치기
    combined_history = existing + new_history

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, ensure_ascii=False, indent=2)
