"""Điểm tin cậy GIẢ cho demo, không phải xác suất model hay vi phạm."""
import hashlib
import json


def demo_confidence(key):
    # Giả ngẫu nhiên theo khóa cố định để restart/retry không đổi điểm.
    digest=hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).digest()
    return {'confidence':(85+int.from_bytes(digest[:4],'big')%14)/100,
            'confidence_source':'demo'}


def api_demo_confidence(event):
    for record in (event.get('clip') or {},event):
        score=record.get('confidence')
        if record.get('confidence_source')=='demo' and isinstance(score,(int,float)) and .85<=score<=.98:
            return {'confidence':score,'confidence_source':'demo'}
    return demo_confidence((event['event_id'],event.get('detected_at')))
