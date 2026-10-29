"""Read optional correlation metadata from both crop message formats."""

import json


def request_id_from_message(body: bytes, content_type: str | None, headers: dict | None) -> str:
    request_id = (headers or {}).get("request_id")
    if request_id is not None:
        return str(request_id)
    if content_type == "image/jpeg":
        return ""
    try:
        payload = json.loads(body.decode("utf-8"))
        if isinstance(payload, dict) and payload.get("request_id") is not None:
            return str(payload["request_id"])
    except (ValueError, UnicodeDecodeError):
        pass
    return ""
