from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


LEADS_PATH = Path(__file__).with_name("leads.json")


def _read_leads() -> list[dict]:
    if not LEADS_PATH.exists():
        return []

    try:
        return json.loads(LEADS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    leads = _read_leads()
    for lead in leads:
        if (
            lead.get("name") == name
            and lead.get("email") == email
            and lead.get("platform") == platform
        ):
            return f"Lead already captured for {name} on {platform} using {email}."

    leads.append(
        {
            "name": name,
            "email": email,
            "platform": platform,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    LEADS_PATH.write_text(json.dumps(leads, indent=2), encoding="utf-8")
    return f"Lead captured successfully for {name} on {platform} using {email}."
