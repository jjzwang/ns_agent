import re
from typing import List
NS_HINTS = [r"\bNetSuite\b", r"\bPO\b", r"\bVendor Bill\b", r"\bSuiteTax\b"]
def extract_entities(text: str) -> List[str]:
    ents = []
    if re.search(r"\bPO(s)?\b", text, re.I): ents.append("Purchase Order")
    if re.search(r"Vendor Bill", text, re.I): ents.append("Vendor Bill")
    return ents
def route(text: str) -> dict:
    entities = extract_entities(text)
    systems = ["netsuite"] if any(re.search(p, text) for p in NS_HINTS) or entities else ["other"]
    flow = "TASK_EXEC" if re.search(r"\bfind\b|\bbill\b|\bsubmit\b", text, re.I) else "QNA"
    return {"systems": systems, "flow": flow, "confidence": 0.75}
