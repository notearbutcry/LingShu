"""Local keyword retrieval with visible source IDs; no fabricated vector search."""
from pathlib import Path
import hashlib
import json
import re


def tokens(text):
    text = text.lower()
    result = set(re.findall(r"[a-z0-9_]+", text))
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        result.update(chunk[i:i+2] for i in range(max(1, len(chunk)-1)))
    return result


class KnowledgeBase:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(__file__).parent / "knowledge_base" / "documents.json"
        raw = self.path.read_bytes()
        if len(raw) > 2_000_000:
            raise ValueError("知识库超过2MB，请分批整理")
        self.version = hashlib.sha256(raw).hexdigest()
        self.documents = json.loads(raw)
        seen = set()
        if not isinstance(self.documents, list) or len(self.documents) > 1000:
            raise ValueError("知识库应为不超过1000项的JSON数组")
        for doc in self.documents:
            if not isinstance(doc, dict) or any(not isinstance(doc.get(k), str) or not doc[k].strip()
                                               for k in ("id", "title", "source", "text")):
                raise ValueError("知识条目缺少id/title/source/text")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", doc["id"]) or doc["id"] in seen:
                raise ValueError("知识条目编号重复或无效")
            if len(doc["text"]) > 12000 or type(doc.get("demo", True)) is not bool:
                raise ValueError("知识条目过长或demo字段非布尔值")
            seen.add(doc["id"])

    def search(self, query, top_k=3):
        if not isinstance(query,str) or not 1 <= len(query) <= 2000 or type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValueError('检索参数无效')
        q = tokens(query)
        hits = []
        for doc in self.documents:
            doc_tokens = tokens(doc["title"] + " " + doc["text"])
            score = len(q & doc_tokens) / max(1, len(q))
            if score:
                hits.append({**doc, "demo": doc.get("demo", True), "score": round(score, 4)})
        return sorted(hits, key=lambda h: (-h["score"], h["id"]))[:top_k]


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, handler):
        if name in self.tools or not callable(handler):
            raise ValueError("工具重复或不可调用")
        self.tools[name] = handler

    def call(self, name, **kwargs):
        if name not in self.tools:
            raise ValueError("工具未注册")
        return self.tools[name](**kwargs)
