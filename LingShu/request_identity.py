"""Identity of an explicit UI request; no model-output cache or shared user state."""
import hashlib
import json
import os
from pathlib import Path


def request_fingerprint(task, *, strategy, max_concurrency, max_revisions,
                        self_loop_test, knowledge_endpoint=None, root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    files = ['config/agents.json', 'knowledge_base/documents.json',
             'platform_runtime.py', 'lingshu_backend.py', 'scheduler.py',
             'registry.py', 'knowledge.py', 'knowledge_api.py',
             'json_contract.py', 'agent_memory.py', 'recovery.py', 'recover_run.py', 'app.py']
    payload = dict(task=task.strip(), strategy=strategy, max_concurrency=max_concurrency,
                   max_revisions=max_revisions, self_loop_test=self_loop_test,
                   knowledge_endpoint=knowledge_endpoint,
                   model=os.environ.get('LINGSHU_MODEL', 'deepseek-chat'),
                   files={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
