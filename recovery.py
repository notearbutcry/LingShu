"""Explicit recovery of local, read-only workflow stages; always re-run review."""
import hashlib
import json
from pathlib import Path

STAGES = ('plan', 'evidence', 'analysis', 'diagnosis')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def execution_signature(options, profiles, knowledge_version):
    root = Path(__file__).resolve().parent
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
    return digest(dict(options=options, profiles=profiles, knowledge=knowledge_version, sources=sources))


def restore_outputs(source, *, task, signature, execution_mode, rollback_to=None):
    if not isinstance(source, dict):
        raise ValueError('恢复文件必须为JSON对象')
    source = source.get('result', source)
    meta = source.get('recovery_context', {})
    if (not isinstance(meta, dict) or meta.get('schema') != 1 or meta.get('task') != task
            or meta.get('signature') != signature or meta.get('execution_mode') != execution_mode
            or meta.get('knowledge_mode') != 'local_plugin'):
        raise ValueError('恢复被拒绝：任务、代码、配置、知识版本或运行模式不匹配；仅支持本地知识库')
    if rollback_to is not None and rollback_to not in STAGES:
        raise ValueError('回退位置必须是 plan/evidence/analysis/diagnosis')
    snapshots = source.get('state_snapshots', source.get('snapshots', []))
    audit = source.get('workflow_audit', source.get('audit', []))
    mains = [s for s in snapshots if isinstance(s, dict) and s.get('workflow') == 'main']
    if not mains:
        raise ValueError('没有可恢复的主流程快照')
    snapshot = mains[-1]
    outputs = snapshot.get('outputs', {})
    restored = {}
    for name in STAGES:
        if name == rollback_to or snapshot.get('states', {}).get(name) != 'DONE':
            break
        if name not in outputs:
            raise ValueError('快照缺少已完成节点的输出')
        value = outputs[name]
        matches = [e for e in audit if e.get('workflow') == 'main' and e.get('node') == name and e.get('state') == 'DONE']
        if not matches or matches[-1].get('output_sha256') != digest(value):
            raise ValueError('快照输出与审计哈希不一致，拒绝复用')
        if ((name == 'plan' and not isinstance(value, dict))
                or (name == 'evidence' and not isinstance(value, list))
                or (name == 'analysis' and (not isinstance(value,dict) or not isinstance(value.get('knowledge'),str)))
                or (name == 'diagnosis' and not isinstance(value,str))):
            raise ValueError('快照节点数据格式错误')
        restored[name] = value
    return restored, meta.get('run_id', ''), rollback_to
