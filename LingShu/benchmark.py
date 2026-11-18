"""Reproducible paired runs. Explicit --offline or --live is required."""
import argparse
import asyncio
import csv
import json
import math
import os
import platform
import hashlib
import importlib.metadata
from pathlib import Path
import sys
from datetime import datetime
from lingshu_backend import execute_pipeline
from privacy import redact


def percentile(values, p):
    """Nearest-rank quantile, shown in milliseconds."""
    return sorted(values)[max(0, math.ceil(len(values)*p)-1)] if values else None


async def evaluate(args):
    dataset = json.loads(Path(args.dataset).read_text(encoding='utf-8'))
    if not isinstance(dataset, list) or not dataset:
        raise ValueError('数据集需要非空JSON数组')
    seen = set()
    for item in dataset:
        if (not isinstance(item,dict) or not isinstance(item.get('id'),str) or not item['id'].strip()
                or item['id'] in seen or not isinstance(item.get('task'),str)
                or not 1 <= len(item['task'].strip()) <= 2000):
            raise ValueError('每个评测案例必须有唯一id和1～2000字task')
        seen.add(item['id'])
    key = '' if args.offline else os.environ.get('DEEPSEEK_API_KEY', '').strip()
    secrets = Path(__file__).parent / '.streamlit' / 'secrets.toml'
    if args.live and not key and secrets.exists():
        import tomllib
        key = str(tomllib.loads(secrets.read_text(encoding='utf-8-sig')).get('DEEPSEEK_API_KEY', '')).strip()
    if args.live and not key:
        raise ValueError('真实评测需要本地配置密钥')
    if args.offline:
        sys.path.insert(0, str(Path(__file__).parent / 'tests'))
        from fixtures import FakeModel
    output = Path(args.output) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    # Record provenance before the run; these are configuration facts, not quality claims.
    root = Path(__file__).resolve().parent
    manifest = {
        'started_at': datetime.now().astimezone().isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'mode': 'offline_fixture' if args.offline else 'live_model',
        'model': os.environ.get('LINGSHU_MODEL', 'deepseek-chat'),
        'repeats': args.repeats,
        'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(root.glob('*.py'))},
        'dependencies': {},
        'human_review': 'pending; text pass is not diagnostic accuracy',
    }
    for package in ('streamlit', 'autogen-agentchat', 'autogen-ext', 'openai'):
        try:
            manifest['dependencies'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest['dependencies'][package] = None
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    variants=[('serial_no_loop','serial',0),('serial','serial',2),('dynamic','dynamic',2)]
    for repeat in range(args.repeats):
        for index, item in enumerate(dataset):
            # Rotate evaluation order to reduce time-of-day/provider load confounding.
            shift=(repeat+index)%len(variants)
            for variant, strategy, revisions in variants[shift:]+variants[:shift]:
                model=FakeModel(domain=item.get('domain','general')) if args.offline else None
                result=await execute_pipeline(item['task'],key,revisions,lambda *a:None,
                                              strategy=strategy,model_call=model)
                run_id=result['run_id']
                (output/(run_id+'.json')).write_text(json.dumps(redact(result),ensure_ascii=False,indent=2),encoding='utf-8')
                m=result['metrics']
                rows.append(dict(case_id=item['id'],repeat=repeat+1,variant=variant,run_id=run_id,
                    expected_behavior=item.get('expected_behavior','需人工评阅'),
                    mode=result['execution_mode'],status=result['status'],elapsed_ms=m['elapsed_ms'],
                    model_calls=m['model_calls'],prompt_tokens=m['prompt_tokens'],completion_tokens=m['completion_tokens'],
                    model_attempts=m['model_attempts'],model_attempt_ms=m['model_attempt_ms'],retries=m['retries'],
                    peak_agents=m['peak_active_agents'],revisions=result['revision_count'],
                    human_quality='',human_notes=''))
    with (output/'runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary={'mode':'OFFLINE_FIXTURE_NOT_MODEL_PERFORMANCE' if args.offline else 'LIVE_MODEL',
             'quantile':'nearest_rank', 'latency_population':'non_ERROR_runs',
             'note':'文本通过率不等于准确率；REJECTED可以是正确的风险升级。人工质量待填。',
             'variants':{}}
    for variant,_,_ in variants:
        rs=[r for r in rows if r['variant']==variant]
        values=[r['elapsed_ms'] for r in rs if r['status']!='ERROR']
        summary['variants'][variant]=dict(count=len(rs),completed_without_error=sum(r['status']!='ERROR' for r in rs),
            text_pass_count=sum(r['status']=='PENDING_HUMAN' for r in rs),
            p50_ms=percentile(values,.5),p95_ms=percentile(values,.95),
            text_pass_rate=sum(r['status']=='PENDING_HUMAN' for r in rs)/len(rs),
            error_count=sum(r['status']=='ERROR' for r in rs),
            mean_model_calls=sum(r['model_calls'] for r in rs)/len(rs),
            mean_tokens=sum(r['prompt_tokens']+r['completion_tokens'] for r in rs)/len(rs),
            mean_revisions=sum(r['revisions'] for r in rs)/len(rs))
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(output.resolve()))
    print(summary['mode'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--offline',action='store_true',help='替身模型，只验证评测脚本')
    mode.add_argument('--live',action='store_true',help='真实模型，产生API费用')
    parser.add_argument('--dataset',default=str(Path(__file__).parent/'evaluation'/'cases.json'))
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--output',default='evaluation_results')
    args=parser.parse_args()
    if not 1<=args.repeats<=20: parser.error('repeats应为1～20')
    asyncio.run(evaluate(args))
