"""Explicit workflow recovery. No previous review or human approval is reused."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from lingshu_backend import execute_pipeline
from run_store import save_record, CHECKPOINT_DIR


def read_source(path):
    path = Path(path)
    if path.stat().st_size > 20_000_000:
        raise ValueError('恢复文件过大')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('恢复文件格式错误')
    return data.get('result', data)


async def recover(source, api_key, emit, *, model_call=None, rollback_to=None, checkpoint_dir=CHECKPOINT_DIR):
    meta = source.get('recovery_context', {})
    options = meta.get('options', {})
    if not meta or not options:
        raise ValueError('此记录不含2.1.0版恢复信息，请先完成一次新运行')
    return await execute_pipeline(meta['task'], api_key, options['max_revisions'], emit,
        strategy=options['strategy'], max_concurrency=options['max_concurrency'],
        self_loop_test=options['self_loop_test'], knowledge_endpoint=options['knowledge_endpoint'],
        model_call=model_call, checkpoint_dir=checkpoint_dir,
        recovery_source=source, rollback_to=rollback_to)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from', dest='source', required=True, help='本机运行记录或检查点JSON')
    parser.add_argument('--rollback-to', choices=['plan','evidence','analysis','diagnosis'])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--offline', action='store_true', help='只允许恢复离线记录')
    mode.add_argument('--live', action='store_true', help='真实模型调用，产生API费用')
    args = parser.parse_args()
    source = read_source(args.source)
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if args.live and not key:
        import tomllib
        config = Path(__file__).parent / '.streamlit/secrets.toml'
        if config.exists():
            key = str(tomllib.loads(config.read_text(encoding='utf-8-sig')).get('DEEPSEEK_API_KEY','')).strip()
    model = None
    if args.offline:
        import sys
        sys.path.insert(0,str(Path(__file__).parent/'tests'))
        from fixtures import FakeModel
        model = FakeModel(domain=source.get('plan',{}).get('domain','general'))
    events = []
    def emit(agent,status,message,round_no=0):
        events.append(dict(agent=agent,status=status,message=message,round=round_no))
    result = asyncio.run(recover(source,key,emit,model_call=model,rollback_to=args.rollback_to))
    print(save_record(result,events))
    print(result['status'], result['execution_mode'], '新run_id='+result['run_id'])
    print('本次模型调用：',result['metrics']['model_calls'],'；审核与人工确认均不继承')


if __name__ == '__main__':
    main()
