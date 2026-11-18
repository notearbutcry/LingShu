"""Dependency/configuration check. --live explicitly makes one paid API call."""
import argparse
import asyncio
import importlib.metadata
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='实际调用一次模型，可能产生费用')
    args = parser.parse_args()
    print('Python：',sys.version.split()[0])
    print('解释器：',sys.executable)
    print('项目目录：',Path(__file__).resolve().parent)
    if sys.version_info < (3,11):
        raise SystemExit('需要Python 3.11或以上，推荐在Anaconda Prompt创建Python 3.12环境。')
    missing = []
    for name in ('streamlit', 'autogen-agentchat', 'autogen-ext', 'autogen-core', 'openai', 'pandas', 'plotly', 'nest-asyncio'):
        try:
            print(name, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    from registry import load_profiles
    from knowledge import KnowledgeBase
    print('角色配置：', len(load_profiles()), '个')
    print('知识条目：', len(KnowledgeBase().documents), '条（默认仅为演示资料）')
    if missing:
        raise SystemExit('缺少依赖：'+', '.join(missing)+'；请先安装 requirements.txt')
    print('本地依赖与配置检查通过；这不等于多Agent完整流程通过。')
    if args.live:
        key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
        config = Path(__file__).parent / '.streamlit' / 'secrets.toml'
        if not key and config.exists():
            import tomllib
            key = str(tomllib.loads(config.read_text(encoding='utf-8-sig')).get('DEEPSEEK_API_KEY', '')).strip()
        if not key:
            raise SystemExit('未配置密钥；请使用环境变量或本地secrets文件，不要写入源码。')
        asyncio.run(live_check(key))


async def live_check(key):
    from platform_runtime import AutoGenRunner
    from registry import load_profiles
    metrics = dict(model_attempts=0, model_attempt_ms=0, prompt_tokens=0, completion_tokens=0, retries=0)
    runner = AutoGenRunner(key, metrics, lambda *args: None)
    try:
        await runner(load_profiles()[0], '返回一个domain为general的有效任务规划JSON。')
        print('单次模型调用成功；请再运行真实案例验证完整平台。')
    finally:
        await runner.close()


if __name__ == '__main__':
    main()
