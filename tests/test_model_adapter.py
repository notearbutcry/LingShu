import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

@unittest.skipUnless(importlib.util.find_spec('openai') and importlib.util.find_spec('autogen_agentchat'),
                     '先安装模型客户端依赖')
class ModelAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_network_retry_is_bounded_and_counted(self):
        from platform_runtime import AutoGenRunner
        from registry import load_profiles
        metrics=dict(model_attempts=0,model_attempt_ms=0,prompt_tokens=0,completion_tokens=0,retries=0)
        runner=AutoGenRunner('offline-placeholder',metrics,lambda *a:None)
        try:
            with patch('autogen_agentchat.agents.AssistantAgent') as agent:
                agent.return_value.run = AsyncMock(side_effect=ConnectionError('offline test'))
                with self.assertRaisesRegex(RuntimeError, 'ConnectionError'):
                    await runner(load_profiles()[0], 'test')
                self.assertEqual(agent.return_value.run.await_count, 2)
            self.assertEqual(metrics['model_attempts'], 2)
            self.assertEqual(metrics['retries'], 1)
        finally:
            await runner.close()

    async def test_autogen_role_with_replay_client(self):
        from autogen_ext.models.replay import ReplayChatCompletionClient
        from platform_runtime import AutoGenRunner
        from registry import load_profiles
        metrics=dict(model_attempts=0,model_attempt_ms=0,prompt_tokens=0,completion_tokens=0,retries=0)
        # Constructor validation exercises the real OpenAI-compatible adapter without a request.
        runner=AutoGenRunner('offline-placeholder',metrics,lambda *a:None)
        self.assertEqual(runner.client._client.max_retries, 0,
                         'SDK retries must not bypass platform attempt accounting')
        await runner.close()
        runner.client=ReplayChatCompletionClient(['offline replay response'])
        try:
            response=await runner(load_profiles()[0],'测试消息')
            self.assertEqual(response,'offline replay response')
            self.assertEqual(metrics['model_attempts'],1)
        finally:
            await runner.close()

if __name__=='__main__':unittest.main()
