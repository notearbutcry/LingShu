import asyncio
from pathlib import Path
import sys
import threading
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knowledge_api import create_server, KnowledgeAPIClient
from lingshu_backend import execute_pipeline
from fixtures import FakeModel


class HTTPToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server=create_server(0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.endpoint=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)

    async def test_actual_http_roundtrip_and_workflow(self):
        client=KnowledgeAPIClient(self.endpoint)
        hits=await asyncio.to_thread(client.search,'水泵振动')
        self.assertTrue(hits)
        result=await execute_pipeline('水泵振动','',1,lambda *a:None,model_call=FakeModel(),
                                     knowledge_endpoint=self.endpoint)
        self.assertEqual(result['status'],'PENDING_HUMAN')
        self.assertEqual(result['knowledge_mode'],'http_api')
        self.assertEqual(result['knowledge_version'],client.version)

    async def test_unconfigured_remote_endpoint_rejected(self):
        with self.assertRaises(ValueError):KnowledgeAPIClient('https://example.com')


if __name__=='__main__':unittest.main()
