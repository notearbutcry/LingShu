import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu_backend import basic_text_check, detect_unsafe_patterns, execute_feedback_loop
from request_identity import request_fingerprint
from fixtures import KNOWLEDGE, DIAGNOSIS


class ReviewRulesTests(unittest.TestCase):
    def test_markdown_does_not_create_missing_safety_phrases(self):
        self.assertEqual(basic_text_check('请**停机**并**隔离能源**；穿戴 **防护用品**。'),(True,[]))
        self.assertFalse(basic_text_check('停机；缺少其他防护说明。')[0])

    def test_prohibited_lists_are_not_dangerous_recommendations(self):
        for text in ['禁止带电拆卸、屏蔽联锁、解除安全联锁。',
                     '禁止绕过联锁、带电拆卸或接触运行部件。',
                     '**严禁**带电维修、带电拆卸。']:
            with self.subTest(text=text):
                self.assertEqual(detect_unsafe_patterns(text),[])

    def test_prohibition_scope_does_not_hide_recommendations(self):
        for text in ['禁止带电拆卸、但可以屏蔽联锁。',
                     '禁止带电拆卸。建议屏蔽联锁。',
                     '禁止带电拆卸\n屏蔽联锁后继续。',
                     '不禁止带电拆卸。', '禁止其他操作、建议带电维修。',
                     '建议带电拆卸、屏蔽联锁。']:
            with self.subTest(text=text):
                self.assertTrue(detect_unsafe_patterns(text))

    def test_request_identity_includes_content_and_configuration(self):
        root=Path(__file__).resolve().parents[1]
        opts=dict(strategy='dynamic',max_concurrency=2,max_revisions=2,self_loop_test=False)
        original=request_fingerprint('水泵振动',**opts)
        self.assertEqual(original,request_fingerprint(' 水泵振动 ',**opts))
        for field,value in [('strategy','serial'),('max_concurrency',1),('max_revisions',1),('self_loop_test',True)]:
            self.assertNotEqual(original,request_fingerprint('水泵振动',**{**opts,field:value}))
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)
            for name in ['config/agents.json','knowledge_base/documents.json',
                         'platform_runtime.py','lingshu_backend.py','scheduler.py',
                         'registry.py','knowledge.py','knowledge_api.py',
                         'json_contract.py','agent_memory.py','recovery.py','recover_run.py','app.py']:
                (target/name).parent.mkdir(parents=True,exist_ok=True)
                shutil.copy(root/name,target/name)
            self.assertEqual(original,request_fingerprint('水泵振动',root=target,**opts))
            with (target/'knowledge_base/documents.json').open('a') as f:f.write('\n')
            self.assertNotEqual(original,request_fingerprint('水泵振动',root=target,**opts))


class FeedbackEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def run_feedback(self,raw,diagnosis=DIAGNOSIS):
        async def review(text): return raw
        async def revise(*args): self.fail('No revision expected')
        return await execute_feedback_loop(task='测试',knowledge=KNOWLEDGE,initial_diagnosis=diagnosis,
            review_call=review,revise_call=revise,max_revisions=0,emit=lambda *args:None)

    async def test_invalid_review_is_format_error_not_equipment_risk(self):
        result=await self.run_feedback('```json broken')
        self.assertEqual(result['status'],'ERROR')
        self.assertEqual(result['stop_code'],'REVIEW_FORMAT_ERROR')
        self.assertEqual(result['revision_history'][0]['review_raw'],'```json broken')

    async def test_valid_unknown_and_high_still_block(self):
        for risk,code in [('high','HIGH_RISK'),('unknown','UNKNOWN_RISK')]:
            result=await self.run_feedback(json.dumps(dict(passed=True,risk_level=risk,issues=[])))
            self.assertEqual(result['status'],'REJECTED')
            self.assertEqual(result['stop_code'],code)
            self.assertTrue(result['blocking_issues'])

    async def test_prohibited_list_passes_but_affirmative_danger_does_not(self):
        review=json.dumps(dict(passed=True,risk_level='low',issues=[]))
        safe=await self.run_feedback(review,DIAGNOSIS+'禁止带电拆卸、屏蔽联锁。')
        unsafe=await self.run_feedback(review,DIAGNOSIS+'建议带电拆卸、屏蔽联锁。')
        self.assertEqual(safe['status'],'PENDING_HUMAN')
        self.assertEqual(unsafe['status'],'REJECTED')
        self.assertTrue(any('危险文本' in x for x in unsafe['blocking_issues']))


if __name__=='__main__': unittest.main()
