import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu_backend import execute_pipeline, execute_feedback_loop, parse_review_json, detect_unsafe_patterns
from platform_runtime import parse_plan
from registry import AgentProfile, AgentRegistry
from scheduler import DAGExecutor, Node
from knowledge import KnowledgeBase
from run_store import save_record, load_record, list_records, save_checkpoint
from privacy import redact
from fixtures import FakeModel, KNOWLEDGE, DIAGNOSIS


class ValidationTests(unittest.TestCase):
    def test_json_rejects_missing_or_wrong_fields(self):
        for value in ['not json', '[]', '{"passed":"true"}',
                      '{"passed":true,"risk_level":"low"}',
                      '{"passed":true,"risk_level":"low","issues":[{}]}']:
            self.assertFalse(parse_review_json(value)['passed'])

    def test_high_or_unknown_cannot_pass(self):
        for risk in ['high', 'unknown', 'unexpected']:
            self.assertFalse(parse_review_json(json.dumps(dict(passed=True, risk_level=risk, issues=[])))['passed'])

    def test_valid_code_fence(self):
        self.assertTrue(parse_review_json('```json\n{"passed":true,"risk_level":"low","issues":[]}\n```')['passed'])

    def test_contrast_does_not_hide_danger(self):
        self.assertIn('带电拆卸', detect_unsafe_patterns('禁止其他行为但建议带电拆卸。'))
        self.assertIn('带电维修', detect_unsafe_patterns('不能排除需要带电维修。'))

    def test_prohibition_and_risk_description(self):
        self.assertEqual(detect_unsafe_patterns('禁止带电拆卸。带电维修可能导致危险。'), [])

    def test_strict_plan_validation(self):
        with self.assertRaises(ValueError):
            parse_plan(json.dumps(dict(domain='pump',priority=True,need_specialist='yes',query='x',reason='x')))

    def test_nested_secret_redaction(self):
        dummy='sk-'+'x'*24
        d=redact({'events':[{'message':'failed '+dummy}], 'deep':{'api_key':'secret'}})
        self.assertNotIn(dummy,str(d))
        self.assertNotIn('api_key',str(d))


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_nodes_and_dependency_barrier(self):
        a_started=asyncio.Event(); b_started=asyncio.Event()
        async def a(c):
            a_started.set(); await asyncio.wait_for(b_started.wait(),.5); return 'A'
        async def b(c):
            b_started.set(); await asyncio.wait_for(a_started.wait(),.5); return 'B'
        async def join(c): return c['a']+c['b']
        out=await DAGExecutor([Node('a',a),Node('b',b),Node('join',join,('a','b'))]).run()
        self.assertEqual(out['join'],'AB')

    async def test_serial_priority_and_skip_propagation(self):
        order=[]
        async def action(c): order.append(len(order)); return len(order)
        d=DAGExecutor([Node('low',action,priority=1),Node('high',action,priority=3),
                       Node('skip',action,condition=lambda c:False),Node('down',action,('skip',))],concurrency=1)
        await d.run()
        self.assertEqual([e['node'] for e in d.audit if e['state']=='RUNNING'],['high','low'])
        self.assertEqual(d.states['down'],'SKIPPED')

    async def test_invalid_graphs_rejected(self):
        async def f(c): pass
        for nodes in [[Node('a',f,('z',))],[Node('a',f,('b',)),Node('b',f,('a',))],
                      [Node('a',f),Node('a',f)]]:
            with self.assertRaises(ValueError): DAGExecutor(nodes)

    async def test_failure_cancels_sibling(self):
        started=asyncio.Event(); cancelled=asyncio.Event()
        async def slow(c):
            started.set()
            try: await asyncio.sleep(10)
            finally: cancelled.set()
        async def fail(c):
            await started.wait(); raise RuntimeError('expected')
        with self.assertRaises(RuntimeError):
            await DAGExecutor([Node('slow',slow),Node('fail',fail)]).run()
        self.assertTrue(cancelled.is_set())

    async def test_registry_domain_load_and_version(self):
        make=lambda n,v,d:AgentProfile(n,v,('diagnose',),d,'test')
        r=AgentRegistry([make('pump','1.0.0',('pump',)),make('pump','2.0.0',('pump',)),
                         make('general','1.0.0',('general',))])
        trace=[]
        async with r.acquire('diagnose','pump',trace,'first') as p:
            self.assertEqual((p.name,p.version),('pump','2.0.0'))
            async with r.acquire('diagnose','pump',trace,'second') as q:
                self.assertEqual(q.name,'general')
        self.assertTrue(all(v==0 for v in r.load.values()))


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, model=None, **kwargs):
        return await execute_pipeline('水泵异常振动，轴承温度升高。','',2,lambda *a:None,
                                      model_call=model or FakeModel(),**kwargs)

    async def test_full_dynamic_workflow_and_snapshots(self):
        with tempfile.TemporaryDirectory() as temp:
            r=await self.run_case(checkpoint_dir=Path(temp))
            self.assertEqual(r['status'],'PENDING_HUMAN')
            self.assertEqual(r['metrics']['peak_active_agents'],2)
            self.assertIn('pump_diagnosis_agent',r['agents_used'])
            self.assertEqual(r['execution_mode'],'offline_fixture')
            self.assertTrue(r['state_snapshots'])
            self.assertTrue((Path(temp)/(r['run_id']+'.json')).exists())

    async def test_serial_baseline_same_work(self):
        a=await self.run_case(strategy='dynamic')
        b=await self.run_case(strategy='serial')
        self.assertEqual(a['diagnosis'],b['diagnosis'])
        self.assertEqual(a['metrics']['model_calls'],b['metrics']['model_calls'])
        self.assertEqual(b['metrics']['peak_active_agents'],1)

    async def test_checkpoint_failure_preserves_memory_result(self):
        with patch('run_store.save_checkpoint',side_effect=OSError('read only')):
            r=await self.run_case(checkpoint_dir=Path('unused'))
        self.assertEqual(r['status'],'PENDING_HUMAN')
        self.assertEqual(r['checkpoint_errors'],['OSError'])
        self.assertTrue(r['state_snapshots'])

    async def test_conditional_specialist_skip(self):
        r=await self.run_case(FakeModel(specialist=False))
        self.assertNotIn('specialist_agent',r['agents_used'])
        self.assertTrue(any(e['state']=='SKIPPED' for e in r['workflow_audit']))

    async def test_invalid_plan_degrades_visibly(self):
        r=await self.run_case(FakeModel(invalid_plan=True))
        self.assertTrue(r['plan']['fallback'])
        self.assertEqual(r['status'],'PENDING_HUMAN')

    async def test_injection_causes_real_callback_revision(self):
        r=await self.run_case(self_loop_test=True)
        self.assertEqual(r['revision_count'],1)
        self.assertEqual(len(r['revision_history']),2)
        self.assertEqual(r['status'],'PENDING_HUMAN')
        self.assertTrue(r['revision_history'][0]['test_injected'])

    async def test_high_risk_stops_without_revision(self):
        r=await self.run_case(FakeModel(high=True),self_loop_test=True)
        self.assertEqual(r['status'],'REJECTED')
        self.assertEqual(r['revision_count'],0)
        self.assertEqual(len(r['revision_history']),1)

    async def test_no_progress_stops_early(self):
        r=await self.run_case(FakeModel(unchanged=True),self_loop_test=True)
        self.assertEqual(r['status'],'REJECTED')
        self.assertEqual(r['revision_count'],1)

    async def test_unavailable_citation_rejected(self):
        r=await self.run_case(FakeModel(bad_citation=True))
        self.assertEqual(r['status'],'REJECTED')
        self.assertTrue(any('知识条目' in i for h in r['revision_history'] for i in h['issues']))

    async def test_failed_model_and_deadline_return_error(self):
        self.assertEqual((await self.run_case(FakeModel(fail_capability='knowledge')))['status'],'ERROR')
        self.assertEqual((await self.run_case(overall_timeout=.001))['status'],'ERROR')

    async def test_review_failure_retains_completed_revision(self):
        n=0
        async def review(d):
            nonlocal n
            n+=1
            if n==2: raise RuntimeError('expected')
            return json.dumps(dict(passed=False,risk_level='low',issues=['补充说明']))
        async def revise(*args): return DIAGNOSIS+'补充了核实要求。'
        r=await execute_feedback_loop(task='test',knowledge=KNOWLEDGE,initial_diagnosis=DIAGNOSIS,
            review_call=review,revise_call=revise,max_revisions=2,emit=lambda *a:None)
        self.assertEqual(r['status'],'ERROR')
        self.assertEqual(r['revision_count'],1)
        self.assertTrue(r['revision_history'][-1]['review_error'])


class StorageTests(unittest.TestCase):
    def test_record_round_trip_and_corrupt_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); rid='abcdef123456'
            save_record(dict(run_id=rid,task='test',status='REJECTED',nested={'api_key':'secret'}),[],folder=root)
            r=load_record(rid,root)
            self.assertEqual(r['result']['status'],'REJECTED')
            self.assertNotIn('api_key',str(r))
            (root/'bad.json').write_text('[]')
            self.assertEqual(len(list_records(folder=root)),1)
            with self.assertRaises(ValueError):load_record('../bad',root)

    def test_knowledge_schema_and_retrieval(self):
        self.assertTrue(KnowledgeBase().search('水泵振动温度'))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'kb.json';p.write_text('[{}]')
            with self.assertRaises(ValueError):KnowledgeBase(p)


if __name__ == '__main__': unittest.main()
