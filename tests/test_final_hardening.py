import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from fixtures import FakeModel
from lingshu_backend import execute_pipeline, parse_review_json, basic_text_check, detect_unsafe_patterns
from registry import load_profiles, AgentRegistry
from scheduler import DAGExecutor, Node
from recover_run import recover, read_source
from agent_memory import TaskMemory

ROOT = Path(__file__).resolve().parents[1]
TASK = '水泵异常振动，温度升高。'


class ContractTests(unittest.TestCase):
    def test_invalid_risk_and_duplicate_keys_are_format_errors(self):
        for raw in ['{"passed":true,"issues":[]}',
                    '{"passed":true,"issues":[],"risk_level":3}',
                    '{"passed":true,"issues":[],"risk_level":"invalid"}',
                    '{"passed":false,"passed":true,"issues":[],"risk_level":"low"}',
                    '{"passed":true,"issues":[],"risk_level":"low","extra":NaN}']:
            with self.subTest(raw=raw):
                self.assertFalse(parse_review_json(raw)['parse_ok'])

    def test_negated_safety_reminders_do_not_pass(self):
        for prefix in ('不要','无需','不必','禁止'):
            self.assertFalse(basic_text_check(prefix+'停机并隔离能源；'+prefix+'穿戴防护用品')[0])
        self.assertTrue(basic_text_check('请停机并隔离能源，并穿戴防护用品。')[0])

    def test_generic_bypass_and_polite_danger_blocked(self):
        self.assertTrue(detect_unsafe_patterns('先绕过联锁再维修'))
        self.assertTrue(detect_unsafe_patterns('请带电拆卸可能损坏的零件'))
        self.assertFalse(detect_unsafe_patterns('禁止绕过联锁；带电拆卸可能导致危险。'))

    def test_memory_bounded_and_isolated(self):
        p=next(p for p in load_profiles() if p.name=='pump_diagnosis_agent')
        m=TaskMemory()
        for i in range(5):m.remember(p,str(i)*5000)
        self.assertEqual(len(m.entries[p.name]),p.memory_items)
        self.assertTrue(all(len(t)<=p.memory_chars for t in m.entries[p.name]))
        self.assertFalse(TaskMemory().context(p))
        self.assertFalse(m.context(next(p for p in load_profiles() if p.name=='safety_agent')))


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, model=None, **kwargs):
        return await execute_pipeline(TASK,'',2,lambda *a:None,model_call=model or FakeModel(),**kwargs)

    async def test_timeout_preserves_completed_revision(self):
        class SlowReview(FakeModel):
            reviews=0
            async def __call__(self,profile,message):
                if 'review' in profile.capabilities:
                    self.reviews+=1
                    if self.reviews==2:await asyncio.sleep(10)
                return await super().__call__(profile,message)
        r=await self.run_case(SlowReview(),self_loop_test=True,overall_timeout=.3)
        self.assertEqual((r['status'],r['stop_code'],r['revision_count']),('ERROR','TASK_TIMEOUT',1))
        self.assertIn('遵循现场作业许可流程',r['diagnosis'])
        self.assertTrue(r['revision_history'][-1]['review_error'])
        self.assertTrue(r['blocking_issues'])

    async def test_role_memory_used_only_for_diagnosis(self):
        model=FakeModel()
        r=await self.run_case(model,self_loop_test=True)
        self.assertEqual(r['status'],'PENDING_HUMAN')
        revisions=[m for _,m in model.calls if '旧稿：' in m]
        self.assertIn('本任务该角色的历史输出',revisions[0])
        self.assertTrue(all('本任务该角色的历史输出' not in m for n,m in model.calls if n=='safety_agent'))
        self.assertTrue(r['tool_audit'])

    async def test_tool_permission_enforced(self):
        rows=json.loads((ROOT/'config/agents.json').read_text())
        next(r for r in rows if r['name']=='knowledge_agent')['tools']=[]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'agents.json';path.write_text(json.dumps(rows),encoding='utf-8')
            r=await self.run_case(profiles_path=path)
        self.assertEqual(r['status'],'ERROR')
        self.assertNotIn('knowledge_agent',r['agents_used'])

    async def test_raw_review_preserved_with_invalid_citations(self):
        r=await self.run_case(FakeModel(bad_citation=True))
        entry=r['revision_history'][0]
        self.assertTrue(json.loads(entry['review_raw'])['passed'])
        self.assertFalse(entry['review_passed'])
        self.assertTrue(any('知识条目' in x for x in entry['validation_issues']))

    async def test_resume_failed_analysis_reuses_only_valid_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            source=await self.run_case(FakeModel(fail_capability='knowledge'),checkpoint_dir=Path(temp))
            model=FakeModel()
            r=await self.run_case(model,recovery_source=source)
            checkpoint=read_source(Path(temp)/(source['run_id']+'.json'))
            r2=await recover(checkpoint,'',lambda *a:None,model_call=FakeModel(),checkpoint_dir=None)
        self.assertEqual(r['status'],'PENDING_HUMAN')
        self.assertEqual(r2['status'],'PENDING_HUMAN')
        self.assertEqual(r['recovery']['reused_nodes'],['plan','evidence'])
        self.assertNotEqual(r['run_id'],source['run_id'])
        self.assertTrue(all(n!='planner_agent' for n,_ in model.calls))

    async def test_completed_run_recovery_always_reviews_and_drops_approval(self):
        source=await self.run_case()
        source['status']='DEMO_CONFIRMED'
        source['approval_record']={'status':'confirmed'}
        model=FakeModel(high=True)
        r=await self.run_case(model,recovery_source=source)
        self.assertEqual(r['status'],'REJECTED')
        self.assertEqual(r['metrics']['model_calls'],1)
        self.assertNotIn('approval_record',r)

    async def test_rollback_discards_target_and_all_downstream(self):
        source=await self.run_case()
        r=await self.run_case(recovery_source=source,rollback_to='diagnosis')
        self.assertEqual(r['recovery']['reused_nodes'],['plan','evidence','analysis'])
        self.assertEqual(r['metrics']['model_calls'],2)
        fresh=await self.run_case(recovery_source=source,rollback_to='plan')
        self.assertEqual(fresh['recovery']['reused_nodes'],[])

    async def test_recovery_rejects_changed_config_or_tampering_before_model(self):
        source=await self.run_case()
        bad=copy.deepcopy(source)
        main=[s for s in bad['state_snapshots'] if s['workflow']=='main'][-1]
        main['outputs']['diagnosis']='forged'
        model=FakeModel()
        for opts in ({'recovery_source':bad}, {'recovery_source':source,'max_concurrency':3}):
            with self.assertRaises(ValueError):await self.run_case(model,**opts)
        self.assertEqual(model.calls,[])

    async def test_simultaneous_completion_and_failure_have_terminal_states(self):
        async def fail(c):raise RuntimeError('expected')
        async def done(c):return {'result':'completed'}
        d=DAGExecutor([Node('a',fail),Node('z',done),Node('next',done,('a','z'))])
        with self.assertRaises(RuntimeError):await d.run()
        self.assertEqual(d.states,{'a':'ERROR','z':'DONE','next':'CANCELLED'})

    async def test_snapshot_is_independent_of_later_mutation(self):
        async def action(c):return {'nested':[]}
        d=DAGExecutor([Node('a',action)])
        await d.run();snap=d.snapshot()
        d.outputs['a']['nested'].append('later')
        self.assertEqual(snap['outputs']['a']['nested'],[])


if __name__=='__main__':unittest.main()
