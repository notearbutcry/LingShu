"""Streamlit AppTest exercises actual UI with offline backend model callbacks."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixtures import FakeModel
from lingshu_backend import execute_pipeline as real_pipeline

@unittest.skipUnless(importlib.util.find_spec('streamlit'), '先安装requirements.txt以运行页面测试')
class AppTests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.model=FakeModel()
        async def offline(task,key,revisions,emit,**kwargs):
            kwargs['checkpoint_dir']=self.root/'checkpoints'
            kwargs.pop('model_call',None)
            return await real_pipeline(task,'',revisions,emit,model_call=self.model,**kwargs)
        self.patches=[patch('lingshu_backend.execute_pipeline',new=offline),
                      patch('recover_run.execute_pipeline',new=offline),
                      patch('run_store.DATA_DIR',self.root/'runs')]
        for p in self.patches:p.start()
        self.at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30)
        self.at.secrets['DEEPSEEK_API_KEY']='offline-placeholder'

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def submit(self):
        self.at.run()
        self.assertFalse(self.at.exception)
        self.at.text_area[0].set_value('水泵异常振动，温度升高。')
        next(b for b in self.at.button if '启动灵枢分析' in b.label).click()
        self.at.run()
        self.assertFalse(self.at.exception)

    def test_submit_confirm_and_history_preserve_current_run(self):
        self.submit()
        self.assertEqual(self.at.session_state['task_status'],'PENDING_HUMAN')
        rid=self.at.session_state['result']['run_id']
        self.at.checkbox(key='approval_confirmed_checkbox').check().run()
        next(b for b in self.at.button if b.label=='确认演示结果').click()
        self.at.run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.at.session_state['task_status'],'DEMO_CONFIRMED')
        self.assertEqual(self.at.session_state['approval_record']['run_id'],rid)
        self.assertEqual(self.at.session_state['result']['run_id'],rid)
        self.assertTrue(list((self.root/'runs').glob('*.json')))
        # A same-text new task must not inherit the old approval. History is read-only.
        self.at.text_area[0].set_value('水泵异常振动，温度升高。')
        self.at.checkbox(key='force_new_run').check()
        next(b for b in self.at.button if '启动灵枢分析' in b.label).click()
        self.at.run()
        new_id=self.at.session_state['result']['run_id']
        self.assertNotEqual(new_id,rid)
        self.assertEqual(self.at.session_state['task_status'],'PENDING_HUMAN')
        self.at.selectbox(key='history_selected_run').select(rid).run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.at.session_state['result']['run_id'],new_id)
        self.assertIsNone(self.at.session_state['approval_record'])
        self.assertFalse(self.at.checkbox(key='force_new_run').value)

    def test_repeated_submit_does_not_reroll_rejected_result(self):
        self.model=FakeModel(high=True)
        self.submit()
        rid=self.at.session_state['result']['run_id']
        calls=len(self.model.calls)
        self.model.high=False  # A fresh model run would now pass.
        for _ in range(3):
            next(b for b in self.at.button if '启动灵枢分析' in b.label).click().run()
            self.assertFalse(self.at.exception)
            self.assertEqual(self.at.session_state['result']['run_id'],rid)
            self.assertEqual(self.at.session_state['task_status'],'REJECTED')
            self.assertEqual(len(self.model.calls),calls)
        self.assertEqual(self.at.session_state['result']['stop_code'],'HIGH_RISK')
        self.assertTrue(any('高风险' in x.value for x in self.at.markdown))

    def test_same_submit_preserves_pass_and_changed_task_starts_new_run(self):
        self.submit()
        rid=self.at.session_state['result']['run_id'];calls=len(self.model.calls)
        next(b for b in self.at.button if '启动灵枢分析' in b.label).click().run()
        self.assertEqual(self.at.session_state['result']['run_id'],rid)
        self.assertEqual(len(self.model.calls),calls)
        self.at.text_area[0].set_value('电机异响，运行状态尚未核实。')
        next(b for b in self.at.button if '启动灵枢分析' in b.label).click().run()
        self.assertFalse(self.at.exception)
        self.assertNotEqual(self.at.session_state['result']['run_id'],rid)

    def test_changed_configuration_starts_new_run(self):
        self.submit()
        rid=self.at.session_state['result']['run_id']
        next(s for s in self.at.slider if s.label=='自动修订上限').set_value(1).run()
        next(b for b in self.at.button if '启动灵枢分析' in b.label).click().run()
        self.assertFalse(self.at.exception)
        self.assertNotEqual(self.at.session_state['result']['run_id'],rid)

    def test_rejected_run_cannot_be_confirmed(self):
        self.model=FakeModel(high=True)
        self.submit()
        self.assertEqual(self.at.session_state['task_status'],'REJECTED')
        confirm=next(b for b in self.at.button if b.label=='无法确认')
        self.assertTrue(confirm.disabled)

    def test_recovery_creates_new_run_and_requires_new_confirmation(self):
        self.submit()
        rid=self.at.session_state['result']['run_id']
        self.at.checkbox(key='approval_confirmed_checkbox').check().run()
        self.at.button(key='confirm_result_button').click().run()
        self.assertEqual(self.at.session_state['task_status'],'DEMO_CONFIRMED')
        self.at.button(key='recover_run_button').click().run()
        self.assertFalse(self.at.exception)
        self.assertNotEqual(self.at.session_state['result']['run_id'],rid)
        self.assertEqual(self.at.session_state['task_status'],'PENDING_HUMAN')
        self.assertIsNone(self.at.session_state['approval_record'])
        self.assertFalse(self.at.checkbox(key='approval_confirmed_checkbox').value)
        self.assertTrue(any('已从检查点复用' in x.value for x in self.at.markdown))


if __name__=='__main__':unittest.main()
