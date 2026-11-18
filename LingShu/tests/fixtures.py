"""Explicit offline fixtures, never presented as real model performance."""
import asyncio
import json

KNOWLEDGE = '【已知事实】用户报告设备异常。【可能原因】仅为待验证假设。【缺少的数据】工况及测量值未提供，需要专业人员核实。'
DIAGNOSIS = '初步判断仍有不确定性，不能确诊。请由专业人员遵守现场规范核实。停机并隔离能源，穿戴防护用品。停止条件与补充数据需要依据现场实际情况确认。'


class FakeModel:
    def __init__(self, *, high=False, invalid_plan=False, specialist=True, domain='pump',
                 unchanged=False, bad_citation=False, fail_capability=None):
        self.high = high
        self.invalid_plan = invalid_plan
        self.specialist = specialist
        self.domain = domain
        self.unchanged = unchanged
        self.bad_citation = bad_citation
        self.fail_capability = fail_capability
        self.calls = []

    async def __call__(self, profile, message):
        await asyncio.sleep(.015)
        cap = profile.capabilities[0]
        self.calls.append((profile.name, message))
        if cap == self.fail_capability:
            raise RuntimeError('offline fixture failure')
        if cap == 'plan':
            return 'bad plan' if self.invalid_plan else json.dumps(dict(
                domain=self.domain, priority=2, need_specialist=self.specialist,
                query='水泵振动温度设备故障', reason='离线测试规划'))
        if cap == 'knowledge':
            return KNOWLEDGE
        if cap == 'specialist':
            return '这是离线专项分析替身，仅列出需要专业人员核实的数据，不构成现场诊断。'
        if cap == 'diagnose':
            suffix = '' if self.unchanged else ('遵循现场作业许可流程。' if '旧稿：' in message else '')
            return DIAGNOSIS + suffix + ('[KB:missing]' if self.bad_citation else '')
        if cap == 'review':
            return json.dumps({'passed': not self.high, 'risk_level': 'high' if self.high else 'low',
                               'issues': ['离线高风险场景'] if self.high else []})
        raise AssertionError(cap)
