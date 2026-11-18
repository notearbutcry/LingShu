import asyncio
import os
import json
from datetime import datetime

from run_store import save_record, load_record, list_records, CHECKPOINT_DIR
from privacy import redact
from registry import profile_summary
from request_identity import request_fingerprint

import nest_asyncio
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from lingshu_backend import execute_pipeline

nest_asyncio.apply()

# -------------------- 页面配置 --------------------
st.set_page_config(
    page_title="灵枢 LingShu | 智能运维指挥台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- 页面样式 --------------------
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 12% 0%, #182a43 0, #0b1220 38%); color: #e8eef8; }
[data-testid="stSidebar"] { background: #0d1728; border-right: 1px solid #223550; }
.block-container { padding-top: 0.8rem; padding-bottom: 1.8rem; max-width: 1500px; }
/* 顶部黑条来自 Streamlit 默认头部背景：保留菜单功能，只让背景透明 */
header[data-testid="stHeader"] {
  background: transparent !important;
  box-shadow: none !important;
  border-bottom: 0 !important;
}
/* 有些 Streamlit 版本在头部工具区额外绘制底色 */
[data-testid="stToolbar"] { background: transparent !important; }
h1, h2, h3 { color: #f3f7ff !important; letter-spacing: -0.02em; }
h1 { font-size: 2.45rem !important; }
div[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(27,47,73,.96), rgba(16,30,49,.96));
  border: 1px solid #304968; padding: 18px; border-radius: 18px;
  box-shadow: 0 8px 24px rgba(0,0,0,.14);
}
/* 提示条顶边文字防裁切：只改内边距和行高，不改变原有绿色/红色 */
div[data-testid="stAlert"] {
  border-radius: 14px;
  border: 1px solid #344b68;
  padding: 0.95rem 1.05rem !important;
  box-sizing: border-box;
  overflow: visible !important;
}
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
  overflow: visible !important;
}
div[data-testid="stAlert"] p {
  margin: 0 !important;
  padding: 0.1rem 0 !important;
  line-height: 1.7 !important;
  white-space: normal !important;
}
.stButton > button, [data-testid="stFormSubmitButton"] button {
  border-radius: 12px; font-weight: 650; min-height: 2.8rem;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #2b405c; }
.stTabs [data-baseweb="tab"] { background: #14243a; border-radius: 10px 10px 0 0; padding: 10px 16px; }
.ls-subtitle { color: #a9bdd7; margin-top: -8px; font-size: 1.02rem; }
.ls-card { background: linear-gradient(145deg, #14243a, #101c2d); border:1px solid #2a405d; border-radius:13px; padding:13px; box-shadow: 0 6px 16px rgba(0,0,0,.10); }
.ls-muted { color:#9eb4d0; font-size:0.9rem; }
.ls-result-card { background: linear-gradient(145deg, rgba(21,37,59,.98), rgba(14,26,43,.98)); border: 1px solid #2b425f; border-radius: 14px; padding: 14px 16px; margin: 6px 0 10px 0; }
.ls-result-kicker { color:#8db9e8; font-size:.78rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; margin-bottom:8px; }
.ls-result-title { color:#f4f8ff; font-size:1.18rem; font-weight:750; margin-bottom:10px; }
.ls-result-body { color:#dbe6f5; line-height:1.8; }
.ls-chip { display:inline-block; padding:5px 10px; border-radius:999px; background:#1d3653; color:#b9dcff; font-size:.82rem; margin:2px 4px 4px 0; border:1px solid #355579; }
.ls-divider { height:1px; background:#2b405c; margin:14px 0; }
</style>
""", unsafe_allow_html=True)

# -------------------- 状态初始化 --------------------
DEFAULT_STATE = {
    "logs": [],
    "result": None,
    "task_status": "等待任务",
    "approval_record": None,
    "events": [],
    "save_message": "",
    # 保存当前正在查看的历史记录信息，避免点击后刷新页面却没有反馈。
    "history_view": None,
    "last_request_fingerprint": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value

AGENT_LABELS = {
    "knowledge_agent": "设备知识分析 Agent",
    "diagnosis_agent": "故障诊断 Agent",
    "safety_agent": "安全审核 Agent",
    "system": "系统",
    "planner_agent": "任务规划 Agent",
    "specialist_agent": "专项分析 Agent",
    "pump_diagnosis_agent": "水泵诊断 Agent",
    "motor_diagnosis_agent": "电机诊断 Agent",
}

def add_event(agent, status, message, round_no=0):
    """记录可审计的执行事件；不记录 API 密钥。"""
    message = redact(str(message))
    event = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "agent_label": AGENT_LABELS.get(agent, agent),
        "status": status,
        "message": message,
        "round": round_no,
    }
    st.session_state.events.append(event)
    progress = globals().get("live_progress")
    if progress is not None and st.session_state.task_status == "正在分析":
        progress.info(f"{event['agent_label']} · {status} · {message}")
    st.session_state.logs.append(
        f"[{event['time']}] [{event['agent_label']}] [第 {round_no} 轮] {status}：{message}"
    )

def get_api_key():
    """优先读取项目 secrets，其次读取环境变量。"""
    try:
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return os.environ.get("DEEPSEEK_API_KEY", "").strip()

async def run_pipeline(task, api_key, max_revisions, self_loop_test):
    """运行完整后台；测试模式只在明确勾选时启用。"""
    return await execute_pipeline(
        task, api_key, max_revisions, add_event,
        self_loop_test=self_loop_test, strategy=scheduling_strategy,
        max_concurrency=max_concurrency, checkpoint_dir=CHECKPOINT_DIR,
        knowledge_endpoint=knowledge_endpoint,
    )

# -------------------- 人工确认回调 --------------------
def confirm_current_run():
    """回调先于页面重新绘制执行，避免在已创建的控件后修改其状态。"""
    current = st.session_state.get("result")
    if not isinstance(current, dict):
        return
    if st.session_state.get("task_status") != "PENDING_HUMAN" or current.get("status") != "PENDING_HUMAN":
        return
    if not st.session_state.get("approval_confirmed_checkbox", False):
        return

    approval = {
        "task": current.get("task", ""),
        "run_id": current.get("run_id", ""),
        "status": "演示确认完成",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state.approval_record = approval
    st.session_state.result = {**current, "status": "DEMO_CONFIRMED"}
    st.session_state.task_status = "DEMO_CONFIRMED"
    add_event("system", "演示确认", "已确认演示结果；不代表现场操作授权", 0)
    try:
        save_record(st.session_state.result,
                    st.session_state.events, approval)
        st.session_state.save_message = "演示确认已写入本地记录"
    except (OSError, ValueError, TypeError):
        st.session_state.save_message = "演示已确认；本地保存失败，请下载本轮 JSON"

# -------------------- 侧边栏 --------------------
with st.sidebar:
    st.markdown("## 🧠 灵枢 LingShu")
    st.caption("多 Agent 智能运维与安全审核")
    st.caption("初赛提交候选版 · 2.1.0")
    st.markdown("---")
    scheduling_strategy = st.selectbox("调度方式", ["dynamic", "serial"],
        format_func=lambda x: {"dynamic": "动态编排（允许并行）", "serial": "同流程串行基线"}[x])
    max_concurrency = st.slider("单任务并发 Agent 上限", 1, 4, 2,
        help="串行基线固定只执行一个节点；动态模式用于独立子任务并行。")
    with st.expander("角色注册与版本"):
        try:
            st.json(profile_summary())
        except (OSError, ValueError, TypeError) as exc:
            st.error('角色配置无法读取：'+str(exc))
            st.stop()
    use_knowledge_api = st.checkbox("通过本地 HTTP API 检索知识", value=False,
        help="需先在另一个终端运行 python knowledge_api.py，默认端口8765。")
    knowledge_endpoint = "http://127.0.0.1:8765" if use_knowledge_api else None
    max_revisions = st.slider("自动修订上限", min_value=0, max_value=5, value=2, step=1,
                              help="0 表示不自动修订；达到上限仍未通过时转人工处理。")
    self_loop_test = st.checkbox(
        "Self-loop 失败注入测试",
        value=False,
        help="仅用于初赛验证。首轮注入一项明确标注的测试性审核缺项，"
             "实际调用诊断 Agent 修订，再由安全 Agent 审核；不伪造通过结果。",
    )
    if self_loop_test:
        st.warning("测试模式已开启：首轮将模拟一项审核缺项，不代表真实诊断不安全。")
    st.markdown("### 工作流")
    st.markdown("1. 规划任务与检索本地资料\n2. 按条件并行知识与专项分析\n3. 按能力和负载选派诊断 Agent\n4. 独立审核、反馈修订与人工确认")
    st.info("本系统为演示工具。AI 输出不能替代现场规程、专业人员判断或正式安全审批。")
    st.markdown("---")
    st.caption("API 密钥通过 .streamlit/secrets.toml 或环境变量读取，不要写进代码。")

# -------------------- 标题和指标 --------------------
st.title("🧠 灵枢 LingShu")
st.markdown('<div class="ls-subtitle">工业设备智能运维 · 多 Agent 协作 · 安全审核与反馈修订</div>',
            unsafe_allow_html=True)
st.write("")
result = st.session_state.result
status_label = {
    "等待任务": "等待任务",
    "正在分析": "正在分析",
    "PENDING_HUMAN": "等待人工确认",
    "DEMO_CONFIRMED": "演示已确认",
    "REJECTED": "审核未通过",
    "ERROR": "执行异常",
}.get(st.session_state.task_status, st.session_state.task_status)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("本轮参与 Agent", (result or {}).get("agent_count", "待运行"))
metric2.metric("自动修订上限", (result or {}).get("max_revisions", max_revisions))
metric3.metric("当前任务状态", status_label)
metric4.metric("事件记录", len(st.session_state.events))

st.divider()

# -------------------- 输入区 --------------------
live_progress = st.empty()
st.subheader("新建任务")
if st.session_state.pop("reset_force_new_run", False):
    st.session_state["force_new_run"] = False
with st.form("task_form", clear_on_submit=False):
    task = st.text_area(
        "故障描述",
        placeholder="例如：水泵异常振动，轴承温度持续升高。",
        height=115,
        max_chars=2000,
    )
    force_new_run = st.checkbox("重新生成（再次调用模型，结论可能变化）", key="force_new_run",
        help="相同输入和配置默认保留当前运行。只有主动勾选才重新生成，不继承上次人工确认。")
    submitted = st.form_submit_button("▶ 启动灵枢分析", type="primary", use_container_width=True)

if submitted:
    try:
        fingerprint = request_fingerprint(task, strategy=scheduling_strategy,
            max_concurrency=max_concurrency, max_revisions=max_revisions,
            self_loop_test=self_loop_test, knowledge_endpoint=knowledge_endpoint)
    except (OSError, ValueError, TypeError) as exc:
        st.error('项目文件缺失或配置错误，请完整解压代码包：'+type(exc).__name__)
        st.stop()
    if not task.strip():
        st.warning("请先输入任务内容。")
    elif self_loop_test and max_revisions < 1:
        st.warning("Self-loop 测试模式至少需要允许修订 1 次，请将左侧自动修订上限调整为 1 或以上。")
    elif (not force_new_run and st.session_state.result is not None and
          fingerprint == st.session_state.last_request_fingerprint):
        st.info("输入和配置未变化，已保留当前运行 " + st.session_state.result.get("run_id", "") +
                "，没有重新调用模型。如需新一次运行，请主动勾选“重新生成”；未通过时先查看下面的具体原因。")
    else:
        api_key = get_api_key()
        if not api_key:
            st.error(
                "没有读取到 DEEPSEEK_API_KEY。请确认文件位置为项目目录下的 "
                ".streamlit/secrets.toml，内容格式为：DEEPSEEK_API_KEY = \"你的密钥\"。"
            )
            st.caption("也可以在启动 Streamlit 的同一个终端中设置环境变量。不要把密钥发到聊天中。")
        else:
            st.session_state.last_request_fingerprint = fingerprint
            st.session_state.reset_force_new_run = True
            st.session_state.logs = []
            st.session_state.events = []
            st.session_state.result = None
            st.session_state.approval_record = None
            st.session_state.history_view = None
            st.session_state["approval_confirmed_checkbox"] = False
            st.session_state.task_status = "正在分析"
            try:
                with st.spinner("灵枢正在调用 Agent，请稍候……"):
                    result = asyncio.run(
                        run_pipeline(task.strip(), api_key, max_revisions, self_loop_test)
                    )
            except Exception as exc:
                # UI 容错：后端未捕获异常也不会导致整个页面白屏；绝不输出密钥。
                message = f"{type(exc).__name__}: {exc}".replace(api_key, "[REDACTED]")[:200]
                add_event("system", "错误", "系统执行异常：" + message, 0)
                result = {
                    "run_id": "", "task": task.strip(), "knowledge": "未完成", "diagnosis": "未完成",
                    "review": "程序异常，请检查本地终端输出。", "status": "ERROR",
                    "missing_phrases": [], "validation_issues": [message],
                    "revision_history": [], "max_revisions": max_revisions,
                    "self_loop_test": bool(self_loop_test), "revision_count": 0,
                }
            st.session_state.result = result
            st.session_state.task_status = result["status"]
            try:
                if result.get("run_id"):
                    path = save_record(result, st.session_state.events)
                    st.session_state.save_message = "本地记录已保存：" + path.name
                else:
                    st.session_state.save_message = "本轮未生成运行编号，请手动导出日志"
            except (OSError, ValueError, TypeError) as exc:
                st.session_state.save_message = "本地记录保存失败，可使用下方下载按钮：" + type(exc).__name__
            st.rerun()

# Explicit local recovery is separate from ordinary repeated-click protection.
with st.expander("恢复中断任务 / 回退后重新执行"):
    st.caption("恢复2.1.0版记录中的已完成节点；回退会丢弃所选节点及后续结果。每次建立新run_id，重新审核，并重新确认。真实模型调用会产生费用。")
    try:
        recovery_records = list_records(limit=30)
    except (OSError, ValueError):
        recovery_records = []
    recovery_ids = [r['run_id'] for r in recovery_records]
    if recovery_ids:
        recovery_id = st.selectbox("恢复来源", recovery_ids, key="recovery_source_id")
        rollback_stage = st.selectbox("重新执行位置", [None, 'diagnosis', 'analysis', 'evidence', 'plan'],
            format_func=lambda x: {None:'从中断处继续（审核一定重做）', 'diagnosis':'回退到诊断', 'analysis':'回退到分析', 'evidence':'回退到检索', 'plan':'从规划重新执行'}[x])
        if st.button("恢复为新运行", key="recover_run_button"):
            from recover_run import recover
            try:
                old = load_record(recovery_id)['result']
                api_key = get_api_key()
                # Prepare recovery before replacing the visible result; a rejected
                # compatibility check must not destroy the previous task/approval.
                buffered = []
                def collect(agent,status,message,round_no=0):
                    buffered.append(dict(time=datetime.now().strftime('%H:%M:%S'),
                        agent=agent,agent_label=AGENT_LABELS.get(agent,agent),
                        status=status,message=redact(message,(api_key,)),round=round_no))
                with st.spinner('正在恢复并重新审核……'):
                    recovered = asyncio.run(recover(old,api_key,collect,rollback_to=rollback_stage))
                st.session_state.result = recovered
                st.session_state.task_status = recovered['status']
                st.session_state.events = buffered
                st.session_state.logs = [e['message'] for e in buffered]
                st.session_state.approval_record = None
                st.session_state['approval_confirmed_checkbox'] = False
                recovery_options = recovered['recovery_context']['options']
                st.session_state.last_request_fingerprint = request_fingerprint(recovered['task'],
                    **{k:recovery_options[k] for k in ('strategy','max_concurrency','max_revisions','self_loop_test','knowledge_endpoint')})
                try:
                    save_record(recovered,buffered)
                    st.session_state.save_message = '恢复结果已保存；请核对新的审核结果'
                except (OSError, ValueError, TypeError):
                    st.session_state.save_message = '恢复已完成，本地保存失败，请下载本次JSON'
                st.rerun()
            except (OSError, ValueError, TypeError, KeyError) as exc:
                st.error('未开始恢复：'+redact(str(exc), (get_api_key(),)))
    else:
        st.caption('完成一次运行后，可在这里选择恢复来源；进程中断的检查点也可用 recover_run.py 恢复。')

# -------------------- 流程可视化 --------------------
st.subheader("执行流程")
events = st.session_state.events
# 每个 Agent 只看「最近一次」事件，不能用任意历史“进行中”判断现在状态。
last_agent_events = {}
for event in events:
    event_agent = event["agent"]
    if event_agent in {"pump_diagnosis_agent", "motor_diagnosis_agent"}:
        event_agent = "diagnosis_agent"
    last_agent_events[event_agent] = event["status"]
terminal_status = st.session_state.task_status in {
    "PENDING_HUMAN", "DEMO_CONFIRMED", "REJECTED", "ERROR"
}
reused_nodes = set((st.session_state.result or {}).get("recovery", {}).get("reused_nodes", []))
restored_stage = {"knowledge_agent": "analysis", "diagnosis_agent": "diagnosis"}
flow_cols = st.columns(4)
flow_items = [
    ("knowledge_agent", "① 知识依据", "证据与未知项"),
    ("diagnosis_agent", "② 故障诊断", "判断与排查步骤"),
    ("safety_agent", "③ 安全审核", "风险与规则检查"),
    ("loop", "④ Self-loop", "失败后修订再审核"),
]
for col, (agent_id, title, description) in zip(flow_cols, flow_items):
    with col:
        if agent_id == "loop":
            loop_count = sum(1 for e in events if e["status"] == "修订中")
            if loop_count:
                node_status = f"已修订 {loop_count} 次"
            elif st.session_state.task_status in {"PENDING_HUMAN", "DEMO_CONFIRMED"}:
                node_status = "审核一次通过 · 无需触发"
            elif st.session_state.task_status == "REJECTED":
                node_status = "未修订 · 请查看审核原因"
            elif st.session_state.task_status == "ERROR":
                node_status = "执行异常"
            else:
                node_status = "等待触发"
        else:
            latest = last_agent_events.get(agent_id)
            if latest in {"完成", "通过"}:
                node_status = "已完成"
            elif latest in {"未通过", "需人工处理"}:
                node_status = "未通过"
            elif latest == "错误":
                node_status = "执行异常"
            elif latest in {"进行中", "修订中"}:
                node_status = "已中断" if terminal_status else "执行中"
            elif restored_stage.get(agent_id) in reused_nodes:
                node_status = "已从检查点复用"
            else:
                node_status = "未启动" if terminal_status else "等待中"
        st.markdown(f"""
        <div class="ls-card">
          <div style="font-size:1.05rem;font-weight:700">{title}</div>
          <div class="ls-muted" style="margin:8px 0">{description}</div>
          <div style="font-weight:700;color:#75c7ff">{node_status}</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- 编排证据 --------------------
current_evidence = st.session_state.result
if isinstance(current_evidence, dict) and current_evidence.get("runtime_version"):
    with st.expander("编排计划、调度依据与知识来源", expanded=False):
        st.write("**任务规划**")
        st.json(current_evidence.get("plan", {}))
        trace = current_evidence.get("schedule_trace", [])
        st.write("**每次分配记录**")
        if trace:
            st.dataframe(pd.DataFrame([{k: row.get(k) for k in
                ("node", "capability", "domain", "selected")} for row in trace]),
                use_container_width=True, hide_index=True)
            with st.expander("候选评分与当前负载"):
                st.json(trace)
        st.write("**本地检索来源**")
        for hit in current_evidence.get("evidence", []):
            st.write(f"[KB:{hit['id']}] {hit['title']}")
            st.caption(hit['source'] + ("（演示资料）" if hit.get('demo') else "（用户提供资料，待核验）"))
            st.write(hit['text'])
        if not current_evidence.get("evidence"):
            st.caption("没有匹配的本地资料，输出需标记为模型假设。")
        st.write("**角色工具与任务内记忆**")
        st.json({'tools': current_evidence.get('tool_audit', []),
                 'memory': current_evidence.get('memory_summary', {}),
                 'recovery': current_evidence.get('recovery', {})})
        st.write("**本轮运行指标**")
        st.json(current_evidence.get("metrics", {}))
        st.caption("elapsed_ms为完整任务耗时；model_attempt_ms为调用耗时之和，并行时不可当作总耗时。")
        with st.expander("节点状态与快照回看"):
            st.json(current_evidence.get("workflow_audit", []))
            snapshots = current_evidence.get("state_snapshots", [])
            if snapshots:
                idx = st.selectbox("选择快照", range(len(snapshots)), key="snapshot_idx")
                st.json(snapshots[idx])
        st.caption("快照支持本地流程恢复与回退后重跑；每次均重新审核，不涉及设备操作回滚。")

# -------------------- 结果面板 --------------------
st.divider()
st.subheader("分析结果")
st.caption("结论优先，详细内容按需展开。")
result = st.session_state.result

if result is None:
    st.markdown("""
    <div class="ls-result-card">
      <div class="ls-result-kicker">READY FOR ANALYSIS</div>
      <div class="ls-result-title">等待提交运维任务</div>
      <div class="ls-result-body">提交设备故障描述后，这里将显示知识分析、故障诊断和安全审核结果。</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.caption(f"本结果对应任务：{result.get('task', '')} ｜ run_id：{result.get('run_id', '')}")
    approved = (
        st.session_state.task_status == "DEMO_CONFIRMED"
        and st.session_state.approval_record is not None
        and st.session_state.approval_record.get("run_id") == result.get("run_id")
        and st.session_state.approval_record.get("task") == result.get("task")
    )
    if result.get("self_loop_test"):
        st.warning("Self-loop 测试模式：首轮包含明确标注的模拟审核缺项。"
                   "后续修订和复审由真实 Agent 执行，未保证一定通过。")
    if approved:
        st.success("当前任务的演示确认已完成。此记录不构成现场操作授权或正式安全审批。")
        status_text = "演示已确认"
        status_icon = "🟢"
    elif result["status"] == "PENDING_HUMAN":
        st.success("审核文字与基础安全提示检查通过：等待人工确认。此状态不代表设备已获准操作。")
        status_text = "待人工确认"
        status_icon = "🟢"
    elif result["status"] == "REJECTED":
        st.error("审核未通过或安全提示检查未通过。系统不开放确认，请由专业人员复核。")
        status_text = "需要复核"
        status_icon = "🟠"
    else:
        st.error("任务执行失败。请查看系统事件记录并检查 API、网络或模型配置。")
        status_text = "执行异常"
        status_icon = "🔴"

    if result["status"] in {"REJECTED", "ERROR"}:
        reasons = {
            "HIGH_RISK": "审核标注高风险，停止自动修订",
            "UNKNOWN_RISK": "审核无法确定文本风险，转人工复核",
            "NO_PROGRESS": "修订没有进展或重复历史稿",
            "REVISION_LIMIT": "修订次数用尽，仍有未解决问题",
            "REVIEW_FORMAT_ERROR": "审核JSON格式错误，未完成有效审核",
            "REVIEW_CALL_ERROR": "审核模型调用失败",
            "REVISION_CALL_ERROR": "修订模型调用失败",
            "TASK_TIMEOUT": "任务整体超时，已保留完成的修订，不能确认",
            "EXECUTION_ERROR": "流程执行异常，尚未完成有效审核",
        }
        st.write("**本次未开放确认的原因：** " + reasons.get(result.get("stop_code"), "流程未完成或校验未通过"))
        issues = result.get("blocking_issues") or result.get("validation_issues", [])
        for issue in issues:
            st.write("• " + issue)

    a, b, c = st.columns([1.1, 1, 1])
    with a:
        st.markdown(f"""
        <div class="ls-result-card">
          <div class="ls-result-kicker">TASK STATUS</div>
          <div class="ls-result-title">{status_icon} {status_text}</div>
          <div class="ls-muted">当前任务处理状态</div>
        </div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown(f"""
        <div class="ls-result-card">
          <div class="ls-result-kicker">SELF-LOOP</div>
          <div class="ls-result-title">最多 {result.get("max_revisions", max_revisions)} 次修订</div>
          <div class="ls-muted">未通过时按配置尝试修订</div>
        </div>
        """, unsafe_allow_html=True)
    with c:
        safety_text = "提示项齐全" if not result.get("missing_phrases") else "存在缺项"
        st.markdown(f"""
        <div class="ls-result-card">
          <div class="ls-result-kicker">SAFETY CHECK</div>
          <div class="ls-result-title">{'🛡️' if not result.get("missing_phrases") else '⚠️'} {safety_text}</div>
          <div class="ls-muted">仅为文字提示检查，不等同于现场安全认证</div>
        </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📚  知识依据", "🛠️  故障诊断", "🛡️  安全审核"])

    with tab1:
        st.markdown("""
        <div class="ls-result-card">
          <div class="ls-result-kicker">KNOWLEDGE / 知识依据</div>
          <div class="ls-result-title">关键依据与待核实项</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(result["knowledge"])
        with st.expander("查看完整知识分析", expanded=False):
            st.code(result["knowledge"], language="text")

    with tab2:
        st.markdown("""
        <div class="ls-result-card">
          <div class="ls-result-kicker">DIAGNOSIS / 故障诊断</div>
          <div class="ls-result-title">初步判断与优先排查</div>
          <div class="ls-muted">辅助建议，执行前须按现场规程核实。</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(result["diagnosis"])
        if result.get("missing_phrases"):
            st.warning("安全提示缺项：" + "、".join(result["missing_phrases"]))
        with st.expander("查看诊断质量检查", expanded=False):
            issues = result.get("validation_issues", [])
            st.write("未发现基础格式问题。" if not issues else "\n".join(f"• {x}" for x in issues))

    with tab3:
        review_pass = result["status"] in {"PENDING_HUMAN", "DEMO_CONFIRMED"}
        review_badge = "审核文字通过" if review_pass else "需人工复核"
        st.markdown(f"""
        <div class="ls-result-card">
          <div class="ls-result-kicker">SAFETY / 安全审核</div>
          <div class="ls-result-title">{'🟢' if review_pass else '🟠'} {review_badge}</div>
          <div class="ls-muted">文字审核不等于现场安全认证。</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(result["review"])

    # Self-loop 闭环证据：展示每轮审核结论、问题与修订差异
    history = result.get("revision_history", [])
    with st.expander(f"Self-loop 过程与修订记录（{len(history)} 条）", expanded=bool(history)):
        if not history:
            st.caption("当前没有记录到审核轮次。")
        else:
            for item in history:
                round_no = item.get("round", 0)
                passed = item.get("review_passed", False)
                st.markdown(f"**第 {round_no + 1} 轮 · {'审核通过' if passed else '审核未通过'}**")
                if item.get("test_injected"):
                    st.warning("本轮为测试注入的审核缺项，仅用于验证真实修订与复审流程。")
                if item.get("issues") and not passed:
                    st.caption("审核问题：" + "；".join(item["issues"]))
                if item.get("missing_phrases"):
                    st.caption("缺少安全提示：" + "、".join(item["missing_phrases"]))
                if item.get("validation_issues"):
                    st.caption("基础检查：" + "；".join(item["validation_issues"]))
                if item.get("review"):
                    with st.expander(f"第 {round_no + 1} 轮审核意见", expanded=False):
                        st.write(item["review"])
                        if item.get("review_raw"):
                            st.code(item["review_raw"], language="json")
                if item.get("diagnosis_before") or item.get("diagnosis_after"):
                    before, after = st.columns(2)
                    with before:
                        st.markdown("**修订前**")
                        st.text(item.get("diagnosis_before", "—"))
                    with after:
                        st.markdown("**修订后**")
                        st.text(item.get("diagnosis_after", "—"))
                st.divider()

    with st.expander("查看完整原始输出", expanded=False):
        st.markdown("**知识分析原文**")
        st.code(result["knowledge"], language="text")
        st.markdown("**故障诊断原文**")
        st.code(result["diagnosis"], language="text")
        st.markdown("**安全审核原文**")
        st.code(result["review"], language="text")

    st.markdown("### 人工确认")
    st.warning("此处仅记录演示确认，不构成真实设备操作授权或正式安全审批。")

    # 不再因状态改变而删除整个复选框/按钮，尽量保证前端节点稳定。
    can_confirm = st.session_state.task_status == "PENDING_HUMAN" and not approved
    confirmed = st.checkbox(
        "我已阅读结果，并理解所有建议都需要按现场规程核实。",
        key="approval_confirmed_checkbox",
        disabled=not can_confirm,
    )
    st.button(
        "确认演示结果" if can_confirm else ("本轮已确认" if approved else "无法确认"),
        disabled=not (can_confirm and confirmed),
        on_click=confirm_current_run,
        use_container_width=True,
        key="confirm_result_button",
    )
    if approved:
        st.success("✓ 本轮已完成演示确认，状态已同步更新。")
    if st.session_state.approval_record:
        st.info(f"记录状态：{st.session_state.approval_record['status']} · {st.session_state.approval_record['time']}")

# -------------------- 事件时间线和导出 --------------------
st.divider()
st.subheader("运行事件时间线")
if events:
    df = pd.DataFrame(events)
    # Plotly 时间线：用颜色区分成功、执行中、失败和人工处理
    status_colors = {
        "完成": "#38bdf8", "通过": "#34d399", "演示确认": "#34d399", "测试注入": "#a78bfa", "进行中": "#fbbf24",
        "修订中": "#a78bfa", "未通过": "#fb923c", "错误": "#f87171",
        "需人工处理": "#fbbf24",
    }
    plot_df = df.copy()
    plot_df["label"] = plot_df["time"] + " · " + plot_df["agent_label"]
    plot_df["color"] = plot_df["status"].map(status_colors).fillna("#94a3b8")
    # 将 Agent 映射为固定数值行，显式给第一行的上方留出空间，防止状态文字被裁切。
    # 仅修改时间线图表的坐标及留白；保留颜色、按钮及原有后台调用方式。
    agent_order = list(dict.fromkeys(plot_df["agent_label"].tolist()))
    agent_to_row = {name: idx for idx, name in enumerate(agent_order)}
    plot_df["agent_row"] = plot_df["agent_label"].map(agent_to_row)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(plot_df))),
        y=plot_df["agent_row"],
        mode="markers+text",
        text=plot_df["status"],
        textposition="top center",
        textfont=dict(color="#f3f7ff", size=12),
        cliponaxis=False,  # 即便文字靠近坐标边缘，也不要裁掉文字上半部分
        marker=dict(size=13, color=plot_df["color"], line=dict(width=1, color="#dbeafe")),
        customdata=plot_df[["agent_label", "time", "round", "status", "message"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>时间：%{customdata[1]}"
            "<br>轮次：%{customdata[2]}<br>状态：%{customdata[3]}"
            "<br>%{customdata[4]}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_layout(
        height=max(330, min(480, 170 + len(agent_order) * 62)),
        # 上方和左右多留空间，避免状态、坐标标签被裁剪
        margin=dict(l=168, r=38, t=68, b=62),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe6f5"),
        xaxis=dict(
            title="事件顺序", showgrid=True, gridcolor="#263b54", zeroline=False,
            range=[-0.65, len(plot_df) - 0.35],
            automargin=True,
        ),
        yaxis=dict(
            title="", showgrid=False,
            tickmode="array", tickvals=list(range(len(agent_order))),
            ticktext=agent_order,
            tickfont=dict(size=12),
            range=[len(agent_order) - 0.35, -0.9],
            automargin=True,
            zeroline=False,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.dataframe(
        df[["time", "agent_label", "round", "status", "message"]].rename(columns={
            "time": "时间", "agent_label": "Agent", "round": "轮次", "status": "状态", "message": "事件"
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "下载运行日志 CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="lingshu_run_log.csv",
        mime="text/csv",
    )
    with st.expander("查看原始日志文本"):
        st.code("\n".join(st.session_state.logs), language="text")
else:
    st.caption("暂无运行事件。提交任务后会记录各 Agent 的执行阶段。")

# -------------------- 历史任务独立预览 --------------------
def history_panel():
    with st.container(border=True):
        st.subheader("📁 本地运行记录 / JSON 导出")
        st.caption("历史记录只在这里预览，不影响上方当前任务。文件保存在本机 .lingshu_runs 目录。")
        if st.session_state.save_message:
            st.caption(st.session_state.save_message)

        current = st.session_state.result
        if isinstance(current, dict):
            record_export = {
                "result": {**current, "status": st.session_state.task_status},
                "events": st.session_state.events,
                "approval_record": st.session_state.approval_record,
            }
            st.download_button(
                "下载当前任务 JSON",
                data=json.dumps(redact(record_export), ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"lingshu_{current.get('run_id') or 'run'}.json",
                mime="application/json",
                key="download_current_json",
            )

        try:
            previous = list_records(limit=30)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            st.warning("读取历史记录目录失败，请检查 .lingshu_runs 是否有权限。")
            return

        if not previous:
            st.caption("目前没有已保存的历史运行。")
            return

        by_id = {r["run_id"]: r for r in previous if r.get("run_id")}
        if not by_id:
            st.caption("没有可读取的历史任务编号。")
            return

        choices = list(by_id)
        selected_id = st.selectbox(
            "选择历史运行（切换后直接在下方预览，不跳到顶部）",
            options=choices,
            format_func=lambda rid: (
                f"{by_id[rid].get('saved_at', '')} · {rid} · "
                f"{by_id[rid].get('status', '')} · {by_id[rid].get('task', '')[:35]}"
            ),
            key="history_selected_run",
        )
        if not selected_id:
            return

        try:
            record = load_record(selected_id)
            hist = record.get("result")
            if not isinstance(hist, dict) or hist.get("run_id") != selected_id:
                raise ValueError("所选任务记录不完整")
            hist_events = record.get("events", [])
            if not isinstance(hist_events, list):
                raise ValueError("历史事件格式错误")
            hist_approval = record.get("approval_record")
            status = hist.get("status", "未知")
            if status == "PENDING_HUMAN" and isinstance(hist_approval, dict):
                if hist_approval.get("run_id") == selected_id or (
                    not hist_approval.get("run_id")
                    and hist_approval.get("task") == hist.get("task")
                ):
                    status = "DEMO_CONFIRMED"
            readable = {
                "PENDING_HUMAN": "待人工确认", "DEMO_CONFIRMED": "演示已确认",
                "REJECTED": "审核未通过", "ERROR": "执行异常",
            }.get(status, status)
            st.success(f"历史任务已显示：{selected_id}｜{readable}")
            st.write("**任务描述**")
            st.write(hist.get("task") or "无")
            with st.expander("查看历史知识分析", expanded=False):
                st.write(hist.get("knowledge") or "无")
            with st.expander("查看历史诊断方案", expanded=True):
                st.write(hist.get("diagnosis") or "无")
            with st.expander("查看历史安全审核", expanded=True):
                st.write(hist.get("review") or "无")
            revisions = hist.get("revision_history") or []
            with st.expander(f"查看历史 Self-loop（{len(revisions)} 轮）", expanded=False):
                for i, item in enumerate(revisions):
                    if not isinstance(item, dict):
                        continue
                    st.markdown(f"**第 {i + 1} 轮 · {'通过' if item.get('review_passed') else '未通过'}**")
                    if item.get("test_injected"):
                        st.caption("本轮包含测试注入，非真实设备审核问题")
                    st.write(item.get("review") or "无审核意见")
                    if item.get("diagnosis_before") is not None:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("**修订前**")
                            st.write(item.get("diagnosis_before"))
                        with c2:
                            st.write("**修订后**")
                            st.write(item.get("diagnosis_after"))
                    st.divider()
            st.caption(f"历史运行事件：{len(hist_events)} 条；本轮确认：{'有记录' if hist_approval else '未确认'}")
            with st.expander("查看历史事件日志", expanded=False):
                if hist_events:
                    st.dataframe(pd.DataFrame(hist_events), hide_index=True,
                                 use_container_width=True)
                else:
                    st.caption("没有事件数据")
            st.download_button(
                "下载这条历史任务 JSON",
                data=json.dumps(redact(record), ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"lingshu_history_{selected_id}.json",
                mime="application/json",
                key=f"download_history_{selected_id}",
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            st.warning("这条历史记录无法读取，其他历史文件不受影响。")

if hasattr(st, "fragment"):
    st.fragment(history_panel)()
else:
    st.warning("当前 Streamlit 版本较旧，请先升级：python -m pip install -U 'streamlit>=1.37,<2'。旧版切换历史记录时可能仍会滚动。")
    history_panel()

st.divider()
st.caption("LingShu 2.1.0 · 结果需要现场核实 · 文字审核不等同于真实安全认证")
