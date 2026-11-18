"""灵枢后台：动态编排入口 + 独立验证 + 可追溯 Self-loop。

教学演示用途；支持本地示例资料检索，无传感器或控制设备权限；任何通过仅为文本审核。
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from datetime import datetime
from typing import Awaitable, Callable
from privacy import redact
from json_contract import json_object

WORKFLOW = ("plan", "evidence", "analysis_subworkflow", "diagnosis", "feedback")
AGENT_PROFILES = {
    "knowledge_agent": "辨别已知事实、假设和缺失数据，使用带来源的本地检索材料",
    "diagnosis_agent": "产生审慎诊断方案、根据审核意见完整修订",
    "safety_agent": "独立审核诊断文本并输出结构化 JSON 意见",
}
REQUIRED_SAFETY = ("停机并隔离能源", "穿戴防护用品")
# 只是最基础的文本风险词检查，不声称能识别所有现场危险。
UNSAFE_PHRASES = (
    "无需停机", "不必停机", "不用停机", "不需要停机", "不停机拆卸", "运行中拆卸",
    "无需断电", "带电拆卸", "带电维修", "直接触摸旋转", "直接接触旋转",
    "无需穿戴防护", "不用穿戴防护", "不必穿戴防护", "不要穿戴防护",
    "绕过联锁", "屏蔽联锁", "解除安全联锁",
)
EventFn = Callable[[str, str, str, int], None]


def get_last_message(result) -> str:
    if getattr(result, "messages", None):
        content = getattr(result.messages[-1], "content", "")
        return content if isinstance(content, str) else str(content)
    return str(result)


def normalize_check_text(text: str) -> str:
    """Ignore Markdown emphasis and horizontal spacing, preserving clause boundaries."""
    return re.sub(r"[^\S\n]+", "", re.sub(r"[*_`]+", "", text or ""))


def basic_text_check(text: str) -> tuple[bool, list[str]]:
    """依旧提供原版接口供 UI 与测试调用。"""
    checked = normalize_check_text(text)
    # A negated reminder must not satisfy the required positive instruction.
    missing = []
    for phrase in REQUIRED_SAFETY:
        matches = list(re.finditer(re.escape(phrase), checked))
        positive = any(not re.search(r'(?:不要|无需|不必|不用|不需要|不应|禁止|不得|不能|避免)$',
                                    checked[max(0,m.start()-12):m.start()]) for m in matches)
        if not positive:
            missing.append(phrase)
    return not missing, missing


def detect_unsafe_patterns(text: str) -> list[str]:
    """谨慎的文本规则：区分危险操作建议与对危险行为的明确禁止。

    这不是工业安全语义分类器。遇到不明确的表述仍按风险项拦截，
    并由独立的安全审核 Agent 继续检查整个诊断方案。
    """
    normalized = normalize_check_text(text)
    clauses = re.split(r"[。！？；;\n，,]", normalized)
    prohibitions = (
        "严禁", "禁止", "不得", "不应", "不能", "不可", "切勿", "避免",
        "不要", "勿", "不允许", "不建议", "杜绝", "防止", "谨防", "警惕",
    )
    negative_re = re.compile("(?:" + "|".join(map(re.escape, prohibitions)) + r")(?:直接|擅自|进行|实施|再次|私自|任何)?$")
    explanatory_re = re.compile(
        r"^(?:等)?(?:行为|操作)?(?:是|属于|可能|易|会|有|存在|引发|造成|导致|风险|危险|隐患|不安全|违规)"
    )
    found = []
    for phrase in UNSAFE_PHRASES:
        for clause in clauses:
            for match in re.finditer(re.escape(phrase), clause):
                before, after = clause[:match.start()], clause[match.end():]
                prohibition = negative_re.search(before)
                # Extend only a simple prohibited list, never across a sentence,
                # contrast, or affirmative instruction. E.g. 禁止带电拆卸、屏蔽联锁.
                list_prohibition = re.search(
                    r"(?:严禁|禁止|不得|不应|不可|切勿|不要|不允许|不建议)"
                    r"(?:直接|擅自|进行|实施|再次|私自)?"
                    r"[^、。！？；;\n，,：:]{1,30}"
                    r"(?:、[^、。！？；;\n，,：:]{1,30})*、$", before)
                if list_prohibition and re.search(
                    r"但|不过|然而|而是|除非|除了|不禁止|不建议禁止|不是禁止|并非禁止|"
                    r"建议|可以|允许|应当|必须|立即|然后|随后|需要",
                    before[list_prohibition.start():].replace("不建议", "禁止").replace("不允许", "禁止")):
                    list_prohibition = None
                for kind, match_scope in (("direct", prohibition), ("list", list_prohibition)):
                    if match_scope and re.search(r"(?:不|并非|不是|无需|取消|撤销)$", before[:match_scope.start()]):
                        if kind == "direct":
                            prohibition = None
                        else:
                            list_prohibition = None
                # 「禁止带电拆卸」不是建议带电拆卸。
                if prohibition or list_prohibition:
                    continue
                # 「带电拆卸可能导致触电」是风险描述，而非操作建议。
                if explanatory_re.search(after) and not re.search(
                    r"(?:建议|可以|允许|直接|先|进行|务必|立即|请|应当|必须)", before[-16:]
                ):
                    continue
                if phrase not in found:
                    found.append(phrase)
    return found


def result_quality_check(knowledge: str, diagnosis: str, review: str) -> list[str]:
    issues: list[str] = []
    if len((knowledge or "").strip()) < 30:
        issues.append("知识分析过短或为空")
    if len((diagnosis or "").strip()) < 50:
        issues.append("故障诊断过短或为空")
    if not (review or "").strip():
        issues.append("审核未返回内容")
    return issues


def parse_review_json(text: str) -> dict:
    """仅接受 JSON 对象或完整 JSON 代码块；解析失败一律拒绝。"""
    if not isinstance(text,str):
        return {"passed": False, "issues": ["审核响应必须为JSON文本"], "risk_level": "unknown", "parse_ok": False}
    raw = (text or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        data = json_object(raw)
    except (ValueError, TypeError):
        return {"passed": False, "issues": ["审核 JSON 无法解析"], "risk_level": "unknown", "parse_ok": False}
    if not isinstance(data, dict) or type(data.get("passed")) is not bool:
        return {"passed": False, "issues": ["审核缺少布尔型 passed 字段"], "risk_level": "unknown", "parse_ok": False}
    issues = data.get("issues")
    if not isinstance(issues, list) or any(not isinstance(x, str) for x in issues):
        return {"passed": False, "issues": ["审核 issues 应为字符串数组且必须提供"], "risk_level": "unknown", "parse_ok": False}
    issues = [x.strip()[:200] for x in issues if x.strip()][:20]
    if not data["passed"] and not issues:
        issues.append("审核未通过，但未提供可操作意见")
    risk = data.get("risk_level")
    if not isinstance(risk, str) or risk.strip().lower() not in ("low", "medium", "high", "unknown"):
        return {"passed": False, "issues": ["审核缺少有效的 risk_level 字段"], "risk_level": "unknown", "parse_ok": False}
    risk = risk.strip().lower()
    if risk == "high":
        issues.append("AI 审核标注高风险：必须人工复核，不自动放行")
    elif risk == "unknown":
        issues.append("无法明确判断风险等级：需人工复核")
    passed = bool(data["passed"] and not issues)
    return {"passed": passed, "issues": issues, "risk_level": risk, "parse_ok": True}


def review_as_markdown(data: dict) -> str:
    label = "文本审核通过（仍待人工确认）" if data["passed"] else "文本审核未通过"
    risks = {"low": "低", "medium": "中", "high": "高", "unknown": "待确定"}
    lines = [f"**审核结果：{label}**", f"风险级别（AI 判断）：{risks[data['risk_level']]}"]
    lines.extend("- " + issue for issue in data["issues"])
    if data["passed"]:
        lines.append("本轮未发现所检查项目的明显缺项；不等于现场安全认证。")
    return "\n\n".join(lines)


def _error_message(exc: Exception, api_key: str = "") -> str:
    msg = f"{type(exc).__name__}: {exc}"
    if api_key:
        msg = msg.replace(api_key, "[REDACTED]")
    return redact(msg, (api_key,))[:350]


def _build_result(*, task, knowledge, diagnosis, review, status,
                  missing, issues, history, max_revisions, test_mode, run_id) -> dict:
    return {
        "run_id": run_id,
        "task": task,
        "knowledge": knowledge,
        "diagnosis": diagnosis,
        "review": review,
        "status": status,
        "missing_phrases": list(missing),
        "validation_issues": list(issues),
        "revision_history": history,
        "max_revisions": max_revisions,
        "self_loop_test": bool(test_mode),
        "revision_count": sum(1 for x in history if x.get("diagnosis_before") is not None),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


async def execute_feedback_loop(
    *, task: str, knowledge: str, initial_diagnosis: str,
    review_call: Callable[[str], Awaitable[str]],
    revise_call: Callable[[str, str, list[str]], Awaitable[str]],
    max_revisions: int, emit: EventFn, self_loop_test: bool = False,
    run_id: str | None = None,
    validate_diagnosis=None, on_progress=None,
) -> dict:
    """真实审核、修订、复审；可用假 Agent 离线验证，不模拟通过。"""
    if type(max_revisions) is not int or not 0 <= max_revisions <= 5:
        raise ValueError("自动修订次数必须是 0～5 的整数")
    run_id = run_id or uuid.uuid4().hex[:12]
    history: list[dict] = []
    diagnosis = initial_diagnosis
    review_display = ""
    missing: list[str] = []
    validation_issues: list[str] = []
    change: dict | None = None
    state = "REJECTED"
    revisions_completed = 0
    stop_code = "REVISION_LIMIT"
    def publish():
        if on_progress:
            partial_history = list(history)
            if change and (not history or history[-1].get('diagnosis') != diagnosis):
                partial_history.append(dict(round=len(history), diagnosis=diagnosis, **change,
                    review_passed=False, review_error=True, issues=['修订已完成，复审尚未完成'], review='待复审'))
            on_progress(_build_result(task=task, knowledge=knowledge, diagnosis=diagnosis,
                review=review_display, status='ERROR', missing=missing, issues=validation_issues,
                history=partial_history, max_revisions=max_revisions, test_mode=self_loop_test, run_id=run_id)
                | {'revision_count': revisions_completed, 'stop_code': 'INTERRUPTED'})
    publish()
    for round_no in range(max_revisions + 1):
        emit("safety_agent", "进行中", f"开始第 {round_no + 1} 次审核", round_no)
        t0 = time.perf_counter()
        try:
            raw = await review_call(diagnosis)
        except Exception as exc:
            emit("safety_agent", "错误", "模型审核失败：" + _error_message(exc), round_no)
            state = "ERROR"
            stop_code = "REVIEW_CALL_ERROR"
            if change:
                history.append({"round": round_no, "diagnosis": diagnosis, **change,
                                "review_passed": False, "review_error": True,
                                "issues": ["修订后复审调用失败"], "review": "未完成复审"})
            review_display = "审核请求失败，请检查模型服务或网络。"
            validation_issues = ["安全审核 API 调用异常"]
            break
        ms = int((time.perf_counter() - t0) * 1000)
        parsed = parse_review_json(raw)
        review_display = review_as_markdown(parsed)
        _, missing = basic_text_check(diagnosis)
        unsafe = detect_unsafe_patterns(diagnosis)
        validation_issues = result_quality_check(knowledge, diagnosis, raw)
        if validate_diagnosis:
            validation_issues.extend(validate_diagnosis(diagnosis))
        if not parsed["parse_ok"]:
            validation_issues.append("审核 JSON 不符合格式要求")
        if unsafe:
            validation_issues.append("存在需要拦截的危险文本表述：" + "、".join(unsafe))
        no_progress = bool(change and (
            diagnosis.strip() == change["diagnosis_before"].strip() or
            any(x.get("diagnosis", "").strip() == diagnosis.strip() for x in history)))
        if no_progress:
            validation_issues.append("修订没有进展或重复历史稿，停止循环")
        # AI 审核与平台独立检查分别展示，避免同一问题重复出现。
        independent_issues = ["缺少安全提示：" + p for p in missing]
        independent_issues.extend(validation_issues)

        # 高风险或风险未知时不再自动修订：应该转专业人员复核。
        must_escalate = parsed["risk_level"] in ("high", "unknown")
        injected = bool(self_loop_test and round_no == 0 and not must_escalate)
        if injected:
            independent_issues.append("【测试注入】本轮要求补充：遵循现场作业许可流程")
            emit("system", "测试注入", "仅演示模式：模拟一项审核缺项，触发真实修订与复审", round_no)
        elif self_loop_test and round_no > 0 and "遵循现场作业许可流程" not in diagnosis:
            independent_issues.append("【测试校验】修订稿仍未包含现场作业许可要求")

        all_issues = list(dict.fromkeys(list(parsed["issues"]) + independent_issues))
        independent_issues = list(dict.fromkeys(
            x for x in independent_issues if x not in parsed["issues"]
        ))
        passed = bool(parsed["passed"] and not all_issues)
        if not passed and independent_issues:
            review_display += "\n\n**平台独立校验：未通过**\n\n" + "\n".join(
                "- " + x for x in independent_issues
            )
        entry = {
            "round": round_no,
            "diagnosis": diagnosis,
            "review": review_display,
            "review_raw": raw,
            "review_passed": passed,
            "model_review_passed": parsed["passed"],
            "risk_level": parsed["risk_level"],
            "test_injected": injected,
            "missing_phrases": list(missing),
            "validation_issues": list(validation_issues),
            "issues": all_issues,
            "duration_ms": ms,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if change:
            entry.update(change)
        history.append(entry)
        publish()
        if not parsed["parse_ok"]:
            state = "ERROR"
            stop_code = "REVIEW_FORMAT_ERROR"
            review_display += "\n\n审核响应格式错误，未完成有效审核；这不是设备风险结论。"
            history[-1]["review"] = review_display
            break
        if passed:
            state = "PENDING_HUMAN"
            stop_code = "TEXT_CHECKS_PASSED"
            emit("safety_agent", "通过", f"第 {round_no + 1} 轮文本审核通过（耗时 {ms}ms），待人工确认", round_no)
            break
        emit("safety_agent", "未通过", f"第 {round_no + 1} 轮未通过：{'；'.join(all_issues)[:160]}", round_no)
        if must_escalate:
            stop_code = "HIGH_RISK" if parsed["risk_level"] == "high" else "UNKNOWN_RISK"
            review_display += "\n\n流程处理：高风险或风险无法确定，已停止自动修订，转人工复核。"
            history[-1]["review"] = review_display
            emit("system", "需人工处理", "高风险或风险未知，停止 Self-loop，转专业人员复核", round_no)
            break
        if no_progress:
            stop_code = "NO_PROGRESS"
            emit("system", "需人工处理", "修订无进展，停止循环", round_no)
            break
        if round_no == max_revisions:
            stop_code = "REVISION_LIMIT"
            emit("system", "需人工处理", "修订次数已达上限，不自动放行", round_no)
            break
        emit("diagnosis_agent", "修订中", f"根据第 {round_no + 1} 轮审核意见修订", round_no + 1)
        try:
            t1 = time.perf_counter()
            updated = await revise_call(diagnosis, review_display, all_issues)
            revise_ms = int((time.perf_counter() - t1) * 1000)
            if not updated or not updated.strip():
                raise ValueError("诊断 Agent 修订结果为空")
        except Exception as exc:
            emit("diagnosis_agent", "错误", "诊断修订失败：" + _error_message(exc), round_no + 1)
            state = "ERROR"
            stop_code = "REVISION_CALL_ERROR"
            validation_issues = ["自动修订调用失败；历史审核记录已保留"]
            break
        change = {"diagnosis_before": diagnosis, "diagnosis_after": updated}
        diagnosis = updated
        revisions_completed += 1
        publish()
        emit("diagnosis_agent", "完成", f"第 {round_no + 1} 次修订完成（耗时 {revise_ms}ms），重新审核", round_no + 1)

    emit("system", "完成" if state == "PENDING_HUMAN" else "需人工处理" if state == "REJECTED" else "错误",
         "流程结束：" + {"PENDING_HUMAN": "文本审核通过，待人工确认", "REJECTED": "未通过，等待人工复核", "ERROR": "执行异常，未完成审核"}[state], 0)
    result = _build_result(task=task, knowledge=knowledge, diagnosis=diagnosis,
                         review=review_display, status=state, missing=missing,
                         issues=validation_issues, history=history,
                         max_revisions=max_revisions, test_mode=self_loop_test, run_id=run_id)

    result["revision_count"] = revisions_completed
    result["stop_code"] = stop_code
    result["blocking_issues"] = (list(dict.fromkeys(
        (history[-1].get("issues", []) if history else []) + validation_issues))
        if state != "PENDING_HUMAN" else [])
    return result


async def execute_pipeline(task, api_key, max_revisions, emit, self_loop_test=False, **options):
    """兼容原前端接口，新增动态编排选项。"""
    from platform_runtime import execute_platform
    return await execute_platform(task, api_key, max_revisions, emit,
                                  feedback_loop=execute_feedback_loop,
                                  self_loop_test=self_loop_test, **options)
