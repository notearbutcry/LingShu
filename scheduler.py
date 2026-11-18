"""Small async DAG scheduler with priorities, conditions and audit snapshots.

Only registered Python handlers execute. Model output never becomes Python code.
Capacity and agent load are scoped to one run; this is not a distributed scheduler.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import time
import copy
from privacy import redact
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass(frozen=True)
class Node:
    name: str
    action: Callable[[dict], Awaitable[object]]
    dependencies: tuple[str, ...] = ()
    priority: int = 0
    condition: Callable[[dict], bool] | None = None
    allow_skipped_dependencies: bool = False


class WorkflowError(RuntimeError):
    pass


class DAGExecutor:
    def __init__(self, nodes, *, concurrency=2, name="main", audit=None, checkpoint=None):
        if type(concurrency) is not int or not 1 <= concurrency <= 8:
            raise ValueError("并发上限应为1～8")
        self.nodes = list(nodes)
        self.concurrency = concurrency
        self.name = name
        self.audit = audit if audit is not None else []
        self.checkpoint = checkpoint
        self.states = {n.name: "WAITING" for n in self.nodes}
        self.outputs = {}
        self._validate()

    def _validate(self):
        if len(self.states) != len(self.nodes) or not self.nodes:
            raise ValueError("工作流节点名重复或为空")
        remaining = {n.name: set(n.dependencies) for n in self.nodes}
        if any(d not in remaining for deps in remaining.values() for d in deps):
            raise ValueError("工作流依赖不存在")
        resolved = set()
        while remaining:
            ready = [k for k, deps in remaining.items() if deps <= resolved]
            if not ready:
                raise ValueError("工作流存在循环依赖；反馈循环应在节点内部有界执行")
            resolved.update(ready)
            for key in ready:
                del remaining[key]

    def snapshot(self):
        return copy.deepcopy({"workflow": self.name, "states": dict(self.states),
                              "outputs": self.outputs})

    def _event(self, node, state, **extra):
        self.states[node] = state
        self.audit.append({"workflow": self.name, "node": node, "state": state,
                           "monotonic_s": round(time.monotonic(), 6), **extra})
        if self.checkpoint:
            self.checkpoint(self.snapshot())

    async def run(self, initial=None):
        context = dict(initial or {})
        pending = {n.name: n for n in self.nodes}
        running = {}
        started = {}
        try:
            while pending or running:
                ready = sorted((n for n in pending.values()
                                if all(self.states[d] in {"DONE", "SKIPPED"} for d in n.dependencies)),
                               key=lambda n: (-n.priority, n.name))
                for node in ready:
                    if len(running) >= self.concurrency:
                        break
                    skipped_dep = any(self.states[d] == "SKIPPED" for d in node.dependencies)
                    if (skipped_dep and not node.allow_skipped_dependencies) or (
                        node.condition is not None and not node.condition(context)
                    ):
                        del pending[node.name]
                        self.outputs[node.name] = context[node.name] = None
                        self._event(node.name, "SKIPPED", reason="条件不满足或前置节点跳过")
                        continue
                    del pending[node.name]
                    started[node.name] = time.perf_counter()
                    self._event(node.name, "RUNNING", priority=node.priority)
                    running[asyncio.create_task(node.action(copy.deepcopy(context)))] = node
                if not running:
                    if pending:
                        # A skip may have made additional nodes ready.
                        if ready:
                            continue
                        raise WorkflowError("无可执行节点")
                    break
                completed, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                # Deterministic tie handling; all pending tasks are cancelled on failure.
                for future in sorted(completed, key=lambda f: running[f].name):
                    node = running[future]
                    try:
                        value = future.result()
                        digest = hashlib.sha256(json.dumps(redact(value), ensure_ascii=False,
                                                          sort_keys=True).encode()).hexdigest()
                    except BaseException:
                        self._event(node.name, "CANCELLED" if future.cancelled() else "ERROR")
                        raise
                    del running[future]
                    context[node.name] = self.outputs[node.name] = value
                    self._event(node.name, "DONE", output_sha256=digest,
                                duration_ms=round((time.perf_counter() - started[node.name]) * 1000, 2))
            return context
        finally:
            for future, node in list(running.items()):
                if not future.done():
                    future.cancel()
                    self._event(node.name, "CANCELLED")
            if running:
                await asyncio.gather(*running, return_exceptions=True)
            # A sibling may finish in the same wait batch as a failure. Give it a
            # terminal state too; do not leave completed tasks marked RUNNING.
            for future, node in running.items():
                if self.states[node.name] != 'RUNNING':
                    continue
                if future.cancelled():
                    self._event(node.name, 'CANCELLED')
                elif future.exception() is not None:
                    self._event(node.name, 'ERROR')
                else:
                    try:
                        value = future.result()
                        digest = hashlib.sha256(json.dumps(redact(value), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                        self.outputs[node.name] = value
                        self._event(node.name, 'DONE', output_sha256=digest)
                    except (ValueError, TypeError):
                        self._event(node.name, 'ERROR')
            for name in pending:
                self._event(name, 'CANCELLED', reason='上游失败或任务中断，尚未执行')
