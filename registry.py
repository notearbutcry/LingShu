"""Versioned agent profiles and deterministic, inspectable capability routing."""
from __future__ import annotations
import asyncio
import json
import re
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentProfile:
    name: str
    version: str
    capabilities: tuple[str, ...]
    domains: tuple[str, ...]
    system_message: str
    capacity: int = 1
    tools: tuple[str, ...] = ()
    memory_strategy: str = 'none'
    memory_items: int = 2
    memory_chars: int = 1200


class AgentRegistry:
    def __init__(self, profiles=()):
        self.profiles = {}
        self.load = {}
        self.changed = asyncio.Condition()
        for profile in profiles:
            self.register(profile)

    def register(self, profile):
        if not isinstance(profile.name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", profile.name):
            raise ValueError("无效Agent名称")
        if not isinstance(profile.version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", profile.version):
            raise ValueError("版本应为 major.minor.patch")
        if not profile.capabilities or type(profile.capacity) is not int or profile.capacity < 1:
            raise ValueError("Agent能力和容量配置无效")
        if (any(x not in {'plan','knowledge','specialist','diagnose','review'} for x in profile.capabilities)
                or any(x not in {'general','pump','motor'} for x in profile.domains)
                or not profile.domains or not profile.system_message.strip()):
            raise ValueError('Agent能力、领域或提示词无效')
        if any(t != 'search_knowledge' for t in profile.tools):
            raise ValueError('角色请求了未实现的工具')
        if (profile.memory_strategy not in {'none', 'task_window'}
                or type(profile.memory_items) is not int or not 1 <= profile.memory_items <= 5
                or type(profile.memory_chars) is not int or not 100 <= profile.memory_chars <= 4000):
            raise ValueError('角色记忆策略或容量无效')
        key = profile.name + "@" + profile.version
        if key in self.profiles:
            raise ValueError("同名同版本Agent已注册")
        self.profiles[key] = profile
        self.load[key] = 0

    def discover(self, capability=None):
        # A newer version replaces an older one for new work, while in-flight leases remain valid.
        latest = {}
        for p in self.profiles.values():
            old = latest.get(p.name)
            if old is None or tuple(map(int, p.version.split('.'))) > tuple(map(int, old.version.split('.'))):
                latest[p.name] = p
        return [p for p in latest.values() if capability is None or capability in p.capabilities]

    @asynccontextmanager
    async def acquire(self, capability, domain, trace, node):
        async with self.changed:
            while True:
                profiles = self.discover(capability)
                if not profiles:
                    raise ValueError("没有匹配能力的Agent：" + capability)
                candidates = []
                for profile in profiles:
                    key = profile.name + "@" + profile.version
                    load = self.load[key]
                    # Domain match and spare capacity both influence assignment.
                    score = 100 + (30 if domain in profile.domains else 0) - 40 * load / profile.capacity
                    candidates.append({"agent": key, "score": score, "load": load,
                                       "capacity": profile.capacity, "domain_match": domain in profile.domains})
                available = [x for x in candidates if x["load"] < x["capacity"]]
                if available:
                    selected = sorted(available, key=lambda x: (-x["score"], x["agent"]))[0]
                    key = selected["agent"]
                    self.load[key] += 1
                    trace.append({"node": node, "capability": capability, "domain": domain,
                                  "selected": key, "candidates": candidates,
                                  "formula": "100 + 30*领域匹配 - 40*当前负载/容量"})
                    break
                await self.changed.wait()
        try:
            yield self.profiles[key]
        finally:
            async with self.changed:
                self.load[key] -= 1
                self.changed.notify_all()


def load_profiles(path=None):
    path = Path(path) if path else Path(__file__).parent / "config" / "agents.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not 3 <= len(data) <= 30:
        raise ValueError("Agent配置必须包含3～30个角色")
    profiles = []
    for row in data:
        if not isinstance(row, dict) or not isinstance(row.get("system_message"), str):
            raise ValueError("Agent配置格式无效")
        for key in ("capabilities", "domains"):
            if not isinstance(row.get(key), list) or any(not isinstance(v, str) for v in row[key]):
                raise ValueError("能力与领域必须为字符串数组")
        if not isinstance(row.get('tools', []), list) or any(not isinstance(t,str) for t in row.get('tools', [])):
            raise ValueError('工具集必须为字符串数组')
        try:
            profiles.append(AgentProfile(**{**row, "capabilities": tuple(row["capabilities"]),
                "domains": tuple(row["domains"]), 'tools': tuple(row.get('tools', []))}))
        except TypeError as exc:
            raise ValueError('角色配置缺少必需字段或包含未知字段') from exc
    AgentRegistry(profiles)  # Validate duplicates, names and versions before any API call.
    return profiles


def profile_summary():
    return [{k: v for k, v in asdict(p).items() if k != "system_message"} for p in load_profiles()]
