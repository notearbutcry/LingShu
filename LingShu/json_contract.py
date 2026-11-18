"""Bounded JSON objects: reject duplicate keys and non-JSON constants."""
import json
import re


def json_object(raw):
    if not isinstance(raw, str) or len(raw) > 100_000:
        raise ValueError('JSON响应为空、类型错误或过长')
    raw = raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.I)
        raw = raw[:-3].strip()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('JSON存在重复字段：' + key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('JSON包含非法常量')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError('需要JSON对象')
    return value
