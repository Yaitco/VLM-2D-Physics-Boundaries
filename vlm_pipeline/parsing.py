from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Optional, Tuple

import yaml

from .specs import PropertySpec, normalize_value


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith('```'):
        t = re.sub(r'^```(?:json)?\s*', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s*```$', '', t)
    return t.strip()


def find_first_json_object(text: str) -> Optional[str]:
    s = strip_code_fences(text)
    start = s.find('{')
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_model_output(raw_output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return None, 'empty_output'

    blob = find_first_json_object(raw_output)
    if blob is None:
        return None, 'json_not_found'

    try:
        parsed = json.loads(blob)
        if isinstance(parsed, dict):
            return parsed, None
        return None, 'json_not_object'
    except Exception as exc_json:
        try:
            parsed = yaml.safe_load(blob)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(blob)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception:
            pass
        return None, f'json_parse_error: {exc_json}'


def _fetch_pred_raw_value(parsed_json: Dict[str, Any], key: str) -> Any:
    if '.' in key:
        group, name = key.split('.', 1)
    else:
        group, name = None, key

    if key in parsed_json:
        return parsed_json[key]
    if name in parsed_json:
        return parsed_json[name]

    props = parsed_json.get('properties', {})
    if isinstance(props, dict):
        if key in props:
            return props[key]
        group_payload = props.get(group) if group is not None else None
        if isinstance(group_payload, dict) and name in group_payload:
            return group_payload[name]

    groups = parsed_json.get('groups', {})
    if isinstance(groups, dict):
        group_payload = groups.get(group) if group is not None else None
        if isinstance(group_payload, dict) and name in group_payload:
            return group_payload[name]
    return None


def normalize_pred(parsed_json: Optional[Dict[str, Any]], property_specs: Dict[str, PropertySpec]) -> Dict[str, Any]:
    if not isinstance(parsed_json, dict):
        return {
            'image_id': None,
            'primary_object': None,
            'notes': None,
            'properties': {
                key: [] if spec.value_type == 'multi_categorical' else 'unknown'
                for key, spec in property_specs.items()
            },
        }

    pred_props: Dict[str, Any] = {}
    for key, spec in property_specs.items():
        pred_props[key] = normalize_value(spec, _fetch_pred_raw_value(parsed_json, key))

    return {
        'image_id': str(parsed_json.get('image_id')).strip() if parsed_json.get('image_id') is not None else None,
        'primary_object': parsed_json.get('primary_object'),
        'notes': parsed_json.get('notes'),
        'properties': pred_props,
    }
