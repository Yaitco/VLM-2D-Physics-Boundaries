from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from .registry import DatasetContext, ROOT_DIR

PDF_COMPACT_SCHEMA_PATH = ROOT_DIR / 'configs' / 'pdf_protocol_properties.yaml'
NARROW_CORE_KEYS_PATH = ROOT_DIR / 'configs' / 'narrow_core_property_keys.yaml'
ABO_NATURAL_BG_V2_SCHEMA_PATH = ROOT_DIR / 'configs' / 'abo_natural_bg_v2_properties.yaml'


@dataclass
class PropertySpec:
    key: str
    group: str
    name: str
    value_type: str  # categorical | boolean | multi_categorical
    allowed_values: List[str]
    description: str


def _normalize_token(value: str) -> str:
    return re.sub(r'\s+', '_', value.strip().lower()).replace('-', '_')


def _load_enums(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    enums = schema.get('enums', {})
    out: Dict[str, List[str]] = {}
    if not isinstance(enums, dict):
        return out
    for enum_name, values in enums.items():
        if isinstance(values, list):
            out[str(enum_name)] = [_normalize_token(str(v)) for v in values]
    return out


def _load_allowed_values(
    cfg: Dict[str, Any],
    enums_map: Dict[str, List[str]],
    enum_ref_key: str = 'enum_ref',
    inline_enum_key: str = 'enum',
) -> List[str]:
    enum_ref = cfg.get(enum_ref_key)
    if enum_ref is not None:
        values = list(enums_map.get(str(enum_ref), []))
        if values:
            return values
    inline_values = cfg.get(inline_enum_key)
    if isinstance(inline_values, list):
        return [_normalize_token(str(v)) for v in inline_values]
    return []


def load_property_specs(
    schema_path: Path,
    include_groups: Optional[Sequence[str]] = None,
) -> Dict[str, PropertySpec]:
    with schema_path.open('r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)

    groups = schema.get('groups', {})
    if not isinstance(groups, dict):
        raise ValueError(f'Invalid schema: groups are missing in {schema_path}')

    enums_map = _load_enums(schema)
    specs: Dict[str, PropertySpec] = {}

    for group_name, group_cfg in groups.items():
        if include_groups and group_name not in include_groups:
            continue
        properties = group_cfg.get('properties', {}) if isinstance(group_cfg, dict) else {}
        if not isinstance(properties, dict):
            continue

        for prop_name, prop_cfg in properties.items():
            if not isinstance(prop_cfg, dict):
                continue

            key = f'{group_name}.{prop_name}'
            desc = str(prop_cfg.get('description') or '').strip()
            prop_type = prop_cfg.get('type')

            if prop_type == 'categorical':
                allowed = _load_allowed_values(prop_cfg, enums_map)
                if 'unknown' not in allowed:
                    allowed.append('unknown')
                specs[key] = PropertySpec(
                    key=key,
                    group=group_name,
                    name=prop_name,
                    value_type='categorical',
                    allowed_values=allowed,
                    description=desc,
                )
                continue

            if prop_type == 'boolean':
                specs[key] = PropertySpec(
                    key=key,
                    group=group_name,
                    name=prop_name,
                    value_type='boolean',
                    allowed_values=['true', 'false', 'unknown'],
                    description=desc,
                )
                continue

            if prop_type == 'list':
                items = prop_cfg.get('items', {})
                if not isinstance(items, dict) or items.get('type') != 'categorical':
                    continue
                allowed = [value for value in _load_allowed_values(items, enums_map) if value != 'unknown']
                specs[key] = PropertySpec(
                    key=key,
                    group=group_name,
                    name=prop_name,
                    value_type='multi_categorical',
                    allowed_values=allowed,
                    description=desc,
                )

    if not specs:
        raise ValueError('No evaluable properties found in schema.')
    return specs


def load_property_subset_specs(schema_path: Path, subset_keys_path: Path) -> Dict[str, PropertySpec]:
    base_specs = load_property_specs(schema_path=schema_path)
    with subset_keys_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    include_keys = cfg.get('include_keys', [])
    if not isinstance(include_keys, list) or not include_keys:
        raise ValueError(f'Invalid subset config: include_keys are missing in {subset_keys_path}')

    subset: Dict[str, PropertySpec] = {}
    missing: List[str] = []
    for key in include_keys:
        key = str(key)
        spec = base_specs.get(key)
        if spec is None:
            missing.append(key)
            continue
        subset[key] = spec

    if missing:
        raise ValueError(f'Unknown property keys in subset config {subset_keys_path}: {missing}')
    return subset


def load_compact_property_specs(schema_path: Path) -> Dict[str, PropertySpec]:
    with schema_path.open('r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)

    properties = schema.get('properties', {})
    if not isinstance(properties, dict):
        raise ValueError(f'Invalid compact schema: properties are missing in {schema_path}')

    specs: Dict[str, PropertySpec] = {}
    for key, cfg in properties.items():
        if not isinstance(cfg, dict) or str(cfg.get('type')) != 'categorical':
            continue
        allowed = _load_allowed_values(cfg, enums_map={})
        if 'unknown' not in allowed:
            allowed.append('unknown')
        specs[str(key)] = PropertySpec(
            key=str(key),
            group='pdf_compact',
            name=str(key),
            value_type='categorical',
            allowed_values=allowed,
            description=str(cfg.get('description') or '').strip(),
        )

    if not specs:
        raise ValueError(f'No properties found in compact schema: {schema_path}')
    return specs


def load_protocol_property_specs(
    protocol_name: str,
    schema_path: Optional[Path] = None,
    include_groups: Optional[Sequence[str]] = None,
) -> Dict[str, PropertySpec]:
    if protocol_name in {'expanded_ontology', 'full_expanded'}:
        if schema_path is None:
            raise ValueError('schema_path is required for expanded ontology protocol')
        return load_property_specs(schema_path=schema_path, include_groups=include_groups)

    if protocol_name == 'narrow_core':
        if schema_path is None:
            raise ValueError('schema_path is required for narrow_core protocol')
        return load_property_subset_specs(schema_path=schema_path, subset_keys_path=NARROW_CORE_KEYS_PATH)

    if protocol_name == 'pdf_compact':
        return load_compact_property_specs(PDF_COMPACT_SCHEMA_PATH)

    if protocol_name == 'natural_bg_v2':
        return load_compact_property_specs(ABO_NATURAL_BG_V2_SCHEMA_PATH)

    raise ValueError(f'Unknown protocol_name: {protocol_name}')


def resolve_protocol_schema_path(dataset: DatasetContext, protocol_name: str) -> Optional[Path]:
    if protocol_name in {'expanded_ontology', 'full_expanded', 'narrow_core'}:
        if dataset.schema_path is None:
            raise ValueError(
                f"Protocol '{protocol_name}' requires schema_path, but dataset '{dataset.dataset_name}' does not provide one."
            )
        return dataset.schema_path
    return None


def _normalize_categorical(value: Any, allowed: Sequence[str]) -> str:
    allowed_set = set(allowed)
    if value is None:
        return 'unknown'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return 'unknown'
    token = _normalize_token(value)
    if token in {'', 'none', 'null', 'nan'}:
        return 'unknown'
    if token in allowed_set:
        return token
    token_alt = token.replace('/', '_')
    return token_alt if token_alt in allowed_set else 'unknown'


def _normalize_boolean(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'unknown'
    token = _normalize_token(str(value))
    if token in {'true', '1', 'yes', 'y'}:
        return 'true'
    if token in {'false', '0', 'no', 'n'}:
        return 'false'
    return 'unknown'


def _normalize_multi_categorical(value: Any, allowed: Sequence[str]) -> List[str]:
    allowed_set = set(allowed)
    if value is None:
        return []
    if isinstance(value, list):
        raw_values: Iterable[Any] = value
    elif isinstance(value, str):
        if value.strip() in {'', 'none', 'null', 'unknown'}:
            return []
        raw_values = [v.strip() for v in value.split(',')]
    else:
        return []

    out: List[str] = []
    for raw in raw_values:
        token = _normalize_token(str(raw))
        if token in allowed_set and token not in out:
            out.append(token)
    return out


def normalize_value(spec: PropertySpec, value: Any) -> Any:
    if spec.value_type == 'categorical':
        return _normalize_categorical(value, spec.allowed_values)
    if spec.value_type == 'boolean':
        return _normalize_boolean(value)
    if spec.value_type == 'multi_categorical':
        return _normalize_multi_categorical(value, spec.allowed_values)
    raise ValueError(f'Unsupported value_type: {spec.value_type}')


def is_known_value(spec: PropertySpec, value: Any) -> bool:
    if spec.value_type == 'multi_categorical':
        return isinstance(value, list) and len(value) > 0
    return value not in {None, 'unknown'}


def values_equal(spec: PropertySpec, pred: Any, gt: Any) -> bool:
    if not is_known_value(spec, gt):
        return False
    if spec.value_type == 'multi_categorical':
        return sorted(pred or []) == sorted(gt or [])
    return pred == gt


def serialize_value(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ''
    return str(value)
