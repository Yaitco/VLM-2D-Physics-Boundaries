from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import DatasetContext
from .specs import PropertySpec, _normalize_token, normalize_value


MASK_FIELD_ALIASES = {
    'mask_path': 'mask_path',
    'masks': 'mask_path',
    'masks_hint': 'mask_path_hint',
    'mask_path_hint': 'mask_path_hint',
    'masks_hint_title': 'mask_path_hint_title',
    'mask_path_hint_title': 'mask_path_hint_title',
}


MASK_PREVIEW_FIELD_ALIASES = {
    'sam_preview_path': 'sam_preview_path',
    'seg_preview_path': 'seg_preview_path',
    'masks_preview': 'seg_preview_path',
    'mask_preview_hint': 'seg_preview_path_hint',
    'masks_preview_hint': 'seg_preview_path_hint',
    'seg_preview_path_hint': 'seg_preview_path_hint',
    'mask_preview_hint_title': 'seg_preview_path_hint_title',
    'masks_preview_hint_title': 'seg_preview_path_hint_title',
    'seg_preview_path_hint_title': 'seg_preview_path_hint_title',
}


def _normalize_field_choice(
    field_name: Optional[str],
    aliases: Dict[str, str],
    field_kind: str,
) -> Optional[str]:
    if field_name is None:
        return None
    raw = str(field_name).strip()
    if not raw:
        return None
    canonical = aliases.get(raw, raw)
    if canonical not in aliases.values():
        allowed = sorted(set(aliases.keys()) | set(aliases.values()))
        raise ValueError(f'Unsupported {field_kind}: {field_name}. Allowed values: {allowed}')
    return canonical


def _default_preview_field_for_mask_field(mask_field: str) -> Optional[str]:
    if mask_field == 'mask_path':
        return 'seg_preview_path'
    if mask_field == 'mask_path_hint':
        return 'seg_preview_path_hint'
    if mask_field == 'mask_path_hint_title':
        return 'seg_preview_path_hint_title'
    return None


def _resolve_dataset_relative_path(dataset_dir: Path, path_raw: Any) -> Optional[Path]:
    if not isinstance(path_raw, str):
        return None
    path_str = path_raw.strip().replace('\\', '/')
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (dataset_dir.parent / path).resolve()
    return path


def _attach_mask_fields(
    sample: Dict[str, Any],
    item: Dict[str, Any],
    dataset_dir: Path,
    mask_field: str,
    mask_preview_field: Optional[str],
) -> None:
    mask_path = _resolve_dataset_relative_path(dataset_dir, item.get(mask_field))
    if mask_path is not None and mask_path.exists():
        sample['mask_path'] = str(mask_path)
        sample['mask_field'] = mask_field

    if mask_preview_field is None:
        return

    preview_path = _resolve_dataset_relative_path(dataset_dir, item.get(mask_preview_field))
    if preview_path is not None and preview_path.exists():
        sample['mask_preview_path'] = str(preview_path)
        sample['mask_preview_field'] = mask_preview_field


def _resolve_panel_path(panel_path_raw: str, dataset_dir: Path, panels_dir: Path, obj_id: str) -> Path:
    candidates: List[Path] = []
    if panel_path_raw:
        p = Path(panel_path_raw)
        if p.is_absolute():
            candidates.extend([p, panels_dir / p.name])
        else:
            candidates.extend([dataset_dir / p, panels_dir / p.name])
    candidates.append(panels_dir / f'{obj_id}.png')

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'Panel image not found for obj_id={obj_id}. Tried: {[str(x) for x in candidates]}')


def _extract_gt_properties(record: Dict[str, Any], property_specs: Dict[str, PropertySpec]) -> Dict[str, Any]:
    groups = record.get('groups', {})
    groups = groups if isinstance(groups, dict) else {}

    gt: Dict[str, Any] = {}
    for key, spec in property_specs.items():
        group_payload = groups.get(spec.group, {}) if '.' in key else record
        raw_value = group_payload.get(spec.name) if isinstance(group_payload, dict) else None
        gt[key] = normalize_value(spec, raw_value)
    return gt


def _map_pdf_material(raw_value: Any) -> str:
    mapping = {
        'wood': 'wood',
        'metal': 'metal',
        'glass': 'glass',
        'plastic': 'plastic',
        'fabric': 'fabric',
        'paper': 'paper/cardboard',
        'cardboard': 'paper/cardboard',
        'ceramic': 'ceramic/stone',
        'stone': 'ceramic/stone',
        'rubber': 'rubber',
        'leather': 'leather',
        'unknown': 'unknown',
    }
    token = _normalize_token(str(raw_value)) if raw_value is not None else 'unknown'
    return mapping.get(token, 'other')


def _map_pdf_reflectance(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else 'unknown'
    if token == 'matte':
        return 'matte'
    if token in {'glossy', 'semi_glossy', 'very_glossy', 'mirror_like'}:
        return 'glossy'
    return 'unknown'


def _map_pdf_surface_roughness(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else 'unknown'
    if token in {'smooth', 'very_smooth'}:
        return 'smooth'
    if token in {'rough', 'very_rough'}:
        return 'rough'
    return 'unknown'


def _map_pdf_rigidity(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else 'unknown'
    if token == 'rigid':
        return 'rigid'
    if token in {'flexible', 'floppy'}:
        return 'deformable'
    return 'unknown'


def _map_pdf_fragility(intrinsic: Dict[str, Any], affordance: Dict[str, Any]) -> str:
    breakable = affordance.get('breakable')
    if isinstance(breakable, bool):
        return 'fragile' if breakable else 'not_fragile'
    token = _normalize_token(str(intrinsic.get('brittleness_class'))) if intrinsic.get('brittleness_class') is not None else 'unknown'
    if token in {'brittle', 'very_brittle', 'slightly_brittle'}:
        return 'fragile'
    if token == 'non_brittle':
        return 'not_fragile'
    return 'unknown'


def _map_pdf_state(state_payload: Dict[str, Any]) -> str:
    is_dirty = state_payload.get('is_dirty')
    if isinstance(is_dirty, bool):
        return 'dirty' if is_dirty else 'clean'
    return 'unknown'


def _map_pdf_weight_hint(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else 'unknown'
    if token in {'very_light', 'light'}:
        return 'light'
    if token in {'heavy', 'very_heavy'}:
        return 'heavy'
    return 'unknown'


def _map_pdf_temperature_hint(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else 'unknown'
    if token in {'room', 'room_temp'}:
        return 'room'
    if token in {'warm', 'hot'}:
        return 'hot'
    if token in {'cold', 'chilled', 'frozen'}:
        return 'cold'
    return 'unknown'


def _map_pdf_phase(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else 'unknown'
    return token if token in {'solid', 'liquid', 'gas'} else 'unknown'


def _map_pdf_filled_state(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else 'unknown'
    if token == 'empty':
        return 'empty'
    if token in {'almost_empty', 'half_full', 'almost_full'}:
        return 'partially_filled'
    if token == 'full':
        return 'filled'
    return 'unknown'


def _map_pdf_slipperiness_hint(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else 'unknown'
    if token == 'slippery':
        return 'slippery'
    if token in {'high', 'normal'}:
        return 'not_slippery'
    return 'unknown'


def extract_pdf_protocol_properties(record: Dict[str, Any], property_specs: Dict[str, PropertySpec]) -> Dict[str, Any]:
    groups = record.get('groups', {}) if isinstance(record.get('groups'), dict) else {}
    intrinsic = groups.get('intrinsic', {}) if isinstance(groups.get('intrinsic'), dict) else {}
    state = groups.get('state', {}) if isinstance(groups.get('state'), dict) else {}
    affordance = groups.get('affordance', {}) if isinstance(groups.get('affordance'), dict) else {}

    compact_values = {
        'material': _map_pdf_material(intrinsic.get('main_material')),
        'transparency': _normalize_token(str(intrinsic.get('transparency_class'))) if intrinsic.get('transparency_class') is not None else 'unknown',
        'reflectance': _map_pdf_reflectance(intrinsic.get('glossiness_class')),
        'surface_roughness': _map_pdf_surface_roughness(intrinsic.get('surface_roughness_class')),
        'rigidity': _map_pdf_rigidity(intrinsic.get('rigidity_class')),
        'fragility': _map_pdf_fragility(intrinsic, affordance),
        'wetness': 'unknown',
        'state': _map_pdf_state(state),
        'weight_hint': _map_pdf_weight_hint(intrinsic.get('mass_class')),
        'temperature_hint': _map_pdf_temperature_hint(state.get('object_temperature_class')),
        'phase': _map_pdf_phase(intrinsic.get('state_of_matter')),
        'filled_state': _map_pdf_filled_state(state.get('fill_state_class')),
        'slipperiness_hint': _map_pdf_slipperiness_hint(intrinsic.get('friction_class')),
    }
    return {key: normalize_value(spec, compact_values.get(key)) for key, spec in property_specs.items()}


def load_abo150_samples(
    annotations_path: Path,
    dataset_dir: Path,
    property_specs: Dict[str, PropertySpec],
    protocol_name: str = 'expanded_ontology',
    max_samples: Optional[int] = None,
    random_seed: int = 42,
    mask_field: str = 'mask_path',
    mask_preview_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    mask_field = _normalize_field_choice(mask_field, MASK_FIELD_ALIASES, 'mask_field') or 'mask_path'
    if mask_preview_field is None:
        mask_preview_field = _default_preview_field_for_mask_field(mask_field)
    mask_preview_field = _normalize_field_choice(
        mask_preview_field,
        MASK_PREVIEW_FIELD_ALIASES,
        'mask_preview_field',
    )

    panels_dir = dataset_dir / 'selected_150_photos' / 'panels'
    rows: List[Dict[str, Any]] = []

    with annotations_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            obj_id = str(record.get('obj_id') or '')
            if not obj_id:
                continue

            panels = record.get('panels', {}) if isinstance(record.get('panels'), dict) else {}
            panel_path = _resolve_panel_path(
                panel_path_raw=str(panels.get('panel_path') or ''),
                dataset_dir=dataset_dir,
                panels_dir=panels_dir,
                obj_id=obj_id,
            )

            if protocol_name == 'pdf_compact':
                gt_properties = extract_pdf_protocol_properties(record, property_specs)
            else:
                gt_properties = _extract_gt_properties(record, property_specs)

            sample = {
                'image_id': obj_id,
                'path': str(panel_path),
                'split': record.get('split'),
                'caption': record.get('caption'),
                'gt_properties': gt_properties,
            }
            _attach_mask_fields(
                sample=sample,
                item=panels,
                dataset_dir=dataset_dir,
                mask_field=mask_field,
                mask_preview_field=mask_preview_field,
            )
            rows.append(sample)

    if max_samples is not None and len(rows) > max_samples:
        rng = random.Random(random_seed)
        rows = rng.sample(rows, k=max_samples)

    if not rows:
        raise RuntimeError('Loaded 0 samples from annotations.')
    return rows


def load_meta_subset_samples(
    meta_path: Path,
    dataset_dir: Path,
    property_specs: Dict[str, PropertySpec],
    max_samples: Optional[int] = None,
    random_seed: int = 42,
    mask_field: str = 'mask_path',
    mask_preview_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows = json.loads(meta_path.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError(f'Expected a JSON list in {meta_path}')
    mask_field = _normalize_field_choice(mask_field, MASK_FIELD_ALIASES, 'mask_field') or 'mask_path'
    if mask_preview_field is None:
        mask_preview_field = _default_preview_field_for_mask_field(mask_field)
        if mask_preview_field == 'seg_preview_path' and not any(
            isinstance(item, dict) and item.get(mask_preview_field) for item in rows
        ):
            mask_preview_field = 'sam_preview_path'
    mask_preview_field = _normalize_field_choice(
        mask_preview_field,
        MASK_PREVIEW_FIELD_ALIASES,
        'mask_preview_field',
    )

    samples: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get('image_id') or '')
        image_rel = str(item.get('path') or '').strip()
        if not image_id or not image_rel:
            continue

        gt_raw = item.get('properties', {})
        if not isinstance(gt_raw, dict):
            gt_raw = {}
        gt_properties = {
            key: normalize_value(spec, gt_raw.get(spec.name))
            for key, spec in property_specs.items()
        }

        image_path = Path(image_rel)
        if not image_path.is_absolute():
            image_path = (dataset_dir.parent / image_path).resolve()

        sample = {
            'image_id': image_id,
            'path': str(image_path),
            'caption': item.get('notes'),
            'primary_object': item.get('primary_object'),
            'gt_properties': gt_properties,
        }
        _attach_mask_fields(
            sample=sample,
            item=item,
            dataset_dir=dataset_dir,
            mask_field=mask_field,
            mask_preview_field=mask_preview_field,
        )

        samples.append(sample)

    if max_samples is not None and len(samples) > max_samples:
        rng = random.Random(random_seed)
        samples = rng.sample(samples, k=max_samples)

    if not samples:
        raise RuntimeError('Loaded 0 samples from meta.json.')
    return samples


def load_samples_for_dataset(
    dataset: DatasetContext,
    property_specs: Dict[str, PropertySpec],
    protocol_name: str,
    max_samples: Optional[int] = None,
    random_seed: int = 42,
    mask_field: str = 'mask_path',
    mask_preview_field: Optional[str] = None,
    meta_override_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    if dataset.dataset_type == 'abo150_annotations':
        if meta_override_path is not None:
            raise ValueError('meta_override_path is only supported for datasets with dataset_type="meta_json"')
        if dataset.annotations_path is None:
            raise ValueError(f"annotations_path is missing for dataset '{dataset.dataset_name}'")
        return load_abo150_samples(
            annotations_path=dataset.annotations_path,
            dataset_dir=dataset.dataset_dir,
            property_specs=property_specs,
            protocol_name=protocol_name,
            max_samples=max_samples,
            random_seed=random_seed,
            mask_field=mask_field,
            mask_preview_field=mask_preview_field,
        )

    if dataset.dataset_type == 'meta_json':
        effective_meta_path = meta_override_path
        if effective_meta_path is None:
            effective_meta_path = dataset.meta_path
        else:
            effective_meta_path = Path(effective_meta_path)
            if not effective_meta_path.is_absolute():
                effective_meta_path = effective_meta_path.resolve()
        if effective_meta_path is None:
            raise ValueError(f"meta_path is missing for dataset '{dataset.dataset_name}'")
        if not effective_meta_path.exists():
            raise FileNotFoundError(f'meta override file not found: {effective_meta_path}')
        return load_meta_subset_samples(
            meta_path=effective_meta_path,
            dataset_dir=dataset.dataset_dir,
            property_specs=property_specs,
            max_samples=max_samples,
            random_seed=random_seed,
            mask_field=mask_field,
            mask_preview_field=mask_preview_field,
        )

    raise ValueError(f'Unsupported dataset_type: {dataset.dataset_type}')
