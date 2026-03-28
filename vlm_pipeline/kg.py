# vlm_pipeline/kg.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import yaml


def _norm(value: Any) -> str:
    if value is None:
        return "unknown"
    s = str(value).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    if s in {"", "none", "null", "nan"}:
        return "unknown"
    return s


@dataclass
class KGPriorConfig:
    properties: List[str]
    aliases: Dict[str, Dict[str, str]]
    material_priors: Dict[str, Dict[str, Dict[str, float]]]
    pairwise_rules: List[Dict[str, Any]]
    keep_original_bias: float = 0.75
    change_penalty: float = 0.55
    max_coordinate_passes: int = 6


def load_kg_prior_config(path: str | Path) -> KGPriorConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {}) or {}

    return KGPriorConfig(
        properties=list(raw.get("properties", [])),
        aliases={k: { _norm(kk): _norm(vv) for kk, vv in (v or {}).items()} for k, v in (raw.get("aliases", {}) or {}).items()},
        material_priors={
            _norm(material): {
                str(prop): { _norm(val): float(score) for val, score in (scores or {}).items() }
                for prop, scores in (prop_map or {}).items()
            }
            for material, prop_map in (raw.get("material_priors", {}) or {}).items()
        },
        pairwise_rules=list(raw.get("pairwise_rules", [])),
        keep_original_bias=float(defaults.get("keep_original_bias", 0.75)),
        change_penalty=float(defaults.get("change_penalty", 0.55)),
        max_coordinate_passes=int(defaults.get("max_coordinate_passes", 6)),
    )


class PhysicalKGReranker:
    """
    Post-hoc reranker for image-level property bundles.

    Input:
        {
            "intrinsic.main_material": "glass",
            "intrinsic.transparency_class": "opaque",
            ...
        }

    Output:
        corrected dict with the same keys.
    """

    def __init__(self, cfg: KGPriorConfig):
        self.cfg = cfg

    def canonicalize_value(self, property_key: str, value: Any) -> str:
        token = _norm(value)
        aliases = self.cfg.aliases.get(property_key, {})
        return aliases.get(token, token)

    def canonicalize_bundle(self, bundle: Mapping[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key, value in bundle.items():
            out[str(key)] = self.canonicalize_value(str(key), value)
        return out

    def _material_score(self, material: str, property_key: str, value: str) -> float:
        mat_block = self.cfg.material_priors.get(material, {})
        prop_block = mat_block.get(property_key, {})
        return float(prop_block.get(value, 0.0))

    def _pairwise_score(
        self,
        left_property: str,
        left_value: str,
        right_property: str,
        right_value: str,
    ) -> float:
        score = 0.0
        for rule in self.cfg.pairwise_rules:
            lp = str(rule.get("left_property"))
            rp = str(rule.get("right_property"))
            if lp != left_property or rp != right_property:
                continue
            weights = rule.get("weights", {}) or {}
            left_block = weights.get(left_value, {}) or {}
            score += float(left_block.get(right_value, 0.0))
        return score

    def _all_pairwise_scores(self, bundle: Mapping[str, str]) -> float:
        score = 0.0
        keys = list(bundle.keys())
        for i, left_key in enumerate(keys):
            for j, right_key in enumerate(keys):
                if i == j:
                    continue
                left_value = bundle[left_key]
                right_value = bundle[right_key]
                score += self._pairwise_score(left_key, left_value, right_key, right_value)
        return score

    def _bundle_score(
        self,
        candidate_bundle: Mapping[str, str],
        original_bundle: Mapping[str, str],
    ) -> float:
        score = 0.0

        # Keep-original bias + small penalty for changes
        for key, value in candidate_bundle.items():
            if value == original_bundle.get(key, "unknown"):
                score += self.cfg.keep_original_bias
            else:
                score -= self.cfg.change_penalty

        # Material-driven priors
        material = candidate_bundle.get("intrinsic.main_material", "unknown")
        if material != "unknown":
            for key, value in candidate_bundle.items():
                score += self._material_score(material, key, value)

        # Pairwise consistency
        score += self._all_pairwise_scores(candidate_bundle)
        return score

    def _candidate_values_for_property(
        self,
        property_key: str,
        current_bundle: Mapping[str, str],
    ) -> List[str]:
        candidates: Set[str] = set()
        current_value = current_bundle.get(property_key, "unknown")
        candidates.add(current_value)

        # candidates from material priors
        material = current_bundle.get("intrinsic.main_material", "unknown")
        if material in self.cfg.material_priors:
            prop_map = self.cfg.material_priors[material].get(property_key, {})
            candidates.update(prop_map.keys())

        # candidates from pairwise rules
        for rule in self.cfg.pairwise_rules:
            lp = str(rule.get("left_property"))
            rp = str(rule.get("right_property"))
            weights = rule.get("weights", {}) or {}
            if lp == property_key:
                candidates.update(weights.keys())
            elif rp == property_key:
                for _, right_map in weights.items():
                    candidates.update((right_map or {}).keys())

        # never return empty
        return sorted(candidates) if candidates else [current_value]

    def rerank_bundle(self, bundle: Mapping[str, Any]) -> Dict[str, str]:
        original = self.canonicalize_bundle(bundle)
        state = dict(original)

        for _ in range(self.cfg.max_coordinate_passes):
            changed = False
            for property_key in self.cfg.properties:
                if property_key not in state:
                    continue

                best_value = state[property_key]
                best_score = None

                for candidate_value in self._candidate_values_for_property(property_key, state):
                    candidate_bundle = dict(state)
                    candidate_bundle[property_key] = candidate_value
                    score = self._bundle_score(candidate_bundle, original)

                    if best_score is None or score > best_score:
                        best_score = score
                        best_value = candidate_value

                if best_value != state[property_key]:
                    state[property_key] = best_value
                    changed = True

            if not changed:
                break

        return state


def make_kg_reranker(config_path: str | Path) -> PhysicalKGReranker:
    return PhysicalKGReranker(load_kg_prior_config(config_path))
