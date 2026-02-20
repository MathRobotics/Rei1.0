from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.mapping import mapping_as_dict
from ...core.state_schema import DTYPE_VISION, canonical_field_name
from .dsl_ops import find_var_dsl, rewrite_get_state_owner_name


@dataclass(frozen=True)
class PreparedVisionCalibrationDsl:
    dsl: dict[str, Any]
    p_var: str
    owner_type: str
    owner_name: str
    field: str
    k: int
    term_name: str
    observations: np.ndarray


def _resolve_nonempty_str(raw: Any, *, where: str) -> str:
    value = str(raw).strip()
    if value == "":
        raise ValueError(f"{where} must be non-empty.")
    return value


def _resolve_observations(raw: Any, *, where: str) -> np.ndarray:
    obs = np.asarray(raw, dtype=float).reshape(-1)
    if obs.size == 0:
        raise ValueError(f"{where} must be non-empty.")
    return obs


def _resolve_int_nonnegative(raw: Any, *, where: str) -> int:
    try:
        value = int(raw)
    except Exception as e:
        raise ValueError(f"{where} must be an integer, got {raw!r}.") from e
    if value < 0:
        raise ValueError(f"{where} must be >= 0, got {value}.")
    return value


def _upsert_term_by_name(
    dsl: dict[str, Any],
    *,
    term_name: str,
    term_dsl: dict[str, Any],
) -> None:
    terms_raw = dsl.get("terms", None)
    if terms_raw is None:
        dsl["terms"] = [term_dsl]
        return
    if not isinstance(terms_raw, list):
        raise ValueError("prepare_vision_calibration_problem_dsl: dsl['terms'] must be a list.")

    terms: list[Any] = list(terms_raw)
    hit_indices: list[int] = []
    for i, term in enumerate(terms):
        if not isinstance(term, Mapping):
            continue
        expr = term.get("expr", None)
        if not isinstance(expr, Mapping):
            continue
        if str(expr.get("name", "")) == term_name:
            hit_indices.append(i)

    if len(hit_indices) == 0:
        terms.append(term_dsl)
        dsl["terms"] = terms
        return

    first = int(hit_indices[0])
    terms[first] = term_dsl
    for idx in reversed(hit_indices[1:]):
        del terms[idx]
    dsl["terms"] = terms


def _build_standardized_vision_term(
    *,
    p_var: str,
    owner_type: str,
    owner_name: str,
    field: str,
    k: int,
    observations: np.ndarray,
    term_name: str,
) -> dict[str, Any]:
    return {
        "expr": {
            "type": "sub",
            "name": str(term_name),
            "a": {
                "type": "get_state",
                "key": {
                    "k": int(k),
                    "owner_type": str(owner_type),
                    "owner_name": str(owner_name),
                    "dtype": DTYPE_VISION,
                    "field": str(field),
                },
                "jac": {"var": str(p_var)},
            },
            "b": {
                "type": "const",
                "var": str(p_var),
                "value": np.asarray(observations, dtype=float).reshape(-1).tolist(),
            },
        },
        "cost": {"type": "l2"},
    }


def prepare_vision_calibration_problem_dsl(
    dsl: Mapping[str, Any],
    *,
    p_var: str | None = None,
    owner_type: str | None = None,
    owner_name: str | None = None,
    field: str | None = None,
    k: int | None = None,
    term_name: str | None = None,
    observations: Sequence[float] | np.ndarray | None = None,
    standardize_terms: bool = True,
) -> PreparedVisionCalibrationDsl:
    """Normalize canonical vision calibration DSL options.

    Canonical source of defaults is top-level ``dsl["vision"]``:
      - ``p_var`` (default: ``"theta"``)
      - ``owner_type`` (default: ``"camera"``)
      - ``owner_name`` (required)
      - ``field`` (default: ``"reproj"``)
      - ``k`` (default: ``0``)
      - ``term_name`` (default: ``"camera_reproj_error"``)
      - ``observations`` (required, 1D numeric)
    """

    dsl_dict = deepcopy(mapping_as_dict(dsl, where="dsl"))
    vision_raw = dsl_dict.get("vision", None)
    if vision_raw is None:
        vision_cfg: dict[str, Any] = {}
    elif isinstance(vision_raw, Mapping):
        vision_cfg = mapping_as_dict(vision_raw, where="dsl.vision")
    else:
        raise ValueError("prepare_vision_calibration_problem_dsl: dsl['vision'] must be a mapping when provided.")

    p_var_use = _resolve_nonempty_str(
        vision_cfg.get("p_var", "theta") if p_var is None else p_var,
        where="prepare_vision_calibration_problem_dsl.p_var",
    )
    owner_type_use = _resolve_nonempty_str(
        vision_cfg.get("owner_type", "camera") if owner_type is None else owner_type,
        where="prepare_vision_calibration_problem_dsl.owner_type",
    )
    owner_name_raw = vision_cfg.get("owner_name", None) if owner_name is None else owner_name
    if owner_name_raw is None:
        raise ValueError(
            "prepare_vision_calibration_problem_dsl: owner_name is required. "
            "Set `vision.owner_name` or pass owner_name=... ."
        )
    owner_name_use = _resolve_nonempty_str(
        owner_name_raw,
        where="prepare_vision_calibration_problem_dsl.owner_name",
    )
    field_use = canonical_field_name(
        vision_cfg.get("field", "reproj") if field is None else field
    )
    k_use = _resolve_int_nonnegative(
        vision_cfg.get("k", 0) if k is None else k,
        where="prepare_vision_calibration_problem_dsl.k",
    )
    term_name_use = _resolve_nonempty_str(
        vision_cfg.get("term_name", "camera_reproj_error") if term_name is None else term_name,
        where="prepare_vision_calibration_problem_dsl.term_name",
    )
    obs_raw = vision_cfg.get("observations", None) if observations is None else observations
    if obs_raw is None:
        raise ValueError(
            "prepare_vision_calibration_problem_dsl: observations are required. "
            "Set `vision.observations` or pass observations=... ."
        )
    observations_use = _resolve_observations(
        obs_raw,
        where="prepare_vision_calibration_problem_dsl.observations",
    )

    if find_var_dsl(dsl_dict, name=p_var_use) is None:
        raise ValueError(
            "prepare_vision_calibration_problem_dsl: variable not found in DSL. "
            f"Expected variable name={p_var_use!r}."
        )

    vision_cfg_norm = dict(vision_cfg)
    vision_cfg_norm["p_var"] = p_var_use
    vision_cfg_norm["owner_type"] = owner_type_use
    vision_cfg_norm["owner_name"] = owner_name_use
    vision_cfg_norm["field"] = field_use
    vision_cfg_norm["k"] = int(k_use)
    vision_cfg_norm["term_name"] = term_name_use
    vision_cfg_norm["observations"] = observations_use.tolist()
    dsl_dict["vision"] = vision_cfg_norm

    _ = rewrite_get_state_owner_name(
        dsl_dict,
        dtype=DTYPE_VISION,
        owner_type=owner_type_use,
        owner_name=owner_name_use,
    )

    if bool(standardize_terms):
        term_dsl = _build_standardized_vision_term(
            p_var=p_var_use,
            owner_type=owner_type_use,
            owner_name=owner_name_use,
            field=field_use,
            k=k_use,
            observations=observations_use,
            term_name=term_name_use,
        )
        _upsert_term_by_name(
            dsl_dict,
            term_name=term_name_use,
            term_dsl=term_dsl,
        )

    return PreparedVisionCalibrationDsl(
        dsl=dsl_dict,
        p_var=p_var_use,
        owner_type=owner_type_use,
        owner_name=owner_name_use,
        field=field_use,
        k=int(k_use),
        term_name=term_name_use,
        observations=observations_use.copy(),
    )


__all__ = [
    "PreparedVisionCalibrationDsl",
    "prepare_vision_calibration_problem_dsl",
]
