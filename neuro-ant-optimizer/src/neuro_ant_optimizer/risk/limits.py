"""Risk limit configuration and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class LimitBreach:
    """Structured representation of a risk limit breach."""

    limit_type: str
    identifier: str
    amount: float
    limit: float
    side: str
    phase: str

    def reason(self) -> str:
        """Human-readable reason token for diagnostics."""

        return f"{self.phase.upper()}:{self.limit_type}:{self.identifier}:{self.side}"

    def to_dict(self) -> Dict[str, object]:
        """Serialize the breach into a JSON-friendly mapping."""

        return {
            "type": self.limit_type,
            "id": self.identifier,
            "amount": float(self.amount),
            "limit": float(self.limit),
            "side": self.side,
            "phase": self.phase,
            "reason": self.reason(),
        }


@dataclass(frozen=True)
class _Bound:
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    @staticmethod
    def from_mapping(mapping: Mapping[str, object]) -> "_Bound":
        minimum = mapping.get("min")
        maximum = mapping.get("max")
        min_val = float(minimum) if minimum is not None else None
        max_val = float(maximum) if maximum is not None else None
        return _Bound(min_val, max_val)


@dataclass(frozen=True)
class _GroupLimit:
    name: str
    members: Tuple[str, ...]
    bound: _Bound


@dataclass(frozen=True)
class _ExposureLimit:
    name: str
    loadings: Mapping[str, float]
    bound: _Bound


@dataclass(frozen=True)
class _LeverageLimit:
    gross: Optional[float] = None
    net: Optional[float] = None
    long: Optional[float] = None
    short: Optional[float] = None
    notional: Optional[float] = None

    @staticmethod
    def from_mapping(mapping: Mapping[str, object]) -> "_LeverageLimit":
        gross = mapping.get("gross")
        net = mapping.get("net")
        long_cap = mapping.get("long") or mapping.get("long_max")
        short_cap = mapping.get("short") or mapping.get("short_max")
        notional = mapping.get("notional") or mapping.get("max")
        return _LeverageLimit(
            gross=float(gross) if gross is not None else None,
            net=float(net) if net is not None else None,
            long=float(long_cap) if long_cap is not None else None,
            short=float(short_cap) if short_cap is not None else None,
            notional=float(notional) if notional is not None else None,
        )


@dataclass(frozen=True)
class RiskLimitSet:
    """Container for normalized risk limit metadata."""

    per_asset: Mapping[str, _Bound] = field(default_factory=dict)
    sectors: Tuple[_GroupLimit, ...] = ()
    groups: Tuple[_GroupLimit, ...] = ()
    exposures: Tuple[_ExposureLimit, ...] = ()
    leverage: Optional[_LeverageLimit] = None

    manifest: Mapping[str, object] = field(default_factory=dict)


class LimitEvaluator:
    """Evaluate portfolio weights against configured risk limits."""

    def __init__(
        self,
        assets: Sequence[str],
        limit_set: RiskLimitSet,
    ) -> None:
        self._assets = list(assets)
        self._index = {name: idx for idx, name in enumerate(self._assets)}
        self._limit_set = limit_set
        self._sector_indices = [
            (limit.name, tuple(self._index[name] for name in limit.members if name in self._index), limit.bound)
            for limit in limit_set.sectors
        ]
        self._group_indices = [
            (limit.name, tuple(self._index[name] for name in limit.members if name in self._index), limit.bound)
            for limit in limit_set.groups
        ]
        self._exposure_specs = [
            (
                limit.name,
                {name: float(weight) for name, weight in limit.loadings.items() if name in self._index},
                limit.bound,
            )
            for limit in limit_set.exposures
        ]

    @property
    def manifest(self) -> Mapping[str, object]:
        return self._limit_set.manifest

    def _coerce_weights(self, weights: Sequence[float]) -> np.ndarray:
        arr = np.asarray(weights, dtype=float)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.shape[0] != len(self._assets):
            raise ValueError("Weight vector dimension mismatch for limit evaluation")
        return arr

    def evaluate(self, weights: Sequence[float], *, phase: str) -> List[LimitBreach]:
        arr = self._coerce_weights(weights)
        breaches: List[LimitBreach] = []
        tol = _TOLERANCE

        # Per-asset bounds
        for name, bound in self._limit_set.per_asset.items():
            idx = self._index.get(name)
            if idx is None:
                continue
            value = float(arr[idx])
            if bound.maximum is not None and value > bound.maximum + tol:
                breaches.append(
                    LimitBreach("ASSET", name, value, float(bound.maximum), "MAX", phase)
                )
            if bound.minimum is not None and value < bound.minimum - tol:
                breaches.append(
                    LimitBreach("ASSET", name, value, float(bound.minimum), "MIN", phase)
                )

        # Sector bounds
        for sector, indices, bound in self._sector_indices:
            if not indices:
                continue
            exposure = float(arr[list(indices)].sum())
            if bound.maximum is not None and exposure > bound.maximum + tol:
                breaches.append(
                    LimitBreach("SECTOR", sector, exposure, float(bound.maximum), "MAX", phase)
                )
            if bound.minimum is not None and exposure < bound.minimum - tol:
                breaches.append(
                    LimitBreach("SECTOR", sector, exposure, float(bound.minimum), "MIN", phase)
                )

        # Group bounds
        for group, indices, bound in self._group_indices:
            if not indices:
                continue
            exposure = float(arr[list(indices)].sum())
            if bound.maximum is not None and exposure > bound.maximum + tol:
                breaches.append(
                    LimitBreach("GROUP", group, exposure, float(bound.maximum), "MAX", phase)
                )
            if bound.minimum is not None and exposure < bound.minimum - tol:
                breaches.append(
                    LimitBreach("GROUP", group, exposure, float(bound.minimum), "MIN", phase)
                )

        # Exposure ladders
        for name, loadings, bound in self._exposure_specs:
            if not loadings:
                continue
            value = 0.0
            for asset, loading in loadings.items():
                idx = self._index.get(asset)
                if idx is None:
                    continue
                value += float(arr[idx]) * loading
            if bound.maximum is not None and value > bound.maximum + tol:
                breaches.append(
                    LimitBreach("EXPOSURE", name, value, float(bound.maximum), "MAX", phase)
                )
            if bound.minimum is not None and value < bound.minimum - tol:
                breaches.append(
                    LimitBreach("EXPOSURE", name, value, float(bound.minimum), "MIN", phase)
                )

        # Leverage / notional caps
        leverage = self._limit_set.leverage
        if leverage is not None:
            long_exposure = float(np.clip(arr, 0.0, None).sum())
            short_exposure = float(np.clip(-arr, 0.0, None).sum())
            gross = long_exposure + short_exposure
            net = float(abs(arr.sum()))
            if leverage.gross is not None and gross > leverage.gross + tol:
                breaches.append(
                    LimitBreach("LEVERAGE", "gross", gross, float(leverage.gross), "MAX", phase)
                )
            if leverage.net is not None and net > leverage.net + tol:
                breaches.append(
                    LimitBreach("LEVERAGE", "net", net, float(leverage.net), "MAX", phase)
                )
            if leverage.long is not None and long_exposure > leverage.long + tol:
                breaches.append(
                    LimitBreach("LEVERAGE", "long", long_exposure, float(leverage.long), "MAX", phase)
                )
            if leverage.short is not None and short_exposure > leverage.short + tol:
                breaches.append(
                    LimitBreach("LEVERAGE", "short", short_exposure, float(leverage.short), "MAX", phase)
                )
            if leverage.notional is not None and gross > leverage.notional + tol:
                breaches.append(
                    LimitBreach("NOTIONAL", "gross", gross, float(leverage.notional), "MAX", phase)
                )

        return breaches

    @classmethod
    def from_spec(
        cls,
        spec: Mapping[str, object],
        *,
        assets: Sequence[str],
        sector_lookup: Optional[Mapping[str, str]] = None,
    ) -> "LimitEvaluator":
        per_asset: Dict[str, _Bound] = {}
        sectors: List[_GroupLimit] = []
        groups: List[_GroupLimit] = []
        exposures: List[_ExposureLimit] = []
        manifest: Dict[str, object] = {}

        if sector_lookup is None:
            sector_lookup = {}
        normalized_sector_lookup = {str(asset): str(label) for asset, label in sector_lookup.items()}

        per_asset_raw = spec.get("per_asset") if isinstance(spec, Mapping) else None
        if isinstance(per_asset_raw, Mapping):
            per_asset = {
                str(asset): _Bound.from_mapping(_ensure_mapping(bounds))
                for asset, bounds in per_asset_raw.items()
            }
        manifest["per_asset"] = {
            key: {"min": val.minimum, "max": val.maximum}
            for key, val in per_asset.items()
        }

        def _group_from_entry(name: str, entry: Mapping[str, object]) -> Optional[_GroupLimit]:
            members_raw = entry.get("members")
            members: Tuple[str, ...]
            if members_raw is None:
                if normalized_sector_lookup:
                    members = tuple(
                        asset
                        for asset, label in normalized_sector_lookup.items()
                        if str(label) == name
                    )
                else:
                    members = tuple()
            else:
                if isinstance(members_raw, (str, bytes)):
                    members = (str(members_raw),)
                elif isinstance(members_raw, Sequence):
                    members = tuple(str(asset) for asset in members_raw)
                else:
                    raise TypeError(
                        f"Group '{name}' members must be a string or sequence"
                    )
            bound = _Bound.from_mapping(entry)
            return _GroupLimit(name, members, bound)

        sector_raw = spec.get("sectors") if isinstance(spec, Mapping) else None
        if isinstance(sector_raw, Mapping):
            for name, entry in sector_raw.items():
                if not isinstance(entry, Mapping):
                    raise TypeError("Sector limits must be mappings with bounds")
                sectors.append(_group_from_entry(str(name), entry))
        sectors = [limit for limit in sectors if limit is not None]
        manifest["sectors"] = {
            limit.name: {"members": list(limit.members), "min": limit.bound.minimum, "max": limit.bound.maximum}
            for limit in sectors
        }

        groups_raw = spec.get("groups") if isinstance(spec, Mapping) else None
        if isinstance(groups_raw, Mapping):
            for name, entry in groups_raw.items():
                if not isinstance(entry, Mapping):
                    raise TypeError("Group limits must be mappings with bounds")
                groups.append(_group_from_entry(str(name), entry))
        groups = [limit for limit in groups if limit is not None]
        manifest["groups"] = {
            limit.name: {"members": list(limit.members), "min": limit.bound.minimum, "max": limit.bound.maximum}
            for limit in groups
        }

        exposures_raw = spec.get("exposures") if isinstance(spec, Mapping) else None
        if isinstance(exposures_raw, Mapping):
            for name, entry in exposures_raw.items():
                if not isinstance(entry, Mapping):
                    raise TypeError("Exposure limits must be mappings with loadings and bounds")
                loadings_raw = entry.get("loadings") or entry.get("weights")
                if loadings_raw is None:
                    raise ValueError(f"Exposure '{name}' missing loadings specification")
                if not isinstance(loadings_raw, Mapping):
                    raise TypeError("Exposure loadings must be provided as a mapping of {asset: value}")
                loadings = {str(asset): float(value) for asset, value in loadings_raw.items()}
                bound = _Bound.from_mapping(entry)
                exposures.append(_ExposureLimit(str(name), loadings, bound))
        manifest["exposures"] = {
            limit.name: {
                "loadings": dict(limit.loadings),
                "min": limit.bound.minimum,
                "max": limit.bound.maximum,
            }
            for limit in exposures
        }

        leverage_raw = spec.get("leverage") if isinstance(spec, Mapping) else None
        leverage_limit: Optional[_LeverageLimit] = None
        if isinstance(leverage_raw, Mapping):
            leverage_limit = _LeverageLimit.from_mapping(leverage_raw)
        manifest["leverage"] = {
            key: value
            for key, value in {
                "gross": getattr(leverage_limit, "gross", None),
                "net": getattr(leverage_limit, "net", None),
                "long": getattr(leverage_limit, "long", None),
                "short": getattr(leverage_limit, "short", None),
                "notional": getattr(leverage_limit, "notional", None),
            }.items()
        }

        limit_set = RiskLimitSet(
            per_asset=per_asset,
            sectors=tuple(sectors),
            groups=tuple(groups),
            exposures=tuple(exposures),
            leverage=leverage_limit,
            manifest=manifest,
        )
        return cls(list(assets), limit_set)


def _ensure_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value  # type: ignore[return-value]
    raise TypeError("Limit specification entries must be mappings")
