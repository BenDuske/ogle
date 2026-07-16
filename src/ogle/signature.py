"""Dataset signatures — the lightweight fingerprint Ogle takes of a DataHub dataset.

A *signature* is the small, comparable summary Ogle persists between runs so it can
notice when a dataset feeding an ML model changed underneath a deployed model. It is
deliberately cheap: schema shape + row count + per-field null fractions. That is enough
to catch the three drifts that actually break production ML:

  * SCHEMA drift   — a feature's source column was renamed / retyped / dropped.
  * VOLUME  drift  — the upstream table stopped filling (row count collapsed) or exploded.
  * QUALITY drift  — a column that used to be populated is now mostly null.

Everything here is pure and deterministic (no DataHub client, no clock): the walker hands
us the aspects it pulled, we fold them into a `DatasetSignature`. That keeps the scoring
logic unit-testable without a live quickstart, and makes signatures reproducible so a
schema_hash computed on Halcyon matches one computed in CI.

Source aspects (when wired to live DataHub in W2):
  * `schema_fields`         <- SchemaMetadata.fields[].{fieldPath,nativeDataType}
  * `row_count`             <- DatasetProfile.rowCount
  * `field_null_fractions`  <- DatasetProfile.fieldProfiles[].{fieldPath,nullProportion}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SchemaField:
    """One column as DataHub reports it: a path and its native (platform) type."""

    path: str
    native_type: str

    def key(self) -> Tuple[str, str]:
        return (self.path, self.native_type)


@dataclass(frozen=True)
class DatasetSignature:
    """An immutable fingerprint of a dataset at one point in time.

    `row_count` and `field_null_fractions` are optional because DataHub may not have a
    profile for every dataset (profiling is opt-in). Scoring degrades gracefully: a
    dimension with no data on either side is simply not scored, never guessed.
    """

    urn: str
    schema_fields: Tuple[SchemaField, ...] = ()
    row_count: Optional[int] = None
    field_null_fractions: Dict[str, float] = field(default_factory=dict)
    # Free-form provenance (e.g. the profile timestamp). Never part of the schema hash.
    computed_at: Optional[str] = None

    @property
    def schema_hash(self) -> str:
        """Stable SHA-256 over the *set* of (path, type) pairs.

        Order-independent: DataHub does not guarantee field ordering across fetches, so
        two fetches of an unchanged schema must hash identically. Only membership and
        types matter for drift.
        """
        canonical = sorted(f.key() for f in self.schema_fields)
        blob = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def field_paths(self) -> frozenset:
        return frozenset(f.path for f in self.schema_fields)

    def to_dict(self) -> dict:
        """Serialize for persistence (Aegis memory store / JSON baseline file)."""
        return {
            "urn": self.urn,
            "schema_fields": [[f.path, f.native_type] for f in self.schema_fields],
            "row_count": self.row_count,
            "field_null_fractions": dict(self.field_null_fractions),
            "computed_at": self.computed_at,
            "schema_hash": self.schema_hash,  # denormalized for quick baseline diffing
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetSignature":
        """Inverse of `to_dict`. Ignores the denormalized `schema_hash` (recomputed)."""
        return cls(
            urn=data["urn"],
            schema_fields=tuple(
                SchemaField(path=p, native_type=t) for p, t in data.get("schema_fields", [])
            ),
            row_count=data.get("row_count"),
            field_null_fractions=dict(data.get("field_null_fractions", {})),
            computed_at=data.get("computed_at"),
        )


def build_signature(
    urn: str,
    schema_fields: Sequence[Tuple[str, str]] = (),
    row_count: Optional[int] = None,
    field_null_fractions: Optional[Dict[str, float]] = None,
    computed_at: Optional[str] = None,
) -> DatasetSignature:
    """Convenience builder from plain tuples (what a DataHub aspect walk yields).

    `schema_fields` is a sequence of (path, native_type). Duplicate paths are collapsed
    to the last occurrence — DataHub occasionally reports nested duplicates and we want a
    single truth per path so the hash and null-fraction lookups stay consistent.
    """
    deduped: Dict[str, str] = {}
    for path, native_type in schema_fields:
        deduped[path] = native_type
    fields = tuple(SchemaField(path=p, native_type=t) for p, t in deduped.items())

    nulls = dict(field_null_fractions or {})
    for path, frac in nulls.items():
        if not 0.0 <= frac <= 1.0:
            raise ValueError(
                f"null fraction for {path!r} must be in [0,1], got {frac!r}"
            )
    if row_count is not None and row_count < 0:
        raise ValueError(f"row_count must be >= 0, got {row_count!r}")

    return DatasetSignature(
        urn=urn,
        schema_fields=fields,
        row_count=row_count,
        field_null_fractions=nulls,
        computed_at=computed_at,
    )
