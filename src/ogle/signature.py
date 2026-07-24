"""Dataset signatures — the lightweight fingerprint Ogle takes of a DataHub dataset.

A *signature* is the small, comparable summary Ogle persists between runs so it can
notice when a dataset feeding an ML model changed underneath a deployed model. It is
deliberately cheap: schema shape + row count + per-field null fractions. That is enough
to catch the three drifts that actually break production ML:

  * SCHEMA drift   — a feature's source column was renamed / retyped / dropped.
  * VOLUME  drift  — the upstream table stopped filling (row count collapsed) or exploded.
  * QUALITY drift  — a column that used to be populated is now mostly null.
  * DISTRIBUTION   — a column's distinct-value fraction collapsed (a categorical feature
                     stuck on one value, or an id/key that lost uniqueness in a bad join).
  * MEAN drift     — a numeric feature's mean shifted (covariate shift): schema, volume,
                     nulls and cardinality all look fine, but the values moved under the
                     model — the classic silent feature-drift that quietly rots accuracy.
  * STDEV drift    — a numeric feature's spread (standard deviation) collapsed or exploded
                     while its mean held steady: a sensor stuck on one reading (variance
                     ->0) or gone noisy (variance blows up) — a scale shift the mean rule,
                     which only sees location, is blind to.
  * RANGE drift    — a numeric feature's observed min/max escaped its historical envelope: a
                     handful of out-of-bounds values (integer overflow, a unit bug on a
                     subset, a new outlier regime) breach the baseline [min, max] band while
                     the mean and stdev — aggregate moments a few extremes barely move — both
                     look fine. The tail signal the moment-based rules cannot see.

Everything here is pure and deterministic (no DataHub client, no clock): the walker hands
us the aspects it pulled, we fold them into a `DatasetSignature`. That keeps the scoring
logic unit-testable without a live quickstart, and makes signatures reproducible so a
schema_hash computed on Halcyon matches one computed in CI.

Source aspects (when wired to live DataHub in W2):
  * `schema_fields`          <- SchemaMetadata.fields[].{fieldPath,nativeDataType}
  * `row_count`              <- DatasetProfile.rowCount
  * `field_null_fractions`   <- DatasetProfile.fieldProfiles[].{fieldPath,nullProportion}
  * `field_unique_fractions` <- DatasetProfile.fieldProfiles[].{fieldPath,uniqueProportion}
  * `field_means`            <- DatasetProfile.fieldProfiles[].{fieldPath,mean}
  * `field_stdevs`           <- DatasetProfile.fieldProfiles[].{fieldPath,stdev}
  * `field_mins`             <- DatasetProfile.fieldProfiles[].{fieldPath,min}
  * `field_maxes`            <- DatasetProfile.fieldProfiles[].{fieldPath,max}
  * `field_quantiles`        <- DatasetProfile.fieldProfiles[].quantiles[].{quantile,value}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple


def parse_iso_epoch(text: Optional[str]) -> Optional[float]:
    """Best-effort parse of a `computed_at` provenance string into epoch seconds.

    `computed_at` is free-form (usually DataHub's profile timestamp, e.g.
    `2026-07-16T00:00:00Z`), so this degrades gracefully: anything that isn't a parseable
    ISO-8601 instant returns None and the caller treats the age as *unknown* rather than
    guessing. A trailing `Z` is normalized to `+00:00` for `fromisoformat`; a naive stamp
    (no offset) is assumed UTC so a bare date still yields a real age.

    Pure and clock-free — the single source of truth both the CLI's staleness views and the
    scorer's freshness dimension read, so a capture age and a freshness finding never disagree.
    """
    if not text:
        return None
    raw = text.strip()
    if raw[-1:] in ("Z", "z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


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
    # Per-field distinct-value fraction (uniqueCount / rowCount), from DataHub's profile.
    # Optional exactly like null fractions — profiling is opt-in and older profiles may lack
    # it. Scoring degrades gracefully: a field with no unique fraction on either side is not
    # scored for distribution drift, never guessed.
    field_unique_fractions: Dict[str, float] = field(default_factory=dict)
    # Per-field numeric mean, from DataHub's profile (`fieldProfiles[].mean`). Optional and
    # unbounded (a mean is a real number — can be negative, has no [0,1] cap unlike the
    # fractions above). Only numeric columns carry one; text/categorical fields simply have
    # no entry. Scoring degrades gracefully: a field with no mean on either side is not scored
    # for mean drift, never guessed.
    field_means: Dict[str, float] = field(default_factory=dict)
    # Per-field standard deviation, from DataHub's profile (`fieldProfiles[].stdev`). Optional
    # exactly like the mean above; only numeric columns carry one. A stdev is non-negative (a
    # dispersion, not a signed location), so unlike `field_means` it is floored at 0 — but it is
    # otherwise unbounded. Scoring degrades gracefully: a field with no stdev on either side is
    # not scored for spread drift, never guessed.
    field_stdevs: Dict[str, float] = field(default_factory=dict)
    # Per-field observed minimum / maximum, from DataHub's profile (`fieldProfiles[].min`/
    # `.max`). Optional exactly like the mean above; only numeric columns carry them. Each is a
    # signed, unbounded finite real (a min can be negative, a max huge) — floored nowhere, only
    # required finite. Together they bound the field's observed value envelope. Scoring degrades
    # gracefully: a field lacking a full min+max on both sides is not scored for range drift,
    # never guessed.
    field_mins: Dict[str, float] = field(default_factory=dict)
    field_maxes: Dict[str, float] = field(default_factory=dict)
    # Per-field empirical quantiles, from DataHub's profile (`fieldProfiles[].quantiles[]` =
    # {quantile, value}). Optional exactly like the moments above; only numeric columns DataHub
    # sampled deeply carry them. Each value is a tuple of (p, v) pairs — p a probability level in
    # [0,1], v the field value at that quantile — kept sorted by p, strictly increasing in p and
    # non-decreasing in v (a quantile function cannot run backwards). This is the ONLY per-field
    # carrier of the distribution's *shape*: mean+stdev fix a Gaussian, but two very different
    # shapes (bimodal vs unimodal, skewed vs symmetric) can share both moments. The empirical
    # distribution-distance scorers read these raw bins to page on the shape shifts a Gaussian
    # summary can't represent. Scoring degrades gracefully: a field lacking a usable (>= 2-point)
    # quantile set on either side is simply not scored empirically — the Gaussian distribution-
    # distances still fire from the moments, never guessed.
    field_quantiles: Dict[str, Tuple[Tuple[float, float], ...]] = field(default_factory=dict)
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
            "field_unique_fractions": dict(self.field_unique_fractions),
            "field_means": dict(self.field_means),
            "field_stdevs": dict(self.field_stdevs),
            "field_mins": dict(self.field_mins),
            "field_maxes": dict(self.field_maxes),
            # Quantiles serialize as a list of [p, v] pairs per field (JSON has no tuples).
            "field_quantiles": {
                path: [[p, v] for p, v in pairs]
                for path, pairs in self.field_quantiles.items()
            },
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
            field_unique_fractions=dict(data.get("field_unique_fractions", {})),
            field_means=dict(data.get("field_means", {})),
            field_stdevs=dict(data.get("field_stdevs", {})),
            field_mins=dict(data.get("field_mins", {})),
            field_maxes=dict(data.get("field_maxes", {})),
            field_quantiles={
                path: tuple((float(p), float(v)) for p, v in pairs)
                for path, pairs in data.get("field_quantiles", {}).items()
            },
            computed_at=data.get("computed_at"),
        )


def build_signature(
    urn: str,
    schema_fields: Sequence[Tuple[str, str]] = (),
    row_count: Optional[int] = None,
    field_null_fractions: Optional[Dict[str, float]] = None,
    field_unique_fractions: Optional[Dict[str, float]] = None,
    field_means: Optional[Dict[str, float]] = None,
    field_stdevs: Optional[Dict[str, float]] = None,
    field_mins: Optional[Dict[str, float]] = None,
    field_maxes: Optional[Dict[str, float]] = None,
    field_quantiles: Optional[Dict[str, Sequence[Tuple[float, float]]]] = None,
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
    uniques = dict(field_unique_fractions or {})
    for path, frac in uniques.items():
        if not 0.0 <= frac <= 1.0:
            raise ValueError(
                f"unique fraction for {path!r} must be in [0,1], got {frac!r}"
            )
    # A mean is an unbounded real (unlike the fractions above): only reject non-finite
    # values (NaN/inf would poison the relative-shift math in the scorer), never a range.
    means = dict(field_means or {})
    for path, mval in means.items():
        if mval != mval or mval in (float("inf"), float("-inf")):
            raise ValueError(
                f"mean for {path!r} must be a finite number, got {mval!r}"
            )
    # A stdev is a non-negative, finite real (a dispersion): reject NaN/inf like the mean, and
    # additionally reject a negative value — a standard deviation below zero is nonsense that
    # would poison the relative-shift math in the scorer.
    stdevs = dict(field_stdevs or {})
    for path, sval in stdevs.items():
        if sval != sval or sval in (float("inf"), float("-inf")):
            raise ValueError(
                f"stdev for {path!r} must be a finite number, got {sval!r}"
            )
        if sval < 0.0:
            raise ValueError(
                f"stdev for {path!r} must be >= 0 (a dispersion), got {sval!r}"
            )
    # A min/max is a signed, unbounded finite real (like the mean): reject only NaN/inf, never
    # a range. Additionally, where a field carries BOTH a min and a max, the min may not exceed
    # the max — an inverted envelope is nonsense that would make the baseline span negative and
    # poison the breach math in the scorer. Reject it up front rather than emit garbage.
    mins = dict(field_mins or {})
    maxes = dict(field_maxes or {})
    for label, mapping in (("min", mins), ("max", maxes)):
        for path, val in mapping.items():
            if val != val or val in (float("inf"), float("-inf")):
                raise ValueError(
                    f"{label} for {path!r} must be a finite number, got {val!r}"
                )
    for path in mins.keys() & maxes.keys():
        if mins[path] > maxes[path]:
            raise ValueError(
                f"min for {path!r} ({mins[path]!r}) must be <= max ({maxes[path]!r})"
            )
    # Quantiles describe an empirical distribution's shape, so they carry stricter structure than
    # a lone moment. A field's set must be a real quantile function: every p in [0,1], every p and
    # v finite, p strictly increasing (no duplicate levels), and v non-decreasing (a quantile
    # function cannot run backwards — Q(0.75) < Q(0.25) is nonsense that would flip the earth-mover
    # integral negative). Fewer than two points can't span a probability band, so it is dropped
    # rather than half-recorded. Reject a malformed set up front rather than emit garbage.
    quantiles = _clean_quantiles(field_quantiles)
    if row_count is not None and row_count < 0:
        raise ValueError(f"row_count must be >= 0, got {row_count!r}")

    return DatasetSignature(
        urn=urn,
        schema_fields=fields,
        row_count=row_count,
        field_null_fractions=nulls,
        field_unique_fractions=uniques,
        field_means=means,
        field_stdevs=stdevs,
        field_mins=mins,
        field_maxes=maxes,
        field_quantiles=quantiles,
        computed_at=computed_at,
    )


def _clean_quantiles(
    field_quantiles: Optional[Dict[str, Sequence[Tuple[float, float]]]],
) -> Dict[str, Tuple[Tuple[float, float], ...]]:
    """Validate and normalize per-field quantile sets into sorted (p, v) tuples.

    A quantile set must be a genuine quantile function: >= 2 points, every p in [0,1], every p
    and v finite, p strictly increasing, and v non-decreasing. A set of fewer than two points is
    dropped (it can't span a probability band); any structural violation raises rather than
    emitting a distribution that would poison the empirical earth-mover integral downstream.
    Input pairs may arrive in any order — they are sorted by p before the monotonicity checks.
    """
    out: Dict[str, Tuple[Tuple[float, float], ...]] = {}
    for path, pairs in (field_quantiles or {}).items():
        cleaned: List[Tuple[float, float]] = []
        for pair in pairs:
            p, v = float(pair[0]), float(pair[1])
            for label, val in (("quantile level", p), ("quantile value", v)):
                if val != val or val in (float("inf"), float("-inf")):
                    raise ValueError(
                        f"{label} for {path!r} must be a finite number, got {val!r}"
                    )
            if not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"quantile level for {path!r} must be in [0,1], got {p!r}"
                )
            cleaned.append((p, v))
        if len(cleaned) < 2:
            # Not enough structure to describe a distribution — degrade to "no quantiles" rather
            # than record a lone point the empirical scorer can't use.
            continue
        cleaned.sort(key=lambda pv: pv[0])
        for (p0, v0), (p1, v1) in zip(cleaned, cleaned[1:]):
            if p1 <= p0:
                raise ValueError(
                    f"quantile levels for {path!r} must be strictly increasing, "
                    f"got {p0!r} then {p1!r}"
                )
            if v1 < v0:
                raise ValueError(
                    f"quantile values for {path!r} must be non-decreasing (a quantile "
                    f"function cannot run backwards), got {v0!r} then {v1!r}"
                )
        out[path] = tuple(cleaned)
    return out
