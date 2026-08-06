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
        """Inverse of `to_dict`. Ignores the denormalized `schema_hash` (recomputed).

        Deserialization runs the persisted payload back through `build_signature` — the SAME
        constructor the walker uses — rather than populating the dataclass field-by-field. That
        is deliberate: `build_signature` is where EVERY per-field invariant the scorers trust is
        enforced (null/unique fractions in [0,1]; means finite; stdevs finite and non-negative;
        min/max finite with min <= max; row_count >= 0; quantiles a genuine quantile function).
        A hand-edited or corrupt baseline can carry any of these violations — a NaN mean, a
        negative stdev, an inverted min>max envelope, an out-of-[0,1] fraction — and every one of
        them would SILENTLY poison its scorer: a NaN moment makes the drift delta NaN, and a NaN
        compares false against every threshold, so real drift is quietly *missed* — the one thing
        a drift detector must never do. `e64e278` closed this hole for quantiles alone via a bare
        `_clean_quantiles` call here; delegating to the builder generalizes that to the full
        invariant set instead of re-validating one field and trusting six others. A round-trip of
        any legitimately-built signature is unaffected (its values already passed these checks on
        the build path — the validators are idempotent on clean data); only a poisoned payload
        raises ValueError, which `BaselineStore.load` already catches to quarantine the file and
        re-baseline — so a corrupt store degrades LOUDLY (drift re-learned) instead of scoring
        garbage. `schema_fields` arrive as [path, type] pairs, exactly the tuple sequence the
        builder expects (its dedup is a no-op on already-deduped persisted fields).

        Each `schema_fields` element must be a genuine JSON array (a [path, type] pair),
        not just any 2-item iterable. A bare `tuple(pair)` char-splits a length-2 STRING
        (`"ab"` -> `('a','b')`) and key-splits a 2-key DICT (`{"foo":1,"bar":2}` ->
        `('foo','bar')`) — both slip through the builder's `for path, native_type in ...`
        unpack SILENTLY and forge a bogus SchemaField, poisoning the schema hash so real
        schema drift is quietly mis-scored (a 3+-char string / other-arity dict raises
        "too many/not enough values to unpack" and is already caught, so ONLY the length-2
        string/dict misreads silently). This is the same char-split silent-misread class the
        store's `muted_urns`/`dict(...)` guards close, one layer deeper. Reject a non-list/
        tuple element with a ValueError, which `BaselineStore.load` catches to quarantine the
        foreign file and re-baseline — loud degradation over a silently corrupted signature.
        """
        raw_fields = data.get("schema_fields", [])
        for pair in raw_fields:
            if not isinstance(pair, (list, tuple)):
                raise ValueError(
                    f"schema_fields entries must be [path, type] arrays, got "
                    f"{type(pair).__name__} (refusing to char-split a truncated/foreign file)"
                )
        return build_signature(
            urn=data["urn"],
            schema_fields=[tuple(pair) for pair in raw_fields],
            row_count=data.get("row_count"),
            field_null_fractions=data.get("field_null_fractions"),
            field_unique_fractions=data.get("field_unique_fractions"),
            field_means=data.get("field_means"),
            field_stdevs=data.get("field_stdevs"),
            field_mins=data.get("field_mins"),
            field_maxes=data.get("field_maxes"),
            field_quantiles=data.get("field_quantiles"),
            computed_at=data.get("computed_at"),
        )


def _require_finite_real(value: object, kind: str, path: str) -> None:
    """Reject a persisted numeric scalar that is a bool or not a finite real number.

    bool is the *silent* laundering case: ``float(True) == 1.0``, so a bare ``true`` in a
    hand-edited or foreign store slips past a plain finiteness check as a real 1.0 — exactly
    the class the store's timestamp path closed with ``_finite_epoch``. A str/None never trips
    the ``x != x`` / inf checks either, so it would pass build here and only blow up later in
    the scorer's arithmetic (a *deferred* crash on a future check run) instead of quarantining
    the file now at load. Reusing the finiteness guard's message keeps existing tests unchanged;
    only the newly-rejected bool/non-number cases are new behavior.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{kind} for {path!r} must be a finite number, got {value!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{kind} for {path!r} must be a finite number, got {value!r}")


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
    # Both halves of a schema field must be genuine strings. A persisted [path, type] pair
    # can carry a non-string (a bare number/null/nested array in a hand-edited or foreign
    # store) that the sibling numeric guards never see, and it corrupts the schema hash in
    # two distinct ways: a UNIFORM non-string type (e.g. `null`) builds silently but hashes
    # to a value that differs from the real string type — schema drift then quietly mis-scores,
    # the one thing this detector must never do — while a MIXED set (an int path beside a str
    # path) DEFERS a `TypeError` to `schema_hash`'s `sorted()`, a crash in the scorer instead
    # of a quarantine at load. Reject a non-str up front with the same ValueError contract the
    # moment/fraction guards raise so `BaselineStore.load` quarantines the foreign file and
    # re-baselines LOUDLY. (bool is not a str subclass, so `isinstance(True, str)` is False —
    # it is rejected here like any other non-string, matching the numeric guards.)
    deduped: Dict[str, str] = {}
    for path, native_type in schema_fields:
        if not isinstance(path, str) or not isinstance(native_type, str):
            raise ValueError(
                f"schema field must be (str path, str type), got "
                f"({path!r}, {native_type!r})"
            )
        deduped[path] = native_type
    fields = tuple(SchemaField(path=p, native_type=t) for p, t in deduped.items())

    # A fraction is a bool-or-non-number away from silent corruption: bool laundering
    # (`float(True) == 1.0`) slips a `true` past the [0,1] range check as a real 1.0 (a
    # "0% / 100%" reading the scorer trusts), so reject the bool type up front. A str/None
    # trips the comparison with a TypeError, which `BaselineStore.load` already catches to
    # quarantine — only the in-range bool misreads silently.
    nulls = dict(field_null_fractions or {})
    for path, frac in nulls.items():
        if isinstance(frac, bool) or not 0.0 <= frac <= 1.0:
            raise ValueError(
                f"null fraction for {path!r} must be in [0,1], got {frac!r}"
            )
    uniques = dict(field_unique_fractions or {})
    for path, frac in uniques.items():
        if isinstance(frac, bool) or not 0.0 <= frac <= 1.0:
            raise ValueError(
                f"unique fraction for {path!r} must be in [0,1], got {frac!r}"
            )
    # A mean is an unbounded real (unlike the fractions above): only reject non-finite
    # values (NaN/inf would poison the relative-shift math in the scorer) and the bool/
    # non-number laundering `_require_finite_real` guards, never a range.
    means = dict(field_means or {})
    for path, mval in means.items():
        _require_finite_real(mval, "mean", path)
    # A stdev is a non-negative, finite real (a dispersion): reject NaN/inf and bool/non-number
    # like the mean, and additionally reject a negative value — a standard deviation below zero
    # is nonsense that would poison the relative-shift math in the scorer.
    stdevs = dict(field_stdevs or {})
    for path, sval in stdevs.items():
        _require_finite_real(sval, "stdev", path)
        if sval < 0.0:
            raise ValueError(
                f"stdev for {path!r} must be >= 0 (a dispersion), got {sval!r}"
            )
    # A min/max is a signed, unbounded finite real (like the mean): reject NaN/inf and bool/
    # non-number, never a range. Additionally, where a field carries BOTH a min and a max, the
    # min may not exceed the max — an inverted envelope is nonsense that would make the baseline
    # span negative and poison the breach math in the scorer. Reject it up front.
    mins = dict(field_mins or {})
    maxes = dict(field_maxes or {})
    for label, mapping in (("min", mins), ("max", maxes)):
        for path, val in mapping.items():
            _require_finite_real(val, label, path)
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
    # A row_count is a non-negative int. A bool laundering (`True < 0` is False) would slip a
    # `true` through as a real 1-row count, so reject the bool type alongside the range — the
    # same silent-misread class the numeric maps above guard.
    if row_count is not None and (isinstance(row_count, bool) or row_count < 0):
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
            # A pair must be a genuine 2-element [level, value] array. A non-sequence or
            # wrong-arity element (a truncated `[0.25]`, a scalar, a 3-tuple) would otherwise
            # raise IndexError/TypeError from `pair[1]` — and IndexError is OUTSIDE
            # `BaselineStore.load`'s (ValueError, KeyError, TypeError) recovery net, so a
            # scheduled `ogle check` crash-loops on exactly the corrupt file the net exists to
            # quarantine. Reject with the same ValueError contract so load() quarantines and
            # re-baselines LOUDLY. (A length-2 string like "ab" is not a list/tuple, so it is
            # rejected here rather than char-splitting into ('a','b') — one layer deeper than the
            # schema_fields guard in `from_dict`.)
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    f"quantile pair for {path!r} must be a [level, value] array, got {pair!r}"
                )
            # bool laundering: `float(True) == 1.0` would forge a p=1.0 (or v=1.0) reading the
            # empirical scorer trusts — the exact silent-misread class every sibling numeric
            # guard (mean/stdev/min/max/fractions/row_count) already rejects; quantiles was the
            # last carrier still coercing a bare `float(...)`. A str/None would defer a crash into
            # the scorer instead of quarantining now. Require a genuine finite real up front, the
            # same contract (and message) `_require_finite_real` raises for the moments.
            _require_finite_real(pair[0], "quantile level", path)
            _require_finite_real(pair[1], "quantile value", path)
            p, v = float(pair[0]), float(pair[1])
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
