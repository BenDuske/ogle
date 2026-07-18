"""W3 tag write-back — Ogle's outbound edit against DataHub.

Everything upstream (`signature`, `scorer`, `store`, `narrative`, `pipeline`, `walker`)
is read-only against DataHub. This module is the one seam that writes: when a drift
check produces findings, it stamps the affected datasets and their downstream
`mlModel`s with an `ogle-drift-flagged` tag so the next person or agent looking at
DataHub inherits the finding.

Two-layer design mirrors `walker`:

  * **Pure core.** `plan_writeback(findings, walk_result)` returns a `WritebackPlan` — a
    deterministic list of `(entity_urn, tag_urn)` actions. `apply(plan, backend)` merges
    each tag onto the target entity via a small `WritebackBackend` protocol that trades
    only in tag-URN sets — the pure code never touches an SDK class. Every test in
    `tests/test_writeback.py` uses a dict-backed `FakeWritebackBackend`.
  * **Live adapter.** `DataHubWritebackBackend` wraps `acryl-datahub` (imported lazily),
    reads the entity's current `GlobalTags` aspect, adds Ogle's tag if missing, and
    re-emits.

Idempotency is a feature, not a coincidence:

  * `plan_writeback` de-duplicates `(entity, tag)` pairs, so the same finding twice
    still produces one action.
  * `apply` fetches the target's *existing* tags, checks for the tag URN, and skips
    (recorded in `unchanged`) when it is already there. Ogle can run every 10 minutes
    without flapping a re-write.

The tag URN is a stable string, `urn:li:tag:ogle-drift-flagged` by default, so it's the
same tag across runs and hosts. DataHub OSS auto-creates the tag entity on first use —
no separate provisioning step required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set

from .scorer import DriftFinding
from .walker import WalkResult

# ---------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------

#: The tag Ogle stamps by default. Stable across runs so the same drift finding across
#: two ticks doesn't produce two different tag entities.
OGLE_DRIFT_TAG = "urn:li:tag:ogle-drift-flagged"

#: Prefix for the optional per-severity tag (`--write-back-severity`). Stamped ALONGSIDE
#: the flat tag so a DataHub operator can filter to `ogle-drift-high` in the UI without
#: losing the coarse `ogle-drift-flagged` grouping. Stable string -> one tag entity per
#: severity across runs and hosts.
OGLE_SEVERITY_TAG_PREFIX = "urn:li:tag:ogle-drift-"


def severity_tag_urn(severity: Any) -> str:
    """`urn:li:tag:ogle-drift-<severity>` for a `Severity` (or its string value).

    Accepts the enum or a raw `"high"`/`"medium"`/`"low"` so callers don't have to
    import `Severity`. An empty/unknown severity yields `...ogle-drift-unknown` rather
    than raising — a write-back should degrade to a still-useful tag, never crash a
    scheduler tick.
    """
    value = getattr(severity, "value", severity)
    value = str(value).strip().lower() or "unknown"
    return f"{OGLE_SEVERITY_TAG_PREFIX}{value}"


# ---------------------------------------------------------------------------------------
# Pure plan + apply
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TagAction:
    """One (entity, tag) pair the write-back wants to apply.

    `reason` is a short human string kept for logs and dry-run output — never emitted
    to DataHub itself (DataHub tag associations carry structured `context`, not free
    text, so we keep the reason local).
    """

    entity_urn: str
    tag_urn: str
    reason: str = ""


@dataclass(frozen=True)
class WritebackPlan:
    """Deterministic set of tag actions the plan wants applied. Empty is a valid plan."""

    actions: List[TagAction] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def to_dict(self) -> dict:
        return {
            "actions": [
                {"entity_urn": a.entity_urn, "tag_urn": a.tag_urn, "reason": a.reason}
                for a in self.actions
            ]
        }


@dataclass(frozen=True)
class WritebackResult:
    """Outcome of applying a plan: which actions actually wrote, which were already done."""

    applied: List[TagAction] = field(default_factory=list)
    unchanged: List[TagAction] = field(default_factory=list)
    # Non-fatal failures per action — apply() reports and moves on so one broken URN
    # doesn't strand a batch.
    failed: List[TagAction] = field(default_factory=list)

    @property
    def tagged_entities(self) -> List[str]:
        """Deduped entity URNs the write actually changed (for the CLI report)."""
        seen: Set[str] = set()
        out: List[str] = []
        for a in self.applied:
            if a.entity_urn in seen:
                continue
            seen.add(a.entity_urn)
            out.append(a.entity_urn)
        return out

    def to_dict(self) -> dict:
        def _dump(actions):
            return [
                {"entity_urn": a.entity_urn, "tag_urn": a.tag_urn, "reason": a.reason}
                for a in actions
            ]

        return {
            "applied": _dump(self.applied),
            "unchanged": _dump(self.unchanged),
            "failed": _dump(self.failed),
            "tagged_entities": list(self.tagged_entities),
        }


def plan_writeback(
    findings: Sequence[DriftFinding],
    walk_result: Optional[WalkResult] = None,
    tag_urn: str = OGLE_DRIFT_TAG,
    severity_tags: bool = False,
) -> WritebackPlan:
    """Decide which entities to tag from a batch of drift findings.

    For every drifted dataset URN, tag:
      1. the dataset itself (so a DataHub browser sees it flagged directly), and
      2. every mlModel that consumes it (via `walk_result.dataset_to_models`) — that is
         the "production-affecting" story the demo is about.

    Duplicates across findings collapse to a single action per (entity, tag) pair. If
    `walk_result` is None (or doesn't cover a URN), only the dataset itself is tagged.

    When ``severity_tags`` is set, each entity ALSO receives a per-severity tag
    (`urn:li:tag:ogle-drift-high`, etc.) alongside the flat tag, so an operator can
    filter DataHub to the worst drift. A dataset's severity tag is the WORST severity
    among its own findings; a model's is the worst among the drifted datasets feeding
    it — the finding that would page you is the one that should colour the model.
    """
    if not findings:
        return WritebackPlan()

    dataset_to_models: Dict[str, List[str]] = (
        walk_result.dataset_to_models if walk_result is not None else {}
    )

    actions: List[TagAction] = []
    seen: Set[str] = set()  # (entity_urn, tag_urn) — collapse duplicates.

    def _push(entity_urn: str, tag: str, reason: str) -> None:
        key = f"{entity_urn}\x1e{tag}"
        if key in seen:
            return
        seen.add(key)
        actions.append(TagAction(entity_urn=entity_urn, tag_urn=tag, reason=reason))

    # Worst severity per drifted dataset (a dataset can carry several findings — the one
    # that pages is the max). Used both for the dataset's own severity tag and to colour
    # its downstream models.
    worst_by_ds: Dict[str, DriftFinding] = {}
    for f in findings:
        cur = worst_by_ds.get(f.urn)
        if cur is None or f.severity.rank > cur.severity.rank:
            worst_by_ds[f.urn] = f

    # Emit dataset actions in findings order, then model actions after — deterministic
    # ordering makes the CLI report + tests stable.
    for f in findings:
        _push(f.urn, tag_urn, reason=f"drift: {f.kind.value} {f.severity.value}")
    if severity_tags:
        # One severity tag per dataset (its worst), emitted after the flat pass so the
        # flat tag always leads for a given entity.
        for f in findings:
            worst = worst_by_ds[f.urn]
            _push(
                f.urn,
                severity_tag_urn(worst.severity),
                reason=f"severity: {worst.severity.value}",
            )

    # De-dup drifted dataset URNs while preserving first-seen order.
    seen_ds: Set[str] = set()
    ordered_datasets: List[str] = []
    for f in findings:
        if f.urn in seen_ds:
            continue
        seen_ds.add(f.urn)
        ordered_datasets.append(f.urn)

    for ds_urn in ordered_datasets:
        for model_urn in dataset_to_models.get(ds_urn, ()):
            _push(model_urn, tag_urn, reason=f"downstream of drifted {ds_urn}")

    if severity_tags:
        # A model inherits the worst severity across every drifted dataset feeding it.
        model_worst: Dict[str, DriftFinding] = {}
        for ds_urn in ordered_datasets:
            worst = worst_by_ds[ds_urn]
            for model_urn in dataset_to_models.get(ds_urn, ()):
                cur = model_worst.get(model_urn)
                if cur is None or worst.severity.rank > cur.severity.rank:
                    model_worst[model_urn] = worst
        for ds_urn in ordered_datasets:
            for model_urn in dataset_to_models.get(ds_urn, ()):
                worst = model_worst[model_urn]
                _push(
                    model_urn,
                    severity_tag_urn(worst.severity),
                    reason=f"downstream severity: {worst.severity.value}",
                )

    return WritebackPlan(actions=actions)


class WritebackBackend(Protocol):
    """The only two operations the pure `apply` needs from the outside world.

    Kept URN-only so `apply` never handles an SDK aspect class — the live adapter
    translates.
    """

    def existing_tag_urns(self, entity_urn: str) -> Set[str]:
        """Set of tag URNs currently on `entity_urn`. Missing entity/aspect -> empty set."""

    def set_tag_urns(self, entity_urn: str, tag_urns: Iterable[str]) -> None:
        """Replace the entity's `GlobalTags` with these URNs. Idempotent from apply's side —
        caller guarantees the set is a superset of the previous one for additive semantics."""


def apply(plan: WritebackPlan, backend: WritebackBackend) -> WritebackResult:
    """Merge each planned tag onto its entity via `backend`.

    Reads the entity's current tag URNs, adds the plan's tag if missing, and writes back
    the union. A tag already present is recorded in `unchanged`, not re-written — the
    scheduled loop can call this every tick without flapping. A backend exception on one
    action is caught and recorded in `failed`; the batch continues.
    """
    applied: List[TagAction] = []
    unchanged: List[TagAction] = []
    failed: List[TagAction] = []

    # Group actions by entity so we merge every tag for a given entity in a single write —
    # avoids racing against ourselves when the plan tags one entity with multiple URNs.
    by_entity: Dict[str, List[TagAction]] = {}
    order: List[str] = []
    for action in plan.actions:
        if action.entity_urn not in by_entity:
            order.append(action.entity_urn)
        by_entity.setdefault(action.entity_urn, []).append(action)

    for entity_urn in order:
        actions = by_entity[entity_urn]
        try:
            current = backend.existing_tag_urns(entity_urn)
        except Exception:
            # Read failure is fatal for THIS entity — never overwrite tags we couldn't read.
            failed.extend(actions)
            continue

        wanted = {a.tag_urn for a in actions}
        missing = wanted - current
        if not missing:
            unchanged.extend(actions)
            continue

        try:
            backend.set_tag_urns(entity_urn, current | wanted)
        except Exception:
            failed.extend(actions)
            continue

        # Split applied vs. unchanged per action: a single write covers all of them, but
        # unchanged tags are still "already present", not "just applied".
        for a in actions:
            (applied if a.tag_urn in missing else unchanged).append(a)

    return WritebackResult(applied=applied, unchanged=unchanged, failed=failed)


# ---------------------------------------------------------------------------------------
# Live adapter — thin `DataHubGraph` wrapper. Imported lazily.
# ---------------------------------------------------------------------------------------


class DataHubWritebackBackend:
    """`WritebackBackend` backed by `acryl-datahub`.

    The SDK is imported inside `__init__` so `ogle.writeback` stays importable on a
    machine without `acryl-datahub` — same rule as `ogle.walker.DataHubBackend`.
    """

    def __init__(self, graph: Optional[Any] = None, gms_server: str = "http://localhost:8080"):
        if graph is None:
            from datahub.ingestion.graph.client import DataHubGraph, DataHubGraphConfig

            graph = DataHubGraph(DataHubGraphConfig(server=gms_server))
        from datahub.emitter.mcp import MetadataChangeProposalWrapper
        from datahub.metadata.schema_classes import (
            GlobalTagsClass,
            TagAssociationClass,
        )

        self._graph = graph
        self._MCP = MetadataChangeProposalWrapper
        self._GlobalTagsClass = GlobalTagsClass
        self._TagAssociationClass = TagAssociationClass

    def existing_tag_urns(self, entity_urn: str) -> Set[str]:
        aspect = self._graph.get_aspect(entity_urn=entity_urn, aspect_type=self._GlobalTagsClass)
        if aspect is None:
            return set()
        return {t.tag for t in (aspect.tags or []) if getattr(t, "tag", None)}

    def set_tag_urns(self, entity_urn: str, tag_urns: Iterable[str]) -> None:
        associations = [self._TagAssociationClass(tag=str(u)) for u in sorted(set(tag_urns))]
        aspect = self._GlobalTagsClass(tags=associations)
        mcp = self._MCP(entityUrn=entity_urn, aspect=aspect)
        self._graph.emit(mcp)
