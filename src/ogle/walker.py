"""DataHub walker — turns a live DataHub graph into signatures Ogle can score.

The rest of Ogle (`signature`, `scorer`, `store`, `pipeline`, `narrative`) is deliberately
pure: it takes `DatasetSignature`s as input and never touches a network. This module is
the one seam that does — it walks DataHub's graph from a deployed model out to its upstream
datasets and folds the aspects it fetches into signatures.

The walk mirrors the wiring Task #2 built:

    mlModelDeployment (IN_SERVICE?)   <- serving flag comes from here
             |
        mlModel  (`MLModelProperties.deployments`, `.mlFeatures`)
             |
        mlFeature  (`MLFeatureProperties.sources`)
             |
        dataset   (`SchemaMetadata`, `DatasetProfile`)  -> signature

Two-layer design so tests never touch a live server:

  * **Pure core** — `build_signature_from_aspects` and the traversal functions take a
    `WalkerBackend` (a small protocol). Signatures are computed from duck-typed aspects.
    Every test in `tests/test_walker.py` uses an in-memory `FakeBackend` — no `datahub`
    import needed at test time.
  * **Live adapter** — `DataHubBackend` wraps `acryl-datahub`'s `DataHubGraph` and is
    imported lazily so the SDK stays an *optional* dependency of Ogle. Users who only
    consume the pure pipeline (feed their own signatures) never pay for the install.

Missing-aspect behavior mirrors `scorer.score_dataset` — never guess. A dataset with
neither schema nor profile yields no signature (skipped from the walk); a feature with no
sources contributes nothing; a model with no properties is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

from .signature import DatasetSignature, SchemaField, build_signature

# ---------------------------------------------------------------------------------------
# The backend protocol — the entire seam to live DataHub.
# ---------------------------------------------------------------------------------------


class WalkerBackend(Protocol):
    """The five aspect fetches the walker needs. Anything that answers these can drive it."""

    def get_model_props(self, urn: str) -> Optional[Any]:
        """Return an `MLModelPropertiesClass`-shaped object, or None if absent."""

    def get_feature_props(self, urn: str) -> Optional[Any]:
        """Return an `MLFeaturePropertiesClass`-shaped object, or None if absent."""

    def get_deployment_props(self, urn: str) -> Optional[Any]:
        """Return an `MLModelDeploymentPropertiesClass`-shaped object, or None if absent."""

    def get_schema_metadata(self, urn: str) -> Optional[Any]:
        """Return a `SchemaMetadataClass`-shaped object, or None if the dataset has no schema."""

    def get_dataset_profile(self, urn: str) -> Optional[Any]:
        """Return the latest `DatasetProfileClass`-shaped object, or None if unprofiled."""


# The exact string DataHub uses for a running deployment (matches DeploymentStatusClass.IN_SERVICE).
IN_SERVICE = "IN_SERVICE"


# ---------------------------------------------------------------------------------------
# Pure fingerprint builder — no SDK import needed, drives every unit test.
# ---------------------------------------------------------------------------------------


def build_signature_from_aspects(
    urn: str,
    schema_metadata: Optional[Any] = None,
    dataset_profile: Optional[Any] = None,
    computed_at: Optional[str] = None,
) -> Optional[DatasetSignature]:
    """Fold DataHub aspects into a `DatasetSignature`.

    Both aspects are optional: DataHub does not guarantee every dataset carries a
    `SchemaMetadata` or a `DatasetProfile`. Returns None only if BOTH are absent — a
    signature with nothing to compare is worse than no signature (the scorer skips
    None-vs-anything anyway).

    The aspects are duck-typed:
      * `schema_metadata.fields[]` -> each has `.fieldPath` and `.nativeDataType`
      * `dataset_profile.rowCount` (Optional[int])
      * `dataset_profile.fieldProfiles[]` -> each has `.fieldPath` and `.nullProportion`

    Missing sub-fields inside the aspect are treated as missing data (skipped), not zero.
    """
    if schema_metadata is None and dataset_profile is None:
        return None

    schema_fields: List[Tuple[str, str]] = []
    if schema_metadata is not None:
        for f in getattr(schema_metadata, "fields", None) or ():
            path = getattr(f, "fieldPath", None)
            native = getattr(f, "nativeDataType", None)
            if path is None or native is None:
                continue
            schema_fields.append((str(path), str(native)))

    row_count: Optional[int] = None
    null_fractions: Dict[str, float] = {}
    if dataset_profile is not None:
        rc = getattr(dataset_profile, "rowCount", None)
        if isinstance(rc, int) and rc >= 0:
            row_count = rc
        for fp in getattr(dataset_profile, "fieldProfiles", None) or ():
            path = getattr(fp, "fieldPath", None)
            frac = getattr(fp, "nullProportion", None)
            if path is None or frac is None:
                continue
            try:
                frac_f = float(frac)
            except (TypeError, ValueError):
                continue
            if 0.0 <= frac_f <= 1.0:
                null_fractions[str(path)] = frac_f

    return build_signature(
        urn=urn,
        schema_fields=schema_fields,
        row_count=row_count,
        field_null_fractions=null_fractions,
        computed_at=computed_at,
    )


# ---------------------------------------------------------------------------------------
# Traversal — deployment/model -> features -> datasets, all through the backend.
# ---------------------------------------------------------------------------------------


def is_model_serving(backend: WalkerBackend, model_urn: str) -> bool:
    """True iff any of the model's deployments is `IN_SERVICE`.

    Uses `MLModelProperties.deployments` (list of deployment URNs) then
    `MLModelDeploymentProperties.status` for each. Missing aspects -> not serving.
    """
    model_props = backend.get_model_props(model_urn)
    if model_props is None:
        return False
    for dep_urn in getattr(model_props, "deployments", None) or ():
        dep_props = backend.get_deployment_props(str(dep_urn))
        if dep_props is None:
            continue
        status = getattr(dep_props, "status", None)
        if status == IN_SERVICE:
            return True
    return False


def dataset_urns_for_model(backend: WalkerBackend, model_urn: str) -> List[str]:
    """Every upstream dataset URN feeding a given `mlModel`.

    Traversal: model -> `mlFeatures` -> each feature's `sources`. Order is preserved
    (first-seen wins), duplicates collapsed. A model with no props / no features returns [].
    """
    model_props = backend.get_model_props(model_urn)
    if model_props is None:
        return []

    ordered: List[str] = []
    seen: Set[str] = set()
    for feature_urn in getattr(model_props, "mlFeatures", None) or ():
        feature_props = backend.get_feature_props(str(feature_urn))
        if feature_props is None:
            continue
        for source_urn in getattr(feature_props, "sources", None) or ():
            s = str(source_urn)
            if s in seen:
                continue
            seen.add(s)
            ordered.append(s)
    return ordered


# ---------------------------------------------------------------------------------------
# Walk result — the shape `pipeline.run_drift_check` consumes.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkResult:
    """What one (or several) walks produce, ready to feed the drift-check pipeline."""

    signatures: List[DatasetSignature] = field(default_factory=list)
    serving_dataset_urns: FrozenSet[str] = frozenset()
    # Diagnostics — populated for debugging live walks; ignored by the pipeline.
    skipped_urns: List[str] = field(default_factory=list)  # dataset URNs with no aspects
    walked_models: List[str] = field(default_factory=list)

    def merge(self, other: "WalkResult") -> "WalkResult":
        """Union two walks, deduping datasets by URN (first-seen signature wins)."""
        by_urn: Dict[str, DatasetSignature] = {s.urn: s for s in self.signatures}
        for sig in other.signatures:
            by_urn.setdefault(sig.urn, sig)
        return WalkResult(
            signatures=list(by_urn.values()),
            serving_dataset_urns=self.serving_dataset_urns | other.serving_dataset_urns,
            skipped_urns=sorted(set(self.skipped_urns) | set(other.skipped_urns)),
            walked_models=sorted(set(self.walked_models) | set(other.walked_models)),
        )


def walk_model(
    backend: WalkerBackend,
    model_urn: str,
    computed_at: Optional[str] = None,
) -> WalkResult:
    """Fingerprint every dataset feeding one `mlModel`.

    The returned `serving_dataset_urns` is either the full dataset set (if the model has
    at least one `IN_SERVICE` deployment) or empty. `pipeline.run_drift_check` uses it to
    escalate the severity of any finding on those datasets.
    """
    dataset_urns = dataset_urns_for_model(backend, model_urn)
    serving = is_model_serving(backend, model_urn)

    signatures: List[DatasetSignature] = []
    skipped: List[str] = []
    for urn in dataset_urns:
        schema_md = backend.get_schema_metadata(urn)
        profile = backend.get_dataset_profile(urn)
        sig = build_signature_from_aspects(urn, schema_md, profile, computed_at=computed_at)
        if sig is None:
            skipped.append(urn)
            continue
        signatures.append(sig)

    return WalkResult(
        signatures=signatures,
        serving_dataset_urns=frozenset(dataset_urns) if serving else frozenset(),
        skipped_urns=skipped,
        walked_models=[model_urn],
    )


def walk_models(
    backend: WalkerBackend,
    model_urns: Iterable[str],
    computed_at: Optional[str] = None,
) -> WalkResult:
    """Walk multiple models and union the result.

    Datasets appearing under more than one model are deduped (first-seen signature wins).
    A dataset that feeds ANY serving model is serving in the union — the severest signal
    wins, matching the pipeline's escalation intent.
    """
    result = WalkResult()
    for urn in model_urns:
        result = result.merge(walk_model(backend, urn, computed_at=computed_at))
    return result


# ---------------------------------------------------------------------------------------
# Live adapter — thin `DataHubGraph` wrapper. Imported lazily to keep the SDK optional.
# ---------------------------------------------------------------------------------------


class DataHubBackend:
    """`WalkerBackend` backed by `acryl-datahub`'s `DataHubGraph`.

    The `datahub` package is imported *inside* the constructor so `ogle.walker` remains
    importable on a machine without the SDK — pure-core users (feed your own signatures)
    do not need to install it.
    """

    def __init__(self, graph: Optional[Any] = None, gms_server: str = "http://localhost:8080"):
        if graph is None:
            from datahub.ingestion.graph.client import DataHubGraph, DataHubGraphConfig

            graph = DataHubGraph(DataHubGraphConfig(server=gms_server))
        # Import the aspect classes here (not at module load) so `import ogle.walker`
        # does not require `datahub` installed.
        from datahub.metadata.schema_classes import (
            DatasetProfileClass,
            MLFeaturePropertiesClass,
            MLModelDeploymentPropertiesClass,
            MLModelPropertiesClass,
            SchemaMetadataClass,
        )

        self._graph = graph
        self._MLModelProps = MLModelPropertiesClass
        self._MLFeatureProps = MLFeaturePropertiesClass
        self._MLDeploymentProps = MLModelDeploymentPropertiesClass
        self._SchemaMetadata = SchemaMetadataClass
        self._DatasetProfile = DatasetProfileClass

    def get_model_props(self, urn: str) -> Optional[Any]:
        return self._graph.get_aspect(entity_urn=urn, aspect_type=self._MLModelProps)

    def get_feature_props(self, urn: str) -> Optional[Any]:
        return self._graph.get_aspect(entity_urn=urn, aspect_type=self._MLFeatureProps)

    def get_deployment_props(self, urn: str) -> Optional[Any]:
        return self._graph.get_aspect(entity_urn=urn, aspect_type=self._MLDeploymentProps)

    def get_schema_metadata(self, urn: str) -> Optional[Any]:
        return self._graph.get_aspect(entity_urn=urn, aspect_type=self._SchemaMetadata)

    def get_dataset_profile(self, urn: str) -> Optional[Any]:
        # DatasetProfile is a TIMESERIES aspect — the SDK refuses `get_aspect` for these
        # and points at `get_latest_timeseries_value`. We take the latest snapshot.
        # `filter_criteria_map={}` (empty dict) means no additional filter — take the
        # latest snapshot as-is. The SDK crashes on None.
        return self._graph.get_latest_timeseries_value(
            entity_urn=urn,
            aspect_type=self._DatasetProfile,
            filter_criteria_map={},
        )

    def discover_deployed_models(self) -> List[str]:
        """Return every `mlModel` URN whose props declare at least one `IN_SERVICE` deployment.

        Uses the SDK's search over MLMODEL to enumerate candidates, then checks each. Only
        needed when a caller wants Ogle to auto-select which models to walk; a scheduled job
        can just pin an explicit list.
        """
        urns: List[str] = list(self._graph.get_urns_by_filter(entity_types=["mlModel"]))
        return [u for u in urns if is_model_serving(self, u)]
