"""Filter DataHub Cloud-only aspects out of a datapack JSON so it can be
ingested into a DataHub OSS server.

Reads: 02-data.json  (a JSON array of MCPs)
Writes: 02-data.oss.json (same array, minus MCPs whose aspectName is on the
        Cloud-only blocklist)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CLOUD_ONLY_ASPECTS = {
    "entityInferenceMetadata",
    "lineageFeatures",
    "usageFeatures",
    "storageFeatures",
    "corpUserUsageFeatures",
    "testResults",
    "assertionsSummary",
    "schemaProposals",
    "proposals",
}

CLOUD_ONLY_BY_ENTITY = {
    ("domain", "status"),
}

def main(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    def keep(mcp: dict) -> bool:
        name = mcp.get("aspectName")
        etype = mcp.get("entityType")
        if name in CLOUD_ONLY_ASPECTS:
            return False
        if (etype, name) in CLOUD_ONLY_BY_ENTITY:
            return False
        return True

    kept = [mcp for mcp in data if keep(mcp)]
    dropped = len(data) - len(kept)
    by_aspect: dict[str, int] = {}
    for mcp in data:
        name = mcp.get("aspectName")
        if name in CLOUD_ONLY_ASPECTS:
            by_aspect[name] = by_aspect.get(name, 0) + 1
    dst.write_text(json.dumps(kept, indent=None), encoding="utf-8")
    print(f"input:   {len(data):>5} MCPs")
    print(f"dropped: {dropped:>5} MCPs (Cloud-only aspects)")
    for name, n in sorted(by_aspect.items(), key=lambda kv: -kv[1]):
        print(f"   - {name}: {n}")
    print(f"output:  {len(kept):>5} MCPs -> {dst}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("02-data.json")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("02-data.oss.json")
    main(src, dst)
