"""Generate 6 narration MP3s from DEMO-VIDEO-SCRIPT.md using Ben's cloned voice."""
import os, urllib.request, urllib.error, json, sys, time

_env = {}
with open(r"C:\Users\bendu\.openclaw\credentials\elevenlabs.env", "rb") as _f:
    for _line in _f.read().decode("utf-8-sig").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _env[_k.strip()] = _v.strip()
API_KEY = _env["ELEVENLABS_API_KEY"]
VOICE_ID = _env["ELEVENLABS_BEN_CLONE_VOICE_ID"]
BASE = "https://api.elevenlabs.io"

SCENES = {
    "scene-0-cold-open.mp3": (
        "Every ML team lives one silent training-data change away from a bad model in "
        "production. The lineage already exists in DataHub. What's missing is an agent that "
        "watches it. This is Ogle."
    ),
    "scene-1-problem-in-graph.mp3": (
        "Here is a real serving model in DataHub — a churn predictor — walked back through "
        "its feature tables to the source tables that feed it. If one of those upstream "
        "tables silently shifts — a schema change, a volume drop, a distribution that "
        "quietly moves — the model degrades and nobody gets paged. Ogle walks this exact "
        "graph on a schedule and catches the shift before the next deploy."
    ),
    "scene-2-alert-fires.mp3": (
        "No Docker. No API key. One command. Ogle seeds healthy baselines — exit zero — "
        "then re-checks a drifted snapshot through the same code path a live DataHub walk "
        "feeds. And it fires. Two serving-path tables drifted at once. One loudly — schema, "
        "volume, and quality. One silently, in its value distributions — cardinality, mean, "
        "standard deviation, and range. That's seven of Ogle's nine drift dimensions in a "
        "single high-severity alert. The silent covariate shift is the one that normally "
        "ships to production unnoticed. Ogle scores it green on schema, red on "
        "distribution. Exit code one — that is what pages your on-call."
    ),
    "scene-3-narrative-writeback.mp3": (
        "Add two flags and you get the other two flagship features from the same keyless "
        "command. First, a root-cause narrative your on-call engineer can act on in about "
        "thirty seconds — what moved, where, and why it matters — written by a local "
        "model. Second, the tag write-back — the exact ogle-drift-flagged and "
        "ogle-drift-high annotations Ogle would stamp back onto the drifted dataset inside "
        "DataHub, shown here as a dry run that touches no catalog. The alert becomes "
        "evidence that lives next to the data."
    ),
    "scene-4-debounce.mp3": (
        "And it remembers what it already flagged. The first run seeds. The drift fires "
        "once — exit one. Run the same drift again and it is debounced to exit zero. You "
        "get paged once per incident, not every tick. That is the difference between an "
        "alert and alert spam."
    ),
    "scene-5-close.mp3": (
        "Ogle. It walks your lineage, catches the silent drift, writes the root cause, "
        "and remembers. Apache two point zero, on GitHub. Thanks for watching."
    ),
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/audio"
os.makedirs(OUT_DIR, exist_ok=True)

for filename, text in SCENES.items():
    out_path = os.path.join(OUT_DIR, filename)
    print(f"Generating {filename} ({len(text.split())} words)...", flush=True)
    url = f"{BASE}/v1/text-to-speech/{VOICE_ID}"
    body = json.dumps({
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            audio = resp.read()
            with open(out_path, "wb") as f:
                f.write(audio)
            print(f"  -> {out_path} ({len(audio)} bytes)", flush=True)
    except urllib.error.HTTPError as e:
        print(f"  ERROR {e.code}: {e.read().decode('utf-8', errors='replace')}", flush=True)
        sys.exit(1)
    time.sleep(0.5)  # be nice to the API

print("All 6 narration files generated.")
