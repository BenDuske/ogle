# Ogle Demo Recording Package

Everything you need to assemble the < 3 min Devpost demo video. Generated 2026-08-08.

## What's in here

```
demo-recording/
├── audio/                              (Ben's cloned voice, ElevenLabs)
│   ├── scene-0-cold-open.mp3           12.86s
│   ├── scene-1-problem-in-graph.mp3    20.67s
│   ├── scene-2-alert-fires.mp3         40.22s
│   ├── scene-3-narrative-writeback.mp3 28.24s
│   ├── scene-4-debounce.mp3            14.91s
│   └── scene-5-close.mp3                9.10s
│                                    TOTAL 2:05  (well under 3:00 cap)
│
├── transcripts/                        (verified command output — exit codes confirmed)
│   ├── scene-2-ogle-demo.txt
│   ├── scene-3-narrate-writeback.txt
│   ├── scene-4a-seed.txt               exit 0
│   ├── scene-4b-fires.txt              exit 1
│   └── scene-4c-debounced.txt          exit 0
│
├── play-demo.sh                        auto-run all commands with typewriter pauses
├── generate_audio.py                   regenerate audio (rerun if narration needs tweaks)
└── README.md                           this file
```

## Recording flow (~10 min hands-on)

### 1. Prep the terminal

- Open **Windows Terminal** (or your preferred term). Font size ~16pt so text reads at 1080p.
- **Maximize** the window (F11 or Win+Up). Clear scrollback.
- `cd C:\Users\bendu\ogle`
- Activate the venv if you have one (or ensure `python -m ogle` works).

### 2. Prep Scene 1's images

Open `docs/screenshots/03-churn-predictor-lineage.png`, `06-demand-forecast-lineage.png`, and `09-feature-table-sources.png` in an image viewer, side-by-side or tabbed, ready to flip through.

### 3. Recorder

Windows: **Win+G** to open Game Bar → start capture (records the active window). Or use **OBS** for a scene switcher.

Give yourself 3–5 seconds of buffer at the start so the first frame isn't the "recording started" toast.

### 4. Record

Two easy paths:

**Path A — one long screen record, cut in post (recommended):**
1. Show title card / README top line — hold 12s (Scene 0)
2. Alt-tab to image viewer, flip through 3 lineage screenshots ~10s each (Scene 1)
3. Alt-tab to terminal, run:
   ```bash
   bash demo-recording/play-demo.sh
   ```
   The script auto-runs Scenes 2, 3, 4 with typewriter typing and natural pauses.
4. Alt-tab back to title/repo URL — hold 10s (Scene 5)
5. Stop recording.

**Path B — six separate takes, one per scene:**
Same as above but split into individual recordings if your editor prefers discrete clips.

### 5. Post-produce

Any video editor works — DaVinci Resolve (free), Clipchamp, Shotcut, iMovie. Steps:

1. Import the screen recording(s) and the six MP3s from `audio/`.
2. Line each MP3 up to its corresponding scene:

   | Scene | Timing (approx) | Audio file |
   |---|---|---|
   | 0 | 0:00 – 0:13 | scene-0-cold-open.mp3 |
   | 1 | 0:13 – 0:34 | scene-1-problem-in-graph.mp3 |
   | 2 | 0:34 – 1:14 | scene-2-alert-fires.mp3 |
   | 3 | 1:14 – 1:42 | scene-3-narrative-writeback.mp3 |
   | 4 | 1:42 – 1:57 | scene-4-debounce.mp3 |
   | 5 | 1:57 – 2:06 | scene-5-close.mp3 |

3. Add ~250ms crossfades between scene audio clips.
4. Trim any terminal dead air > 2s between commands so audio drives the pace.
5. Render 1080p MP4.

### 6. Upload

- YouTube → Upload → **Unlisted** (or Public — either satisfies Devpost).
- Copy the URL.
- **Open the URL in an incognito window and verify it plays without sign-in.** (This is the "most common thing people miss" per DataHub's checklist.)

### 7. Paste into Devpost

Paste the URL into:
- The Devpost submission's Video field
- Row "Demo video" in `docs/DEVPOST-SUBMISSION.md` (nice to have, not required)

## Regenerating audio

If you tweak a narration line in `docs/DEMO-VIDEO-SCRIPT.md` and need fresh audio, edit the matching `SCENES` entry in `generate_audio.py` and rerun:

```bash
python demo-recording/generate_audio.py
```

Uses Ben's ElevenLabs cloned voice (`VMEzE95otIZyOaJaKcL0`, `eleven_multilingual_v2`, stability 0.5 / similarity 0.8). API key read from `C:\Users\bendu\.openclaw\credentials\elevenlabs.env`.

## Not committing raw MP3s / recordings

Everything under `demo-recording/audio/`, `demo-recording/screen/`, and any `.mp4` renders should stay LOCAL — add to gitignore if not already. Final YouTube URL is what goes in the submission.
