#!/usr/bin/env bash
# Ogle demo playback — auto-runs the demo script's command sequence with
# natural pauses so you can hit RECORD once and get a clean screen capture.
#
# Usage:
#   1. Open a clean, MAXIMIZED terminal (Windows Terminal / cmder / iTerm).
#      Font size big enough to read at 1080p — ~16pt is a good starting point.
#   2. cd C:/Users/bendu/ogle
#   3. Start your screen recorder (Win+G on Windows, OBS, whatever).
#      Give yourself 3 seconds to focus the terminal.
#   4. Run: bash demo-recording/play-demo.sh
#   5. When done, stop recording, then post-produce with the audio in demo-recording/audio/.
#
# Pauses are tuned so total wall-clock ~= narration budget (2:05 audio + ~55s screen pauses).

set -e
cd "$(dirname "$0")/.."  # jump to ogle repo root

typewrite() {
  # Print a command like the user is typing it, then run it.
  local cmd="$1"
  echo -n "$ "
  for ((i=0; i<${#cmd}; i++)); do
    echo -n "${cmd:$i:1}"
    sleep 0.04
  done
  echo ""
  sleep 0.4
  eval "$cmd" || true  # allow drift exit codes (1) without stopping playback
  echo ""
}

banner() {
  echo ""
  echo "============================================================"
  echo "  $1"
  echo "============================================================"
  sleep 1.5
}

# Scene 0 — cold open handled by title-card overlay in post; no terminal work.
# Scene 1 — screenshots flipped in post; no terminal work.

banner "SCENE 2 — one command, the alert fires"
sleep 2
typewrite "ogle demo"
sleep 6   # pause on the alert block for narration to catch up

banner "SCENE 3 — root cause + write-back, still keyless"
sleep 2
typewrite "ogle demo --narrate --write-back --write-back-severity"
sleep 5   # hold on narrative + write-back preview

banner "SCENE 4 — it remembers (debounce)"
sleep 2
rm -f demo.json
typewrite "ogle check --store demo.json --signatures examples/demo/healthy-signatures.json"
sleep 1
typewrite "ogle check --store demo.json --signatures examples/demo/drifted-signatures.json"
sleep 1
typewrite "ogle check --store demo.json --signatures examples/demo/drifted-signatures.json"
sleep 4   # highlight the 0 -> 1 -> 0 sequence
rm -f demo.json

# Scene 5 — close screen handled by title card in post.

banner "DONE — stop recording"
