#!/usr/bin/env bash
# Build final Ogle demo MP4 from the raw screen recording + narration audio + DataHub screenshots.
# Produces demo-recording/final/ogle-demo.mp4 (target ~2:07, under 3:00 cap).
set -e

ROOT="$(dirname "$0")"
cd "$ROOT"
mkdir -p final work
rm -f work/*.mp4 final/ogle-demo.mp4

RAW=screen/demo-raw.mp4
AUD=audio
SHOT=../docs/screenshots
FONT="/c/Windows/Fonts/consola.ttf"
FONT_SANS="/c/Windows/Fonts/segoeui.ttf"

W=1920; H=1080; FPS=30

# ---------- SCENE 0: Ogle promo image (12.86s) ----------
ffmpeg -y -loop 1 -t 12.86 -i "$SHOT/ogle-linkedin-promo.png" \
  -vf "scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:color=0x0a1929,setsar=1" \
  -c:v libx264 -preset medium -pix_fmt yuv420p -r ${FPS} work/s0.mp4 > work/s0.log 2>&1

# ---------- SCENE 1: three DataHub screenshots, 6.89s each = 20.67s ----------
# Use ffmpeg concat with three still-image segments
for i in 03 06 09; do
  case $i in
    03) src=$SHOT/03-churn-predictor-lineage.png ;;
    06) src=$SHOT/06-demand-forecast-lineage.png ;;
    09) src=$SHOT/09-feature-table-sources.png ;;
  esac
  ffmpeg -y -loop 1 -t 6.89 -i "$src" \
    -vf "scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1" \
    -c:v libx264 -preset medium -pix_fmt yuv420p -r ${FPS} work/s1-${i}.mp4 > work/s1-${i}.log 2>&1
done
# concat the three
printf "file 's1-03.mp4'\nfile 's1-06.mp4'\nfile 's1-09.mp4'\n" > work/s1-list.txt
ffmpeg -y -f concat -safe 0 -i work/s1-list.txt -c copy work/s1.mp4 > work/s1-concat.log 2>&1

# CROP_TASKBAR: strip the 48px Windows taskbar off the bottom, then pad back to 1080 with black
CROP_TB="crop=1920:1032:0:0,pad=1920:1080:0:0:color=black,setsar=1"

# ---------- SCENE 2: ogle demo active (2-15s of raw = 13s) + freeze final frame to fill 40.22s ----------
ffmpeg -y -ss 2 -to 15 -i "$RAW" -vf "$CROP_TB" -c:v libx264 -preset medium -pix_fmt yuv420p -r ${FPS} -an work/s2a.mp4 > work/s2a.log 2>&1
# Extract the last frame at t=14.9s of raw as an image (also crop the taskbar)
ffmpeg -y -ss 14.9 -i "$RAW" -vf "$CROP_TB" -frames:v 1 -q:v 2 work/s2-freeze.png > work/s2-freeze.log 2>&1
ffmpeg -y -loop 1 -t 27.22 -i work/s2-freeze.png \
  -vf "scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1" \
  -c:v libx264 -preset medium -pix_fmt yuv420p -r ${FPS} work/s2b.mp4 > work/s2b.log 2>&1
printf "file 's2a.mp4'\nfile 's2b.mp4'\n" > work/s2-list.txt
ffmpeg -y -f concat -safe 0 -i work/s2-list.txt -c copy work/s2.mp4 > work/s2-concat.log 2>&1

# ---------- SCENE 3: last 18s of Scene 3 recording (raw 40-58s, narrate+writeback visible) + screenshot 10 hold (10.24s) = 28.24s ----------
ffmpeg -y -ss 40 -to 58 -i "$RAW" -vf "$CROP_TB" -c:v libx264 -preset medium -pix_fmt yuv420p -r ${FPS} -an work/s3a.mp4 > work/s3a.log 2>&1
ffmpeg -y -loop 1 -t 10.24 -i "$SHOT/10-churn-predictor-tagged.png" \
  -vf "scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1" \
  -c:v libx264 -preset medium -pix_fmt yuv420p -r ${FPS} work/s3b.mp4 > work/s3b.log 2>&1
printf "file 's3a.mp4'\nfile 's3b.mp4'\n" > work/s3-list.txt
ffmpeg -y -f concat -safe 0 -i work/s3-list.txt -c copy work/s3.mp4 > work/s3-concat.log 2>&1

# ---------- SCENE 4: 14.91s slice of the debounce sequence (raw 76-91s captures the 3 checks + exits) ----------
ffmpeg -y -ss 76 -to 90.91 -i "$RAW" -vf "$CROP_TB" -c:v libx264 -preset medium -pix_fmt yuv420p -r ${FPS} -an work/s4.mp4 > work/s4.log 2>&1

# ---------- SCENE 5: Ogle promo image + small repo URL overlay (9.10s) ----------
ffmpeg -y -loop 1 -t 9.10 -i "$SHOT/ogle-linkedin-promo.png" \
  -vf "scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:color=0x0a1929,setsar=1,\
drawtext=fontfile=${FONT}:text='github.com/BenDuske/ogle  •  Apache 2.0':fontcolor=0xdddddd:fontsize=32:x=(w-text_w)/2:y=h-90" \
  -c:v libx264 -preset medium -pix_fmt yuv420p -r ${FPS} work/s5.mp4 > work/s5.log 2>&1

# ---------- CONCAT ALL VIDEO SCENES ----------
printf "file 's0.mp4'\nfile 's1.mp4'\nfile 's2.mp4'\nfile 's3.mp4'\nfile 's4.mp4'\nfile 's5.mp4'\n" > work/all-list.txt
ffmpeg -y -f concat -safe 0 -i work/all-list.txt -c copy work/video-only.mp4 > work/video-concat.log 2>&1

# ---------- CONCAT AUDIO ----------
# Use concat demuxer for MP3s (re-encode via filter to keep sample rate consistent)
ffmpeg -y \
  -i "$AUD/scene-0-cold-open.mp3" \
  -i "$AUD/scene-1-problem-in-graph.mp3" \
  -i "$AUD/scene-2-alert-fires.mp3" \
  -i "$AUD/scene-3-narrative-writeback.mp3" \
  -i "$AUD/scene-4-debounce.mp3" \
  -i "$AUD/scene-5-close.mp3" \
  -filter_complex "[0:a][1:a][2:a][3:a][4:a][5:a]concat=n=6:v=0:a=1[out]" \
  -map "[out]" -c:a aac -b:a 192k work/audio-full.aac > work/audio-concat.log 2>&1

# ---------- MUX VIDEO + AUDIO ----------
ffmpeg -y -i work/video-only.mp4 -i work/audio-full.aac \
  -c:v copy -c:a copy -shortest final/ogle-demo.mp4 > work/final-mux.log 2>&1

echo "=========================================="
echo "FINAL: $(pwd)/final/ogle-demo.mp4"
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_type,codec_name,width,height -of default=noprint_wrappers=1 final/ogle-demo.mp4
