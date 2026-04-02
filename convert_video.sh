cd /home/haole/data/khanh/projects/gesture_recognition/datasets/wlasl-2000/wlasl-complete/wlasl100_clean

find test_browser -type f -name "*.mp4" -print0 | while IFS= read -r -d '' f; do

  full_path="$(pwd)/$f"

  
  # Skip if already processed (optional)
  if ffprobe -v error -select_streams v:0 \
     -show_entries stream=codec_name \
     -of default=noprint_wrappers=1:nokey=1 "$full_path" | grep -q "h264"; then
    echo "✔ Already OK: $full_path"
    continue
  fi

  tmp="${full_path%.mp4}_tmp.mp4"

  echo "Processing: $full_path"

  if [ ! -f "$full_path" ]; then
    echo "⚠️ $full_path file does not exist"
    continue
  fi  

  ffmpeg -y -loglevel error -i "$full_path" \
    -vcodec libx264 -acodec aac \
    -movflags faststart \
    "$tmp"

  if [ -f "$tmp" ]; then
    mv "$tmp" "$full_path"
    echo "✅ Converted"
  else
    echo "❌ Failed: $full_path"
  fi

done