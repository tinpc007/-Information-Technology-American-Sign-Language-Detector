import os
import json
import shutil
from collections import defaultdict
from tqdm import tqdm
import cv2

# =========================
# CONFIG
# =========================

root_data_dir = "/home/haole/data/khanh/projects/gesture_recognition/datasets/wlasl-2000/wlasl-complete"


JSON_PATH = f"{root_data_dir}/WLASL_v0.3.json"
VIDEO_ROOT = f"{root_data_dir}/videos"
OUTPUT_ROOT = f"{root_data_dir}/wlasl100_clean"
GLSS_LIST_PATH = f"{root_data_dir}/wlasl_class_list.txt"

TOP_K = 100
MIN_FRAMES = 16
MIN_SAMPLES_PER_CLASS = 10

# =========================
# LOAD CLASS LIST
# =========================

class_list = []
with open(GLSS_LIST_PATH) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            class_list.append(parts[1])

top_classes = set(class_list[:TOP_K])

print(f"Using {len(top_classes)} classes")

# =========================
# LOAD JSON
# =========================

with open(JSON_PATH) as f:
    data = json.load(f)

# =========================
# PREPARE STRUCTURES
# =========================

samples = {"train": [], "val": [], "test": []}
class_count = defaultdict(int)

# =========================
# FILTER
# =========================

for entry in tqdm(data):
    gloss = entry["gloss"]

    if gloss not in top_classes:
        continue

    for inst in entry["instances"]:
        split = inst["split"]
        video_id = inst["video_id"]

        video_path = os.path.join(VIDEO_ROOT, f"{video_id}.mp4")

        # skip missing
        if not os.path.exists(video_path):
            continue

        # check video validity
        cap = cv2.VideoCapture(video_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frames < MIN_FRAMES:
            continue

        samples[split].append((video_path, gloss))
        class_count[gloss] += 1

# =========================
# FILTER LOW-SAMPLE CLASSES
# =========================

valid_classes = {
    c for c, cnt in class_count.items()
    if cnt >= MIN_SAMPLES_PER_CLASS
}

print(f"Valid classes after filtering: {len(valid_classes)}")

# =========================
# CREATE OUTPUT
# =========================

for split in ["train", "val", "test"]:
    for c in valid_classes:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, c), exist_ok=True)

# =========================
# COPY DATA
# =========================

stats = {"train": 0, "val": 0, "test": 0}

for split in ["train", "val", "test"]:
    for path, gloss in tqdm(samples[split], desc=split):

        if gloss not in valid_classes:
            continue

        filename = os.path.basename(path)
        dst = os.path.join(OUTPUT_ROOT, split, gloss, filename)

        shutil.copy(path, dst)
        stats[split] += 1

# =========================
# SAVE LABEL MAP
# =========================

label2id = {c: i for i, c in enumerate(sorted(valid_classes))}
id2label = {i: c for c, i in label2id.items()}

with open(os.path.join(OUTPUT_ROOT, "label2id.json"), "w") as f:
    json.dump(label2id, f, indent=2)

with open(os.path.join(OUTPUT_ROOT, "id2label.json"), "w") as f:
    json.dump(id2label, f, indent=2)

# =========================
# REPORT
# =========================

print("\n✅ DONE")
print(stats)
print(f"Total classes: {len(valid_classes)}")