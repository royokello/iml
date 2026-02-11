#!/usr/bin/env python
import os
import csv
import sys
import re
import argparse
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

app = Flask(__name__)

album_dir = ""
src_dir = ""
labels_path = ""
labels = {}
image_ids = []
image_sizes = {}
resolution_buckets = {}
LABEL_OPTIONS = ["out_of_focus", "motion_blur", "too_dark", "too_bright"]

@app.route('/')
def index():
    global src_dir, labels, album_dir, image_ids
    total_images = len(image_ids)
    total_labels = 0
    label_stats = {'left': 0, 'right': 0}
    label_reason_stats = {opt: 0 for opt in LABEL_OPTIONS}
    for v in labels.values():
        choice = (v.get("preference") or "").lower()
        if choice in label_stats:
            label_stats[choice] += 1
            total_labels += 1
        reason = (v.get("label") or "").lower()
        if reason in label_reason_stats:
            label_reason_stats[reason] += 1

    album_name = os.path.basename(album_dir)
    return render_template(
        'index.html',
        total_images=total_images,
        total_labels=total_labels,
        label_stats=label_stats,
        label_reason_stats=label_reason_stats,
        label_options=LABEL_OPTIONS,
        album_name=album_name,
    )

@app.route('/image/<path:img_id>')
def get_image(img_id):
    global src_dir
    for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
        filename = f"{img_id}{ext}"
        p = os.path.join(src_dir, filename)
        if os.path.exists(p):
            return send_from_directory(src_dir, filename)
    return "Image not found", 404

@app.route('/label', methods=['POST'])
def label_image():
    global labels, labels_path, image_ids
    try:
        data = request.json or {}
        if 'img_1' not in data or 'img_2' not in data or 'choice' not in data or 'label' not in data:
            missing = []
            if 'img_1' not in data: missing.append('img_1')
            if 'img_2' not in data: missing.append('img_2')
            if 'choice' not in data: missing.append('choice')
            if 'label' not in data: missing.append('label')
            return jsonify(success=False, error=f"Missing required fields: {', '.join(missing)}"), 400

        img_1_id = data['img_1']
        img_2_id = data['img_2']
        choice = data['choice']
        label = data['label']
        valid_choices = ['left', 'right']
        valid_labels = set(LABEL_OPTIONS)

        if choice.lower() not in valid_choices:
            return jsonify(success=False, error=f"Invalid choice '{choice}'. Must be one of: {', '.join(valid_choices)}"), 400
        if label.lower() not in valid_labels:
            return jsonify(success=False, error=f"Invalid label '{label}'. Must be one of: {', '.join(LABEL_OPTIONS)}"), 400
        if img_1_id not in image_ids:
            return jsonify(success=False, error=f"Image ID '{img_1_id}' not found in available images"), 400
        if img_2_id not in image_ids:
            return jsonify(success=False, error=f"Image ID '{img_2_id}' not found in available images"), 400

        if choice.lower() == "left":
            winner_id = img_1_id
            loser_id = img_2_id
        else:
            winner_id = img_2_id
            loser_id = img_1_id

        s = sorted([winner_id, loser_id])
        pair_id = f"{s[0]}|||{s[1]}"
        labels[pair_id] = {
            "winner": winner_id,
            "loser": loser_id,
            "preference": choice.lower(),
            "label": label.lower(),
        }

        # when saving CSV in /label
        with open(labels_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['winner', 'loser', 'label', 'preference'])
            for pair, info in labels.items():
                w.writerow([
                    info.get("winner", ""),
                    info.get("loser", ""),
                    info.get("label", ""),
                    info.get("preference", ""),
                ])


        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route('/random', methods=['GET'])
def get_random():
    global image_ids, resolution_buckets
    import random
    if len(image_ids) <= 1:
        return jsonify({"error": "Not enough images", "total": len(image_ids)}), 400
    candidate_buckets = [b for b in resolution_buckets.values() if len(b) >= 2]
    if not candidate_buckets:
        return jsonify({"error": "Not enough images with matching resolution", "total": len(image_ids)}), 400
    bucket = random.choice(candidate_buckets)
    img_1, img_2 = random.sample(bucket, 2)
    return jsonify({"img_1": img_1, "img_2": img_2})

@app.route('/following', methods=['POST'])
def get_following():
    global image_ids, resolution_buckets, image_sizes
    import random
    if len(image_ids) <= 1:
        return jsonify({"error": "Not enough images", "total": len(image_ids)}), 400
    candidate_buckets = [b for b in resolution_buckets.values() if len(b) >= 2]
    if not candidate_buckets:
        return jsonify({"error": "Not enough images with matching resolution", "total": len(image_ids)}), 400
    bucket = random.choice(candidate_buckets)
    idx = random.randint(0, len(bucket) - 1)
    next_idx = (idx + 1) % len(bucket)
    return jsonify({"img_1": bucket[idx], "img_2": bucket[next_idx]})

@app.route('/random_single', methods=['GET'])
def get_random_single():
    global image_ids, resolution_buckets, image_sizes
    import random
    if len(image_ids) <= 0:
        return jsonify({"error": "No images available", "total": len(image_ids)}), 400
    anchor = request.args.get('anchor', '').strip()
    if anchor and anchor in image_sizes:
        res = image_sizes.get(anchor)
        bucket = resolution_buckets.get(res, [])
        if len(bucket) > 1:
            choices = [i for i in bucket if i != anchor]
            if choices:
                img = random.choice(choices)
                return jsonify({"img": img})
        if len(bucket) == 1:
            return jsonify({"error": "Not enough images with matching resolution", "total": len(image_ids)}), 400
    img = random.choice(image_ids)
    return jsonify({"img": img})

@app.route('/neighbor', methods=['POST'])
def get_neighbor():
    global image_ids, resolution_buckets, image_sizes
    data = request.json or {}
    img_id = (data.get('img') or '').strip()
    direction = (data.get('direction') or '').strip().lower()
    if not img_id or img_id not in image_sizes:
        return jsonify({"error": "Invalid image identifier"}), 400
    if direction not in ("prev", "next"):
        return jsonify({"error": "Invalid direction"}), 400

    res = image_sizes.get(img_id)
    bucket = resolution_buckets.get(res, [])
    if len(bucket) < 2:
        return jsonify({"error": "Not enough images with matching resolution"}), 400

    try:
        idx = bucket.index(img_id)
    except ValueError:
        return jsonify({"error": "Image not found in resolution bucket"}), 400

    if direction == "prev":
        next_idx = (idx - 1) % len(bucket)
    else:
        next_idx = (idx + 1) % len(bucket)

    return jsonify({"img": bucket[next_idx]})

@app.route('/switch', methods=['POST'])
def switch_images():
    global image_ids
    data = request.json or {}
    img_1 = data.get('img_1')
    img_2 = data.get('img_2')
    if img_1 is None or img_2 is None or img_1 not in image_ids or img_2 not in image_ids:
        return jsonify({"error": "Invalid image identifiers"}), 400
    return jsonify({"img_1": img_2, "img_2": img_1})

def load_labels_from_csv(csv_path):
    d = {}
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['winner', 'loser', 'label', 'preference'])
        return d
    with open(csv_path, 'r', newline='') as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return d
        header_lower = [h.strip().lower() for h in header]
        if 'winner' in header_lower and 'loser' in header_lower:
            iw = header_lower.index('winner')
            il = header_lower.index('loser')
            ir = header_lower.index('label') if 'label' in header_lower else None
            ip = header_lower.index('preference') if 'preference' in header_lower else None
            for row in r:
                if len(row) > max(iw, il):
                    winner_id = row[iw].strip()
                    loser_id = row[il].strip()
                    if not winner_id or not loser_id:
                        continue
                    label_val = ""
                    if ir is not None and len(row) > ir:
                        label_val = row[ir].strip().lower()
                    pref_val = ""
                    if ip is not None and len(row) > ip:
                        pref_val = row[ip].strip().lower()
                    s = sorted([winner_id, loser_id])
                    pid = f"{s[0]}|||{s[1]}"
                    d[pid] = {
                        "winner": winner_id,
                        "loser": loser_id,
                        "label": label_val,
                        "preference": pref_val,
                    }
        elif 'img_1' in header_lower and 'img_2' in header_lower and 'preference' in header_lower:
            i1 = header_lower.index('img_1')
            i2 = header_lower.index('img_2')
            ip = header_lower.index('preference')
            il = header_lower.index('label') if 'label' in header_lower else None
            for row in r:
                if len(row) > max(i1, i2, ip):
                    img_1_id = row[i1].strip()
                    img_2_id = row[i2].strip()
                    pref = row[ip].strip().lower()
                    if pref not in ('left', 'right'):
                        continue
                    label_val = ""
                    if il is not None and len(row) > il:
                        label_val = row[il].strip().lower()
                    winner_id = img_1_id if pref == 'left' else img_2_id
                    loser_id = img_2_id if pref == 'left' else img_1_id
                    s = sorted([winner_id, loser_id])
                    pid = f"{s[0]}|||{s[1]}"
                    d[pid] = {
                        "winner": winner_id,
                        "loser": loser_id,
                        "label": label_val,
                        "preference": pref,
                    }
    return d


def get_image_ids(directory):
    ids = []
    for root, _, files in os.walk(directory):
        for f in files:
            n, ext = os.path.splitext(f)
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                rel_path = os.path.relpath(os.path.join(root, f), directory)
                rel_path = rel_path.replace("\\", "/")
                rel_no_ext, _ = os.path.splitext(rel_path)
                ids.append(rel_no_ext)
    return ids

def build_resolution_buckets(directory, ids):
    sizes = {}
    buckets = {}
    for img_id in ids:
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
            p = os.path.join(directory, f"{img_id}{ext}")
            if os.path.exists(p):
                img_path = p
                break
        if not img_path:
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            sizes[img_id] = (w, h)
            buckets.setdefault((w, h), []).append(img_id)
        except Exception:
            continue
    for res in buckets.keys():
        buckets[res] = sorted(buckets[res])
    return sizes, buckets

def start_labeler(project_dir, stage):
    global album_dir, src_dir, labels_path, labels, image_ids, image_sizes, resolution_buckets
    album_dir = project_dir
    if not os.path.isdir(album_dir):
        os.makedirs(album_dir, exist_ok=True)
    stage_dir = f"stage_{stage}"
    src_dir = os.path.join(album_dir, stage_dir)
    if not os.path.isdir(src_dir):
        os.makedirs(src_dir, exist_ok=True)
    labels_path = os.path.join(album_dir, f"{stage_dir}_rank_labels.csv")
    labels = load_labels_from_csv(labels_path)
    image_ids = get_image_ids(src_dir)
    image_sizes, resolution_buckets = build_resolution_buckets(src_dir, image_ids)
    print(f"Found {len(image_ids)} images in {src_dir}")
    print(f"Starting labeler for project: {os.path.basename(album_dir)}, stage: {stage}")

def get_latest_stage(project_dir):
    p = re.compile(r'stage_(\d+)')
    stages = []
    for item in os.listdir(project_dir):
        item_path = os.path.join(project_dir, item)
        if os.path.isdir(item_path):
            m = p.match(item)
            if m:
                stages.append((int(m.group(1)), item))
    if not stages:
        print(f"No stage directories found in {project_dir}")
        return None
    stages.sort(reverse=True)
    return stages[0][1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the IML Ranker labeling tool.')
    parser.add_argument('--project', required=True, help='Project directory path where files are kept')
    parser.add_argument('--stage', type=str, help='Stage number (optional, will use latest stage if not provided)')
    args = parser.parse_args()
    if args.stage is None:
        stage_dir = get_latest_stage(args.project)
        if stage_dir is None:
            print("Error: No stage directories found in the project directory.")
            sys.exit(1)
        args.stage = stage_dir.replace('stage_', '')
    start_labeler(args.project, args.stage)
    app.run(debug=True)
