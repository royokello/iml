#!/usr/bin/env python
import os
import csv
import sys
import re
import argparse
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

album_dir = ""
src_dir = ""
labels_path = ""
labels = {}
image_ids = []

@app.route('/')
def index():
    global src_dir, labels, album_dir
    total_images = sum(
        1 for e in os.scandir(src_dir)
        if e.is_file() and e.name.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    total_labels = 0
    # inside index()
    label_stats = {'left': 0, 'right': 0}
    for v in labels.values():
        k = v.lower()
        if k in label_stats:
            label_stats[k] += 1
            total_labels += 1

    album_name = os.path.basename(album_dir)
    return render_template(
        'index.html',
        total_images=total_images,
        total_labels=total_labels,
        label_stats=label_stats,
        album_name=album_name,
    )

@app.route('/image/<path:img_id>')
def get_image(img_id):
    global src_dir
    for ext in ['.png', '.jpg', '.jpeg']:
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
        if 'img_1' not in data or 'img_2' not in data or 'choice' not in data:
            missing = []
            if 'img_1' not in data: missing.append('img_1')
            if 'img_2' not in data: missing.append('img_2')
            if 'choice' not in data: missing.append('choice')
            return jsonify(success=False, error=f"Missing required fields: {', '.join(missing)}"), 400

        img_1_id = data['img_1']
        img_2_id = data['img_2']
        choice = data['choice']
        # inside /label
        valid_choices = ['left', 'right']

        if choice.lower() not in valid_choices:
            return jsonify(success=False, error=f"Invalid choice '{choice}'. Must be one of: {', '.join(valid_choices)}"), 400
        if img_1_id not in image_ids:
            return jsonify(success=False, error=f"Image ID '{img_1_id}' not found in available images"), 400
        if img_2_id not in image_ids:
            return jsonify(success=False, error=f"Image ID '{img_2_id}' not found in available images"), 400

        s = sorted([img_1_id, img_2_id])
        pair_id = f"{s[0]}|||{s[1]}"
        labels[pair_id] = choice

        # when saving CSV in /label
        with open(labels_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['img_1', 'img_2', 'preference'])
            for pair, c in labels.items():
                a, b = pair.split('|||')
                w.writerow([a, b, c])


        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route('/random', methods=['GET'])
def get_random():
    global image_ids
    import random
    if len(image_ids) <= 1:
        return jsonify({"error": "Not enough images", "total": len(image_ids)}), 400
    img_1, img_2 = random.sample(image_ids, 2)
    return jsonify({"img_1": img_1, "img_2": img_2})

@app.route('/random_single', methods=['GET'])
def get_random_single():
    global image_ids
    import random
    if len(image_ids) <= 0:
        return jsonify({"error": "No images available", "total": len(image_ids)}), 400
    img = random.choice(image_ids)
    return jsonify({"img": img})

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
            csv.writer(f).writerow(['img_1', 'img_2', 'preference'])
        return d
    with open(csv_path, 'r', newline='') as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return d
        i1 = header.index('img_1')
        i2 = header.index('img_2')
        ip = header.index('preference')
        for row in r:
            if len(row) > max(i1, i2, ip):
                a = row[i1].strip()
                b = row[i2].strip()
                c = row[ip].strip().lower()
                if c not in ('left', 'right'):
                    continue
                s = sorted([a, b])
                pid = f"{s[0]}|||{s[1]}"
                d[pid] = c
    return d


def get_image_ids(directory):
    ids = []
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            n, ext = os.path.splitext(f)
            if ext.lower() in ['.jpg', '.jpeg', '.png']:
                ids.append(n)
    return ids

def start_labeler(project_dir, stage):
    global album_dir, src_dir, labels_path, labels, image_ids
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
