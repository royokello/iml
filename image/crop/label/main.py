
import argparse
import csv
import json
import os
import random
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from utils.stages import find_latest_stage
from PIL import Image

app = Flask(__name__)

# Globals
image_dir = ""
image_files = []
# box tuple: (class_id, x1, y1, x, y) where x,y are x2,y2 (bottom-right)
labels = {}         # Dict[int, List[Tuple[int, float, float, float, float]]]
labels_file = ""

ratios = []
ratios_file = ""
current_image_index = -1
labelled_idxs = []


def _get_class_stats(labels_dict):
    stats = {}
    for _idx, boxes in labels_dict.items():
        for (cls_id, *_coords) in boxes:
            stats[cls_id] = stats.get(cls_id, 0) + 1
    return stats


@app.route('/')
def index():
    return render_template(
        "index.html",
        total_images=len(image_files),
        total_labels=sum(len(b) for b in labels.values()),
        class_stats=_get_class_stats(labels),
        current_index=current_image_index,
        labelled_count=len(labelled_idxs)
    )


@app.route('/image/<int:index>')
def get_image(index):
    try:
        img_name = image_files[index]
    except (IndexError, TypeError):
        return jsonify(error='Image not found'), 404

    label_data = []
    if index in labels:
        for (cls_id, x1, y1, x, y) in labels[index]:
            label_data.append({'class': cls_id, 'x1': x1, 'y1': y1, 'x': x, 'y': y})

    response = make_response(send_from_directory(image_dir, img_name))
    if label_data:
        response.headers['X-Label'] = json.dumps(label_data)
    return response


@app.route('/labels/<int:index>')
def get_labels(index):
    try:
        _ = image_files[index]
    except (IndexError, TypeError):
        return jsonify(error='Image not found'), 404
    lst = []
    for i, (cls_id, x1, y1, x, y) in enumerate(labels.get(index, [])):
        lst.append({'id': i, 'class': cls_id, 'x1': x1, 'y1': y1, 'x': x, 'y': y})
    return jsonify(labels=lst)


@app.route('/stats')
def get_stats():
    return jsonify(
        total_images=len(image_files),
        labelled_images=len(labelled_idxs),
        total_labels=sum(len(b) for b in labels.values()),
        class_stats=_get_class_stats(labels),
        current_index=current_image_index
    )


@app.route('/label', methods=['POST'])
def label_image():
    global labels, labelled_idxs
    data = request.json
    idx = data.get('index')
    is_delete = data.get('is_delete', False)

    if not isinstance(idx, int) or idx < 0 or idx >= len(image_files):
        return jsonify(error='Invalid image index'), 400

    if is_delete:
        if idx in labels:
            labels.pop(idx)
            if idx in labelled_idxs:
                labelled_idxs.remove(idx)
        _rewrite_labels_csv()
        return jsonify(_stats_payload())

    try:
        cls_id = int(data['class'])
        x1 = float(data['x1'])
        y1 = float(data['y1'])
        x  = float(data['x'])
        y  = float(data['y'])
    except (KeyError, TypeError, ValueError):
        return jsonify(error='Invalid label data'), 400

    labels.setdefault(idx, []).append((cls_id, x1, y1, x, y))
    if idx not in labelled_idxs:
        labelled_idxs.append(idx)

    _rewrite_labels_csv()
    return jsonify(_stats_payload())


@app.route('/label/update', methods=['POST'])
def update_label():
    global labels
    data = request.json
    try:
        idx = int(data['index'])
        box_id = int(data['box_id'])
        cls_id = int(data['class'])
        x1 = float(data['x1'])
        y1 = float(data['y1'])
        x  = float(data['x'])
        y  = float(data['y'])
    except (KeyError, TypeError, ValueError):
        return jsonify(error='Invalid payload'), 400

    if idx not in labels or box_id < 0 or box_id >= len(labels[idx]):
        return jsonify(error='Label not found'), 404

    labels[idx][box_id] = (cls_id, x1, y1, x, y)
    _rewrite_labels_csv()
    return jsonify(_stats_payload())


@app.route('/label/delete', methods=['POST'])
def delete_one_label():
    global labels, labelled_idxs
    data = request.json
    try:
        idx = int(data['index'])
        box_id = int(data['box_id'])
    except (KeyError, TypeError, ValueError):
        return jsonify(error='Invalid payload'), 400

    if idx not in labels or box_id < 0 or box_id >= len(labels[idx]):
        return jsonify(error='Label not found'), 404

    del labels[idx][box_id]
    if not labels[idx]:
        labels.pop(idx)
        if idx in labelled_idxs:
            labelled_idxs.remove(idx)
    _rewrite_labels_csv()
    return jsonify(_stats_payload())


@app.route('/navigate', methods=['POST'])
def navigate():
    global current_image_index
    data = request.json
    action = data.get('action')

    if not image_files:
        return jsonify(error='No images'), 400

    if action == 'next':
        current_image_index = (current_image_index + 1) % len(image_files)
    elif action == 'prev':
        current_image_index = (current_image_index - 1) % len(image_files)
    elif action == 'random':
        current_image_index = random.randint(0, len(image_files) - 1)
    elif action == 'next_labelled':
        if not labelled_idxs:
            return jsonify(error='No labelled images'), 400
        nxt = [i for i in labelled_idxs if i > current_image_index]
        current_image_index = nxt[0] if nxt else labelled_idxs[0]
    elif action == 'prev_labelled':
        if not labelled_idxs:
            return jsonify(error='No labelled images'), 400
        prev = [i for i in labelled_idxs if i < current_image_index]
        current_image_index = prev[-1] if prev else labelled_idxs[-1]
    else:
        return jsonify(error='Invalid action'), 400

    return jsonify(index=current_image_index, **_stats_payload())


def _rewrite_labels_csv():
    global labels, image_files, image_dir, labels_file
    dims_cache = {}
    with open(labels_file, 'w', newline='') as f:
        w = csv.writer(f)
        # YOLO-normalized center + half-size, but keep headers as x1,y1,x,y per your spec
        w.writerow(['img', 'class', 'x1', 'y1', 'x', 'y'])
        for img_idx in sorted(labels.keys()):
            img_name = image_files[img_idx]
            if img_name not in dims_cache:
                with Image.open(os.path.join(image_dir, img_name)) as im:
                    dims_cache[img_name] = im.size  # (iw, ih)
            iw, ih = dims_cache[img_name]
            for (cls_id, x1, y1, x2, y2) in labels[img_idx]:
                cx_px = (x1 + x2) / 2.0
                cy_px = (y1 + y2) / 2.0
                hx_px = (x2 - x1) / 2.0
                hy_px = (y2 - y1) / 2.0
                cx = cx_px / iw
                cy = cy_px / ih
                hx = hx_px / iw
                hy = hy_px / ih
                w.writerow([img_name, cls_id,
                            f'{cx:.6f}', f'{cy:.6f}', f'{hx:.6f}', f'{hy:.6f}'])

def _stats_payload():
    return dict(
        total_labels=sum(len(b) for b in labels.values()),
        class_stats=_get_class_stats(labels),
        labelled_images=len(labelled_idxs)
    )


def main(project: str, stage: int = None):
    global image_dir, image_files, labels, labels_file
    global ratios, ratios_file, current_image_index, labelled_idxs

    if stage is None:
        stage = find_latest_stage(project)
    print(f"Stage {stage} ...")
    stage_dir = f"stage_{stage}"
    image_dir = os.path.join(project, stage_dir)
    if not os.path.exists(image_dir):
        print(f"Error: Stage directory '{stage_dir}' not found in {project}")
        return

    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    image_files.sort()
    current_image_index = 0 if image_files else -1

    labels_file = os.path.join(project, f'stage_{stage}_crop_labels.csv')

    if not os.path.exists(labels_file):
        with open(labels_file, 'w', newline='') as f:
            csv.writer(f).writerow(['img', 'class', 'x1', 'y1', 'x', 'y'])

    labels.clear()
    name_to_idx = {name: i for i, name in enumerate(image_files)}

    if os.path.exists(labels_file):
        with open(labels_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                try:
                    img_name, cls_s, cx_s, cy_s, hx_s, hy_s = row
                    if img_name not in name_to_idx:
                        continue
                    idx_i = name_to_idx[img_name]
                    cls_id = int(cls_s)
                    cx, cy, hx, hy = float(cx_s), float(cy_s), float(hx_s), float(hy_s)
                    with Image.open(os.path.join(image_dir, img_name)) as im:
                        iw, ih = im.size
                    x1 = (cx - hx) * iw
                    y1 = (cy - hy) * ih
                    x2 = (cx + hx) * iw
                    y2 = (cy + hy) * ih
                    labels.setdefault(idx_i, []).append((cls_id, x1, y1, x2, y2))
                except Exception:
                    continue

    labelled_idxs = list(sorted(labels.keys()))

    print(f"Starting app: {len(image_files)} images, {sum(len(b) for b in labels.values())} labels.")
    app.run(host='0.0.0.0', port=5051, debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--stage', type=int, default=None)
    args = parser.parse_args()
    main(args.project, args.stage)
