# image/crop/train_yolo.py
import argparse
import os
import shutil
from ultralytics import YOLO
from utils.stages import find_latest_stage

def infer_run_from_model_path(model_path: str):
    abspath = os.path.abspath(model_path)
    weights_dir = os.path.dirname(abspath)
    run_dir = os.path.dirname(weights_dir)
    name = os.path.basename(run_dir)
    project_dir = os.path.dirname(run_dir)
    results_csv = os.path.join(run_dir, "results.csv")
    if os.path.isfile(results_csv):
        return project_dir, name, run_dir
    return None, None, None

def ensure_paths(project: str, stage: int):
    data_yaml = os.path.join(project, f"stage_{stage}_yolo.yaml")
    runs_root = os.path.join(project, f"stage_{stage}_yolo_runs")
    ul_project = os.path.join(runs_root, "detect")
    os.makedirs(ul_project, exist_ok=True)
    if not os.path.isfile(data_yaml):
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")
    return data_yaml, ul_project

def copy_fixed_outputs(run_dir: str, project: str, stage: int, variant: str):
    best_src = os.path.join(run_dir, "weights", "best.pt")
    best_dst = os.path.join(project, f"stage_{stage}_crop_{variant}.pt")
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)

    csv_src = os.path.join(run_dir, "results.csv")
    csv_dst = os.path.join(project, f"stage_{stage}_crop_{variant}_epoch_log.csv")
    if os.path.exists(csv_src):
        shutil.copy2(csv_src, csv_dst)
    return best_dst, csv_dst

def latest_run_dir(ul_project: str):
    if not os.path.isdir(ul_project):
        return None
    candidates = [
        os.path.join(ul_project, d)
        for d in os.listdir(ul_project)
        if os.path.isdir(os.path.join(ul_project, d))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--model", type=str, default=None, help="absolute path to a trained model (for resume/continue)")
    ap.add_argument("--variant", required=True, help="e.g. yolo11n, yolov8n, ...")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    stage = args.stage or find_latest_stage(args.project)
    data_yaml, ul_project = ensure_paths(args.project, stage)

    resume = False
    model_path = None
    ul_name = None

    if args.model:
        proj_found, name_found, run_found = infer_run_from_model_path(args.model)
        if proj_found and name_found and run_found:
            ul_project = proj_found
            ul_name = name_found
            model_path = args.model
            resume = True
        else:
            model_path = args.model
            resume = False
    else:
        model_path = f"{args.variant}.pt"
        resume = False

    # Train
    model = YOLO(model_path)
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project=ul_project,
        name=ul_name,
        resume=resume,
        device="cuda",
        verbose=True,
        rect=True,
        augment=False,
        patience=16,
    )

    # figure out which run directory to pull artifacts from
    run_dir = os.path.join(ul_project, ul_name) if ul_name else latest_run_dir(ul_project)
    if not run_dir or not os.path.isdir(run_dir):
        raise SystemExit("Could not locate the Ultralytics run directory to collect outputs.")

    best_out, csv_out = copy_fixed_outputs(run_dir, args.project, stage, args.variant)
    print("Training complete.")
    print(f"Best model → {best_out}")
    print(f"Epoch log  → {csv_out}")
    print(f"Run dir    → {run_dir}")

if __name__ == "__main__":
    main()
