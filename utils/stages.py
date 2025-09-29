import os

def find_latest_stage(project: str) -> int:
    stage_dirs = []
    for item in os.listdir(project):
        if os.path.isdir(os.path.join(project, item)) and item.startswith('stage_'):
            try:
                stage_num = int(item.split('_')[1])
                stage_dirs.append(stage_num)
            except (IndexError, ValueError):
                continue
    
    if stage_dirs:
        return max(stage_dirs)
    else:
        raise ValueError(f"No stage directories found in {project}. Create at least one stage directory (e.g., 'stage_1').")