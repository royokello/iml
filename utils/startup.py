import os


def prepare_directories(root_path):
    # create knowledge dir with raw and graph subdir
    os.makedirs(os.path.join(root_path, 'knowledge', 'raw'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'knowledge', 'graph'), exist_ok=True)
    # create a logs dir
    os.makedirs(os.path.join(root_path, 'logs'), exist_ok=True)
    # create a recordings dir
    os.makedirs(os.path.join(root_path, 'recordings'), exist_ok=True)
    # create a skills dir
    os.makedirs(os.path.join(root_path, 'skills'), exist_ok=True)
    # create a tools dir
    os.makedirs(os.path.join(root_path, 'tools'), exist_ok=True)
   
