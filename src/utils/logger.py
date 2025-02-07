from loguru import logger
import os
import sys
import random
import string
from datetime import datetime

current_dir = os.path.dirname(__file__) # 获取当前文件的目录
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) # 获取上一级目录的路径
log_file_path = os.path.join(parent_dir, 'log', 'out.log') # 设置日志文件路径为上一级目录中的 log 文件夹下的 out.log


logger.add(log_file_path, level='INFO', rotation='10 MB', format='{time} | {level} | {message}')

def find_save_path(experiment_id):
    """
    Find the directory path for saving results associated with a given `experiment_id`. This function constructs a directory path within the
    'log/results' directory relative to the script's location.

    :param experiment_id: model identifier, (str).
    :return: directory path, (str).
    """
    root_path = parent_dir
    dir_path = os.path.join(root_path, "log", "results", experiment_id)
    return dir_path

def generate_id(name, target_stock):
    """
    Generate a unique experiment identifier based on the input `name` and the current timestamp in the format "YYYY-MM-DD_HH_MM_SS".
    Create a directory path using this identifier within the 'loggers/results'  directory relative to the script's location, and if
    it doesn't exist, create it.

    :param name: name of the DL model to be used in the experiment, (str).
    :return: experiment_id: unique experiment identifier, (str).
    """
    random_string_part = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(7))
    init_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_id = f"{target_stock}_{name}_{init_time}_{random_string_part}"

    root_path = sys.path[0]
    dir_path = f"{root_path}/log/results/{experiment_id}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return experiment_id