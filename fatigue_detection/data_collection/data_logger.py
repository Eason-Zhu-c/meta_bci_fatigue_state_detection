# data_logger.py
import os
from datetime import datetime


class DataLogger:
    def __init__(self, subject_name):
        self.folder_name = "data_collection_label"
        os.makedirs(self.folder_name, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.folder_name, f"subject_{subject_name}_{now}.txt")
        self.file = open(self.file_path, 'w')

    def write_start_time(self):
        start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"Data collection started at: {start_time_str}\n")
        self.file.flush()

    def write_prompt_shown(self, time_str):
        self.file.write(f"Prompt shown at: {time_str}\n")
        self.file.flush()

    def write_user_input(self, level, time_str):
        note = ""
        if int(level) == -1:
            level = 3
            note = " (timeout)"

        self.file.write(f"User input fatigue level: {level} at {time_str}{note}\n")
        self.file.flush()

    def write_end_time(self):
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"Data collection ended by user at: {end_time_str}\n")
        self.file.flush()

    def close(self):
        self.file.close()
