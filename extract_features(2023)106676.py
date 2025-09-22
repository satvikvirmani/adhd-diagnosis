import os
import pandas as pd
from tqdm import tqdm
import matlab.engine

from helpers.python.extraction_pipe import extract_info

eng = matlab.engine.start_matlab()
eng.clear(nargout=0)
eng.addpath(r'/Users/satvik/Documents/projects/adhd/helpers/matlab', nargout=0)
print("MATLAB started. Test sqrt(16):", eng.sqrt(16.0))

adhd_dir1 = "data/ADHD_part1"
adhd_dir2 = "data/ADHD_part2"
control_dir1 = "data/Control_part1"
control_dir2 = "data/Control_part2"

final_file = pd.DataFrame()

for file in tqdm(os.listdir(adhd_dir1)):
    if not file.endswith(".mat"):
        continue
    filepath = os.path.join(adhd_dir1, file)
    time_length = eng.get_time_length(filepath)
    print(f"Processing {file}, Time length: {time_length} seconds")
    # final_features = extract_info(eng, filepath, 1)
    # final_file = pd.concat([final_file, final_features], ignore_index=True)

for file in tqdm(os.listdir(adhd_dir2)):
    if not file.endswith(".mat"):
        continue
    filepath = os.path.join(adhd_dir2, file)
    time_length = eng.get_time_length(filepath)
    print(f"Processing {file}, Time length: {time_length} seconds")
    # final_features = extract_info(eng, filepath, 1)
    # final_file = pd.concat([final_file, final_features], ignore_index=True)

for file in tqdm(os.listdir(control_dir1)):
    if not file.endswith(".mat"):
        continue
    filepath = os.path.join(control_dir1, file)
    time_length = eng.get_time_length(filepath)
    print(f"Processing {file}, Time length: {time_length} seconds")
    # final_features = extract_info(eng, filepath, 0)
    # final_file = pd.concat([final_file, final_features], ignore_index=True)

for file in tqdm(os.listdir(control_dir2)):
    if not file.endswith(".mat"):
        continue
    filepath = os.path.join(control_dir2, file)
    time_length = eng.get_time_length(filepath)
    print(f"Processing {file}, Time length: {time_length} seconds")
    # final_features = extract_info(eng, filepath, 0)
    # final_file = pd.concat([final_file, final_features], ignore_index=True)

final_file.to_csv("data/features_extracted.csv", index=False)