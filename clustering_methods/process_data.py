import pandas as pd
import json
import os

def process_trace_file(filepath, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data, columns=["ID", "Timestamp"])
    df["Timestamp_Diff"] = df["Timestamp"].diff().fillna(0).astype(int)
    timestamp_diffs = df["Timestamp_Diff"].tolist()[1:]  
    ids = df["ID"].tolist()[1:]

    # Finding how many chunks of 50 elements we need to create
    if (len(ids) % 50) == 0:
        num_lengths = len(ids) // 50
    else:
        num_lengths = len(ids) // 50 + 1

    for i in range(num_lengths):
        id_trim = ids[i * 50:(i + 1) * 50]
        time_diff_trim = timestamp_diffs[i * 50:(i + 1) * 50]

        if id_trim and time_diff_trim:
            data = [id_trim, time_diff_trim]
            output_filepath = os.path.join(output_dir, f"{i + 1}.json")
            with open(output_filepath, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Saved: {output_filepath}")

def process_multiple_trace_files(filepaths, output_base_dir):
    for file_idx, filepath in enumerate(filepaths, start=1):
        trace_output_dir = os.path.join(output_base_dir, f"trace_{file_idx}")
        process_trace_file(filepath, trace_output_dir)

        
train_data_path = r"C:\Uni Bremen\Job\Comnets\Anomaly Detection\Anomaly_Detection\trace_data\theft_protection\single_thread\version_3\normal\train_data/"
train_data_path_files = [os.path.join(train_data_path, f) for f in os.listdir(train_data_path) if f.endswith(".json")]
print("The files are : ", train_data_path_files)


process_multiple_trace_files(train_data_path_files,"output_folder")
