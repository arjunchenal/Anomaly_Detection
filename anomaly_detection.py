import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler   
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import LSTM, Dense, Dropout   
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay  


# Read trace files and extracts event data
# Paramters : trace_path (str) path to trace file
def read_traces(trace_path):
    with open(trace_path, 'r') as f:
        data = json.load(f)
    return data


# To test a single test data file with a trained LSTM model to detect anomalies
def test_single(file_path, model, scaler, sequence_length, threshold):
    anomalies = []
    if file_path.find('.npy') != -1:
        sample_data = np.load(file_path)
    else:
        sample_data = read_traces(file_path)

    sample_data_scaled = scaler.transform(sample_data)              # Scaling the test data

    # Prepare sequences for the LSTM model
    X_test, y_test = [], []
    for i in range(len(sample_data_scaled) - sequence_length):
        X_test.append(sample_data_scaled[i:i + sequence_length])
        y_test.append(sample_data_scaled[i + sequence_length])
    X_test, y_test = np.array(X_test), np.array(y_test)

    predictions = model.predict(X_test)                             # Make predictions
    errors = np.mean(np.abs(predictions - y_test), axis=1)          # Calculate reconstruction errors

    # Determine anomalies based on the threshold
    for i in range(len(errors)):
        if errors[i] > threshold:
            start_idx = i
            end_idx = i + sequence_length
            anomalies.append([(start_idx, end_idx), (sample_data[start_idx][1], sample_data[end_idx][1]), os.path.basename(file_path)])

    return anomalies
    
    
# To merge overlapping or closely coming anomaly detections
# Parameters : detections of type list -> List of detected anomalies
# diff_val(int) -> time difference threshold in seconds to merge detections    
def merge_detections(detections, diff_val):
    DIFF_VAL = diff_val
    pred = detections
    pred = sorted(pred, key=lambda x: x[1][0])              # Sort the list using the first timestamp of every detection

    # Group detections based on time difference
    group = []
    group_ind = []
    aggregated_ts = []
    aggregated_ts_ind = []
    ymax = 0

    for xi, (x1, x2, y1, y2) in enumerate(zip([x[1][0] for x in pred], [x[1][0] for x in pred[1:]], [x[1][1] for x in pred], [x[1][1] for x in pred[1:]])):
        if y1 > ymax:
            ymax = y1

        cond1 = x2 < ymax if group else x2 < y1
        diff_ts2 = abs(x2 - y1)
        cond2 = diff_ts2 <= DIFF_VAL

        if cond1 or cond2:
            group.append(pred[xi])
            group_ind.append(xi)
            if xi == len(pred) - 2:  
                group.append(pred[xi + 1])
                group_ind.append(xi + 1)
                aggregated_ts_ind.append(group_ind)
                aggregated_ts.append(group)
        else:
            group_ind.append(xi)
            group.append(pred[xi])
            aggregated_ts_ind.append(group_ind)
            aggregated_ts.append(group)
            group = []
            ymax = 0
            if xi == len(pred) - 2:  
                group.append(pred[xi + 1])
                group_ind.append(xi + 1)
                aggregated_ts_ind.append(group_ind)
                aggregated_ts.append(group)

    merge_detection = []
    for gp in aggregated_ts:
        gp = sorted(gp, key=lambda x: x[1][0])
        lowest_ts = gp[0][1][0]
        highest_ts = gp[-1][1][1]
        first_event = gp[0][0][0]
        last_event = gp[-1][0][0]

        merge_detection.append([(first_event, last_event), (lowest_ts, highest_ts), gp[0][2]])

    return merge_detection, aggregated_ts

# To evaluate detection accuracy by comparing with ground truth values
# Parameters : detection(list) -> List of detected anomalies
# ground_truth(list) -> List of actual anomalies    
def get_correct_detections(detection, ground_truth):
    correct_pred = []   # True detections
    rest_pred = []      # False positives
    false_neg = []      # False negatives
    y_pred = []
    y_true = []
    
    for gt in ground_truth:
        gt_start, gt_end, gt_ts1, gt_ts2, gt_class = gt
        gt_set = set(range(gt_start, gt_end + 1))  
        matched_elements = set()
        matched = False
        
        print(f"\nGround Truth: ID Range=({gt_start}, {gt_end}), Timestamps=({gt_ts1}, {gt_ts2}), Class={gt_class}")
        
        for det in detection:
            det_start, det_end = det[0]
            det_ts1, det_ts2 = det[1]
            det_set = set(range(det_start, det_end + 1))  
            overlap = gt_set & det_set  
            if overlap:
                print(f"Detection: ID Range=({det_start}, {det_end}), Timestamps=({det_ts1}, {det_ts2}), Overlap={overlap}")
                if gt_set == det_set:
                    correct_pred.append(det)
                    y_pred.append(1)
                    y_true.append(1)
                    matched = True
                    break
                
                # If partial match, append only the overlapping elements
                else:
                    matched_elements.update(overlap)
                    matched = True

        if matched:
            for elem in matched_elements:
                correct_pred.append(((elem, elem), (gt_ts1, gt_ts2)))  # Saving each matched element as a separate tuple
                y_pred.append(1)
                y_true.append(1)

        unmatched_elements = gt_set - matched_elements
        for elem in unmatched_elements:
            false_neg.append(((elem, elem), (gt_ts1, gt_ts2)))
            y_pred.append(0)
            y_true.append(1)
    
    # For false positives
    for det in detection:
        det_start, det_end = det[0]
        det_set = set(range(det_start, det_end + 1))
        unmatched_elements = det_set - {elem[0][0] for elem in correct_pred} 
        for elem in unmatched_elements:
            rest_pred.append(((elem, elem), (det[1][0], det[1][1])))
            y_pred.append(1)
            y_true.append(0)
    
    return correct_pred, rest_pred, y_pred, y_true, false_neg
