import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler   
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import LSTM, Dense, Dropout   
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay  
from collections import defaultdict


# Read trace files and extracts event data
# Paramters : trace_path (str) path to trace file
def read_traces(trace_path):
    with open(trace_path, 'r') as f:
        data = json.load(f)
    return data


def load_data(file_paths):
    data = []
    for file in file_paths:
        print("File from load data : ", file)
        traces = read_traces(file)
        if isinstance(traces, list):
            data.append(traces)
    return data


# To test a single test data file with a trained LSTM model to detect anomalies
def test_single(file_path, model, sequence_length):
    anomalies = []

    test_data = load_data([file_path])[0]                       # Load test data

    # Prepare sequences for the LSTM model
    X_test, y_test = [], []
    for i in range(0, len(test_data) - sequence_length, sequence_length):
        id_value = [int(trace[0]) for trace in test_data[i:i + sequence_length]]
        X_test.append(id_value)
        y_test.append(int(test_data[i + sequence_length][0]))

    X_test, y_test = np.array(X_test), np.array(y_test)
    predictions = model.predict(X_test)                               # Make predictions
    predictions = np.round(predictions).astype(int)
    errors = np.abs(predictions.flatten() - y_test.flatten())         # Calculate reconstruction errors

    for i in range(len(errors)):
        if errors[i] > 0:
            anomaly_id = (i * sequence_length) + sequence_length
            if anomaly_id < len(test_data):
                anomalies.append([
                    test_data[anomaly_id][0],
                    test_data[anomaly_id][1],
                    os.path.basename(file_path)
                ])

    print("\nAnomalies detected:", anomalies)
    return anomalies
   
    
def merge_detections(detections, diff_val=5):
    '''
    Merge detections that are within a certain time difference
    :param all_detections: list of tuples (filename, detections, labels) where detections is a list of tuples (id, timestamp)
    :param diff_val: time difference in seconds
    :return: list of merged detections
    '''
    sorted_detections = sorted(detections, key=lambda x: x[1])
    timestamps = [x[1] / 1000 for x in sorted_detections]  # Convert to seconds

    merged_detections = []
    current_group = []

    for i, detection in enumerate(sorted_detections):
        if not current_group:
            current_group.append(detection)
        else:
            prev_ts = timestamps[i - 1]
            curr_ts = timestamps[i]
            if abs(curr_ts - prev_ts) <= diff_val:
                current_group.append(detection)
            else:
                ts_start = current_group[0][1]
                ts_end = current_group[-1][1]
                id_start = current_group[0][0]
                filename = current_group[0][2]
                merged_detections.append([(id_start, 0), (ts_start, ts_end), filename])
                current_group = [detection]

    if current_group:
        ts_start = current_group[0][1]
        ts_end = current_group[-1][1]
        id_start = current_group[0][0]
        filename = current_group[0][2]
        merged_detections.append([(id_start, 0), (ts_start, ts_end), filename])

    return merged_detections



def bbox_iou(b1_x1, b1_x2, b2_x1, b2_x2,):
    """
    Returns the IoU of two bounding boxes
    """

    # Get the coordinates of bounding boxes
    b1_y1 = b2_y1 = 0
    b1_y2 = b2_y2 = 1

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 , a_min=0, a_max=None) * np.clip(
        inter_rect_y2 - inter_rect_y1 , a_min=0, a_max=None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 ) * (b1_y2 - b1_y1 )
    b2_area = (b2_x2 - b2_x1 ) * (b2_y2 - b2_y1 )
    print('inter:', inter_area)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou




def get_correct_detections(detection, ground_truth):
        '''
        detection -> list: detections from the  -> [[(var1, 0), (ts1, ts2), file_name], [], [], ...., []]
        ground_truth -> list: ground truth labels -> [[(ind1, ind2), (ts1, ts2), class], [], [], ...., []]

        return:
        y_pred -> list: [1, 1, 0, 1, 0, 0, ...., 1]
        y_true -> list: [1, 1, 1, 1, 0, 0, ...., 0]
        '''
        
        gt_pred = defaultdict(list)      ### list of detections for each gt instance. The index of list denote its respective pred at that index, and list contains gt for that pred.
        rest_pred = [] ### list of detections that are not associated with any gt instance
        correct_pred = [] ### list of correct predictions
        false_negatives = [] ### list of false negatives
        y_true_ind = []
        y_pred_ind = []
        y_true = []
        y_pred = []
        # print('gt_pred:', gt_pred)
        # print(y_pred, y_true)

        if len(ground_truth) != 0:
            for gt_ind ,gt in enumerate(ground_truth):
                ind1 = gt[0]
                ind2 = gt[1]
                gt_ts1 = gt[2]
                gt_ts2 = gt[3]
                class_label = gt[4]

                if len(detection) != 0:
                    tmp_pred = []
                    for im, pred in enumerate(detection):
                        state1, state2 = pred[0]
                        pd_ts1, pd_ts2 = pred[1]
                        filename = pred[2]
                        # print('(gt_ts1, gt_ts2), (pd_ts1, pd_ts2)', (gt_ts1, gt_ts2), (pd_ts1, pd_ts2))

                        cond_1 = pd_ts1 >= gt_ts1 and pd_ts2 <= gt_ts2  ### check if the detection timestamp is within the ground truth timestamp (case 1)
                        cond_2 = pd_ts1 <= gt_ts1 and pd_ts2 >= gt_ts2  ### check if the gorund truth timestamp is within the detection timestamp (case 2)
                        cond_3 = pd_ts1 >= gt_ts1 and pd_ts1 <= gt_ts2 and pd_ts2 >= gt_ts2    ### partial detection on right of the ground truths, check 5 second difference after this (case 3)
                        cond_4 = pd_ts2 <= gt_ts2 and pd_ts2 >= gt_ts1 and pd_ts1 <= gt_ts1   ### partial detection on left of the ground truths, check 5 second difference after this (case 4)

                        
                        if cond_1 or cond_2 or cond_3 or cond_4:
                            print(gt_ind, im, cond_1, cond_2, cond_3, cond_4)
                            tmp_pred += [(im, pred, cond_1, cond_2, cond_3, cond_4)]    ### store all correct predictions that match with current gt      ### if cond_1 is TRUE, that means the detection is inside the gt and even multiple pred can be correct                  

                    if tmp_pred != []:
                        # print('tmp_pred', tmp_pred)
                        y_true_ind += [gt_ind]
                        ### if there are multiple correct predictions for a single gt, then choose the one that is closest to gt
                        if len(tmp_pred) > 1:
                            # print('if:', tmp_pred)
                            iou_pred = []
                            for ip, pred, case_1, case_2, case_3, case_4 in tmp_pred:
                                print('ip, pred', ip, pred)
                                state1, state2 = pred[0]
                                pd_ts1, pd_ts2 = pred[1]
                                filename = pred[2]
                                ### get IOU for all detections with gt
                                if not case_1:
                                    iou = bbox_iou(gt_ts1, gt_ts2, pd_ts1, pd_ts2)    ### calculate the IOU between the ground truth and the detection to evaluate which is the best detection for the given gt
                                else:
                                    iou = 1.0
                                iou_pred += [iou]

                            ### include all the detection with case1, even if there are multiple detections, as they are all correct
                            perfect_pred = False
                            for io, iou in enumerate(iou_pred):
                                if iou == 1.0:
                                    best_pred = tmp_pred[io]
                                    iou_pred[io] = 0
                                    if best_pred[0] not in y_pred_ind:
                                        perfect_pred = True
                                        y_pred_ind += [best_pred[0]]
                                        correct_pred += [best_pred[1]]
                                        y_true += [1]
                                        y_pred += [1]

                            if not perfect_pred:    ### skip selecting the best detection if there is a perfect detection (case 1)
                                best_pred_ind = iou_pred.index(max(iou_pred))
                                best_pred = tmp_pred[best_pred_ind]
                                print('ground_truth', gt)
                                print('best_pred:', best_pred)
                                print('y_pred_ind:', y_pred_ind)
                                # gt_pred[gt_ind] += [pred]  ### store the best detection for the given gt
                                if best_pred[0] not in y_pred_ind:
                                    y_pred_ind += [best_pred[0]]
                                    correct_pred += [best_pred[1]]
                                    y_true += [1]
                                    y_pred += [1]
                        else:
                            # print('else:', tmp_pred)
                            print('y_pred_ind:', y_pred_ind)
                            if tmp_pred[0][0] not in y_pred_ind:
                                y_pred_ind += [tmp_pred[0][0]]
                                correct_pred += [tmp_pred[0][1]]  
                                y_true += [1]
                                y_pred += [1]     
                    else:
                        ### this means no detection for this gt, denots FN
                        y_true += [1]
                        y_pred += [0]
                        false_negatives += [gt]
            
        ### calculate FP
        for im, pred in enumerate(detection):
            if im not in y_pred_ind:
                rest_pred += [pred]
                y_true += [0]
                y_pred += [1]

        return correct_pred, rest_pred, y_pred, y_true, false_negatives


def get_correct_detections_id_timestamp(detection, ground_truth):
    '''
    detection -> list: detections from the  -> [[(var1, 0), (ts1, ts2), file_name], [], [], ...., []]
    ground_truth -> list: ground truth labels -> [[ind1, ind2, ts1, ts2, class], [], [], ...., []]

    return:
    y_pred -> list: [1, 1, 0, 1, 0, 0, ...., 1]
    y_true -> list: [1, 1, 1, 1, 0, 0, ...., 0]
    '''
    rest_pred = []  # Detections not associated with any gt
    correct_pred = []  # Correct predictions
    false_negatives = []  # False negatives
    y_true_ind = []  # Ground truth indices matched
    y_pred_ind = []  # Prediction indices matched
    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels

    if len(ground_truth) != 0:
        for gt_ind, gt in enumerate(ground_truth):
            ind1, ind2, gt_ts1, gt_ts2, class_label = gt  # Unpack ground truth

            if len(detection) != 0:
                tmp_pred = []
                for im, pred in enumerate(detection):
                    state1, _ = pred[0]  # Unpack ID (state1) from (var1, 0), ignore 0
                    pd_ts1, pd_ts2 = pred[1]  # Detection timestamps
                    filename = pred[2]

                    # Timestamp overlap conditions
                    cond_1 = pd_ts1 >= gt_ts1 and pd_ts2 <= gt_ts2  # Detection inside GT
                    cond_2 = pd_ts1 <= gt_ts1 and pd_ts2 >= gt_ts2  # GT inside detection
                    cond_3 = pd_ts1 >= gt_ts1 and pd_ts1 <= gt_ts2 and pd_ts2 >= gt_ts2  # Partial right
                    cond_4 = pd_ts2 <= gt_ts2 and pd_ts2 >= gt_ts1 and pd_ts1 <= gt_ts1  # Partial left

                    # ID match condition: state1 must be within the range [ind1, ind2]
                    id_match = ind1 <= state1 <= ind2

                    # Combine timestamp and ID conditions
                    if (cond_1 or cond_2 or cond_3 or cond_4) and id_match:
                        print(gt_ind, im, cond_1, cond_2, cond_3, cond_4, id_match)
                        tmp_pred += [(im, pred, cond_1, cond_2, cond_3, cond_4)]  # Store matching predictions

                if tmp_pred != []:
                    y_true_ind += [gt_ind]
                    if len(tmp_pred) > 1:  # Multiple matches, use IoU to pick best
                        iou_pred = []
                        for ip, pred, case_1, case_2, case_3, case_4 in tmp_pred:
                            print('ip, pred', ip, pred)
                            state1, _ = pred[0]
                            pd_ts1, pd_ts2 = pred[1]
                            filename = pred[2]
                            if not case_1:  # If not perfect containment, calculate IoU
                                iou = bbox_iou(gt_ts1, gt_ts2, pd_ts1, pd_ts2)
                            else:
                                iou = 1.0
                            iou_pred += [iou]

                        perfect_pred = False
                        for io, iou in enumerate(iou_pred):
                            if iou == 1.0:
                                best_pred = tmp_pred[io]
                                iou_pred[io] = 0  # Mark as used
                                if best_pred[0] not in y_pred_ind:
                                    perfect_pred = True
                                    y_pred_ind += [best_pred[0]]
                                    correct_pred += [best_pred[1]]
                                    y_true += [1]
                                    y_pred += [1]

                        if not perfect_pred:  # Pick best IoU if no perfect match
                            best_pred_ind = iou_pred.index(max(iou_pred))
                            best_pred = tmp_pred[best_pred_ind]
                            print('ground_truth', gt)
                            print('best_pred:', best_pred)
                            print('y_pred_ind:', y_pred_ind)
                            if best_pred[0] not in y_pred_ind:
                                y_pred_ind += [best_pred[0]]
                                correct_pred += [best_pred[1]]
                                y_true += [1]
                                y_pred += [1]
                    else:  # Single match
                        print('y_pred_ind:', y_pred_ind)
                        if tmp_pred[0][0] not in y_pred_ind:
                            y_pred_ind += [tmp_pred[0][0]]
                            correct_pred += [tmp_pred[0][1]]
                            y_true += [1]
                            y_pred += [1]
                else:
                    # No match for this ground truth (False Negative)
                    y_true += [1]
                    y_pred += [0]
                    false_negatives += [gt]

    # Calculate False Positives
    for im, pred in enumerate(detection):
        if im not in y_pred_ind:
            rest_pred += [pred]
            y_true += [0]
            y_pred += [1]

    return correct_pred, rest_pred, y_pred, y_true, false_negatives



# Load test data
def load_data_separate_sequence(file_paths):
    id_data = []
    timestamp_data = []
    original_timestamp = []

    for file in file_paths:
        traces = read_traces(file)
        if isinstance(traces, list):
            id_sequence = [int(trace[0]) for trace in traces]
            timestamp_sequence = [int(trace[1]) for trace in traces]

            time_difference = []
            for i in range(1, len(timestamp_sequence)):
                difference = timestamp_sequence[i] - timestamp_sequence[i-1]
                time_difference.append(difference)


            new_id_sequence = id_sequence[1:]
            new_timestamp_sequence = timestamp_sequence[1:]
            id_data.append(new_id_sequence)
            timestamp_data.append(time_difference)
            original_timestamp.append(new_timestamp_sequence)
            
    return id_data, timestamp_data, original_timestamp



def test_single_id_timestamp(file_path, id_model, timestamp_model, sequence_length):
    """
    Test a single test data file with trained LSTM models to detect anomalies in ID and timestamp difference sequences.

    :param file_path: Path to the test data file.
    :param id_model: Trained LSTM model for ID sequences.
    :param timestamp_model: Trained LSTM model for timestamp difference sequences.
    :param sequence_length: Length of the sequence for LSTM.
    :return: List of anomalies -> format [detected_id, actual_timestamp, original_timestamp, filename, anomaly_type].
    """
    anomalies = []

    # Load test data for a single file (no need to handle lists)
    id_data, timestamp_data, actual_timestamps = load_data_separate_sequence([file_path])



    id_data = id_data[0]  # Extract first (and only) item
    timestamp_data = timestamp_data[0]
    actual_timestamps = actual_timestamps[0]

    # Prepare sequences for the ID model
    X_test_id, y_test_id = [], []
    for i in range(0, len(id_data) - sequence_length, sequence_length):
        X_test_id.append(id_data[i:i + sequence_length])
        y_test_id.append(id_data[i + sequence_length])

    # Prepare sequences for the timestamp difference model
    X_test_timestamp, y_test_timestamp = [], []
    for i in range(0, len(timestamp_data) - sequence_length, sequence_length):
        X_test_timestamp.append(timestamp_data[i:i + sequence_length])
        y_test_timestamp.append(timestamp_data[i + sequence_length])

    # Convert to numpy arrays
    X_test_id, y_test_id = np.array(X_test_id), np.array(y_test_id)
    X_test_timestamp, y_test_timestamp = np.array(X_test_timestamp), np.array(y_test_timestamp)

    # Make predictions
    id_predictions = np.round(id_model.predict(X_test_id, verbose=0)).astype(int)
    timestamp_predictions = np.round(timestamp_model.predict(X_test_timestamp, verbose=0)).astype(int)


    

    # Calculate errors
    id_errors = np.abs(id_predictions.flatten() - y_test_id.flatten())
    timestamp_errors = np.abs(timestamp_predictions.flatten() - y_test_timestamp.flatten())

    # Detect anomalies for ID model
    for i in range(len(id_errors)):
        if id_errors[i] > 0:
            anomaly_index = (i * sequence_length) + sequence_length
            if anomaly_index < len(id_data):
                anomalies.append([
                    anomaly_index,  # Detected ID
                    actual_timestamps[anomaly_index],  
                    os.path.basename(file_path)
                ])

    # Detect anomalies for Timestamp Difference model
    for i in range(len(timestamp_errors)):
        if timestamp_errors[i] > 0:
            anomaly_index = (i * sequence_length) + sequence_length
            if anomaly_index < len(timestamp_data):
                anomalies.append([
                    anomaly_index,  # Detected ID
                    actual_timestamps[anomaly_index],  
                    os.path.basename(file_path)
                ])               

    print("\nAnomalies detected:", anomalies)
    return anomalies    