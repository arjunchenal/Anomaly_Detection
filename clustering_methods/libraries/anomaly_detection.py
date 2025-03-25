import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import time
from sklearn.preprocessing import MinMaxScaler


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
def test_single_id_timestamp(file_path, model, sequence_length, scaler):
    anomalies = []

    if file_path.find('.npy') != -1:
        test_data = np.load(file_path)
    else:
        test_data = read_traces(file_path)

    # Prepare sequences for the LSTM model
    X_test, y_test = [], []
    for i in range(0, len(test_data) - sequence_length, sequence_length):
        X_test.append(test_data[i:i + sequence_length])
        y_test.append(test_data[i + sequence_length])


    X_test, y_test = np.array(X_test), np.array(y_test)

    #Scaling the test data because the model was trained on scaled data
    X_test_new = X_test.reshape(-1,2)
    X_test_scaled = scaler.transform(X_test_new)
    X_test = X_test_scaled.reshape(X_test.shape)

    y_test_new = y_test.reshape(-1,2)
    y_test_scaled = scaler.transform(y_test_new)
    y_test = y_test_scaled.reshape(y_test.shape)

    start_time = time.time()
    predictions = model.predict(X_test)                               # Make predictions

    end_time = time.time()
    inference_time = (end_time - start_time) * 1000 # inference time in ms

    predictions = scaler.inverse_transform(predictions)
    predictions = np.round(predictions).astype(int)

    y_test = scaler.inverse_transform(y_test)
    y_test = np.round(y_test).astype(int)

    # print("predictions :", predictions)
    print("y_test :", y_test)

    errors = np.abs(predictions.flatten() - y_test.flatten())         # Calculate reconstruction errors
    for i in range(len(errors)):
        if errors[i] > 0:
            anomaly_seq_end_ind = (i * sequence_length) + sequence_length
            anomaly_seq_start_index = 0 if  i == 0  else anomaly_seq_end_ind - sequence_length + 1

            if anomaly_seq_end_ind < len(test_data):
                anomalies.append([
                    (test_data[anomaly_seq_start_index][0], test_data[anomaly_seq_end_ind][0]),
                    (test_data[anomaly_seq_start_index][1], test_data[anomaly_seq_end_ind][1]),
                    os.path.basename(file_path)
                ])
          
    return anomalies, inference_time



# To test a single test data file with a trained LSTM model to detect anomalies
def test_single_id(file_path, model, sequence_length, scaler):
    anomalies = []

    if file_path.find('.npy') != -1:
        test_data = np.load(file_path)
    else:
        test_data = read_traces(file_path)

    # Prepare sequences for the LSTM model
    X_test, y_test = [], []
    for i in range(0, len(test_data) - sequence_length, sequence_length):
        id_value = [int(trace[0]) for trace in test_data[i:i + sequence_length]]
        X_test.append(id_value)
        y_test.append(int(test_data[i + sequence_length][0]))

    X_test, y_test = np.array(X_test), np.array(y_test)

    X_test_new = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_new)
    X_test = X_test_scaled.reshape(X_test.shape)

    start_time = time.time()
    predictions = model.predict(X_test)                               # Make predictions

    predictions = np.round(predictions).astype(int)
    print("predictions :", predictions)

    end_time = time.time()
    inference_time = (end_time - start_time) * 1000 # inference time in ms
    predictions = np.round(predictions).astype(int)
    errors = np.abs(predictions.flatten() - y_test.flatten())         # Calculate reconstruction errors
    for i in range(len(errors)):
        if errors[i] > 0:
            anomaly_seq_end_ind = (i * sequence_length) + sequence_length
            anomaly_seq_start_index = 0 if  i == 0  else anomaly_seq_end_ind - sequence_length + 1

            if anomaly_seq_end_ind < len(test_data):
                anomalies.append([
                    (test_data[anomaly_seq_start_index][0], test_data[anomaly_seq_end_ind][0]),
                    (test_data[anomaly_seq_start_index][1], test_data[anomaly_seq_end_ind][1]),
                    os.path.basename(file_path)
                ])
          
    return anomalies, inference_time

def merge_detections(detections, diff_val=5):
    '''
    This fucntions merges multiple detections that are less the 2 seconds apart. 
    These multiple detections can be caused because of multiple variables or even multiple anomalies that are colser
    Each anomaly that occurs, affects group of variables, resulting in multiple detections for single ground truth
    This function groups the detections based on the time difference between them and selects one from each group

    detections: list of detected anomalies -> list -> in format: [[(var1, 0), (ts1, ts2), filename], [(var2, 0), (ts1, ts2), filename], ....]
    diff_val: time difference threshold in seconds to group detections -> int/float

    return:
    dedup_detection: list of deduplicated detections -> list
    '''
    DIFF_VAL = diff_val
    pred = detections
    ### sort the list using the first timestamp of every detection
    pred = sorted(pred, key=lambda x: x[1][0])
    print('sorted detecions:', pred)
    det_ts1 = [ x[1][0]/1000 for x in pred]  ### get first timestamp of every detection and convert from mili second to second
    det_ts2 = [ x[1][1]/1000 for x in pred]  ### get first timestamp of every detection and convert from mili second to second
    # print('merge ts:', pred[0][1], det_ts1[0], det_ts2[0])
    group = []
    group_ind = []
    aggregated_ts = []
    aggregated_ts_ind = []
    ymax = 0
    cond1 = False
    cond2 = False
    for xi, (x1, x2, y1, y2) in enumerate(zip(det_ts1[0:-1], det_ts1[1:], det_ts2[0:-1], det_ts2[1:])):    ### get the first and last timestamp of every detection
        # print(xi)
        ### diff between start points of two detections
        # diff_ts1 = abs(x2 - x1)
        ### diff between start of first detection and end of second detection
        if y1 > ymax:
            ymax = y1
                
        if group != []:
            cond1 = x2 < ymax
        else:
            cond1 = x2 < y1
        diff_ts2 = abs(x2 - y1)
        cond2 = diff_ts2 <= DIFF_VAL
        # print('Merge diff:', diff_ts1, x1, x2)
        ### decision to wether or not group detections. If the difference between the detections is less than diff_val seconds, then group them
        ### if the difference between the detections is more than diff_val seconds, 
        ### then check if the second detection has started before first detection ends or 
        ### the second detecion starts withing diff_val seconds from end of rist detection. If yes, then group them
        if cond1 or cond2:
            group += [pred[xi]]
            group_ind += [xi]
            if xi == len(det_ts1)-2:  ### for last pair
                group += [pred[xi+1]]
                group_ind += [xi+1]
                aggregated_ts_ind += [group_ind]
                aggregated_ts += [group]
            ### store the highest ts that shows end of the groupped detections
        else:
            group_ind += [xi]
            group += [pred[xi]]   ### group the predictions which have time diff less than 2 seconds
            # print(group)
            aggregated_ts_ind += [group_ind]   
            aggregated_ts += [group]    ### collect all the groups
            group = []
            ymax = 0
            if xi == len(det_ts1)-2:   ### for last pair
                group += [pred[xi+1]]
                group_ind += [xi+1]
                aggregated_ts_ind += [group_ind]
                aggregated_ts += [group]

    ### merge the detections (starting TS from first detection and ending TS from last detection from each group)
    merge_detection = []
    for gp in aggregated_ts:    #### read single group of detections. in format list of detections
        ### sort the detections in ascending order based on the first timestamp
        gp = sorted(gp, key=lambda x: x[1][0])
        #### scan through the group and get the lowest first timestamp and highest last timestamp
        lowest_ts = gp[0][1][0]
        highest_ts = gp[-1][1][1]
        first_event = gp[0][0][0]
        last_event = gp[-1][0][0]
        for i, (_, gts, _) in enumerate(gp):
            if gts[0] < lowest_ts:
                lowest_ts = gts[0]
                first_event = gp[i][0][0]
            if gts[1] > highest_ts:
                highest_ts = gts[1]
                last_event = gp[i][0][0]


        merge_detection += [[(first_event, last_event), (lowest_ts, highest_ts), gp[0][2]]]    
        ### eg. [(10, 8), (27249, 30675), 'trace2-sensor'], where first tuple contains variable of first and last detection in group, second tuple is the first TS of first detection and second ts of last detection, and the file name is taken from first detection

    return merge_detection, aggregated_ts


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
