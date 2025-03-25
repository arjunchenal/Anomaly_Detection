import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from collections import defaultdict 

import configparser
from io import StringIO


#### config for plotting #####
FONTSIZE = 15
# PLOTWIDTH = 2000
PLOTHEIGHT = 1500



def get_config(file_name='theft_protection_config'):
    '''
    read the configuration file and extract the values
    file_name: path to the configuration file -> str without extension
    return:
    CODE: code name -> str
    BEHAVIOUR: behaviour name -> str
    THREAD: thread name -> str
    VER: version number -> int
    '''

    # Read the file
    with open(f'libv3/{file_name}.txt', 'r') as f:
        config_string = '[dummy_section]\n' + f.read()

    # Parse the configuration
    config = configparser.ConfigParser(allow_no_value=True, comment_prefixes='#')
    config.read_string(config_string)

    # Get the configuration values
    CODE = config.get('dummy_section', 'CODE')
    CODE = CODE.split('#')[0].strip()
    BEHAVIOUR = config.get('dummy_section', 'BEHAVIOUR')
    BEHAVIOUR = BEHAVIOUR.split('#')[0].strip()
    THREAD = config.get('dummy_section', 'THREAD')
    THREAD = THREAD.split('#')[0].strip()   

    # Extract the integer part of the 'VER' value
    ver_string = config.get('dummy_section', 'VER')
    VER = int(ver_string.split('#')[0])

    print(f'CODE: {CODE}')
    print(f'BEHAVIOUR: {BEHAVIOUR}')
    print(f'THREAD: {THREAD}')
    print(f'VER: {VER}')

    return CODE, BEHAVIOUR, THREAD, VER


def get_paths(log_path: str):
    '''
    log_path: path to the folder containing the log files -> str

    return: 
    paths to the log files -> list;
    traces files -> list;
    varlist files -> list;
    '''
    label_path = os.path.join(log_path, 'labels')
    if os.path.exists(label_path):
        labels = os.listdir(label_path)
    else:
        labels = []
    all_files = os.listdir(log_path)
    all_files.sort()
    logs = []
    traces = []
    varlist = []
    unknown = []
    for i in all_files:
        # if i.find('label') == 0:
        #     labels += [i]
        if i.find('log') == 0:
            logs += [i]
        elif i.find('trace') == 0 and i.find('.txt') == -1:
            traces += [i]
        elif i.find('varlist') == 0:
            varlist += [i]
        else:
            unknown += [i]

    ######### path to files
    paths_log = [os.path.join(log_path, x) for x in logs]
    paths_traces = [os.path.join(log_path, x) for x in traces]
    varlist_path = [os.path.join(log_path,x) for x in varlist]
    paths_label = [os.path.join(label_path, x) for x in labels]
    paths_log.sort()
    paths_traces.sort()
    varlist_path.sort()
    paths_label.sort()

    return paths_log, paths_traces, varlist_path, paths_label


def read_json(path: str):
    '''
    read the trace files and extract variable names
    data: json format
    '''
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, path):
    '''
    save the data in json format
    data: data to save -> dict
    path: path to save the data -> str

    return:
    None
    '''
    with open(path, 'w') as f:
        json.dump(data, f)

def read_traces(trace_path):
    '''
    read the trace files and extract variable names
    trace_path: path to the trace files -> str

    return:
    data = [ [event, timestamp], [], [],......,[] ]
    '''
    with open(trace_path, 'r') as f:
        data = json.load(f)
    return data


def load_sample(file_path):
    '''
    for numpy files
    '''
    data = np.load(file_path, allow_pickle=False)
    return data


def is_consistent(varlist_paths: list):
    '''
    check if the varlists for all traces are consistent
    varlist_paths: list of paths to the varlist files -> list

    return:
    True, varlist -> bool, dict
    '''
    
    varlist = read_json(varlist_paths[0])

    if len(varlist_paths) == 1:
        return True, varlist
    # print(varlist)
    inconsistent = []
    for ind, varlist_p in enumerate(varlist_paths[1:]):
        varlist_ = read_json(varlist_p)
        if varlist != varlist_:
            print(f'varlist {ind+1} is not consistent varlist 0')
            inconsistent += [varlist_p]
        else:
            print(f'varlist {ind+1} is consistent with varlist 0')
    
    if inconsistent != []:
        return False, inconsistent
    else:
        return True, varlist


def mapint2var(vartoint):
        '''
        map the integers back to variable names
        vartoint: dict of variables to map to integer -> dict

        return:
        dict of integers to map to variable names -> dict
        '''
        inttovar = {}
        for key, value in vartoint.items():
            inttovar[value] = key
        return inttovar


def preprocess_traces(paths_traces, var_list=None):
    '''
    This function takes multiple trace paths as input and stores it in one place
    var_list: list of variable names -> list
    paths_traces: list of paths to the trace files -> list(str)

    return:
    col_data: list of trace data -> list
    '''

    col_data = []
    for p in paths_traces:
        trace = read_traces(p)
        w = p.split('/')[-1].split('.')[0]
        print(p,w)
        num_trace = []
        time_stamp = []
        for (t, ts) in trace:
            num_trace.extend([t])
            time_stamp.extend([ts])
            # ### take limited samples
            # if ts > 250000:
            #     break
        col_data += [(w, time_stamp, num_trace, var_list, p)]   ### in the format (trace_name, x_data, y_data, y_labels, trace_path) 
    return col_data


def get_dataframe(col_data):
    '''
    This function takes the trace data and converts it into a data frame for plotting
    col_data: list of trace data -> list

    return:
    all_df: list of data frames, one for each trace -> list
    '''

    all_df = []
    for col in col_data:
        # print(col)
        plot_data = dict()
        plot_data['time'] = col[1]   ### x_data
        plot_data[col[0]] = col[2]   ### y_data (traces)

        ### convert the list to data frame and store it for plotting
        df = pd.DataFrame(plot_data)
        all_df += [df]

    return all_df


def plot_single_trace(df, 
                      var_list, 
                      with_time=False, 
                      ground_truths=None, 
                      is_xticks=False, 
                      gt_classlist=['gt_communication', 'gt_sensor', 'gt_bitflip', 'gt_unhandled-interupt', 'expected behaviour'],
                      detections=None,
                      dt_classlist=['detection'],
                      ):
    '''
    This function plots the traces with dataframe as input
    ground_truths: list of ground_truths -> list, format: ( list([start index, end index]), list([start timestamp, end timestamp]), list(class) )
    
    return:
    fig: plotly figure -> go.Figure
    '''
    gt_colour_list = ['skyblue', 'blue', 'goldenrod', 'teal', 'lawngreen'] ### , 'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow'
    dt_colour_list= ['red', 'lightslategray',]
    # Create figure
    fig = go.Figure()

    df_col = df.columns ### df_col = ['time', 'tracename']
    # print('df:', list(df.index))

    if with_time == False:
        fig.add_trace(
                    go.Scatter(x=df.index, y=list(df[df_col[1]]), name=df_col[1], mode='markers', marker=dict(size=10, color='midnightblue')),   ### equivalent to: y=list(df['trace1'])
                    )
    else:
        fig.add_trace(
                    go.Scatter(x=list(df[df_col[0]]), y=list(df[df_col[1]]), name=df_col[1], mode='markers', marker=dict(size=10, color='midnightblue')),   ### equivalent to: y=list(df['trace1'])
                    )
        
    ### plot ground_truths
    if ground_truths != None:
        gt_class_list = gt_classlist
        ### sperate the content of ground_truths
        ground_truths_values = ground_truths[0]
        ground_truths_xticks = ground_truths[1]
        ground_truths_class = ground_truths[2]

        for (start_ind, end_ind), (start_ts, end_ts), cls in zip(ground_truths_values, ground_truths_xticks, ground_truths_class):
            ### check if time on x-axis
            if with_time:
                start = start_ts
                end = end_ts
            else:
                start = start_ind
                end = end_ind

            #### select colour based on class
            if cls == -1:
                fill_colour = gt_colour_list[cls]
            else:
                fill_colour = gt_colour_list[cls-1]
            fig.add_shape(type="rect", # specify the shape type "rect"
                    xref="x", # reference the x-axis
                    yref="paper", # reference the y-axis
                    x0=start, # the x-coordinate of the left side of the rectangle
                    y0=0, # the y-coordinate of the bottom of the rectangle
                    x1=end, # the x-coordinate of the right side of the rectangle
                    y1=1, # the y-coordinate of the top of the rectangle
                    fillcolor=fill_colour, # the fill color
                    opacity=0.5, # the opacity
                    layer="below", # draw shape below traces
                    line_width=0, # outline width
                    )
            
            # Add dotted lines on the sides of the rectangle
            for x in [start, end]:
                fig.add_shape(type="line",
                        xref="x",
                        yref="paper",
                        x0=x,
                        y0=0,
                        x1=x,
                        y1=1,
                        line=dict(
                            color=fill_colour,
                            width=2,
                            dash="dot",
                        ),
                    )
        
        ##### add legend for anomaly based on colours/class
        for colour, name in zip(gt_colour_list, gt_class_list):
            fig.add_trace(go.Scatter(
                x=[None],  # these traces will not have any data points
                y=[None],
                mode='markers',
                marker=dict(size=10, color=colour),
                showlegend=True,
                name=name,
            ))

    ### plot detections
    if detections != None:
        dt_class_list = dt_classlist
        ### sperate the content of detections
        detections_values = detections[0]   ### index values
        detections_xticks = detections[1]   ### timestamp values
        detections_class = detections[2]    ### class values

        for (start_ind, end_ind), (start_ts, end_ts), cls in zip(detections_values, detections_xticks, detections_class):
            ### check if time on x-axis
            if with_time:
                start = start_ts
                end = end_ts
            else:
                start = start_ind
                end = end_ind

            #### select colour based on class
            fill_colour = dt_colour_list[cls]
            fig.add_shape(type="rect", # specify the shape type "rect"
                    xref="x", # reference the x-axis
                    yref="paper", # reference the y-axis
                    x0=start, # the x-coordinate of the left side of the rectangle
                    y0=0, # the y-coordinate of the bottom of the rectangle
                    x1=end, # the x-coordinate of the right side of the rectangle
                    y1=1, # the y-coordinate of the top of the rectangle
                    fillcolor=fill_colour, # the fill color
                    opacity=0.5, # the opacity
                    layer="below", # draw shape below traces
                    line_width=0, # outline width
                    )
            
            # Add dotted lines on the sides of the rectangle
            for x in [start, end]:
                fig.add_shape(type="line",
                        xref="x",
                        yref="paper",
                        x0=x,
                        y0=0,
                        x1=x,
                        y1=1,
                        line=dict(
                            color=fill_colour,
                            width=2,
                            dash="dot",
                        ),
                    )
        
        ##### add legend for anomaly based on colours/class
        for colour, name in zip(dt_colour_list, dt_class_list):
            fig.add_trace(go.Scatter(
                x=[None],  # these traces will not have any data points
                y=[None],
                mode='markers',
                marker=dict(size=10, color=colour),
                showlegend=True,
                name=name,
            ))


    ### generate x ticks with timestamp and index num  
    x_data = df[df_col[0]]
    # print('x_data:', x_data)
    #get index of first element of x_data
    start_index = x_data.index[0]  
    end_index = x_data.index[-1]  
    # print('x_data:', x_data)
    # print('start_index:', start_index, 'end_index:', end_index)
    if is_xticks == True and with_time == False:
        x_ticks = [(i,x_data[i]) for i in range(start_index,end_index,10) ]
        x_tickvals = [k for k in range(start_index,end_index,10)]
    elif is_xticks == True and with_time == True:
        x_ticks = [(i,x_data[i]) for i in range(start_index,end_index,10) ]
        x_tickvals = [x_data[k] for k in range(start_index,end_index,10)]
    elif is_xticks == False:
        x_ticks = None
        x_tickvals = None

    # Add range slider, title, yticks, axes labels
    fig.update_layout(
        title_text="Event Trace without Time",
        xaxis=dict(
            title="Number of events",
            rangeslider=dict(visible=True),
            type='linear',
            tickvals=x_tickvals,
            ticktext=x_ticks,
            tickfont = dict(size = FONTSIZE),
            titlefont = dict(size = FONTSIZE),
            color='black',
        ),
        yaxis=dict(
            title="Events",
            tickvals=[k for k in range(0,len(var_list))],
            ticktext= var_list,
            tickfont = dict(size = FONTSIZE),
            titlefont = dict(size = FONTSIZE),
            color='black',
        ),
        autosize=True,
        # width=PLOTWIDTH,
        height=PLOTHEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        
    )

    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        autorange=True,
    )

    # style all the traces
    fig.update_traces(
        #hoverinfo="name+x+text",
        line={"width": 0.5},
        marker={"size": 8},
        mode="lines+markers",
        showlegend=True,
        
    )

    # fig.show()
    return fig

def prepare_gt(path):
    '''
    path: path to the test data file which is labelled -> str

    return:
    label_content: list of labels to plot (for required for plot_single_trace)-> list
    '''
    labels = read_json(path)
    ########## extract indeices of the labels for plotting ##########
    labels_content = labels['labels']
    label_trace_name = list(labels_content.keys())[0]
    label_content = labels_content[label_trace_name]

    label_plot_values = [(x[0], x[1]) for x in label_content]
    label_plot_x_ticks = [(x[2], x[3]) for x in label_content]
    label_plot_class = [x[4] for x in label_content]

    label_content = (label_plot_values, label_plot_x_ticks, label_plot_class)   ####### in format (values, x_ticks(timestamp), class)

    return label_content


def index2timestamp(df, index):
    '''
    This function takes the index of the event and returns the timestamp
    df: data frame -> pd.DataFrame
    index: index of the event -> int

    return:
    timestamp: timestamp of the event -> int
    '''
    return df['time'][index]


def get_var_timestamps(paths_traces=None, df=None, config=None):
    '''
    This function takes the trace paths and extracts the variable timestamps
    paths_traces: list of paths to the trace files -> list(str)
    df: data frame -> pd.DataFrame. in format: [time, trace]
    
    return:
    var_timestamps: list of variable timestamps -> list(dict)'''
    

    var_timestamps = []     ### [log1, log,2, ... ] --> [{var1:[], var2:[], ....}, {}  ]

    if paths_traces:
        for p in paths_traces:
            #print(p,w)
            trace = read_traces(p)
            #print(trace)
            var_timelist = defaultdict(list)
            # for var_name in _var_list:
            #     print(var_name)
            for ind, (t, ts) in enumerate(trace):
                #print(t,ts)
                var_timelist[t] += [[ts, ind]]     ### format: {var1:[[ts, ind], ...], var2:[[ts, ind], ...]}

            var_timestamps += [(p, var_timelist)]   ### in the format (trace_path, {var1:[[ts, ind], ...], var2:[[ts, ind], ...})
        return var_timestamps
    
    elif not df.empty and config != None:
        var_timelist = defaultdict(list)
        for df_row in df.itertuples():
            # print('in utils:', ind, 'time:', df_row[1], 'trace', df_row[2])
            ind = df_row[0]   ### index
            t = df_row[2]  ### trace
            ts = df_row[1]   ### time
            var_timelist[t] += [[ts, ind]]     ### format: {var1:[[ts, ind], ...], var2:[[ts, ind], ...]}

        var_timestamps += [(config, var_timelist)]   ### in the format (trace_path, {var1:[[ts, ind], ...], var2:[[ts, ind], ...})
        return var_timestamps


def preprocess_variable_plotting(var_timestamps, var_list, from_number, trace_number=0):
    '''
    This function preprocesses the variable timestamps for plotting
    var_timestamps: list of variable timestamps -> list(dict)
    var_list: list of variable names -> list
    from_number: dict of integers to map to variable names -> dict
    
    return:
    to_plot: list of variables to plot -> list
    '''

    to_plot = []   ### in format -> [var_name, ( [[<exe inters of var in log1>], [timestamps ]  )]
    ### collect data for each variable from each log file
    for v in range(len(var_list)):
        xy_data = [] ### execution intervals
        trace_names = []
        for i, (p, data) in enumerate(var_timestamps):
            w = p.split('/')[-1].split('.')[0]   ### trace name
            # print(i,p,data)
            ##############################
            if i==trace_number:   ### take data of single log for plotting
            #################################
                try:
                    #_, data = read_logs(p)   ### data of each log file
                    time_list = data[v]   ### get the timestamps
                    #print(data)
                    exe_time = []
                    timestamp = []
                    indices = []
                    for _, (td1,td2) in enumerate(zip(time_list[0:-1], time_list[1:])):
                        t2 = td2[0]
                        t1 = td1[0]
                        ind = td2[1]
                        tdiff = t2-t1
                        #print(tdiff)
                        exe_time+=[tdiff]
                        timestamp+=[t2] 
                        indices+=[ind]  ### index of the event in trace files | to maps events between full trace plots and variable plots
                        
                        # print(ind,t2)
                        # ##### testing limited samples  
                        # if t2 > 250000:
                        #     break
                        # ###### testing limited samples

                    print('length of exe_time:', len(exe_time))
                    assert(len(exe_time)==len(timestamp))
                    trace_names += [w]
                    xy_data += [(exe_time,timestamp, indices)]
                except Exception as e:
                    print(f'{w}:{v}', e)    ### will execute if any variable has only one execution time
            
        assert(len(trace_names)==len(xy_data))
        to_plot += [(p.replace(w,from_number[v]),trace_names,xy_data)]  ### [name of the file to write plots(variable name), labels for legend (log names), execution intervals for respective variables(y_data), timestamps(x_data, indexs)]
    
    return to_plot


def plot_execution_interval_single(to_plot, 
                                    ground_truths=None, 
                                    is_xticks=False, 
                                    gt_classlist=['gt_communication', 'gt_sensor', 'gt_bitflip', 'gt_unhandled-interupt', 'expected behaviour'],
                                    detections=None,
                                    dt_classlist=['detection'],
                                    thresholds=None):
    '''
    This function plots the execution intervals for each variable
    to_plot: list of variables to plot -> list ; output of preprocess_variable_plotting()

    return:
    fig_list: list of plotly figure objects -> list
    '''
    # CODE, BEHAVIOUR, THREAD, VER = get_config()
    gt_colour_list = ['skyblue', 'blue', 'goldenrod', 'teal', 'lawngreen'] ### , 'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow'
    dt_colour_list= ['red', 'lightslategray',]

    ### name represents the name of respective variable with which file will be saved
    fig_list = []    ### plotly figure objects to plot later
    for (name, log_names, xy_data) in to_plot:
        ### path to save the plots
        to_write_name = name.replace('trace_data', 'exe plots')
        file_name = os.path.basename(to_write_name)
        # file_name = f'{THREAD}_version{VER}_{BEHAVIOUR}_{file_name}'
        dir_name = os.path.dirname(to_write_name)
        to_write_name = os.path.join(dir_name, file_name)
        var_name = os.path.basename(name)
        
        ########## make data frame to be able to plot ################
        df = dict()
        _y_all = [] ### to adjust y-ticks
        legend_lab = [] ### collect names of the plots only
        line_style = ['solid', 'dashed', 'dashdot', 'dotted']
        markers = ['.','o','*','+','^','x','d','h',',','H','D']

        # Create figure
        fig = go.Figure()
        # print(xy_data)
        for (num, (l,xy)) in enumerate(zip(log_names, xy_data)):
            x = xy[1]
            #x = [i-x[0] for i in x]   ### get timestamps relative to first timestamp
            y = xy[0]
            ind = xy[2]
            # print(len(x),len(y),len(ind))

            if is_xticks == True:
                x_ticks = [(ind[i],x[i]) for i in range(0,len(x),5) ]
                x_ticksval = [ind[k] for k in range(0,len(x),5)]

                # x_ticks = [(ind[i]) for i in range(0,len(x)) ]
                # x_ticksval = tickvals=[k for k in range(0,len(x))],
            else:
                x_ticks = None
                x_ticksval = None

            ### ignore all the variables that are only executed once
            if xy[1]!= []:
                #print(x,y)
                df[l]=xy
                _y_all.extend(y)
                legend_lab.append(l)
                
                fig.add_trace(
                    go.Scatter(x=ind, y=y, name=l, mode='markers', marker=dict(size=10, color='midnightblue'))
                )

                # fig.add_trace(
                #     go.Scatter(y=y, name=l, mode='markers', marker=dict(size=10, color='midnightblue'))
                # )
                
                # Add range slider, title, yticks, axes labels
                fig.update_layout(
                    title_text=f"Execution Interval for '{var_name}'",
                    xaxis=dict(
                        title="Number of events",
                        rangeslider=dict(visible=True),
                        type='linear',
                        tickvals=x_ticksval,
                        ticktext=x_ticks,
                        tickfont = dict(size = 20),
                        titlefont = dict(size = 20),
                        color='black',
                        ),
                    yaxis=dict(
                        title="Execution interval (ms)",
                        # tickvals=[k for k in range(1,len(_var_list)+1)],
                        # ticktext=_var_list,
                        tickfont = dict(size = 20),
                        titlefont = dict(size = 20),
                        color='black'
                        ),
                    autosize=True,
                    #width=500,
                    #height=600,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                fig.update_xaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey'
                )
                
                fig.update_yaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey'
                )
            
                if thresholds != None:
                    if var_name not in thresholds.keys():
                        lower_th = 0
                        upper_th = 1000
                    else:
                        lower_th, upper_th = thresholds[var_name]
                        lower_th *= 1000
                        upper_th *= 1000
                    print('plot fun:', var_name, lower_th, upper_th)

                    #### select colour based on class
                    fig.add_shape(type="rect", # specify the shape type "rect"
                            # xref="x", # reference the x-axis
                            # yref="paper", # reference the y-axis
                            x0=ind[0], # the x-coordinate of the left side of the rectangle
                            y0=lower_th, # the y-coordinate of the bottom of the rectangle
                            x1=ind[-1], # the x-coordinate of the right side of the rectangle
                            y1=upper_th, # the y-coordinate of the top of the rectangle
                            opacity=0.3, # the opacity
                            layer="below", # draw shape below traces
                            line_width=0, # outline width
                            fillcolor='LightSeaGreen', # the fill color
                            )
                    
                    # Add dotted lines on the sides of the rectangle
                    for y in [lower_th, upper_th]:
                        fig.add_shape(type="line",
                                # xref="x",
                                # yref="paper",
                                x0=ind[0],
                                y0=y,
                                x1=ind[-1],
                                y1=y,
                                line=dict(
                                    color='LightSeaGreen',
                                    width=2,
                                    dash="dot",
                                ),
                            )
                    ##### add legend for threshold based on colours/class
                    fig.add_trace(go.Scatter(
                        x=[None],  # these traces will not have any data points
                        y=[None],
                        mode='markers',
                        marker=dict(size=10, color='LightSeaGreen'),
                        showlegend=True,
                        name='Threshold',
                    ))

            ### plot ground_truths
            if ground_truths != None:
                gt_class_list = gt_classlist
                ### sperate the content of ground_truths
                ground_truths_values = ground_truths[0]
                ground_truths_xticks = ground_truths[1]
                ground_truths_class = ground_truths[2]

                for (start_ind, end_ind), (start_ts, end_ts), cls in zip(ground_truths_values, ground_truths_xticks, ground_truths_class):
                    ### check if time on x-axis
                    start = start_ind
                    end = end_ind

                    #### select colour based on class
                    if cls == -1:
                        fill_colour = gt_colour_list[cls]
                    else:
                        fill_colour = gt_colour_list[cls-1]
                    fig.add_shape(type="rect", # specify the shape type "rect"
                            xref="x", # reference the x-axis
                            yref="paper", # reference the y-axis
                            x0=start, # the x-coordinate of the left side of the rectangle
                            y0=0, # the y-coordinate of the bottom of the rectangle
                            x1=end, # the x-coordinate of the right side of the rectangle
                            y1=1, # the y-coordinate of the top of the rectangle
                            fillcolor=fill_colour, # the fill color
                            opacity=0.5, # the opacity
                            layer="below", # draw shape below traces
                            line_width=0, # outline width
                            )
                    
                    # Add dotted lines on the sides of the rectangle
                    for x in [start, end]:
                        fig.add_shape(type="line",
                                xref="x",
                                yref="paper",
                                x0=x,
                                y0=0,
                                x1=x,
                                y1=1,
                                line=dict(
                                    color=fill_colour,
                                    width=2,
                                    dash="dot",
                                ),
                            )
                
                ##### add legend for anomaly based on colours/class
                for colour, name in zip(gt_colour_list, gt_class_list):
                    fig.add_trace(go.Scatter(
                        x=[None],  # these traces will not have any data points
                        y=[None],
                        mode='markers',
                        marker=dict(size=10, color=colour),
                        showlegend=True,
                        name=name,
                    ))

            ### plot detections
            if detections != None:
                dt_class_list = dt_classlist
                ### sperate the content of detections
                detections_values = detections[0]   ### index values
                detections_xticks = detections[1]   ### timestamp values
                detections_class = detections[2]    ### class values

                for (start_ind, end_ind), (start_ts, end_ts), cls in zip(detections_values, detections_xticks, detections_class):
                    ### check if time on x-axis
                    start = start_ind
                    end = end_ind

                    #### select colour based on class
                    fill_colour = dt_colour_list[cls]
                    fig.add_shape(type="rect", # specify the shape type "rect"
                            xref="x", # reference the x-axis
                            yref="paper", # reference the y-axis
                            x0=start, # the x-coordinate of the left side of the rectangle
                            y0=0, # the y-coordinate of the bottom of the rectangle
                            x1=end, # the x-coordinate of the right side of the rectangle
                            y1=1, # the y-coordinate of the top of the rectangle
                            fillcolor=fill_colour, # the fill color
                            opacity=0.5, # the opacity
                            layer="below", # draw shape below traces
                            line_width=0, # outline width
                            )
                    
                    # Add dotted lines on the sides of the rectangle
                    for x in [start, end]:
                        fig.add_shape(type="line",
                                xref="x",
                                yref="paper",
                                x0=x,
                                y0=0,
                                x1=x,
                                y1=1,
                                line=dict(
                                    color=fill_colour,
                                    width=2,
                                    dash="dot",
                                ),
                            )
                
                ##### add legend for anomaly based on colours/class
                for colour, name in zip(dt_colour_list, dt_class_list):
                    fig.add_trace(go.Scatter(
                        x=[None],  # these traces will not have any data points
                        y=[None],
                        mode='markers',
                        marker=dict(size=10, color=colour),
                        showlegend=True,
                        name=name,
                    ))

        if _y_all != []:

            # style all the traces
            fig.update_traces(
                #hoverinfo="name+x+text",
                line={"width": 0.5},
                marker={"size": 8},
                mode="lines+markers",
                showlegend=True,
                )

            # fig.show()
            fig_list += [fig]

    return fig_list



def write_to_csv(data, name):
    '''
    data in dict format, where keys form the column names
    '''
    df = pd.DataFrame(data)
    df.to_csv(name+'.csv', index=False)