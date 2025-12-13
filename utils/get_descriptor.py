# -*- coding: utf-8 -*-
"""
Battery Time-Series Descriptor Generation

Module for generating descritors of lithium-ion battery time-series data (ZJU dataset),
including voltage, temperature, SOC, and current characteristics at both individual cell and module level.

"""

from openai import OpenAI
import json
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import convert_timeseries_to_descriptor as t4


def ensemble_entropy(voltage_values, bins=3):
    """
    Calculate Shannon entropy of voltage distribution
    
    Args:
        voltage_values (np.array): Array of voltage measurements
        bins (int): Number of bins for histogram calculation (default:3)
    
    Returns:
        float: Calculated entropy value
    """
    hist, _ = np.histogram(voltage_values, bins=bins)
    probabilities = hist / len(voltage_values)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def cal_entro(signals):
    """
    Calculate entropy for multiple signals across samples
    
    Args:
        signals (np.array): 3D array of shape (samples, variables, cells)
        
    Returns:
        np.array: 2D array of entropy values (samples × variables)
    """
    entro_list = []
    sam_num, var_num, _ = np.shape(signals)
    
    for j in range(var_num):
        entro_per_list = []
        for i in range(sam_num):
            voltages = signals[i,j,:]
            entro_per_list.append(ensemble_entropy(voltages))
        entro_list.append(np.hstack(entro_per_list))
    return np.array(entro_list).T


def generate_descriptor_zju(data_v, data_c, window=10):
    """
    Generate comprehensive descriptions for battery time-series data
    
    Processes voltage, temperature, SOC and current data to produce:
    - Individual cell descriptions (grouped in sets of 4 cells)
    - Module-level statistical descriptions (mean, max, min)
    - Inconsistency metrics (standard deviation and entropy)
    
    Args:
        data_v (np.array): 3D array of voltage/temperature/SOC data 
                          (samples × variables × cells)
        data_c (np.array): 1D array of current data
        window (int): Smoothing window size for trend analysis (default:10)
    
    Returns:
        dict: Dictionary containing:
            - ind1: Descriptions for cells 1-4
            - ind2: Descriptions for cells 5-8  
            - ind3: Descriptions for cells 9-12
            - ind4: Descriptions for cells 13-16
            - clu: Module-level cluster descriptions
    
    Note:
        Threshold parameters (thre1-thre6) control sensitivity for:
        - thre1: Rise/fall detection
        - thre2: Stable/varying phase detection  
        - thre3: Fluctuation detection
        - thre4/thre5: Inconsistency level thresholds
        - thre6: Outlier detection
    """
    sample_num, _, cell_num = np.shape(data_v)

    # Initialize description containers for cell groups
    description_chunk_ind1 = []
    description_chunk_ind2 = [] 
    description_chunk_ind3 = []
    description_chunk_ind4 = []

    # Generate descriptions for each individual cell

    for cell_id in range(cell_num):
        cell_remider = f"The descriptions for the Cell #{cell_id+1} are below: "
        
        # voltage
        description_v = t4.time_description(data_v[:,1,cell_id], thre1 = 0.00025, thre2 = 0.01, thre3 = 0.1, thre6 = 0.005, window=window,
                              outlier_allow = 0.05, signal_name='voltage', normal_range = [0,5], description_pick = [0,1,2,4])
        
        # temperature
        description_t = t4.time_description(data_v[:,0,cell_id], thre1 = 0.00325, thre2 = 0.01, thre3 = 1.3, thre6 = 0.065, window=window,
                              outlier_allow = 0.65, signal_name='temperature', normal_range=[0,50], description_pick = [0,1,2,4])
        
        # SOC
        description_s = t4.time_description(data_v[:,2,cell_id], thre1 = 0.025, thre2 = 0.01, thre3 = 10, thre6 = 0.25, window=window,
                              outlier_allow = 5, signal_name='SOC', normal_range=[0,100], description_pick = [0,1,2,4])
        
        # Combine descriptions for cells
        descriptions = "\n".join([cell_remider, description_v, description_t, description_s])
        
        # Assign to appropriate group
        if cell_id < 4:
            description_chunk_ind1.append(descriptions)
        elif cell_id < 8:
            description_chunk_ind2.append(descriptions)
        elif cell_id < 12:
            description_chunk_ind3.append(descriptions)
        else:
            description_chunk_ind4.append(descriptions)
    
    # Join descriptions for each cell group
    description_chunk_ind1 = "\n".join(description_chunk_ind1)
    description_chunk_ind2 = "\n".join(description_chunk_ind2)
    description_chunk_ind3 = "\n".join(description_chunk_ind3)
    description_chunk_ind4 = "\n".join(description_chunk_ind4)
    
    # Generate module-level descriptions
    descriptions_all = []
    
    enesmble_reminder = ("For each sampling time, the statistical indicators (e.g., mean and standard deviation) "
                         "are calculated based on each variable (e.g., voltage) among all the cells: ")
    descriptions_all.append(enesmble_reminder)
    
    # Calculate statistical metrics
    vara_mean = np.mean(data_v[:,:3,:], axis = 2)
    vara_std = np.std(data_v[:,:3,:], axis = 2)
    vara_ent = cal_entro(data_v[:,:3,:])
    vara_max = np.max(data_v[:,:3,:], axis = 2)
    vara_min = np.min(data_v[:,:3,:], axis = 2)
    
    vara_trend = [vara_mean, vara_max, vara_min]
    
    # Generate descriptions of mean, max., min.
    ensemble_trend_name = ["average ", "maximal ", "minimal "]
    
    for i in range(len(vara_trend)):
        vara_trend_per = vara_trend[i]
        description_v2 = t4.time_description(vara_trend_per[:,1], thre1 = 0.00025, thre2 = 0.01, thre3 = 0.1, thre6 = 0.005, outlier_allow = 0.05,window=window,
                                  signal_name = ensemble_trend_name[i] + 'voltage', normal_range = [0,5], description_pick = [0,1,2,4])
        description_s2 = t4.time_description(vara_trend_per[:,2], thre1 = 0.025, thre2 = 0.01, thre3 = 10, thre6 = 0.25, outlier_allow = 5,window=window,
                                signal_name = ensemble_trend_name[i] + 'SOC', normal_range=[0,100], description_pick = [0,1,2,4])
        
        description_t2 = t4.time_description(vara_trend_per[:,0], thre1 = 0.00325, thre2 = 0.01, thre3 = 1.3, thre6 = 0.065, outlier_allow = 0.65,window=window,
                                signal_name = ensemble_trend_name[i] + 'temperature', normal_range=[0,50], description_pick = [0,1,2,4])
        descriptions = "\n".join([description_v2, description_t2, description_s2])
        descriptions_all.append(descriptions)
        
    # Generate inconsistency descriptions (std and entropy)
    
    description_v3 = t4.time_description(vara_std[:,1], thre4 = 0.05, thre5 = 0.1, outlier_allow = 1,thre3=0.01, window=window,
                              signal_name='voltage std', normal_range = [0,5], description_pick = [2,3])
    
    description_t3 = t4.time_description(vara_std[:,0], thre4 = 0.65, thre5 = 1.3, outlier_allow = 1,thre3=0.13, window=window,
                              signal_name='temperature std', normal_range = [0,5], description_pick = [2,3])
    
    description_s3 = t4.time_description(vara_std[:,2], thre4 = 5, thre5 = 10, outlier_allow = 1,thre3=1, window=window,
                              signal_name='SOC std', normal_range = [0,5], description_pick = [2,3])
    
    description_v4 = t4.time_description(vara_ent[:,1], thre4 = 1.5, thre5 = 3, outlier_allow = 1,window=window,
                              signal_name='voltage entropy', normal_range = [0,5], description_pick = [2,3])
    
    description_t4 = t4.time_description(vara_ent[:,0], thre4 = 1.5, thre5 = 3, outlier_allow = 1,window=window,
                              signal_name='temperature entropy', normal_range = [0,5], description_pick = [2,3])
    
    description_s4 = t4.time_description(vara_ent[:,2], thre4 = 1.5, thre5 = 3, outlier_allow = 1,window=window,
                              signal_name='SOC entropy', normal_range = [0,5], description_pick = [2,3])
    
    descriptions = "\n".join([description_v3, description_t3, description_s3, 
                             description_v4, description_t4, description_s4])
    
    descriptions_all.append(descriptions)
    
    # Current description
    description_i = t4.time_description(data_c, thre1 = 0.06725, thre2 = 0.01, thre3 = 26.9, thre6 = 1.345, outlier_allow = 13.45,window=window,
                                    signal_name='current', normal_range=[-1000,1000], description_pick = [0,1,2,4])
    
    descriptions_all.append(description_i)

    descriptions_all = "\n".join(descriptions_all)
    
    # Combine all module-level descriptions
    description_chunk_clu = descriptions_all

    return {"ind1": description_chunk_ind1, "ind2": description_chunk_ind2, 
            "ind3": description_chunk_ind3, "ind4": description_chunk_ind4,
             "clu": description_chunk_clu}

def generate_descriptor_mit(data_v, window=10):
    """
    Generate descriptive text analysis for battery cell time-series data.
    
    Analyzes voltage, temperature, capacity and current data for each cell in a battery module,
    producing human-readable descriptions of time-series characteristics and anomalies.
    
    Args:
        data_v (numpy.ndarray): 3D array of voltage data with shape:
                               [sample_num, metrics_num, cell_num]
                               metrics typically include:
                               0: temperature, 1: voltage, 2: charge capacity, 
                               3: discharge capacity, -1: current
        data_c (numpy.ndarray): Current data (currently unused in function)
        window (int): Size of the rolling window for time-series analysis (default: 10)
        
    Returns:
        dict: Dictionary containing generated descriptions with key "ind1" and value
              being the concatenated descriptions for all cells
    """
    
    # Get array dimensions
    sample_num, _, cell_num = np.shape(data_v)
    description_chunk_ind1 = []

    for cell_id in range(cell_num):
        cell_remider = f"The descriptions for the Cell #{cell_id+1} are below: "

        # 1. Voltage description
        description_v = t4.time_description(data_v[:,1,cell_id], thre1 = 0.0005, thre2 = 0.01, thre3 = 0.2, thre6 = 0.0025, window=window,
                              outlier_allow = 0.1, signal_name='voltage', normal_range = [0,5], description_pick = [0,1,2,4])
        
        # 2. Temperature description
        description_t = t4.time_description(data_v[:,0,cell_id], thre1 = 0.0025, thre2 = 0.01, thre3 = 1, thre6 = 0.025, window=window,
                              outlier_allow = 0.5, signal_name='temperature', normal_range=[0,50], description_pick = [0,1,2,4])
        
        # 3. charge capacity description
        description_s = t4.time_description(data_v[:,2,cell_id], thre1 = 0.001, thre2 = 0.01, thre3 = 0.1, thre6 = 0.00125, window=window,
                              outlier_allow = 0.05, signal_name='charge capacity', normal_range=[0,10], description_pick = [0,1,2,4])
        
        # 4. discharge capacity description
        description_s2 = t4.time_description(data_v[:,3,cell_id], thre1 = 0.001, thre2 = 0.01, thre3 = 0.1, thre6 = 0.00125, window=window,
                              outlier_allow = 0.05, signal_name="discharge capacity", normal_range=[0,10], description_pick = [0,1,2,4])
        
        # 5. Current description
        description_i = t4.time_description(data_v[:,-1,cell_id], thre1 = 0.002, thre2 = 0.01, thre3 = 0.8, thre6 = 0.01, outlier_allow = 0.4,window=window,
                                        signal_name='current', normal_range=[-1000,1000], description_pick = [0,1,2,4])
        
        
        descriptions = "\n".join([cell_remider, description_v, description_t, description_s, description_s2, description_i])
        description_chunk_ind1.append(descriptions)
    
    description_chunk_ind1 = "\n".join(description_chunk_ind1)


    return {"ind1": description_chunk_ind1}

def generate_descriptor_tju(data_v, window=10):
    sample_num, _, cell_num = np.shape(data_v)
    description_chunk_ind1 = []

    for cell_id in range(cell_num):
        cell_remider = f"The descriptions for the Cell #{cell_id+1} are below: "
        description_v = t4.time_description(data_v[:,0,cell_id], thre1 = 0.0003, thre2 = 0.01, thre3 = 0.2, thre6 = 0.005, window=window,
                              outlier_allow = 0.1, signal_name='voltage', normal_range = [0,5], description_pick = [0,1,2,4])
        
        description_s = t4.time_description(data_v[:,1,cell_id], thre1 = 0.00045, thre2 = 0.01, thre3 = 0.3, thre6 = 0.003, window=window,
                              outlier_allow = 0.15, signal_name='charge capacity', normal_range=[0,5], description_pick = [0,1,2,4])
        
        description_s2 = t4.time_description(data_v[:,2,cell_id], thre1 = 0.00045, thre2 = 0.01, thre3 = 0.3, thre6 = 0.003, window=window,
                              outlier_allow = 0.15, signal_name='discharge capacity', normal_range=[0,5], description_pick = [0,1,2,4])
        
        description_i = t4.time_description(data_v[:,3,cell_id], thre1 = 0.00075, thre2 = 0.01, thre3 = 0.5, thre6 = 0.0125, outlier_allow = 0.25,window=window,
                                        signal_name='current', normal_range=[-1000,1000], description_pick = [0,1,2,4])
        
        descriptions = "\n".join([cell_remider, description_v, description_s, description_s2, description_i])
        description_chunk_ind1.append(descriptions)
    
    description_chunk_ind1 = "\n".join(description_chunk_ind1)


    return {"ind1": description_chunk_ind1}