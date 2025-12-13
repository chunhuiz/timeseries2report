"""
This module provides tools for analyzing time-series signals and generating natural language descriptions of the signal characteristics including:
- Trends (increasing, decreasing, stable)
- Outliers
- Fluctuations
- Amplitude levels
- Transition points

Key Components:
1. dict_list_to_text class - Converts analysis results to text descriptions
2. Signal detection functions:
   - detect_trend_simple - Identifies signal trends
   - detect_outlier_fluct - Detects outliers and fluctuations
   - detect_amplitude - Classifies amplitude levels
   - detect_change - Identifies transition points
3. time_description - Main function that coordinates all analyses
"""

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class dict_list_to_text:
    """Converts signal analysis results into natural language descriptions.
    
    Attributes:
        sn (str): Signal name (e.g., 'voltage', 'temperature')
    """
    
    def __init__(self, signal_name):
        """Initialize the text converter.
        
        Args:
            signal_name (str): Name of the signal being analyzed
        """
        self.sn = signal_name

    def trend_text(self, data_list):
        """Generate trend descriptions from analysis results.
        
        Args:
            data_list (list): List of trend analysis dictionaries
            
        Returns:
            str: Formatted trend description text
        """
        descriptions = []
        target_signals = ["temperature", "SOC"]
        decimal_places = 1 if self.sn in target_signals else 4
        unit = "%" if "SOC" in self.sn else ""
        for item in data_list:
            if item['trend'] != "is stable":
                desc = (f"From {item['begin time']}th to {item['end time']}th time, "
                        f"{self.sn} {item['trend']} from {item['signal begin']:.{decimal_places}f}{unit} to "
                        f"{item['signal end']:.{decimal_places}f}{unit}.")
            else:
                desc = (f"From {item['begin time']}th to {item['end time']}th time, "
                        f"{self.sn} {item['trend']} around {item['signal min']:.{decimal_places}f}{unit}~{item['signal max']:.{decimal_places}f}{unit} with mean of "
                        f"{item['signal mean']:.{decimal_places}f}{unit} and std. of {item['signal std']:.4f}.")
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def outlier_text(self, data_list):
        """Generate outlier descriptions from analysis results.
       
       Args:
           data_list (list): List of outlier analysis dictionaries
           
       Returns:
           str: Formatted outlier description text
       """
        descriptions = []
        target_signals = ["temperature", "SOC"]
        decimal_places = 1 if self.sn in target_signals else 4
        for item in data_list:
            formatted_values = np.char.mod(f'%.{decimal_places}f', item['outlier value'])
            formatted_values_list = formatted_values.tolist()
            desc = (f"At time {item['happen time']}, outliers ({', '.join(formatted_values_list)}) are observed from {self.sn}.")
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def fluct_text(self, data_list):
        """Generate fluctuation descriptions from analysis results.
        
        Args:
            data_list (list): List of fluctuation analysis dictionaries
            
        Returns:
            str: Formatted fluctuation description text
        """
        descriptions = []
        for item in data_list:
            desc = (f"From {item['begin time']}th to {item['end time']}th time, "
                    f"{self.sn} shows noticeable fluctuations.")
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def amp_text(self, data_list):
        """Generate amplitude level descriptions from analysis results.
        
        Args:
            data_list (list): List of amplitude analysis dictionaries
            
        Returns:
            str: Formatted amplitude description text
        """
        descriptions = []
        for item in data_list:
            desc = (f"From {item['begin time']}th to {item['end time']}th time, "
                    f"{self.sn} shows {item['amp level']} with mean value of {item['signal amp']:.4f} and std of {item['signal amp std']:.4f}.")
            descriptions.append(desc)
        return "\n".join(descriptions)
        
    def pick_transition(self, tran_time):
        """Format transition time points into readable text.
       
       Args:
           tran_time (list): List of transition time points
           
       Returns:
           str: Formatted transition time description
       """
        l = len(tran_time)
        tran_des = []
        tran_des.append("At time")
        i = 0
        while i < l - 1:
            if tran_time[i+1] - tran_time[i] == 1:
                tran_des.append(f"{tran_time[i]}~{tran_time[i+1]},")
                i = i + 2
            else:
                tran_des.append(f"{tran_time[i]},")
                i = i + 1
        tran_des.append(f"{tran_time[-1]}")
        return " ".join(tran_des)
        

    def change_text(self, data_list):
        """Generate transition point descriptions from analysis results.
        
        Args:
            data_list (list): List of transition analysis dictionaries
            
        Returns:
            str: Formatted transition description text
        """
        descriptions = []
        for item in data_list:
            desc = (f"{self.pick_transition(item['happen time'])}, transition points are observed from {self.sn}.")
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def text_all(self, trend_list=[], outlier_list=[], fluct_list=[], amp_list=[], change_list=[]):
        """Generate comprehensive text description from all analysis results.
        
        Args:
            trend_list (list): Trend analysis results
            outlier_list (list): Outlier analysis results
            fluct_list (list): Fluctuation analysis results
            amp_list (list): Amplitude analysis results
            change_list (list): Transition point analysis results
            
        Returns:
            str: Combined description text
        """
        descriptions_all = []
        if trend_list != []:
            descriptions_all.append(self.trend_text(trend_list))
        if outlier_list != []:
            descriptions_all.append(self.outlier_text(outlier_list))
        else:
            descriptions_all.append(f"no outliers observed from {self.sn}.")
        if fluct_list != []:
            descriptions_all.append(self.fluct_text(fluct_list))
        else:
            descriptions_all.append(f"The fluctuation level of {self.sn} is slight.")
        if amp_list != []:
            descriptions_all.append(self.amp_text(amp_list))
        if change_list != []:
            descriptions_all.append(self.change_text(change_list))
        
        return "\n".join(descriptions_all)


def detect_change(signal, window = 10, detect_size = 5, thre6 = 0.2, flag = 0):
    """Detect significant change points in a signal.
    
    Args:
        signal (np.array): Input signal
        window (int): Analysis window size
        detect_size (int): Sub-window size for slope detection
        thre6 (float): Threshold for change point detection
        flag (int): Enable/disable detection (0=disabled)
        
    Returns:
        list: List of dictionaries containing change point information
    """
    if flag == 0:
        return []
    else:
        detect_size = max(int((detect_size/10) * window),3)
        detect_num = int(len(signal)/window)
        state_list = []
        for i in range(detect_num):
            detect_sig = signal[i*window:(i+1)*window]
            t_range = np.linspace(1, window, num=window)
            
            slopes = []
            for j in range(len(t_range) - detect_size + 1):
                t_window = t_range[j:j+detect_size]
                s_window = detect_sig[j:j+detect_size]
                slope = np.polyfit(t_window, s_window, 1)[0]
                slopes.append(slope)
            
            change_indices = []
            for j in range(1, len(slopes)):
                if (slopes[j] - slopes[j-1]) > thre6:
                    change_indices.append(j + detect_size // 2)  
                if (slopes[j] - slopes[j-1]) < -thre6:
                    change_indices.append(j + detect_size // 2)
            change_indices = np.unique(change_indices)
            
            if len(change_indices) > 0:
                state_list.append([i, change_indices])
                
        description_change = []
        for i in range(len(state_list)):
            happen_time = state_list[i][0] * window + state_list[i][1]
            des_list = {"happen time": happen_time}
            description_change.append(des_list)
        
        return description_change

def detect_trend_simple(signal, window = 10, thre1 = 0.001, thre2 = 0.005, flag = 0):
    """Detect trends in a signal (increasing, decreasing, stable).
   
   Args:
       signal (np.array): Input signal
       window (int): Analysis window size
       thre1 (float): Threshold for trend detection
       thre2 (float): Threshold for stable phase confirmation
       flag (int): Enable/disable detection (0=disabled)
       
   Returns:
       list: List of dictionaries containing trend information
   """
    if flag == 0:
        return []
    else:
        detect_num = int(len(signal)/window)
        state = ['increases', 'decreases', 'is stable']
        state_list = []
        t_range = np.linspace(1, window,window)
        for i in range(detect_num):
            detect_sig = signal[i*window:(i+1)*window]
            slope = np.polyfit(t_range, detect_sig, 1)[0]
            
            if (slope > thre1):
                state_list.append(0)
            elif (slope < -thre1):
                state_list.append(1)
            else:
                state_list.append(2)
        seg_list = []
        begin = 0
        end = 0
        for i in range(detect_num-1):
            if state_list[i+1] != state_list[i]:
                seg_list.append([begin,end])
                begin = i + 1
                end = i + 1
            else:
                if state_list[i] == 2:
                    signal_range1 = signal[begin * window:(i+1) *window]
                    signal_range2 = signal[begin * window:(i+2) *window]
                    max_min_temp1 = np.mean(signal_range1)
                    max_min_temp2 = np.mean(signal_range2)
                    if np.abs(max_min_temp1-max_min_temp2) <= thre2 * (abs(max_min_temp1)+0.0001):
                        end = end + 1
                    else:
                        seg_list.append([begin,end])
                        begin = i + 1
                        end = i + 1
                else:
                    end = end + 1
        seg_list.append([begin,end])
        description = []
        
        for i in range(len(seg_list)):
            time_begin = seg_list[i][0] * window + 1
            time_end = (seg_list[i][-1] + 1)* window
            signal_pick = signal[time_begin - 1: time_end]
            signal_begin = signal_pick[0]
            signal_end = signal_pick[-1]
            signal_max = np.max(signal_pick)
            signal_min = np.min(signal_pick)
            signal_mean = np.mean(signal_pick)
            signal_std = np.std(signal_pick)
            trend_detect = state[state_list[seg_list[i][0]]]
            if trend_detect == "is stable" and seg_list[i][-1]-seg_list[i][0]>1:
                t_range_new = np.linspace(time_begin, time_end,time_end-time_begin+1)
                slope_new = np.polyfit(t_range_new,signal_pick,1)[0]
                if (slope_new > thre1):
                    trend_detect = "increases"
                elif (slope_new < -thre1):
                    trend_detect = "decreases"
                else:
                    trend_detect = "is stable"
            des_list = {"begin time": time_begin, "end time": time_end,
                        "signal begin": signal_begin, "signal end": signal_end,
                        "signal max": signal_max, "signal min": signal_min,
                        "signal mean": signal_mean, "signal std": signal_std,
                        "trend": trend_detect}
            description.append(des_list)
            
        return description

def detect_outlier_fluct(signal, window = 10, thre3 = 0.5, outlier_allow = 0.001, flag = [0,0]):
    """Detect outliers and fluctuations in a signal.
   
   Args:
       signal (np.array): Input signal
       window (int): Analysis window size
       thre3 (float): Fluctuation threshold
       outlier_allow (float): Minimum outlier threshold
       flag (list): Enable flags for [outlier, fluctuation] detection
       
   Returns:
       tuple: (outlier_descriptions, fluctuation_descriptions)
   """
    if flag.all() == 0:
        return [], []
    else:
        detect_num = int(len(signal)/window)
        state_list = []
        state_list2 = []
        for i in range(detect_num):
            detect_sig = signal[i*window:(i+1)*window]
            mean_sig = np.mean(detect_sig)
            x = np.arange(len(detect_sig))
            slope, intercept = np.polyfit(x, detect_sig, 1)
            detrended = detect_sig - (slope * x + intercept)
            cv_detrended = np.std(detrended)
            std_sig = np.std(detect_sig)
            if std_sig != 0:
                z_score = np.abs(detect_sig - mean_sig)
                outlier_idx = np.where((z_score > 3 * (std_sig)) & (z_score > outlier_allow))[0]

                if len(outlier_idx) > 0:
                    state_list.append([i, outlier_idx])
                
                if cv_detrended > thre3:
                    state_list2.append(i)
                
        seg_list = []
        if len(state_list2) > 0:
            begin = state_list2[0]
            end = state_list2[0]
            for i in range(len(state_list2)-1):
                if state_list2[i+1] != 1 + state_list2[i]:
                    seg_list.append([begin,end])
                    begin = state_list2[i + 1]
                    end = state_list2[i + 1]
                else:
                    end = state_list2[i + 1]
            seg_list.append([begin,end])
        
        description_outlier = []
        for i in range(len(state_list)):
            happen_time = state_list[i][0] * window + state_list[i][1]
            outlier_value = signal[happen_time]
            des_list = {"happen time": happen_time, "outlier value": outlier_value}
            description_outlier.append(des_list)
        
        
        description_fluct = []
        for i in range(len(seg_list)):
            time_begin = seg_list[i][0] * window + 1
            time_end = (seg_list[i][-1] + 1)* window
            des_list = {"begin time": time_begin, "end time": time_end}
            description_fluct.append(des_list)
        
        if flag[0] == 0:
            return [], description_fluct
        elif flag[-1] == 0:
            return description_outlier, []
        else:
            return description_outlier, description_fluct

def detect_amplitude(signal, window = 10, thre4 = 0.005, thre5 = 0.01, flag = 0):
    """Detect and classify signal amplitude levels.
    
    Args:
        signal (np.array): Input signal
        window (int): Analysis window size
        thre4 (float): Threshold for slight amplitude
        thre5 (float): Threshold for moderate amplitude
        flag (int): Enable/disable detection (0=disabled)
        
    Returns:
        list: List of dictionaries containing amplitude information
    """
    if flag == 0:
        return []
    else:
        detect_num = int(len(signal)/window)
        state = ['slight', 'moderate', 'significant']
        state_list = []
        for i in range(detect_num):
            detect_sig = signal[i*window:(i+1)*window]
            mean_sig = np.mean(detect_sig)
            if mean_sig < thre4:
                state_list.append(0)
            elif mean_sig < thre5:
                state_list.append(1)
            else:
                state_list.append(2)
                
        seg_list = []
        begin = 0
        end = 0
        for i in range(detect_num-1):
            if state_list[i+1] != state_list[i]:
                seg_list.append([begin,end])
                begin = i + 1
                end = i + 1
            else:
                end = end + 1
        seg_list.append([begin,end])
        
        description_amp = []
        for i in range(len(seg_list)):
            time_begin = seg_list[i][0] * window + 1
            time_end = (seg_list[i][-1] + 1)* window
            amp_detect = state[state_list[seg_list[i][0]]]
            amp = np.mean(signal[time_begin - 1:time_end])
            amp_std = np.std(signal[time_begin - 1:time_end])
            des_list = {"begin time": time_begin, "end time": time_end,
                        "amp level": amp_detect,"signal amp": amp, "signal amp std": amp_std}
            description_amp.append(des_list)
    
        return description_amp

def time_description(signal, 
                     thre1 = 0.001, thre2 = 0.005, thre3 = 0.5, thre4 = 0.005, thre5 = 0.01, thre6 = 0.2,
                     outlier_allow = 0.001, detect_size = 5,
                     signal_name='voltage', normal_range = [0,5], window = 10,
                     description_pick = [0,1,2,3,4]):
    """Main function for comprehensive time-series signal descriptor generation.
   
   Args:
       signal (np.array): Input signal
       thre1 (float): Trend detection threshold
       thre2 (float): Stable phase confirmation threshold
       thre3 (float): Fluctuation threshold
       thre4 (float): Slight amplitude threshold
       thre5 (float): Moderate amplitude threshold
       thre6 (float): Change point detection threshold
       outlier_allow (float): Minimum outlier threshold
       detect_size (int): Sub-window size for change detection
       signal_name (str): Name of the signal
       normal_range (list): Valid range for signal values
       window (int): Analysis window size
       description_pick (list): Flags for which analyses to perform
       
   Returns:
       str: Comprehensive signal description text
   """
    signal = signal.flatten()
    out_of_range_indices = np.where((signal < normal_range[0]) | (signal > normal_range[1]))[0]
    signal[out_of_range_indices] = signal[out_of_range_indices+1]
    flags = np.zeros(5)
    for i in description_pick:
        flags[i] = 1
    
    description_trend = detect_trend_simple(signal, window, thre1, thre2, flags[0])
    description_outlier, description_fluct = detect_outlier_fluct(signal, window, thre3, outlier_allow, flags[1:3])
    description_amp = detect_amplitude(signal, window, thre4, thre5, flags[3])
    description_change = detect_change(signal, window, detect_size, thre6, flags[4])
    d2t = dict_list_to_text(signal_name)
    description_final = d2t.text_all(trend_list=description_trend, 
                                     outlier_list=description_outlier, 
                                     fluct_list=description_fluct,
                                     amp_list=description_amp,
                                     change_list=description_change)
    
    return description_final

