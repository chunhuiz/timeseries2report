"""
Battery Time-Series Downstream Task -- anomaly detection

This system performs real-time anomaly detection on lithium-ion battery modules using text-based LLM
"""

import os
from openai import OpenAI
import base64
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import numpy as np
from zai import ZhipuAiClient
import pickle

# API Configuration for LLM service
OPENAI_API_KEY = "sk-e76205eae508447dbe165ad68b158858"
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Anomaly detection criteria instructions
ins_att = """
Attention: 
1. you should precisely marked the monitoring conclusion, for instance: "<ans>abnormal</ans>", so that the user can find it easily.
2. There is certain inconsistency in the batteries, which is a normal phenomenon.
Similarly, only SIGNIFICANT difference observed from specific cell will be considered as an anomaly.
examples of normal individual operations: **Cells #1** have **SOC values** (5%) compared to the rest (6%): normal! these difference in SOC is slight! 
 **Cells #1** have **temperature values** (28 °C) compared to the rest (25 °C) IS normal! inconsistency like 3°C or less is actually normal.
 **Cells #1** have **voltage values** (3.4V) compared to the rest (3.3V) IS normal! inconsistency like 0.1V or less is actually normal.
 **Cells #1** have **SOC values** (20%) compared to the rest (25%) IS normal! inconsistency like 5% or less is actually normal.

"""

# OPENAI_MODEL = "glm-4.5"
OPENAI_MODEL = "qwen3-14b"

# Initialize appropriate API client based on model
if 'glm' in OPENAI_MODEL: 
    client = ZhipuAiClient(api_key="13ab570616de4823b2777e9b74041126.Xkj0uHXFaGHGH1Fn")
else:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

def TimeSeriesText(signal):
    """
    Formats battery cell measurements into human-readable text report
    
    Args:
        signal: Array containing [temperature, voltage, SOC] for 16 cells + current
        
    Returns:
        str: Formatted text report with all cell measurements
    """
    signalTextAll = []
    for i in range(16):
        signalText = (f"time series of Cell {i+1}:\n"
                     f"temperature (°C): {signal[i*3]}; "
                     f"voltage (V): {signal[i*3+1]}; "
                     f"SOC (%): {signal[i*3+2]}.")
        signalTextAll.append(signalText)
    signalCur = f"time series of current: {signal[-1]}"
    signalTextAll.append(signalCur)
    return "\n".join(signalTextAll)

def process_sample(idx, sample, instruct, client, OPENAI_MODEL):
    """
    Processes a single battery sample through LLM API (thread worker function)
    
    Args:
        idx: Sample index for tracking
        sample: Dictionary containing 'data' with time-series measurements
        instruct: Analysis instructions template
        client: API client instance
        OPENAI_MODEL: Model identifier
        
    Returns:
        tuple: (index, API response content)
    """
    try:
        # API call with battery data and instructions
        
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": instruct + TimeSeriesText(sample["data"]) + ins_att},
            ],
            extra_body={"enable_thinking": False},
            temperature=0.7,
        )
        
        answers = completion.choices[0].message.content.strip()
        print(f"Sample {idx} processed: {answers[:50]}...")  # Truncated output
        return (idx, answers)  
    
    except Exception as e:
        print(f"Sample {idx} failed: {str(e)}")
        return (idx, None)

class PreprocessedQADataset:
    """Handles loading and preprocessing of battery monitoring data"""
    
    def __init__(self, jsonl_file):
        self.samples = self._load_data(jsonl_file)
        
    def _load_data(self, jsonl_file):
        """Loads and cleans JSONL data from specified path"""
        current_file_path = Path(__file__)
        jsonl_file = current_file_path.parents[0] / "data" / jsonl_file
        samples = []
        
        with open(jsonl_file, 'r', encoding='latin-1') as f:
            for line in f:
                # Clean special characters in temperature units
                cleaned_line = line.replace('¡ãC', '°C').replace('¡ã', '°')
                data = json.loads(cleaned_line)
                timeData = data['time series']
                samples.append({
                    'data': timeData,  # Store raw time-series data
                })
        return samples

if __name__ == "__main__":
    # Primary instruction template for anomaly detection
    
    instruct1 = ("""Please analyze the following Time Series of battery module (consist of 16 series-connected LIB cells), and determine whether there are any anomalies in this time period.
    Here are some basic informations of the involved LIB cells: Nominal Voltage: 3.2V, Operating Voltage: 2.5~3.65V, Charging/Discharging Protocal: Constant Power, Ambient Temperature: 25degC.
    These basic informations indicate the expected normal operating ranges of LIB cells, which is important for your fault diagnosis tasks.
    You should conduct monitoring task through cluster-level (cluster behavior) and individual-level (each specific LIB cell).
    If a noticable anomaly is detected, you should further select the most probable anomaly type from these options:
    A. Temperature Anomaly in specific individual cell
    B. Temperature Anomaly of whole LIB module
    C. Voltage Anomaly in specific individual cell
    D. Voltage Anomaly of whole LIB module.
    Also, you should identify the beginning sample and the ending sample of the detected anomaly, e.g., "1th~50th sample"
    The output format: you should return your answers in the format with "The operation is <ans>normal</ans>." or "The operation is <ans>abnormal</ans>." in the begining, then the related analysis should be given.
    If the anomaly is detected, you should output its fault types, beginning time and the endding time.
    """)
    
    print("Model in use:", OPENAI_MODEL)
    
    # Dataset initialization
    
    current_file_path = Path(__file__)
    test_set = '20251210'
    
    # the path to save
    dir_path = current_file_path.parents[0] / f"downstream_result_{test_set}"
    
    # dataset_name = f'C:\\Users\\14525\\Desktop\\XS-REPORT\\2predict1128_{OPENAI_MODEL}.jsonl'
    dataset_name = current_file_path.parents[0] / "output_data" / f"report_{OPENAI_MODEL}_zju_100_{test_set}.jsonl"
    dataset = PreprocessedQADataset(dataset_name)
    
    # dataset = PreprocessedQADataset(f"template_ad_{OPENAI_MODEL}_xs_13_20251113.jsonl")
    
    # dataset = PreprocessedQADataset("template_ad_qa_dsv3_10.jsonl")

    sample_pick = dataset.samples
    # Parallel processing configuration
    max_workers = 10  # Number of concurrent API calls
    results = [None] * len(sample_pick)  # Pre-allocated results array
    
    # Process samples in parallel (testing with sample 45)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit analysis tasks
        futures = [
            executor.submit(
                process_sample,
                idx=i,
                sample=sample_pick[i],
                instruct=instruct1,
                client=client,
                OPENAI_MODEL=OPENAI_MODEL
            )
            for i in [45]  # Single sample for testing
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            idx, answers = future.result()
            results[idx] = answers

        print("\n" + "-"*50 + " All samples processed " + "-"*50 + "\n")

    # Parse and format results
    result_list = []
    for idx, answers in enumerate(results):
        if answers is not None:
            match = re.search(r'<ans>(.*?)</ans>', answers)
            result_list.append(match.group(1) if match else -1)
        else:
            print(f"Sample {idx} processing failed\n")
    
    # Convert results to numerical labels
    result_list = np.array(result_list)
    one_hot_result = []
    for i in range(len(result_list)):
        if result_list[i] == 'abnormal':
            one_hot_result.append(1)
        elif result_list[i] == 'normal':
            one_hot_result.append(0)
        else:
            one_hot_result.append(2)  # Unknown/invalid result
    
    one_hot_result = np.array(one_hot_result).reshape(1,-1)