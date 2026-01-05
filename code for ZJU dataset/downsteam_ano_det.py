"""
Battery Time-Series Downstream Task -- anomaly detection

This script performs automated analysis of lithium-ion battery time-series data using LLM APIs.
Key functionalities:
1. Processes battery monitoring data (voltage, temperature, SOC)
2. Uses AI models for anomaly detection
3. Supports parallel processing for efficient API calls

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
from zai import ZhipuAiClient
import re
import numpy as np
import pickle
import sys

# Local module imports
original_sys_path = sys.path.copy()
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
utils_dir = parent_dir / "utils"
sys.path.append(str(utils_dir))

import prompt_engineering as pe

sys.path = original_sys_path


OPENAI_MODEL = "qwen3-14b"
# OPENAI_MODEL = "glm-4.5"

if 'glm' in OPENAI_MODEL: 
    client = ZhipuAiClient(api_key="YOUR API KEY")

else:
    OPENAI_API_KEY = "sk-ab07cd0c228c49aab7897b6e1e8382a5"
    OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(
        api_key= OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

# Instruction template for anomaly detection
ins_att = """
Attention: 
1. you should precisely marked the monitoring conclusion, for instance: "<ans>abnormal</ans>", so that the user can find it easily.
2. There is certain inconsistency in the batteries, which is a normal phenomenon. Only when SIGNIFICANT inconsistency are detected will it be considered as an abnormal situation.
Similarly, only SIGNIFICANT difference observed from specific cell will be considered as an anomaly.
examples of normal individual operations: **Cells #1** have **SOC values** (5%) compared to the rest (6%): normal! these difference in SOC is slight! 
 **Cells #1** have **temperature values** (28 °C) compared to the rest (25 °C) IS normal! inconsistency like 3°C or less is actually normal.
 **Cells #1** have **voltage values** (3.4V) compared to the rest (3.3V) IS normal! inconsistency like 0.1V or less is actually normal.
 **Cells #1** have **SOC values** (20%) compared to the rest (25%) IS normal! inconsistency like 5% or less is actually normal.

"""

def process_sample(idx, sample, instruct, client, OPENAI_MODEL):
    """
    Process single sample through LLM API (thread worker function)
    
    Args:
        idx (int): Sample index for tracking
        sample (dict): Contains 'context' with battery data description
        instruct (str): Analysis instructions for LLM
        client: API client instance
        OPENAI_MODEL (str): Model identifier
        
    Returns:
        tuple: (index, API response content)
    """
    try:
        
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": instruct + sample["context"]+ins_att},
            ],
            extra_body={"enable_thinking": False},
            temperature=0.7,
        )
        
        answers = completion.choices[0].message.content.strip()
        print(f"Sample {idx} processed: {answers[:50]}...")
        return (idx, answers)
    
    except Exception as e:
        print(f"Sample {idx} failed: {str(e)}")
        return (idx, None)

class PreprocessedQADataset:
    """Handles loading and preprocessing of battery monitoring data"""
    
    def __init__(self, json_file):
        """
        Initialize dataset from json file
        
        Args:
            json_file (str): Path to data file
        """
        self.samples = self._load_data(json_file)
        
    def _load_data(self, json_file):
        """
        Load and preprocess json data
        
        Args:
            json_file (str): File path
            
        Returns:
            list: Processed samples with context strings
        """
        current_file_path = Path(__file__)
        json_file = current_file_path.parents[0] / "data" / json_file
        samples = []
        with open(json_file, 'r', encoding='latin-1') as f:
            for line in f:
                print(len(samples))
                cleaned_line = line.replace('¡ãC', '°C').replace('¡ã', '°')
                data = json.loads(cleaned_line)
                if "description text" in data:
                    context = data['description text']
                elif "descript c1" in data:
                    description1 = pe.restore_clu_text(data['descript c1'], "2023-12-01 08:00:00 UTC", "2023-12-01 09:40:02 UTC", "Module 1, Cluster 1, Stack 1")
                    description2 = pe.restore_ind_text(data['descript c2'], data['descript c3'], data['descript c4'], data['descript c5'])
                    context = description1 + "\n" + description2
                else:
                    continue
                
                samples.append({
                    'context': context,
                })
        return samples

if __name__ == "__main__":
    # Analysis instruction templates
    
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
    
    # Dataset configuration
    
    current_file_path = Path(__file__)
    test_set = '20251210'
    
    # the path to save
    dir_path = current_file_path.parents[0] / f"downstream_result_{test_set}"
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # for normally operation cases
    dataset_name = current_file_path.parents[0] / "output_data" / f"report_{OPENAI_MODEL}_zju_100_{test_set}.json"
    
    # for abnormally operation cases
    # dataset_name = current_file_path.parents[0] / "output_data" / f"report_ano_{OPENAI_MODEL}_zju_3_{test_set}.json"
    
    dataset = PreprocessedQADataset(dataset_name)
    print(f'info: anomaly detection by {OPENAI_MODEL}')
    

    # Parallel processing setup
    sample_pick = dataset.samples
    max_workers = 10
    results = [None] * len(sample_pick)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all sample tasks (each sample corresponds to one future)
        futures = [
            executor.submit(
                process_sample,
                idx=i,
                sample=sample_pick[i],
                instruct=instruct1,  # Original instruct from the loop (assumed defined)
                client=client,  # Original client from the loop (assumed thread-safe)
                OPENAI_MODEL=OPENAI_MODEL  # Original model name from the loop
            )
            for i in range(len(sample_pick))  # Original loop range: range(len(sample_pick)-1)
        ]
        
        # Get completed task results and store by index (maintains sample order)
        for future in as_completed(futures):
            idx, answers = future.result()  # Get returned (idx, answers)
            results[idx] = answers  # Store in results list by sample index
        
        # Print separator
        print("\n" + "-"*50 + " All samples processed " + "-"*50 + "\n")

    # Output all results (in sample order)
    result_list = []
    for idx, answers in enumerate(results):
        if answers is not None:
            match = re.search(r'<ans>(.*?)</ans>', answers)
            if match:
                first_value = match.group(1)
                result_list.append(first_value)
            else:
                result_list.append(-1)
        else:
            print(f"Sample {idx} processing failed\n")
    
    result_list = np.array(result_list)
    one_hot_result = []
    for i in range(len(result_list)):
        if result_list[i] == 'abnormal':
            one_hot_result.append(1)
        elif result_list[i] == 'normal':
            one_hot_result.append(0)
        else:
            one_hot_result.append(2)
    one_hot_result = np.array(one_hot_result).reshape(1,-1)
    
    OUTPUT_DATASET = f'{dir_path}\\ada_res_{OPENAI_MODEL}.pkl'
    
    with open(OUTPUT_DATASET, 'wb') as f:  
        pickle.dump(results, f)
        
