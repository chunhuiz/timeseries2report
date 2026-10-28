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
                {"role": "user", "content": instruct + sample["context"]},
            ],
            extra_body={"enable_thinking": False},
            temperature=0.3,
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
    
    instruct1 = ("""Please analyze the following Time Series of battery module (consist of 16 series-connected LIB cells), and determine whether the operation of the concerned LIB module is normal or abnormal during this time period.
Here are some basic informations of the involved LIB cells: Nominal Voltage: 3.2V, Charging/Discharging Protocol: Constant Power, Ambient Temperature: 25 degC.
These basic informations indicate the expected normal operating ranges of LIB cells, which is important for your fault diagnosis tasks.

Follow the diagnostic procedure below strictly and in the given order.

Step 1: Individual-cell screening

Inspect Cell #1 through Cell #16 separately. For every cell, examine its voltage, temperature, and SOC over the entire time window. Brief extreme events must not be ignored.

First determine whether one specific cell, or a small number of specific cells, clearly stands out from most of the remaining cells.
The difference may appear in either the cell's dynamic behavior or in its temporal mean/std. values, such as a persistently higher or lower voltage, temperature, or SOC.

For instance, for voltage, approximately 3.6 V is a safety-relevant upper cutoff. If any individual cell reaches or exceeds approximately 3.6 V while other cells maintain a lower voltage,
it must be treated as an individual-cell voltage anomaly, even if the event is brief, the module-level voltage standard deviation is slight, or all cells have similar overall trends.

Step 2: Module-level inconsistency screening

When no single cell or small group of cells is the dominant outlier, further determine whether a module-level inconsistency exists.

A module-level inconsistency means that the cell values are broadly dispersed across multiple cells, with substantial differences among the cells, but no single cell is clearly responsible for the dispersion.

Therefore:

- One clearly different cell indicates an individual-cell anomaly.
- Several cells distributed across noticeably different levels, without one dominant outlier, indicate a module-level inconsistency.
- Similar changing trends do not by themselves prove that the cells are consistent; their actual values must also be compared.

Step 3: Final diagnosis

After completing the individual-cell screening and the module-level inconsistency screening, determine whether the module is normal or abnormal.

If no sufficiently supported anomaly is found, output exactly:

The operation is <ans>normal</ans>.

If an anomaly is found, select ONLY ONE most probable and dominant anomaly type from the following options:

1. Temperature Anomaly in specific individual cell
2. Temperature Inconsistency Anomaly of whole LIB module
3. Voltage Anomaly in specific individual cell
4. Voltage Inconsistency Anomaly of whole LIB module
5. SOC Anomaly in specific individual cell
6. SOC Inconsistency Anomaly of whole LIB module

Then output:

The operation is <ans>abnormal</ans>.
Anomaly Type: [exactly one option from the list]
Abnormal Cell: [one specific cell number, a small number of cells, or the whole module]
Time Period: [beginning sample~ending sample]

Attention: 
1. Moderate cell-to-cell differences are normal in real battery modules and must not be reported as inconsistency anomalies by themselves.

Typical acceptable instantaneous **max-minus-min** differences among the 16 cells include approximately:

- Temperature: 1–3 degC
- Voltage: 0.05 V
- SOC: 3–5 percentage points

Here, max-minus-min means the difference between the highest and lowest cell values at the same timestamp. 
It measures cell-to-cell difference and must not be confused with standard deviation or the temporal standard deviation of one cell over the whole window.

2. Evidence priority:
If one cell is clearly responsible for most of the observed dispersion, classify it as an individual-cell anomaly and identify that cell. Do not misclassify this one-cell-standing-out pattern as module-level inconsistency.

3. Before producing the final answer, verify that the selected variable, anomaly scope, abnormal cell, and time period are directly supported by the strongest quantitative evidence in the analysis.
""")
    
    # Dataset configuration
    
    current_file_path = Path(__file__)
    test_set = '20251210'
    
    # the path to save
    dir_path = current_file_path.parents[0] / f"downstream_result_{test_set}"
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # take normally operation cases for example
    dataset_name = current_file_path.parents[0] / "output_data" / f"report_{OPENAI_MODEL}_zju_100_{test_set}.json"
    
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
        
