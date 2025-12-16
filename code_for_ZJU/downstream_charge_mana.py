"""
Battery Time-Series Downstream Task -- Charge/Discharge Management System

This system analyzes battery time-series data to provide:
1. State of Charge (SOC) monitoring
2. Operational recommendations
3. Safety cutoff predictions
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
import sys
from zai import ZhipuAiClient
import re
import numpy as np

# Local module imports
original_sys_path = sys.path.copy()
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
utils_dir = parent_dir / "utils"
sys.path.append(str(utils_dir))

import prompt_engineering as pe

sys.path = original_sys_path

# Model selection (Qwen by default, GLM alternative commented out)
OPENAI_MODEL = "qwen3-14b"
# OPENAI_MODEL = "glm-4.5"

# Initialize appropriate API client
if 'glm' in OPENAI_MODEL: 
    client = ZhipuAiClient(api_key="YOUR API KEY")
    
else:
    OPENAI_API_KEY = "YOUR API KEY"
    OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(
        api_key= OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def process_sample(idx, sample, instruct, client, OPENAI_MODEL):
    """
    Processes a single battery sample through LLM API (thread worker function)
    
    Args:
        idx: Sample index for tracking
        sample: Dictionary containing 'context' with battery data
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
                {"role": "user", "content": instruct + sample["context"]},
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
        current_file_path = Path(__file__)
        jsonl_file = current_file_path.parents[0] / "data" / jsonl_file
        samples = []
        with open(jsonl_file, 'r', encoding='latin-1') as f:
            for line in f:
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
    # Primary instruction template for charge/discharge management


    instruct3 = ("""Please analyze the following Time Series of battery module to generate precise, actionable forecasts for charge and discharge cycle management.
                Your Primary Task:
1. State and Trend Analysis (Descriptive):
**Describe the current charging/discharging state of the battery module. 
**Report the recent state of charge (SOC) (Mean SOC, Maximal SOC, and Minimal SOC across all the cells within the module).
**Identify the overall trend of the module's SOC (e.g., increasing, decreasing, or stable).

2. Operational Forecast and Actionable Guidance:
**Based on the current SOC parameters and trend, predict the time until the module's critical SOC cutoff condition is met.
**Provide clear, practical operational advice regarding charge/discharge actions.

#### Specific Cutoff Conditions:
Charging Scenario: The charging porcess of LIB module must be terminated when the Maximal SOC reaches 97%.
Discharging Scenario: The discharging process of LIB module must be terminated when the Minimal SOC reaches 3%.
The maximal (minimal) SOC refer to the highest (lowest) SOC value among all individual battery cells within the module.

#### sampling interval: 
The sampling interval of involved battery time series is 1 minute.

There are some output examples:
Output Example 1 (Charging):
"Based on the observed time-series data, the battery module experiences the Charging Phase. At the most recent timestamp, the module’s Mean SOC is <ans>mean_soc</ans>%, the Maximum SOC is <ans>max_soc</ans>%, and the Minimum SOC is <ans>min_soc</ans>%. The overall SOC exhibits an upward trend. Our forecast indicates that the module's Maximum SOC will reach the 97% cutoff condition in approximately <ans>pred_time</ans> minutes. To prevent overcharging and ensure battery longevity, it is advised to terminate charging after <ans>pred_time</ans> minutes. Detailed Analysis Follows:..."

Output Example 2 (Discharging):
"Based on the observed time-series data, the battery module experiences the Discharging Phase. At the most recent timestamp, the module’s Mean SOC is <ans>mean_soc</ans>%, the Maximum SOC is <ans>max_soc</ans>%, and the Minimum SOC is <ans>min_soc</ans>%. The overall SOC exhibits a downward trend. Our forecast indicates that the module's Minimum SOC will reach the 3% cutoff condition in approximately <ans>pred_time</ans> minutes. To prevent deep discharge and irreversible degradation, it is advised to terminate discharging after <ans>pred_time</ans> minutes. Detailed Analysis Follows:..."

### Attention:
1. Output Format Specification: The final output must strictly adhere to the following structure, using the <ans></ans> tag to explicitly highlight all critical numerical data.
2. You should first present all the key conclusions, and then provide the detailed analysis.
3. The output answers should include a Detailed Analysis that fully describes the reasoning for your conclusions, which is crucial for operators to understand the basis of the operational decisions. Furthermore, provide any additional practical recommendations or safety warnings derived from the analysis.
4. The maximal, minimal, average SOC refer to the highest, lowest, and average SOC value among all individual battery cells within the module.
                """)
                 
    
    # Initialize dataset
    current_file_path = Path(__file__)
    test_set = '20251210'
    
    # the path to save
    dir_path = current_file_path.parents[0] / f"downstream_result_{test_set}"
    
    dataset_name = current_file_path.parents[0] / "output_data" / f"report_{OPENAI_MODEL}_zju_100_{test_set}.jsonl"

    dataset = PreprocessedQADataset(dataset_name)

    # Prepare samples (testing with first sample repeated)
    sample_pick = dataset.samples
    
    # 5 independent API call for the 47th sample (you can pick other samples)
    sample_pick = [sample_pick[46]] + [sample_pick[46]] + [sample_pick[46]] + [sample_pick[46]] + [sample_pick[46]]

    max_workers = 10  # Concurrent API call limit
    
    # Initialize results storage
    results = [None] * len(sample_pick)

    # Process samples in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all analysis tasks
        futures = [
            executor.submit(
                process_sample,
                idx=i,
                sample=sample_pick[i],
                instruct=instruct3,
                client=client,
                OPENAI_MODEL=OPENAI_MODEL
            )
            for i in range(len(sample_pick))
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
    

    result_list = np.array(result_list)
