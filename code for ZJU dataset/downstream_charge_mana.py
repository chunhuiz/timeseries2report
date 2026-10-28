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
            temperature=0.3,
        )
        
        answers = completion.choices[0].message.content.strip()
        print(f"Sample {idx} processed: {answers[:50]}...")  # Truncated output
        return (idx, answers)
    
    except Exception as e:
        print(f"Sample {idx} failed: {str(e)}")
        return (idx, None)  

class PreprocessedQADataset:
    """Handles loading and preprocessing of battery monitoring data"""
    
    def __init__(self, json_file):
        self.samples = self._load_data(json_file)
        
    def _load_data(self, json_file):
        current_file_path = Path(__file__)
        json_file = current_file_path.parents[0] / "data" / json_file
        samples = []
        with open(json_file, 'r', encoding='latin-1') as f:
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


    instruct3 = ("""Please analyze the following Time Series of battery module to generate precise, actionable forecasts for charge or discharge cycle management.

Your primary objective is to analysis the charging (or discharging) state of battery module and predict the remaining time until the battery module reaches its SOC cutoff condition.

Step 1: Determine the operating state.
Use the current trend and SOC trend to decide whether the module is charging or discharging. The latest continuous active operating segment is the most important.

Step 2: Find the correct cutoff SOC.
For charging:
- The cutoff is reached when the highest-SOC cell in the module reaches 97%.
- At every timestamp, "highest SOC" means the maximum SOC among the 16 cells at that timestamp. It does NOT mean the maximum value over the entire time series.

For discharging:
- The cutoff is reached when the lowest-SOC cell in the module reaches 3%.
- At every timestamp, "lowest SOC" means the minimum SOC among the 16 cells at that timestamp. It does NOT mean the minimum value over the entire time series.

Step 3: Estimate the SOC rate.
For charging, estimate the positive slope of the module-level maximum-SOC trajectory during the latest continuous charging segment.

For discharging, estimate the negative slope of the module-level minimum-SOC trajectory during the latest continuous discharging segment.

Use SOC percentage points per minute. Do not confuse 50% with 0.50 when calculating the slope.

Step 4: Calculate the remaining time.

The sampling interval is 1 minute, so the result is in minutes.

Sanity checks:
- If charging and the current maximum SOC is below 97%, the result must be positive.
- If discharging and the current minimum SOC is above 3%, the result must be positive.
- Do not confuse the module maximum/minimum SOC with a temporal maximum/minimum.

Unit reminder: 
Treat all SOC values and SOC changes as percentages (%), not fractions. When estimating the SOC change rate and remaining charging/discharging time, consistently use % and % per minute.
Using the wrong units in calculations can mess up the magnitude of the predicted remaining time.

The output format:
Your answers should include a Detailed Analysis that fully describes the reasoning for your conclusions, which is crucial for operators to understand the basis of the operational decisions. 
Furthermore, please output one numerical answer in the end of your response:
The predicted remaining time is <ans>XX</ans> minutes.
Finally, provide any additional practical recommendations or safety warnings derived from the analysis.
""")
                 
    
    # Initialize dataset
    current_file_path = Path(__file__)
    test_set = '20251210'
    
    # the path to save
    dir_path = current_file_path.parents[0] / f"downstream_result_{test_set}"
    
    dataset_name = current_file_path.parents[0] / "output_data" / f"report_{OPENAI_MODEL}_zju_100_{test_set}.json"

    dataset = PreprocessedQADataset(dataset_name)

    # Prepare samples (testing with first sample repeated)
    sample_pick = dataset.samples
    
    # 5 independent API call for the 47th sample (you can pick other samples)
    sample_pick = [sample_pick[46]]

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
