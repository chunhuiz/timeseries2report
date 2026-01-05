"""
Battery Time-Series Downstream Task -- SOC prediction

This system processes battery time-series data, and uses large language models for SOC forecasting.
Supports parallel processing for efficient API calls.
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
import pickle

# Local module imports
original_sys_path = sys.path.copy()
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
utils_dir = parent_dir / "utils"
sys.path.append(str(utils_dir))

import prompt_engineering as pe

sys.path = original_sys_path

# Configuration for LLM API access

OPENAI_MODEL = "qwen3-14b"
# OPENAI_MODEL = "glm-4.5"

# Initialize appropriate API client based on model selection
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
    Processes a single sample through LLM API (thread worker function)
    
    Args:
        idx: Sample index for tracking
        sample: Dictionary containing 'context' with battery data
        instruct: Analysis instructions for LLM
        client: API client instance
        OPENAI_MODEL: Model identifier
        
    Returns:
        Tuple of (index, API response content)
    """
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": instruct + sample["context"]},
            ],
            extra_body={"enable_thinking": False},
            temperature=0.7,
        )
        
        answers = completion.choices[0].message.content.strip()
        print(f"Sample {idx} processed: {answers[:50]}...")   # Truncated output for readability
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

    # Instruction templates for LLM prompts
    instruct2 = ("""Please analyze the following Time-Series of battery module, and answer the question:
                please forecast the possible value of average SOC of all the cells in the next 10 minutes. (the sampling interval of involved time series is 1 minute.)
                Note:
                please answer in the format like 'The average SOC of LIB cells: <ans>xx</ans>%.' in the begining,(xx is your predicted value) then the related analysis should be given.
                (the sign <ans> and </ans> are necessary to locate the answer!)
                """)
                 
    # Dataset initialization
    
    current_file_path = Path(__file__)
    test_set = '20251210'
    
    # the path to save
    dir_path = current_file_path.parents[0] / f"downstream_result_{test_set}"
  
    dataset_name = current_file_path.parents[0] / "output_data" / f"report_pred_{OPENAI_MODEL}_zju_100_{test_set}.jsonl"
    
    dataset = PreprocessedQADataset(dataset_name)

    # Parallel processing setup
    sample_pick = dataset.samples

    max_workers = 5  # Adjust based on API rate limits
    results = [None] * len(sample_pick)
    
    # Process samples in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_sample,
                idx=i,
                sample=sample_pick[i],
                instruct=instruct2,  
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

    # Process and format results
    result_list = []
    for idx, answers in enumerate(results):
        if answers is not None:
            match = re.search(r'<ans>(.*?)</ans>', answers)
            result_list.append(match.group(1) if match else -1)
        else:
            print(f"Sample {idx} processing failed\n")
    
    result_list = np.array(result_list).reshape(1,-1)
    
    OUTPUT_DATASET = f'{dir_path}\\soc_res_{OPENAI_MODEL}.pkl'
    
    with open(OUTPUT_DATASET, 'wb') as f:  
        pickle.dump(results, f)
