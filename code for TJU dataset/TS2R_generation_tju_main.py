"""
Main script for generating descriptive report from battery time-series data using LLM APIs --- for TJU dataset.

Key Functionality:
- Data loading and preprocessing of battery time-series data
- API integration with multiple LLM providers (Qwen, GLM, DeepSeek)
- Parallel generation of descriptive texts using thread pools
- Result validation and JSON dataset creation
- Reproducibility through seed setting

Dependencies:
- PyTorch, NumPy, OpenAI, scipy, tqdm, sklearn, etc.

Note: API keys should be managed securely and not committed to version control.
"""

import random
from tqdm import tqdm
import json
import os
import sys
from typing import *
from pathlib import Path
import torch
import tiktoken
import pdb
from openai import OpenAI
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
import pdb
from zai import ZhipuAiClient
import concurrent.futures
from typing import Optional 
import markdown
import datetime
import pickle
import time


original_sys_path = sys.path.copy()
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
utils_dir = parent_dir / "utils"
sys.path.append(str(utils_dir))

import prompt_engineering as pe
from get_descriptor import generate_descriptor_tju
import process_report as prre

sys.path = original_sys_path

# Load prompt templates from prompt_engineering module
# These define the different description styles we generate
SYSTEM_PROMPT2, template2 = pe.expert_prompt_for_report_tju()

# List of supported LLM variants
llm_list = ['qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b-instruct-2507', 'deepseek-chat', 'glm-4.5']

# API configuration

OPENAI_MODEL = llm_list[4]  # Default to qwen3-14b

if 'qwen' in OPENAI_MODEL:
    OPENAI_API_KEY = "YOUR API KEY"
    OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    MAX_TOKEN = 8192  # Standard token limit
    
elif 'deepseek' in OPENAI_MODEL:
    OPENAI_API_KEY = "YOUR API KEY"
    OPENAI_BASE_URL = "https://api.deepseek.com"
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    MAX_TOKEN = 8192  # Standard token limit

else:
    client = ZhipuAiClient(api_key="YOUR API KEY")
    MAX_TOKEN = 40000  # Higher limit for GLM models

class PreprocessedDataset:
    """Handle loading and preprocessing of JSONL dataset files.
    
    Attributes:
        samples (List[Dict]): List of processed data samples
        
    Methods:
        _load_data: Internal method for loading and cleaning data
    """
    def __init__(self, jsonl_file: str):
        """Initialize dataset by loading from specified JSONL file.
        
        Args:
            jsonl_file: Path to JSONL data file
        """
        self.samples = self._load_data(jsonl_file)
        
    def _load_data(self, jsonl_file: str) -> List[Dict]:
        """Load and preprocess data from JSONL file.
        
        Args:
            jsonl_file: Relative path to JSONL file
            
        Returns:
            List[Dict]: Processed samples containing time series data
            
        Note:
            Handles character encoding issues in source data
            Maintains original sample count
        """
        # Resolve full file path
        current_file_path = Path(__file__)
        jsonl_file = current_file_path.parents[0] / "time_series_data" / jsonl_file
        
        samples = []
        with open(jsonl_file, 'r', encoding='latin-1') as f:
            for line in f:
                # Track progress
                print(f"Loading sample {len(samples)}")
                
                # Clean problematic characters in JSON
                cleaned_line = (line
                    .replace('¡ãC', '°C')
                    .replace('¡ã', '°')
                    .replace("“", '"')
                    .replace("”", '"'))
                
                # Parse JSON and extract time series
                data = json.loads(cleaned_line)
                timeData = data['time series']
                
                # Store sample data
                samples.append({
                    'data': timeData,
                })
        return samples

def convert_descriptor_to_report(data_v_seen, data_desp_i):
    """Generate multiple descriptive texts for multivariate battery time series.
    
    Args:
        data_v_seen: Observed voltage, charge capacity, discharge capacity, and current time series (3D array)
        data_desp_i: Dictionary of pre-generated descriptors
        
    Returns:
        Dict: Contains:
            - time_series: Flattened time series data
            - descript_c2: Descriptor variants
            
    Note:
        Uses parallel execution to generate multiple description variants
        Currently ignores predicted data parameters
    """
    instruction = ("There are time series collected from LIB cells.")

    time_series_list = [
        data_v_seen[:, i, j]  # Extract each individual time series
        for j in range(data_v_seen.shape[2])  # Loop through third dimension
        for i in range(data_v_seen.shape[1])  # Loop through second dimension
    ]

    # Prepare prompt templates and description keys
    SYSTEM_PROMPT_list = [SYSTEM_PROMPT2]
    template_list = [template2]
    des_pick = ["ind1"]
    des_json_list = []

    # Prepare tasks for parallel execution
    tasks = []
    for sys_prompt, template, des_key in zip(SYSTEM_PROMPT_list, template_list, des_pick):
        # Combine base instruction with specific description template
        user_content = instruction + data_desp_i[des_key]  
        sys_prompt_combined = sys_prompt + template
        tasks.append((sys_prompt_combined, user_content)) 
    
    # Execute all tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Process each task using process_task function
        results = executor.map(
            lambda task: prre.process_task(
                client,
                task[0],  # system_prompt
                task[1],  # user_content
                OPENAI_MODEL,  
                0.3,   # temperature
                MAX_TOKEN,     
                dataset_name = 'tju'
            ),
            tasks  
        )
        
        # Collect successful results
        des_json_list = [res for res in results if res is not None]

    result = {
        'original list': "\n".join(data_desp_i),
        'time series': time_series_list,
        'descript c2': des_json_list[0],
    }

    return result


def generate_report_from_dataset(data_v_win_train):

    with open(OUTPUT_DATASET, 'wt') as f:
        for i in range(TOTAL_CNT):
            print(f"the {i+1} th generation!")
            data_v_i = data_v_win_train[i]

            data_desp_i = generate_descriptor_tju(data_v_i, window=10)

            item = convert_descriptor_to_report(data_v_i, data_desp_i)
            
            cur_data = {
                "sample id": i+1,
                'original list': item['original list'],
                'time series': prre.timeseries_to_list(item['time series']),
                'descript c2': item['descript c2'],
            }
            f.write(json.dumps(cur_data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    
    # Set random seeds for reproducibility
    prre.set_all_seeds()
    TOTAL_CNT = 40 # Number of samples to generate
   
    current_file_path = Path(__file__)
    dir_path = current_file_path.parents[0] / "output_data"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DATASET = f'{dir_path}\\report_{OPENAI_MODEL}_tju_{TOTAL_CNT}_{timestamp[:8]}.jsonl'
    
    """
    read data from "expert-reference20251210.jsonl". Note that this file contains both raw time series and 
    attached expert-written reports. we only read the time-series part here. please refer to "evaluate_report_tju.py"
    for the performance assignment procedure using the expert-wrtten reports.
    """
    
    dataset = PreprocessedDataset("expert-reference20251210.jsonl")
    data_v_win_train, _ = prre.read_data_from_json(dataset, dataset_name = 'tju')

    print("info: TJU generation, model: ", OPENAI_MODEL)
    generate_report_from_dataset(data_v_win_train)
