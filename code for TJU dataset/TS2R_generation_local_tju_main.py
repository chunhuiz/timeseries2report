"""
Main script for generating descriptive report from battery time-series data using LLM APIs. (locally deployed LLM)

Key Functionality:
- Data loading and preprocessing of battery time-series data
- API integration with multiple LLM providers (Qwen, GLM, DeepSeek)
- Parallel generation of descriptive texts using thread pools
- Result validation and JSON dataset creation
- Reproducibility through seed setting

Dependencies:
- PyTorch, NumPy, OpenAI, scipy, tqdm, sklearn, etc.

Note: you need to run local_api.py script (in our utils folder) to deploy the local LLM. 
For simplier start, you can use remote API and run "TS2R_generation_tju_main.py"
"""

# Standard library imports
import random
import json
import os
import copy
import pdb
import time
import datetime
from typing import *
from pathlib import Path
import concurrent.futures

# Third-party imports
from tqdm import tqdm
import torch
import tiktoken
from openai import OpenAI
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import markdown
from zai import ZhipuAiClient
import requests
import sys

# Local module imports
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
SYSTEM_PROMPT, template2 = pe.expert_prompt_for_report_tju()

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


def get_completion(prompt, sys_prompt="", require_json=False):
    """
    Sends a completion request to the LLM API endpoint with error handling
    
    Args:
        prompt (str): The user input/question to send to the LLM
        sys_prompt (str, optional): System message to guide LLM behavior. Defaults to empty string.
        require_json (bool, optional): Whether to require JSON formatted response. Defaults to False.
    
    Returns:
        dict/str: The parsed API response content (JSON dict if require_json=True, else str)
    """
    headers = {'Content-Type': 'application/json'}  # Set JSON content type
    data = {
        "prompt": prompt,
        "sys_prompt": sys_prompt,
        "require_json": require_json  # Flag for structured JSON response
    }
    
    # Send POST request to LLM API endpoint
    response = requests.post(
        url='http://xxxx:6006',  # API endpoint URL, replace by your own API address
        headers=headers, 
        data=json.dumps(data)  # Serialize data to JSON string
    )
    
    # Return the 'response' field from JSON response
    return response.json()['response']

def process_task(
    system_prompt: str,
    user_content: str,
) -> Optional[str]:
    """
    Thread-safe task processor with retry logic for LLM API calls
    
    Implements:
    - Exponential backoff retry
    - Response validation
    - Error handling
    
    Args:
        system_prompt (str): System instruction for the LLM
        user_content (str): User query/content to process
        temperature (float): Creativity parameter (0-1)
        max_tokens (int): Max tokens to generate
        
    Returns:
        Optional[str]: Parsed response if successful, None after max retries
    """
    max_retries = 3  # Maximum retry attempts
    retry_delay_base = 1.0  # Base delay in seconds (exponential backoff)
    
    for attempt in range(max_retries + 1):
        try:
            # Format system prompt with isolation notice
            augmented_sys_prompt = f"(IMPORTANT: This is an isolated request.){system_prompt}"
            
            # Get API completion (forcing JSON response)
            result = get_completion(
                user_content,
                augmented_sys_prompt,
                require_json=True
            )
            
            validate1, validate2 = prre.validate_result(result, dataset_name='tju')
            if validate1:
                if len(validate2) == 0:
                    return result
                else:
                    return validate2[0]
                return result
            else:
                print(f"Validation failed (attempt {attempt+1}/{max_retries}): {result[:100]}...")
        
        except Exception as e:
            # Log API call failures
            print(f"API call failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {str(e)[:100]}")
        
        # Implement exponential backoff before retry
        if attempt < max_retries:
            delay = retry_delay_base * (2 ** attempt)
            time.sleep(delay)
            
    # If all retries failed
    if attempt == max_retries:
        print(f"⚠️ Task failed after maximum retries: system_prompt='{system_prompt[:20]}...'")
        return None



def convert_descriptor_to_report(data_v_seen, data_desp_i):
    
    instruction = ("There are time series collected from LIB cells.")

    time_series_list = [
    data_v_seen[:, i, j]  # 每个元素是形状 (200,) 的时间序列
    for j in range(data_v_seen.shape[2])
    for i in range(data_v_seen.shape[1])]
    
    SYSTEM_PROMPT_list = [SYSTEM_PROMPT]
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
            lambda task: process_task(
                task[0],  # system_prompt
                task[1],  # user_content
            ),
            tasks 
        )
        
        des_json_list = [res for res in results if res is not None]

    result = {
        'original list': "\n".join(data_desp_i),
        'time series': time_series_list,
        'descript c2': des_json_list[0],
    }

    return result


def generate_report_from_dataset(data_v_win_train):
    
    """Generate and save a complete report from time series windows.
    
    Args:
        data_v_win_train: List of temperature, voltage, etc. time series windows
        
    Writes:
        JSONL file specified by OUTPUT_DATASET global variable
        
    Note:
        Processes TOTAL_CNT samples from input data
        Creates a time-series report as JSON object per sample in output file
    """
    
    with open(OUTPUT_DATASET, 'wt') as f:
        for i in range(TOTAL_CNT):
            print(f"the {i+1} th generation!")
            data_v_i = data_v_win_train[i]
            
            data_desp_i = generate_descriptor_tju(data_v_i, window=10)

        
            item = convert_descriptor_to_report(data_v_i, data_desp_i)
            
            cur_data = {
                'sample id': i+1,
                'original list': item['original list'],
                'time series': prre.timeseries_to_list(item['time series']),
                'descript c2': item['descript c2'],
            }
            f.write(json.dumps(cur_data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    prre.set_all_seeds()  
    TOTAL_CNT = 40
   
    current_file_path = Path(__file__)
    dir_path = current_file_path.parents[0] / "output_data"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DATASET = f'{dir_path}\\report_llama3_tju_{TOTAL_CNT}_{timestamp[:8]}.jsonl'
    
    """
    read data from "expert-reference20251210.jsonl". Note that this file contains both raw time series and 
    attached expert-written reports. we only read the time-series part here. please refer to "evaluate_report_tju.py"
    for the performance assignment procedure using the expert-wrtten reports.
    """
    
    dataset = PreprocessedDataset("expert-reference20251210.jsonl")
    data_v_win_train, _ = prre.read_data_from_json(dataset, dataset_name = 'tju')
    print("info: TJU generation, model: llama3")
    generate_report_from_dataset(data_v_win_train)