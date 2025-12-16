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

# Local module imports
import prompt_engineering as pe
from get_descriptor import generate_descriptor_zju
import process_report as prre

# Load prompt templates from prompt_engineering module
# These define the different description styles we generate
SYSTEM_PROMPT, SYSTEM_PROMPT2, SYSTEM_PROMPT3, SYSTEM_PROMPT4, SYSTEM_PROMPT5 = pe.expert_prompt_for_report()

# List of supported LLM variants
llm_list = ['qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b-instruct-2507', 'deepseek-chat', 'glm-4.5']

# API configuration

OPENAI_MODEL = llm_list[4]  # Default to qwen3-14b

if 'qwen' in OPENAI_MODEL:
    OPENAI_API_KEY = "sk-ab07cd0c228c49aab7897b6e1e8382a5"
    OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    MAX_TOKEN = 8192  # Standard token limit
    
elif 'deepseek' in OPENAI_MODEL:
    OPENAI_API_KEY = "sk-4eaade62e88a40c09af5b4c139674d26"
    OPENAI_BASE_URL = "https://api.deepseek.com"
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    MAX_TOKEN = 8192  # Standard token limit

else:
    # client = ZhipuAiClient(api_key="13ab570616de4823b2777e9b74041126.Xkj0uHXFaGHGH1Fn")
    client = ZhipuAiClient(api_key="0eaf5ebe67a94c7bbd66d2373bc7d53d.xglPgSVXC2EZg0Rr")
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

def convert_descriptor_to_report(
    data_v_seen,
    data_c_seen
    ):
    """Generate multiple descriptive texts for multivariate battery time series.
    
    Args:
        data_v_seen: Observed temperature, voltage, SOC time series (3D array)
        data_c_seen: Observed current time series (1D array)
        
    Returns:
        Dict: Contains:
            - time_series: Flattened time series data
            - descript_c1-c5: Different description variants
            
    Note:
        Uses parallel execution to generate multiple description variants
        Currently ignores predicted data parameters
    """   
    instruction = ("There are time series collected from a LIB module consisting of 16 series-connected cells.")

    time_series_list = [
    data_v_seen[:, i, j] 
    for j in range(data_v_seen.shape[2])
    for i in range(data_v_seen.shape[1])]
    time_series_list.append(data_c_seen.flatten())
    
    data_desp_i = {"clu": f"{prre.TimeSeriesText(time_series_list, begin_idx=0, cell_num=16)}"+"the current of the battery module: "+f"{prre.delLineBreaks(time_series_list[-1])}.",
                "ind1": f"{prre.TimeSeriesText(time_series_list, begin_idx=0)}",
                "ind2": f"{prre.TimeSeriesText(time_series_list, begin_idx=4)}",
                "ind3": f"{prre.TimeSeriesText(time_series_list, begin_idx=8)}",
                "ind4": f"{prre.TimeSeriesText(time_series_list, begin_idx=12)}",
                }

    SYSTEM_PROMPT_list = [SYSTEM_PROMPT, SYSTEM_PROMPT2, SYSTEM_PROMPT3, SYSTEM_PROMPT4, SYSTEM_PROMPT5]
    des_pick = ["clu", "ind1", "ind2", "ind3", "ind4"]
    des_json_list = []
    
    # Prepare tasks for parallel execution
    tasks = []
    for sys_prompt, des_key in zip(SYSTEM_PROMPT_list, des_pick):
        # Combine base instruction with specific description template
        user_content = instruction + data_desp_i[des_key]
        tasks.append((sys_prompt, user_content))
    
    # Execute all tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Process each task using process_task function
        results = executor.map(
            lambda task: prre.process_task(
                client,
                task[0],  # system_prompt
                task[1],  # user_content
                OPENAI_MODEL,
                0.3,      # temperature
                MAX_TOKEN
            ),
            tasks
        )
        
        # Collect successful results
        des_json_list = [res for res in results if res is not None]

    # Package all results into dictionary
    result = {
        'time series': time_series_list,
        'descript c1': des_json_list[0],
        'descript c2': des_json_list[1],
        'descript c3': des_json_list[2],
        'descript c4': des_json_list[3],
        'descript c5': des_json_list[4],
    }

    return result


def generate_report_from_dataset(data_v_win_train, data_c_win_train):
    """Generate and save a complete QA dataset from time series windows.
    
    Args:
        data_v_win_train: List of temperature, voltage, SOC time series windows
        data_c_win_train: List of current time series windows
        
    Writes:
        JSONL file specified by OUTPUT_DATASET global variable
        
    Note:
        Processes TOTAL_CNT samples from input data
        Creates a time-series report as JSON object per sample in output file
    """

    # Open output file for writing
    with open(OUTPUT_DATASET, 'wt', encoding='utf-8') as f:
        # Process each requested sample
        for i in range(46,TOTAL_CNT):
            print(f"Generating sample {i+1} of {TOTAL_CNT}")
            
            # Get current window
            data_v_i = data_v_win_train[i]
            data_c_i = data_c_win_train[i]
            
            # Get all descriptive reports
            item = convert_descriptor_to_report(data_v_i, data_c_i)
            
            # Prepare output structure
            cur_data = {
                'sample id': i+1,
                'time series': prre.timeseries_to_list(item['time series']),
                'descript c1': item['descript c1'],
                'descript c2': item['descript c2'],
                'descript c3': item['descript c3'],
                'descript c4': item['descript c4'],
                'descript c5': item['descript c5'],
            }
            
            # Write as JSON line
            f.write(json.dumps(cur_data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    prre.set_all_seeds()

    TOTAL_CNT = 47
    dataset_name = "human-description20251210.jsonl"
    dataset = PreprocessedDataset(dataset_name)
    
    print(f"Report generation info: Model={OPENAI_MODEL}, File={dataset_name}")
    current_file_path = Path(__file__)
    dir_path = current_file_path.parents[0] / "output_data"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DATASET = f'{dir_path}\\report_ow{OPENAI_MODEL}_zju_{TOTAL_CNT}_{timestamp[:8]}.jsonl'
    
    # === Data Processing Pipeline ===
    
    # Load and reshape time series data
    data_v_win_train, data_c_win_train = prre.read_data_from_json(dataset)
    
    # Generate and save QA dataset
    generate_report_from_dataset(data_v_win_train, data_c_win_train)