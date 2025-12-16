"""
Battery State of Charge (SOC) Prediction System using Text-Based LLMs

This system predicts the average SOC of battery modules by:
1. Processing time-series battery data (temperature, voltage, SOC) for 16 cells
2. Using large language models for SOC forecasting
3. Supporting parallel processing for efficient batch predictions

Key Features:
- Multi-threaded API calls to LLM services (OpenAI, ZhipuAI)
- Configurable for different LLM providers
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

# OPENAI_MODEL = "qwen3-14b"  
OPENAI_MODEL = "glm-4.5" 

# Initialize appropriate client based on model type
if 'glm' in OPENAI_MODEL: 
    client = ZhipuAiClient(api_key="YOUR API KEY")
else:
    OPENAI_API_KEY = "YOUR API KEY"
    OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
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
                     f"temperature (°C): {signal[i*3][:-10]}; "
                     f"voltage (V): {signal[i*3+1][:-10]}; "
                     f"SOC (%): {signal[i*3+2][:-10]}.")
        signalTextAll.append(signalText)
    signalCur = f"time series of current: {signal[-1][:-10]}"
    signalTextAll.append(signalCur)
    return "\n".join(signalTextAll)

def process_sample(idx, sample, instruct, client, OPENAI_MODEL):
    """
    Process a single battery sample through LLM for SOC prediction
    
    Args:
        idx: Sample index for tracking
        sample: Dictionary containing 'data' key with time-series
        instruct: Prediction prompt template
        client: Initialized LLM client
        OPENAI_MODEL: Model name to use
        
    Returns:
        Tuple of (index, LLM response) or (index, None) if failed
    """
    try:
        # Call LLM API with formatted battery data
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": instruct + TimeSeriesText(sample["data"])}],
            extra_body={"enable_thinking": False},  # Disable chain-of-thought for faster response
            temperature=0.7,  # Controls response randomness
        )
        
        answers = completion.choices[0].message.content.strip()
        print(f"Sample {idx} processed: {answers[:50]}...")  # Print truncated response
        return (idx, answers)
    
    except Exception as e:
        print(f"Sample {idx} failed: {str(e)}")
        return (idx, None)

class PreprocessedTimeseries:
    """Handles loading and preprocessing of battery time-series data from JSONL files"""
    def __init__(self, jsonl_file):
        """
        Initialize dataset from JSONL file
        
        Args:
            jsonl_file: Path to JSONL data file
        """
        self.samples = self._load_data(jsonl_file)
        
    def _load_data(self, jsonl_file):
        """
        Internal method to load and clean JSONL data
        
        Args:
            jsonl_file: Path to data file
            
        Returns:
            List of processed samples containing time-series data
        """
        current_file_path = Path(__file__)
        jsonl_file = current_file_path.parents[0] / "data" / jsonl_file
        samples = []
        
        with open(jsonl_file, 'r', encoding='latin-1') as f:
            for line in f:
                # Clean encoding artifacts (common in exported data)
                cleaned_line = line.replace('¡ãC', '°C').replace('¡ã', '°')
                data = json.loads(cleaned_line)
                
                # Store time series data
                samples.append({
                    'data': data['time series'],
                })
        return samples
    

if __name__ == "__main__":

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

    # pick the test dataset
    dataset_name = current_file_path.parents[0] / "output_data" / f"report_{OPENAI_MODEL}_zju_100_{test_set}.jsonl"

    # pick the time-series data from dataset by PreprocessedTimeseries function 
    dataset = PreprocessedTimeseries(dataset_name)
    sample_pick = dataset.samples
    
    sample_pick = dataset.samples
    # Configure parallel processing
    max_workers = 5  # Adjust based on API rate limits
    results = [None] * len(sample_pick)  # Pre-allocate results list
    
    # Process samples in parallel using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all samples for processing
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

    # Parse and store numerical results
    result_list = []
    for idx, answers in enumerate(results):
        if answers is not None:
            # Extract values between <ans> tags using regex
            match = re.search(r'<ans>(.*?)</ans>', answers)
            if match:
                result_list.append(match.group(1))  # Store predicted SOC
            else:
                result_list.append(-1)  # Error marker if no match
        else:
            print(f"Sample {idx} failed\n")
    
    # Reshape results for analysis
    result_list = np.array(result_list).reshape(1,-1)

    print("Model used:", OPENAI_MODEL)
