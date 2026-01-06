# -*- coding: utf-8 -*-
"""
tools and functions for report generation procedure
"""

from typing import *
import json
import prompt_engineering as pe
from pathlib import Path
import copy
import time
import random
import numpy as np
import torch
import os

def set_all_seeds(seed=6):
    """Initialize all random number generators with specified seed for reproducibility.
    
    Args:
        seed (int): Seed value for random number generators (default: 6)
        
    Affects:
        - Python built-in random module
        - NumPy random state
        - PyTorch random states (CPU and CUDA)
        - Python hash randomization
        - CUDA benchmarking behavior
    """
    # Set Python's built-in random module seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # CUDA-specific seeds
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # Configure CUDA for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Set Python hash seed for consistent dictionary ordering
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Special configuration for PyTorch CUDA operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def read_data_from_json(dataset, dataset_name='zju'):
    """Extract and reshape voltage and current time series from dataset.
    
    Args:
        dataset: Loaded PreprocessedQADataset instance
        
    Returns:
        Tuple containing:
            - List of temperature, voltage, SOC windows (3D arrays)
            - List of current windows (2D arrays)
            
    Note:
        Assumes specific structure of 16 cells with 3 measurements each
        Performs transposition to get correct dimensionality
    """
    sample_pick = dataset.samples
    data_v_win = []
    data_c_win = []
    
    if dataset_name == 'zju':
        cell_num = 16
        var_num = 3
    elif dataset_name == 'mit':
        cell_num = 6
        var_num = 5
    elif dataset_name == 'tju':
        cell_num = 4
        var_num = 4

    for i in range(len(sample_pick)):
        sample_target = sample_pick[i]["data"]
        sample_per = []
        
        # Process each of the 16 cells
        for j in range(cell_num):
            # Stack measurements for each cell
            sample_sample_per = np.stack(sample_target[var_num*j:var_num*(j+1)])
            sample_per.append(sample_sample_per)
        
        # Stack all cells and transpose dimensions
        sample_per = np.stack(sample_per)
        sample_per = np.transpose(sample_per, (2, 1, 0))
        data_v_win.append(sample_per)
        
        # Process current data
        current = np.array(sample_target[-1]).reshape(-1,1)
        data_c_win.append(current)
    
    return data_v_win, data_c_win


def validate_result(result, dataset_name='zju'):
    
    """Validate the structure and content of LLM response JSON.
    
    Args:
        result (str): Raw JSON string response from LLM API
        
    Returns:
        bool: True if JSON is valid and contains required fields,
              False otherwise
              
    Note:
        Uses separate validation functions from prompt_engineering module
        for different description types ('clu' and 'ind' variants)
    """
    if dataset_name == 'mit':
        cell_num = 6
    else:
        cell_num = 4

    if not result:
        return False, []
    try:
        output_data = []
        # Attempt to parse JSON
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result
            
        if type(data) is list:
            data = {'cells_info': data}
            output_data = [json.dumps(data, ensure_ascii=False)]
        if "cells_info" in data.keys():
            description2 = pe.check_ind_des(data, cell_num=cell_num, dataset_name=dataset_name)
        else:
            description2 = pe.check_clu_des(data)
        return True, output_data
    except:
        return False, []

def timeseries_to_list(timeseries, digits: int=6, cp=True):
    """Convert numpy arrays to nested lists with rounding.

    Args:
        timeseries: Input time series data (numpy array or list)
        digits: Number of decimal places to round to
        cp: Whether to make a deep copy
        
    Returns:
        Nested list representation of the time series
    """
    if cp:
        result = copy.deepcopy(timeseries)
    else:
        result = timeseries
    if type(result) == np.ndarray:
        result = result.tolist()
    
    if type(result[0]) == float:
        for i in range(len(result)):
            result[i] = round(float(result[i]), 4)
    else:
        for i in range(len(result)):
            result[i] = timeseries_to_list(result[i], digits, cp=False)

    return result

def process_task(
    client,
    system_prompt,
    user_content,
    model,
    temperature,
    max_tokens,
    dataset_name = 'zju'
    ):
    """Execute a single LLM API call with robust error handling and retries.
    
    Args:
        system_prompt: System message defining LLM behavior
        user_content: User query/prompt content
        model: Identifier of the LLM model to use
        temperature: LLM temperature
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Optional[str]: Validated JSON response string if successful,
                      None if all retries failed
                      
    Implements:
        - Exponential backoff retry strategy
        - Response validation
        - Error logging
    """
    # Retry configuration
    max_retries = 3  # Maximum number of retry attempts
    retry_delay_base = 1.0  # Base delay in seconds for exponential backoff
    
    for attempt in range(max_retries + 1):
        try:
            # Construct message payload for API
            messages = [
                {
                    "role": "system", 
                    "content": f"(IMPORTANT: This is an isolated request.){system_prompt}"
                },
                {
                    "role": "user", 
                    "content": user_content
                }
            ]
            
            # Make API call with specified parameters
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                extra_body={"enable_thinking": False}
            )
            
            # Extract and clean response content
            result = response.choices[0].message.content.strip()
            
            # Validate response structure
            validate1, validate2 = validate_result(result, dataset_name=dataset_name)
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


def delLineBreaks(signal):
    """
    Remove line breaks and extra spaces from text
    
    Args:
        signal: Input text string
        
    Returns:
        Cleaned text without line breaks
    """
    text = str(signal)
    return text.replace('\n', ' ').replace('  ', ' ')

def TimeSeriesText(signal, begin_idx=0, cell_num=4, dataset_name = 'zju'):
    """
    Format battery time-series data into textual records
    
    Args:
        signal: List of time-series data arrays
        begin_idx: Starting index for cells
        cell_num: Number of cells to include
        
    Returns:
        Formatted text describing cell states
    """
    signalTextAll = []

    if dataset_name == 'zju':
        vara_num = 3
        for i in range(begin_idx, begin_idx+cell_num):
            signalText = (f"time series of Cell {i+1}:\n"
                        f"temperature: {delLineBreaks(signal[i*vara_num])}; "
                        f"voltage: {delLineBreaks(signal[i*vara_num+1])}; "
                        f"SOC: {delLineBreaks(signal[i*vara_num+2])}.")
            signalTextAll.append(signalText)

    elif dataset_name == 'mit':
        vara_num = 5
        for i in range(begin_idx, begin_idx+cell_num):
            signalText = (f"time series of Cell {i+1}:\n"
                        f"temperature: {delLineBreaks(signal[i*vara_num])}; "
                        f"voltage: {delLineBreaks(signal[i*vara_num+1])}; "
                        f"charge capacity: {delLineBreaks(signal[i*vara_num+2])};"
                        f"discharge capacity: {delLineBreaks(signal[i*vara_num+3])};"
                        f"current: {delLineBreaks(signal[i*vara_num+4])}.")
            signalTextAll.append(signalText)

    elif dataset_name == 'tju':
        vara_num = 4
        for i in range(begin_idx, begin_idx+cell_num):
            signalText = (f"time series of Cell {i+1}:\n"
                        f"voltage: {delLineBreaks(signal[i*vara_num])}; "
                        f"charge capacity: {delLineBreaks(signal[i*vara_num+1])}; "
                        f"discharge capacity: {delLineBreaks(signal[i*vara_num+2])};"
                        f"current: {delLineBreaks(signal[i*vara_num+3])};"
                        )
            signalTextAll.append(signalText)

    else:
        raise("no such dataset!")

    return "\n".join(signalTextAll)
