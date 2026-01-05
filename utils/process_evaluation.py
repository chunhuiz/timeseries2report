# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 09:04:53 2025

@author: 14525
"""

from openai import OpenAI
import threading
import prompt_engineering as pe
import json
import time
import random
import numpy as np
import torch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import chardet
import logging

# Thread synchronization locks for shared resources
response_lock = threading.Lock()  # For API response handling
results_lock = threading.Lock()  # For results aggregation
all_results = []  # Stores evaluation results from all threads
failed_samples = []  # Tracks failed sample indices

# Load evaluation prompt template
evaluate_prompt = pe.evaluate_prompt()

# API configuration for evaluation LLM
OPENAI_API_KEY = "sk-ab07cd0c228c49aab7897b6e1e8382a5"
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_MODEL = 'qwen3-235b-a22b-instruct-2507'

# Initialize API client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

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

def read_jsonl(file_path, pass_sample=[]):
    """
    Robust JSONL file reader with encoding detection
    
    Args:
        file_path: Path to JSONL file
        pass_sample: List of line numbers to skip
        
    Returns:
        Tuple of (parsed_data, error_logs)
    """
    data = []
    errors = []
    
    try:
        # Detect file encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            
        encoding_detection = chardet.detect(raw_data)
        detected_encoding = encoding_detection['encoding'] or 'utf-8'
        
        # Read with detected encoding
        with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
            for line_number, line in enumerate(f, 1):
                if line_number in pass_sample:
                    continue
                    
                line = line.strip()
                if not line:
                    continue
                    
                # Clean common formatting issues
                line = line.replace("“", '"').replace("”", '"').replace("mean_std", "mean_and_std")
                
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    errors.append({
                        'line_number': line_number,
                        'line_content': line[:200],
                        'error': str(e)
                    })
        
        return data, errors
        
    except Exception as e:
        logging.error(f"File processing error: {e}")
        try:
            debug_file = file_path + '.debug.bin'
            with open(debug_file, 'wb') as df:
                df.write(raw_data)
            logging.info(f"Debug data saved to: {debug_file}")
        except:
            pass
        raise

def get_key_value(data, key_name, case_sensitive=False):
    """
    Flexible dictionary value lookup with substring matching
    
    Args:
        data: Dictionary to search
        key_name: Substring to match in keys
        case_sensitive: Whether to match case
        
    Returns:
        Matched value or alarming errors
    """
    search_str = key_name if case_sensitive else key_name.lower()
    
    for key in data:
        key_str = key if case_sensitive else str(key).lower()
        if search_str in key_str:
            value_to_return = data[key]
            if len(value_to_return) == 0:
                return 'none'
            else:
                return value_to_return
            
    raise KeyError('not found '+key_name)

def process_sample(sample, evaluate_prompt, client, sample_idx):

    """
    Processes a single evaluation sample with retry mechanism
    
    Args:
        sample: Tuple of (reference_texts, generated_texts)
        evaluate_prompt: Template for evaluation
        client: OpenAI client instance
        sample_idx: Index for tracking
        
    Returns:
        Tuple of (evaluation_result, sample_idx) or error dict
    """
    max_retries = 5
    retry_delay_base = 1.0
    
    for attempt in range(max_retries + 1):
        try:
            reference_texts, generated_texts = sample
            
            # Validate input lengths
            if len(reference_texts) != len(generated_texts):
                raise ValueError("Reference and generated text lists must have equal length!")
            
            # Format evaluation prompts
            response_list = []
            for i in range(len(reference_texts)):
                response = f"id: {i+1}: LLM response: {generated_texts[i][0]} fact: {reference_texts[i][0]}"
                response_list.append(response)
            
            prompt_all = evaluate_prompt + f"{response_list}"
            
            # Call evaluation LLM
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": prompt_all}],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=24192,
                extra_body={"enable_thinking": False}
            )
            
            # Parse and validate response
            response_result = response.choices[0].message.content.strip()
            response_list = json.loads(response_result)["evaluate results"]
            
            if len(response_list) != len(reference_texts):
                print(f"Result length mismatch (expected: {len(reference_texts)}, got: {len(response_list)})")
            else:
                return response_result, sample_idx
        
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                delay = retry_delay_base * (2 ** attempt)  # Exponential backoff
                time.sleep(delay)
            else:
                return {"error": last_error, "sample_idx": sample_idx}

        delay = retry_delay_base * (2 ** attempt)
        time.sleep(delay)
        
    # If all retries failed
    if attempt == max_retries:
        print("⚠️ Task failed after maximum retries.")
        return {"error": last_error, "sample_idx": sample_idx}

def parallel_evaluation(evaluate_list):
    # Parallel evaluation with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:

        with tqdm(total=len(evaluate_list), desc="processing progress", unit="sample") as pbar:
            
            # submit the task
            futures = {
                executor.submit(process_sample, sample, evaluate_prompt, client, idx+1): idx+1
                for idx, sample in enumerate(evaluate_list)
            }
            
            # finished tasks
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                
                with results_lock:
                    if isinstance(result, tuple):
                        
                        all_results.append(result)
                        pbar.set_postfix({"status": "OK", "current id": result[1]})
                    else:
                       
                        failed_samples.append(result["sample_idx"])
                        pbar.write(f"❗ sample {result['sample_idx']} failed: {result['error']}")
                        pbar.set_postfix({"status": "ERROR", "current id": result["sample_idx"]})
                    
                    pbar.update(1)  # 更新进度条
    return all_results