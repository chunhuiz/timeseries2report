"""
Battery Time-Series Downstream Task -- Charge/Discharge Management System

This system analyzes battery time-series data through text-based LLMs.
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

# Configuration for LLM API
OPENAI_API_KEY = "sk-e76205eae508447dbe165ad68b158858"
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_MODEL = "qwen3-14b"  # Default model (can be changed to 'glm-4.5' etc.)

# Initialize appropriate client based on model type
if 'glm' in OPENAI_MODEL: 
    client = ZhipuAiClient(api_key="13ab570616de4823b2777e9b74041126.Xkj0uHXFaGHGH1Fn")
else:
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
                     f"temperature (°C): {signal[i*3]}; "
                     f"voltage (V): {signal[i*3+1]}; "
                     f"SOC (%): {signal[i*3+2]}.")
        signalTextAll.append(signalText)
    signalCur = f"time series of current: {signal[-1]}"
    signalTextAll.append(signalCur)
    return "\n".join(signalTextAll)

def process_sample(idx, sample, instruct, client, OPENAI_MODEL):
    """
    Processes a single battery sample through LLM API (thread worker function)
    
    Args:
        idx: Sample index for tracking
        sample: Dictionary containing 'data' with time-series measurements
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
                {"role": "user", "content": instruct + TimeSeriesText(sample["data"])},
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
        """Loads and cleans JSONL data from specified path"""
        current_file_path = Path(__file__)
        jsonl_file = current_file_path.parents[0] / "data" / jsonl_file
        samples = []
        
        with open(jsonl_file, 'r', encoding='latin-1') as f:
            for line in f:
                # Clean special characters in temperature units
                cleaned_line = line.replace('¡ãC', '°C').replace('¡ã', '°')
                data = json.loads(cleaned_line)
                timeData = data['time series']
                samples.append({
                    'data': timeData,  # Store raw time-series data
                })
        return samples

if __name__ == "__main__":
    # Primary instruction template for anomaly detection
    
    
    instruct3 = ("""Please analyze the following Time-Series of battery module to generate precise, actionable forecasts for charge and discharge cycle management.
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
    
    print("Model in use:", OPENAI_MODEL)
    
    dataset = PreprocessedQADataset("template_mana_qwen3-32b_xs_20251021.jsonl")
    sample_pick = dataset.samples
    
    sample_pick = [sample_pick[0]] + [sample_pick[0]] + [sample_pick[0]] + [sample_pick[0]] + [sample_pick[0]]
    
    # Parallel processing configuration
    max_workers = 10  # Number of concurrent API calls
    results = [None] * len(sample_pick)  # Pre-allocated results array
    
    # Process samples in parallel (testing with sample 45)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit analysis tasks
        futures = [
            executor.submit(
                process_sample,
                idx=i,
                sample=sample_pick[i],
                instruct=instruct3,
                client=client,
                OPENAI_MODEL=OPENAI_MODEL
            )
            for i in [45]  # Single sample for testing
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            idx, answers = future.result()
            results[idx] = answers

        print("\n" + "-"*50 + " All samples processed " + "-"*50 + "\n")