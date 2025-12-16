"""
Battery Time-Series Downstream Task -- anomaly detection using vision-based LLM

This script performs automated analysis of lithium-ion battery time-series data using LLM APIs.
Key functionalities:
1. Processes battery monitoring data (voltage, temperature, SOC)
2. Uses vision-based LLMs for charging/discharging management

"""

import os
from openai import OpenAI
import base64
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from zai import ZhipuAiClient
import re
import numpy as np
import pickle


OPENAI_MODEL = "qwen3-vl-plus"
# OPENAI_MODEL = "glm-4.5v"

if 'glm' in OPENAI_MODEL: 
    client = ZhipuAiClient(api_key="YOUR API KEY")
else:
    OPENAI_API_KEY = "YOUR API KEY"
    OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(
        api_key= OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def TimeSeriesImage(signal, cell_num=16):
    """
    Generates time-series visualization plots for each cell in a battery module and returns 
    them as base64-encoded image strings (thread-safe implementation).
    
    This function creates a composite plot for each battery cell containing three subplots:
    temperature, voltage, and State of Charge (SOC) over time. The plots are generated using
    a non-interactive backend to prevent GUI window creation, making it suitable for server
    environments. Each plot is serialized to a base64-encoded PNG image for web transmission.
    
    Parameters:
        signal (list): Input time-series data array structured as [cell0_temp, cell0_voltage, 
                       cell0_SOC, cell1_temp, ...]. The array length must be 3 * cell_num.
        cell_num (int): Number of battery cells in the module (default=16)
    
    Returns:
        list: Base64-encoded image strings prefixed with data URI scheme ("data:image/png;base64,...")
    
    Implementation Details:
        1. Configures plot parameters for consistent visualization
        2. Uses tight_layout to optimize subplot spacing
        3. Employs BytesIO buffer for in-memory image storage
        4. Explicitly closes figures to prevent memory leaks
        5. Encodes binary image data as UTF-8 compatible string
    """
     # Define plot metadata for axis labeling
    cols_name = ["temperture", "voltage", "SOC"]
    unit_name = ["°C", "V","%"]
    base64_images = []
    
    # Iterate through each battery cell in the module
    for i in range(cell_num):
        # Create figure with 3 vertically stacked subplots (non-interactive mode)
        fig, ax = plt.subplots(3, 1, figsize=(6, 4))
        
        # Generate subplots for each measurement type (temperature, voltage, SOC)
        for ii in range(3):
            ax[ii].plot(signal[i*3 + ii]) 
            ax[ii].set_title(f"{cols_name[ii]} of Cell {i+1}")
            ax[ii].set_xlabel("Time (min)")
            ax[ii].set_ylabel(f"Value ({unit_name[ii]})")
            
        fig.tight_layout() 
        
        # Serialize plot to in-memory binary buffer
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",      # Use PNG format for lossless compression
            dpi=100,           # 100 DPI resolution balance quality/file size
            bbox_inches="tight",  # Minimize whitespace around plot
            pad_inches=0.1     # Small padding to prevent label clipping
        )
        plt.close(fig)  # Critical: Release figure resources to prevent memory leaks
        
        # Convert binary image to base64-encoded string with data URI prefix
        buffer.seek(0)  # Reset buffer position for reading
        base64_str = base64.b64encode(buffer.read()).decode("utf-8")
        base64_images.append(f"data:image/png;base64,{base64_str}")
        
    return base64_images

def TimeSeriesImageCur(signal):
    """
    Generates a time-series visualization plot for the module current and returns it as a 
    base64-encoded image string.
    
    This function creates a single-plot visualization of current measurements over time
    for the entire battery module. The plot is optimized for web transmission through
    base64 encoding and includes appropriate axis labeling.
    
    Parameters:
        signal (list): Input time-series data array where the last element contains
                       current measurements for the entire module
        
    Returns:
        str: Base64-encoded image string prefixed with data URI scheme
    
    Implementation Details:
        - Uses compact figure size (6x2 inches) for efficient space utilization
        - Implements same memory management and encoding techniques as TimeSeriesImage
        - Assumes current data is the last element in the signal array
    """
    # Create single-plot figure for current visualization
    fig, ax = plt.subplots(figsize=(6, 2))  

    # Extract and plot module current data (last element in signal array)
    ax.plot(signal[-1])
    ax.set_title("current of module")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Value (A)")

    # Optimize layout and serialize to in-memory buffer
    fig.tight_layout()
    
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=100,
        bbox_inches="tight",
        pad_inches=0.1
    )
    plt.close(fig)  # Release figure resources

    # Encode as base64 string with data URI prefix
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def process_sample(idx, sample, instruct, client, OPENAI_MODEL):
    """
    Processes a single battery data sample through multimodal AI analysis (thread-safe).
    
    This function orchestrates the complete processing pipeline for a battery data sample:
    1. Generates visualization images for all cells and module current
    2. Constructs a multimodal prompt with textual instructions and image data
    3. Submits the prompt to an AI model via API
    4. Handles API responses and error conditions
    
    Parameters:
        idx (int): Sample index for tracking and result ordering
        sample (dict): Data sample dictionary containing 'data' key with time-series
        instruct (str): Primary textual instruction for the AI model
        client (object): Initialized API client instance (must be thread-safe)
        OPENAI_MODEL (str): Identifier for the AI model to use
    
    Returns:
        tuple: (index, analysis_result) where:
            index: Original sample index for ordering
            analysis_result: AI-generated analysis text or None on failure
    
    Implementation Details:
        - Combines 16 cell visualizations + 1 current visualization
        - Uses multimodal message format (text + image_url)
        - Implements robust error handling for API failures
        - Includes progress logging with truncated output
        - Maintains index association for parallel processing
    """
    try:
        # Generate visualization images for all cells and module current
        prompt = TimeSeriesImage(sample["data"])
        prompt_cur = TimeSeriesImageCur(sample["data"])
        prompt.append(prompt_cur)  # 添加电流图像
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user","content": [
                    {"type": "text", "text": instruct},
                    *[{"type": "image_url", "image_url": {"url": prompt[idx]}} for idx in range(17)],
                    {"type": "text", "text": ins2},
                    ]}],
            temperature=0.3
            )
        answers = completion.choices[0].message.content.strip()
        
         # Log processing completion with truncated output
        print(f"Sample {idx} processed successfully. Initial response: {analysis_result[:200]}...")
        
        return (idx, answers)
    
    except Exception as e:
        # Handle exceptions with detailed error reporting
        print(f"Processing failed for sample {idx}: {str(error)}")
        
        return (idx, None)
 

class PreprocessedTimeseries:

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
                timeData = data['time series']
                samples.append({
                    'data': timeData,
                })
        return samples
if __name__ == "__main__":

    instruct2 = ("""Please analyze the following Time-Series plots of battery module to generate precise, actionable forecasts for charge and discharge cycle management.
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

    dataset = PreprocessedTimeseries(dataset_name)

    # Prepare samples (testing with first sample repeated)
    sample_pick = dataset.samples
    
    # 5 independent API call for the 47th sample (you can also pick other samples)
    sample_pick = [sample_pick[46]] + [sample_pick[46]] + [sample_pick[46]] + [sample_pick[46]] + [sample_pick[46]]
    # print(len(sample_pick))
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
                instruct=instruct2,  
                client=client,  
                OPENAI_MODEL=OPENAI_MODEL
            )
            for i in range(len(sample_pick)) # 原循环范围：range(len(sample_pick)-1)
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
