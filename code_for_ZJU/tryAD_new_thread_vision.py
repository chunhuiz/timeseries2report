"""
Battery Time-Series Downstream Task -- anomaly detection using vision-based LLM

This script performs automated analysis of lithium-ion battery time-series data using LLM APIs.
Key functionalities:
1. Processes battery monitoring data (voltage, temperature, SOC)
2. Uses AI models for anomaly detection

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

# Instruction template for anomaly detection
ins_att = """
Attention: 
1. you should precisely marked the monitoring conclusion, for instance: "<ans>abnormal</ans>", so that the user can find it easily.
2. There is certain inconsistency in the batteries, which is a normal phenomenon.
Similarly, only SIGNIFICANT difference observed from specific cell will be considered as an anomaly.
examples of normal individual operations: **Cells #1** have **SOC values** (5%) compared to the rest (6%): normal! these difference in SOC is slight! 
 **Cells #1** have **temperature values** (28 °C) compared to the rest (25 °C) IS normal! inconsistency like 3°C or less is actually normal.
 **Cells #1** have **voltage values** (3.4V) compared to the rest (3.3V) IS normal! inconsistency like 0.1V or less is actually normal.
 **Cells #1** have **SOC values** (20%) compared to the rest (25%) IS normal! inconsistency like 5% or less is actually normal.

"""

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
        
        return (idx, answers)  # 返回索引和结果（用于后续排序）
    
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

    instruct1 = ("""Please analyze the following Time Series of battery module (consist of 16 series-connected LIB cells), and determine whether there are any anomalies in this time period.
    Here are some basic informations of the involved LIB cells: Nominal Voltage: 3.2V, Operating Voltage: 2.5~3.65V, Charging/Discharging Protocal: Constant Power, Ambient Temperature: 25degC.
    These basic informations indicate the expected normal operating ranges of LIB cells, which is important for your fault diagnosis tasks.
    You should conduct monitoring task through cluster-level (cluster behavior) and individual-level (each specific LIB cell).
    If a noticable anomaly is detected, you should further select the most probable anomaly type from these options:
    A. Temperature Anomaly in specific individual cell
    B. Temperature Anomaly of whole LIB module
    C. Voltage Anomaly in specific individual cell
    D. Voltage Anomaly of whole LIB module.
    Also, you should identify the beginning sample and the ending sample of the detected anomaly, e.g., "1th~50th sample"
    The output format: you should return your answers in the format with "The operation is <ans>normal</ans>." or "The operation is <ans>abnormal</ans>." in the begining, then the related analysis should be given.
    If the anomaly is detected, you should output its fault types, beginning time and the endding time.
    """)
    
    current_file_path = Path(__file__)
    test_set = '20251210'
    
    # the path to save
    dir_path = current_file_path.parents[0] / f"downstream_result_{test_set}"

    # pick the test dataset
    dataset_name = current_file_path.parents[0] / "output_data" / f"report_qwen3-14b_zju_100_{test_set}.jsonl"

    # pick samples from the dataset, note that only time-seires data are picked by PreprocessedTimeseries function.
    dataset = PreprocessedTimeseries(dataset_name)
    
    input_texts = []
    true_values = []
    sample_pick = dataset.samples
    max_workers = 10 
    print(f"a total of {len(sample_pick)} samples")
    results = [None] * len(sample_pick)  

    # Parallel processing setup
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all sample tasks (each sample corresponds to one future)
        futures = [
            executor.submit(
                process_sample,
                idx=i,
                sample=sample_pick[i],
                instruct=instruct1,  # Original instruct2 from the loop (assumed defined)
                client=client,  # Original client from the loop (assumed thread-safe)
                OPENAI_MODEL=OPENAI_MODEL  # Original model name from the loop
            )
            for i in range(len(sample_pick))  # Original loop range
        ]
        
        # Get completed task results and store by index (maintains sample order)
        for future in as_completed(futures):
            idx, answers = future.result()  # Get returned (idx, answers)
            results[idx] = answers  # Store in results list by sample index
        
        # Print separator
        print("\n" + "-"*50 + " All samples processed " + "-"*50 + "\n")

    # Output all results (in sample order)
    result_list = []
    for idx, answers in enumerate(results):
        if answers is not None:
            match = re.search(r'<ans>(.*?)</ans>', answers)
            if match:
                first_value = match.group(1)
                result_list.append(first_value)
            else:
                result_list.append(-1)
        else:
            print(f"Sample {idx} processing failed\n")
    
    result_list = np.array(result_list)
    one_hot_result = []
    for i in range(len(result_list)):
        if result_list[i] == 'abnormal':
            one_hot_result.append(1)
        elif result_list[i] == 'normal':
            one_hot_result.append(0)
        else:
            one_hot_result.append(2)
    one_hot_result = np.array(one_hot_result).reshape(1,-1)
